#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
//#include "tiny_gltf.h"
#include <cuda_runtime.h>
#include "bvh.h"
#include "scene.h"



#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define PiOver2 1.57079632679489661923
#define PiOver4 0.785398163397448309616

__device__ bool environmentMapEnabled;
__device__ __constant__ bool d_environmentMapEnabled;


void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

struct PathTerminated
{
    __host__ __device__
        bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces == 0;
    }
};

struct MaterialIdComparator {
    __host__ __device__
        bool operator()(const PathSegment& a, const PathSegment& b) const {
        return a.materialId < b.materialId;
    }
};


//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
__device__ glm::vec3* d_environmentMap = NULL;
__device__ int d_envWidth;
__device__ int d_envHeight;
// BVH

// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;
	//currentScene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // --- ENVIRONMENT MAP ---
    if (environmentMapEnabled) {
        std::string hdrPath = "../scenes/" + scene->state.environmentMapFile;

        int envWidth, envHeight, envChannels;
        float* h_envPixels = stbi_loadf(hdrPath.c_str(), &envWidth, &envHeight, &envChannels, 0);

        if (h_envPixels == NULL) {
            std::cout << "Failed to load environment map: " << hdrPath << std::endl;
            std::cout << "STB failure reason: " << stbi_failure_reason() << std::endl;
            bool f = false;
            cudaMemcpyToSymbol(environmentMapEnabled, &f, sizeof(bool));
        }
        if (envChannels < 3) {
            std::cout << "Environment map must have at least 3 channels (RGB)." << std::endl;
            bool f = false;
            cudaMemcpyToSymbol(environmentMapEnabled, &f, sizeof(bool));
            stbi_image_free(h_envPixels);  // clean up what was loaded
            return;  // PREVENT CRASH
        }
        // std::cout << "HDR loaded: " << envWidth << "x" << envHeight << " channels=" << envChannels << std::endl;

        std::vector<glm::vec3> h_environmentMap;

        cudaMemcpyToSymbol(d_environmentMapEnabled, &environmentMapEnabled, sizeof(bool));



        // initialize memory
        if (h_envPixels != NULL) {
            // fill envBuffer from h_envPixels
            std::vector<glm::vec3> envBuffer(envWidth * envHeight);

            for (int i = 0; i < envWidth * envHeight; ++i) {
                envBuffer[i] = glm::vec3(
                    h_envPixels[i * envChannels + 0],
                    h_envPixels[i * envChannels + 1],
                    h_envPixels[i * envChannels + 2]
                );
            }

            // set env dims on device
            cudaMemcpyToSymbol(d_envWidth, &envWidth, sizeof(int));
            cudaMemcpyToSymbol(d_envHeight, &envHeight, sizeof(int));

            // allocate env map
            glm::vec3* d_env_dev_ptr = nullptr;

            // allocate device memory and copy host envBuffer into it
            size_t envBytes = envWidth * envHeight * sizeof(glm::vec3);
            cudaMalloc(&d_env_dev_ptr, envBytes);
            cudaMemcpy(d_env_dev_ptr, envBuffer.data(), envBytes, cudaMemcpyHostToDevice);

            // store device pointer into the device symbol so kernels can use it
            cudaMemcpyToSymbol(d_environmentMap, &d_env_dev_ptr, sizeof(glm::vec3*));

            //cudaMemcpy(d_environmentMap, envBuffer.data(), envWidth * envHeight * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            stbi_image_free(h_envPixels);
        }
    }
	// --- ENVIRONMENT MAP ---



    std::vector<glm::vec3> h_vertices;
    std::vector<glm::ivec3> h_indices;

    Geom h_geom;
    Mesh h_mesh;

    std::vector<Geom> h_geoms = scene->geoms;

    // Extract mesh data from scene
    for (Geom& g : h_geoms) {
        if (g.type == MESH) {
            h_vertices = std::vector<glm::vec3>(g.mesh.vertices, g.mesh.vertices + g.mesh.vertexCount);
            glm::vec3* d_vertices = nullptr;
            cudaMalloc(&d_vertices, h_vertices.size() * sizeof(glm::vec3));
            cudaMemcpy(d_vertices, h_vertices.data(), h_vertices.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);


            h_indices = std::vector<glm::ivec3>(g.mesh.indices, g.mesh.indices + g.mesh.indexCount);
            glm::ivec3* d_indices = nullptr;
            cudaMalloc(&d_indices, h_indices.size() * sizeof(glm::ivec3));
            cudaMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(glm::ivec3), cudaMemcpyHostToDevice);


            Mesh d_mesh;
            d_mesh.vertices = d_vertices;
            d_mesh.indices = d_indices;
            d_mesh.vertexCount = g.mesh.vertexCount;
            d_mesh.indexCount = g.mesh.indexCount;
            d_mesh.bbox = g.mesh.bbox;
            d_mesh.materialId = g.mesh.materialId;

            // Update geom's mesh with device pointers
            g.mesh = d_mesh;
        }
    }

    // Copy all Geoms to device
    cudaMalloc(&dev_geoms, h_geoms.size() * sizeof(Geom));
    
    cudaMemcpy(dev_geoms, h_geoms.data(), h_geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    // BVH

    //BuildBVH();
    



    // ---

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    // cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    // cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    if (dev_geoms != nullptr) {
        // Copy geoms back to host to get device pointers to mesh buffers
        std::vector<Geom> h_geoms(hst_scene->geoms.size());
        cudaMemcpy(h_geoms.data(), dev_geoms, hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyDeviceToHost);

        // Free each mesh’s device buffers
        for (const Geom& g : h_geoms) {
            if (g.type == MESH) {
                if (g.mesh.vertices != nullptr) {
                    cudaFree(g.mesh.vertices);
                }
                if (g.mesh.indices != nullptr) {
                    cudaFree(g.mesh.indices);
                }
            }
        }

        // Free the device geom array itself
        cudaFree(dev_geoms);
        dev_geoms = nullptr;
    }

    glm::vec3* h_env_dev_ptr = nullptr;
    cudaMemcpyFromSymbol(&h_env_dev_ptr, d_environmentMap, sizeof(glm::vec3*));
    if (h_env_dev_ptr != nullptr) {
        cudaFree(h_env_dev_ptr);
    }
    //FreeBVH();
    
    checkCUDAError("pathtraceFree");
}

__host__ __device__ glm::vec2 sampleUniformDiskConcentric(float u1, float u2) {
    glm::vec2 uOffset = 2.f * glm::vec2(u1, u2) - glm::vec2(1, 1);
    if (uOffset.x == 0 && uOffset.y == 0)
        return { 0, 0 };

    float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(std::cos(theta), std::sin(theta));
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= cam.resolution.x || y >= cam.resolution.y) return;

    cam.focalDistance = glm::length(cam.lookAt - cam.position);

    int index = x + (y * cam.resolution.x);
    PathSegment& segment = pathSegments[index];

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // TODO: implement antialiasing by jittering the ray
    float jitterX = u01(rng);
    float jitterY = u01(rng);
    float px = (float)x + jitterX;
    float py = (float)y + jitterY;

    glm::vec3 rayOrigin = cam.position;
    segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);

    
    glm::vec3 rayDirection = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * (px - (float)cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y * (py - (float)cam.resolution.y * 0.5f)
    );

    if (cam.lensRadius > 0.0f) {
		// Depth of field
        glm::vec2 pLens = cam.lensRadius * sampleUniformDiskConcentric(
            u01(rng),
            u01(rng)
        );


        float ft = cam.focalDistance / glm::dot(cam.view, rayDirection);
        glm::vec3 pFocus = cam.position + ft * rayDirection;
        rayOrigin += pLens.x * cam.right + pLens.y * cam.up;
        rayDirection = glm::normalize(pFocus - rayOrigin);
    }
    else
    {
		// Pinhole camera
        

    }
    segment.ray.origin = rayOrigin;
    segment.ray.direction = rayDirection;
    //segment.ray.origin += EPSILON * segment.ray.direction;

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
    
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?
            else if (geom.type == MESH)
            {
				t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}



__device__ glm::vec3 sampleEnvironmentMap(const glm::vec2& uv) {
    // Clamp UVs
    float u = fminf(fmaxf(uv.x, 0.0f), 1.0f);
    float v = fminf(fmaxf(uv.y, 0.0f), 1.0f);

    int x = static_cast<int>(u * (d_envWidth - 1));
    int y = static_cast<int>(v * (d_envHeight - 1));
    int idx = y * d_envWidth + x;

    return d_environmentMap[idx];
}


__host__ __device__
glm::vec2 sampleSphericalMap(const glm::vec3& dir) {
    float u = 0.5f + atan2(dir.z, dir.x) / (2.0f * PI);
    float v = 0.5f - asin(dir.y) / PI;
    return glm::vec2(u, v);
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].radiance = (materialColor * material.emittance);
				//pathSegments[idx].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                pathSegments[idx].radiance = glm::vec3(0.0f);
				scatterRay(pathSegments[idx], pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t , intersection.surfaceNormal, material, rng);
                //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                //pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                //pathSegments[idx].throughput *= u01(rng); // apply some noise because why not
                //pathSegments[idx].throughput *= material.throughput;
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            if (d_environmentMapEnabled) {
                glm::vec2 uv = sampleSphericalMap(pathSegments[idx].ray.direction);
                glm::vec3 environmentColor = sampleEnvironmentMap(uv);
                // Reinhard hdr color correction
                environmentColor = environmentColor / (environmentColor + glm::vec3(1.0f));
                pathSegments[idx].radiance = environmentColor;// * pathSegments[idx].throughput;
                pathSegments[idx].remainingBounces = -1;
            }
            else {
                pathSegments[idx].throughput = glm::vec3(0.0f);
				pathSegments[idx].radiance = glm::vec3(0.0f);
                pathSegments[idx].remainingBounces = -1;
            }
        }

    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.throughput * iterationPath.radiance;
    }
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        // ERROR HAPPENING HERE
        //std::vector<Geom> h_geoms_copy(hst_scene->geoms.size());
        //cudaMemcpy(h_geoms_copy.data(), dev_geoms, hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyDeviceToHost);

        //for (size_t i = 0; i < h_geoms_copy.size(); i++) {
        //    std::cout << "Geom " << i << ": type = " << h_geoms_copy[i].type
        //        << ", vertexCount = " << h_geoms_copy[i].mesh.vertexCount
        //        << ", indexCount = " << h_geoms_copy[i].mesh.indexCount
        //        << ", materialId = " << h_geoms_copy[i].materialid
        //        << std::endl;

        //    // You cannot directly print pointers usefully, but you can check if they are null:
        //    std::cout << "  vertices ptr: " << h_geoms_copy[i].mesh.vertices << std::endl;
        //    std::cout << "  indices ptr: " << h_geoms_copy[i].mesh.indices << std::endl;
        //}


        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            depth
        );
		checkCUDAError("shade material");
		cudaDeviceSynchronize();

		// Stream compact away terminated paths
        PathSegment* new_end = thrust::remove_if(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            PathTerminated()
        );
        num_paths = new_end - dev_paths;
        checkCUDAError("thrust::remove_if");

		// Sort paths by material ID
        if (num_paths >= 0 && num_paths <= pixelcount)
            thrust::sort_by_key(
                thrust::device,
                dev_paths,
                dev_paths + num_paths,
                dev_intersections,
                MaterialIdComparator()
            );
        checkCUDAError("thrust::sort");

        // TODO: should be based off stream compaction results.
        if (depth > traceDepth || num_paths == 0)
            iterationComplete = true;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
