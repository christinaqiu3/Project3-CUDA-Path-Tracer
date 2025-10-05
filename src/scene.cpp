#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>


//#define TINYGLTF_NO_INCLUDE_STB_IMAGE
//#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"



using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else if (ext == ".gltf" || ext == ".glb")
    {
        loadFromGLTF(filename);
		return;
    }
	else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

AABB computeAABB(const std::vector<glm::vec3>& verts) {
    AABB box;
    box.min = glm::vec3(FLT_MAX);
    box.max = glm::vec3(-FLT_MAX);
    for (auto& v : verts) {
        box.min = glm::min(box.min, v);
        box.max = glm::max(box.max, v);
    }
    return box;
}


void Scene::loadFromGLTF(const std::string& gltfName)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    //bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, "../scenes/box.gltf");
    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, "../scenes/box.glb");


    std::vector<glm::vec3> vertices;
    std::vector<glm::ivec3> indices;

    if (model.meshes.empty()) {
        std::cerr << "GLTF Error: No meshes found in the model." << std::endl;
        exit(EXIT_FAILURE);
    }

    const tinygltf::Mesh& mesh = model.meshes[0]; // assuming first mesh
    for (const auto& primitive : mesh.primitives) {
        auto posAttr = primitive.attributes.find("POSITION");
        if (posAttr == primitive.attributes.end()) {
            std::cerr << "GLTF Error: 'POSITION' attribute missing in primitive." << std::endl;
            continue;
        }

        /*const float* buf = reinterpret_cast<const float*>(
            &model.buffers[model.bufferViews[primitive.attributes.find("POSITION")->second].buffer].data[0]);
        int stride = model.bufferViews[primitive.attributes.find("POSITION")->second].byteStride;
        int count = model.accessors[primitive.attributes.find("POSITION")->second].count;*/

        int posAccessorIndex = posAttr->second;
        const tinygltf::Accessor& posAccessor = model.accessors[posAccessorIndex];
        const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
        const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];

        int count = posAccessor.count;
        int stride = posBufferView.byteStride ? posBufferView.byteStride : sizeof(float) * 3; // fallback

        const unsigned char* dataPtr = posBuffer.data.data() + posBufferView.byteOffset + posAccessor.byteOffset;
        const float* buf = reinterpret_cast<const float*>(dataPtr);

        for (int i = 0; i < count; i++) {
            glm::vec3 v;
            v.x = buf[i * stride / sizeof(float) + 0];
            v.y = buf[i * stride / sizeof(float) + 1];
            v.z = buf[i * stride / sizeof(float) + 2];
            vertices.push_back(v);
        }

        const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
        const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
        const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

        const unsigned char* indexData = indexBuffer.data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset;

        for (size_t i = 0; i < indexAccessor.count; i += 3) {
            uint32_t i0, i1, i2;

            switch (indexAccessor.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                i0 = static_cast<uint32_t>(reinterpret_cast<const uint8_t*>(indexData)[i + 0]);
                i1 = static_cast<uint32_t>(reinterpret_cast<const uint8_t*>(indexData)[i + 1]);
                i2 = static_cast<uint32_t>(reinterpret_cast<const uint8_t*>(indexData)[i + 2]);
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                i0 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(indexData)[i + 0]);
                i1 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(indexData)[i + 1]);
                i2 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(indexData)[i + 2]);
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                i0 = static_cast<uint32_t>(reinterpret_cast<const uint32_t*>(indexData)[i + 0]);
                i1 = static_cast<uint32_t>(reinterpret_cast<const uint32_t*>(indexData)[i + 1]);
                i2 = static_cast<uint32_t>(reinterpret_cast<const uint32_t*>(indexData)[i + 2]);
                break;
            default:
                std::cerr << "Unsupported index component type!" << std::endl;
                return;
            }

            indices.push_back(glm::ivec3(i0, i1, i2));
        }

    }
	Geom newGeom;
	Mesh newMesh;
    newMesh.vertexCount = vertices.size();
    newMesh.indexCount = indices.size();

    newMesh.vertices = new glm::vec3[newMesh.vertexCount];
    std::copy(vertices.begin(), vertices.end(), newMesh.vertices);

    newMesh.indices = new glm::ivec3[newMesh.indexCount];
    std::copy(indices.begin(), indices.end(), newMesh.indices);

    Material defaultMat;
    defaultMat.color = glm::vec3(0.7f, 0.7f, 0.7f); // light gray
    defaultMat.specular.exponent = 0;
    defaultMat.specular.color = glm::vec3(0.0f);
    defaultMat.hasReflective = 0;
    defaultMat.hasRefractive = 0;
    defaultMat.indexOfRefraction = 1.0f;
    defaultMat.emittance = 0.0f;

    int matID = materials.size();
    materials.push_back(defaultMat);

	newGeom.type = MESH;
	newGeom.materialid = matID;
	newGeom.mesh = newMesh;

    Camera& camera = state.camera;
    RenderState& state = this->state;
	geoms.push_back(newGeom);

    // Hardcoded camera config
    camera.resolution = glm::ivec2(800, 600);
    float fovy = 45.0f;  // in degrees
    state.iterations = 100;
    state.traceDepth = 5;
    state.imageName = "rendered_output.png";

    camera.position = glm::vec3(0.0f, 5.0f, 10.5f);
    camera.lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f);

    // Camera basis
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view)); // re-orthogonalize

    // Calculate FOV
    float yscaled = tan(glm::radians(fovy));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = glm::degrees(atan(xscaled));
    camera.fov = glm::vec2(fovx, fovy);

    // Pixel length in world units
    camera.pixelLength = glm::vec2(
        2.0f * xscaled / static_cast<float>(camera.resolution.x),
        2.0f * yscaled / static_cast<float>(camera.resolution.y)
    );

    // Depth of field
    camera.lensRadius = 0.05f;
    camera.focalDistance = glm::length(camera.lookAt - camera.position);

    // Set up output buffer
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3(0));

    state.environmentMapFile = "../scenes/qwantani_dusk_2_puresky_4k.hdr";
    environmentMapEnabled = true;//!state.environmentMapFile.empty();

    /*std::cout << "Loaded " << vertices.size() << " vertices:\n";
    for (size_t i = 0; i < vertices.size(); ++i) {
        const glm::vec3& v = vertices[i];
        std::cout << "v[" << i << "]: " << v.x << ", " << v.y << ", " << v.z << "\n";
    }

    std::cout << "Loaded " << indices.size() << " triangles:\n";
    for (size_t i = 0; i < indices.size(); ++i) {
        const glm::ivec3& tri = indices[i];
        std::cout << "tri[" << i << "]: " << tri.x << ", " << tri.y << ", " << tri.z << "\n";
    }*/


}


void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);

        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.specular.color = glm::vec3(1.0f);
        }
        else if (p["TYPE"] == "Reflective")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.f;
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.f;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        else if (p["TYPE"] == "Subsurface")
        {

            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);

            //subsurface
			newMaterial.hasSubsurface = p["SUBSURFACE"];
			newMaterial.thickness = p["THICKNESS"];
			newMaterial.distortion = p["DISTORTION"];
			newMaterial.glow = p["GLOW"];
			newMaterial.bssrdfScale = p["BSSRDF_SCALE"];
			newMaterial.ambient = p["AMBIENT"];
		}
        else
        {
            std::cerr << "Unknown material type: " << p["TYPE"] << std::endl;
            continue;
		}
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
		else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else
        {
            newGeom.type = MESH;
        }

        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    if (data.contains("Environment") && data["Environment"].is_array() && !data["Environment"].empty()) {
        state.environmentMapFile = data["Environment"][0]["FILE"];
        environmentMapEnabled = true;//!state.environmentMapFile.empty();
    }
}
