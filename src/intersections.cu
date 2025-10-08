#include "intersections.h"
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "bvh.h"
#include <iostream>
#include <fstream>
#include <string>

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}


__host__ __device__
void solveQuadratic(float A, float B, float C, float& t0, float& t1) {
    float invA = 1.0f / A;
    B *= invA;
    C *= invA;

    float neg_halfB = -B * 0.5f;
    float u2 = neg_halfB * neg_halfB - C;
    float u = sqrt(u2);

    t0 = neg_halfB - u;
    t1 = neg_halfB + u;
}


__host__ __device__ bool intersectAABB(const Ray& ray, const AABB& box, float& tNear, float& tFar) {
    tNear = -FLT_MAX;
    tFar = FLT_MAX;
    for (int i = 0; i < 3; i++) {
        float t1 = (box.min[i] - ray.origin[i]) / ray.direction[i];
        float t2 = (box.max[i] - ray.origin[i]) / ray.direction[i];
        float tmin = fminf(t1, t2);
        float tmax = fmaxf(t1, t2);
        tNear = fmaxf(tNear, tmin);
        tFar = fminf(tFar, tmax);
        if (tNear > tFar) return false;
    }
    return true;
}

__host__ __device__ bool intersectRayTriangle(
    const glm::vec3& orig,
    const glm::vec3& dir,
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    float& t, glm::vec3& n)
{
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(dir, edge2);
    float a = glm::dot(edge1, h);

    if (fabs(a) < EPSILON) return false; // Ray parallel to triangle

    float f = 1.0f / a;
    glm::vec3 s = orig - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * glm::dot(edge2, q);
    if (t > EPSILON) {
        n = glm::normalize(glm::cross(edge1, edge2));
        return true;
    }

    return false;
}

//__device__ Tri* d_tris;
//__device__ unsigned int* d_triIdxs;
//__device__ BVHNode* d_bvhNodes;

__host__ __device__ void intersectBVH(Ray& ray, const int nodeIdx, float& closestT, glm::vec3& hitNormal) {
    const BVHNode& node = d_bvhNodes[nodeIdx];
	AABB bbox = node.bbox;
    float tFar;
    if (!intersectAABB(ray, bbox, closestT, tFar)) return;

    if (node.isLeaf()) {
        for (int i = 0; i < node.triCount; ++i) {
            const Tri& t = d_tris[d_triIdxs[node.firstTriIdx + i]];
            float tHit;
            glm::vec3 normal;
            if (intersectRayTriangle(ray.origin, ray.direction, t.vertex0, t.vertex1, t.vertex2, tHit, normal)) {
                if (tHit < closestT) {
                    closestT = tHit;
                    hitNormal = normal;
                }
            }
        }
    }
    else {
        intersectBVH(ray, node.leftNode, closestT, hitNormal);
        intersectBVH(ray, node.leftNode+1, closestT, hitNormal);
    }
}

__host__ __device__ float meshIntersectionTest(
    const Geom geom,
    const Ray r,
    glm::vec3& intersect,
    glm::vec3& normal,
    bool& outside
) {
    float t_min = FLT_MAX;
    bool hit = false;

#if USE_AABB_CULLING
    float t0, t1;
    if (!intersectAABB(r, geom.mesh.bbox, t0, t1)) {
        return -1.0f;
    }
#endif
	
    // Loop triangles
    for (int i = 0; i < geom.mesh.indexCount; i++) {
        glm::ivec3 tri = geom.mesh.indices[i];

        glm::vec3 v0 = geom.mesh.vertices[tri.x];
        glm::vec3 v1 = geom.mesh.vertices[tri.y];
        glm::vec3 v2 = geom.mesh.vertices[tri.z];

        glm::vec3 bary;
        float t;
        if (intersectRayTriangle(r.origin, r.direction, v0, v1, v2, t, bary)) {

            if (t > 0.0f && t < t_min) {
                t_min = t;
                intersect = r.origin + t * r.direction;
                normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                outside = glm::dot(r.direction, normal) < 0.0f;
                hit = true;
            }
        }
    }

    //Ray rayCopy = r; // Since intersectBVH updates ray.t
    //intersectBVH(rayCopy, 0, t_min, normal);

    //if (t_min < FLT_MAX) {
    //    intersect = rayCopy.origin + t_min * rayCopy.direction;
    //    outside = glm::dot(rayCopy.direction, normal) < 0.0f;
    //    hit = true;
    //}


    return hit ? t_min : -1.0f;
}



__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
    

    /*float radius = 0.5f;

    // Transform ray to object space
    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 pos = glm::vec3(0.0f); // Center of sphere in object space
    glm::vec3 diff = ro - pos;

    float a = 1.0f; // since rd is normalized
    float b = 2.0f * glm::dot(rd, diff);
    float c = glm::dot(diff, diff) - radius * radius;

    float t0 = -1.0f, t1 = -1.0f;
    solveQuadratic(a, b, c, t0, t1); // Po-Shen Loh solver

    // Find the closest valid t
    float t = -1.0f;
    if (t0 > 0.0f && t1 > 0.0f) {
        t = glm::min(t0, t1);
        outside = true;
    }
    else if (t0 > 0.0f || t1 > 0.0f) {
        t = glm::max(t0, t1);
        outside = false;
    }
    else {
        return -1.0f; // No valid intersection
    }

    // Compute intersection point and normal in object space
    glm::vec3 objSpaceIntersection = ro + t * rd;
    glm::vec3 objSpaceNormal = glm::normalize(objSpaceIntersection);

    // Transform back to world space
    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objSpaceIntersection, 1.0f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objSpaceNormal, 0.0f)));

    // if (!outside) normal = -normal;

    return glm::length(r.origin - intersectionPoint);*/
}