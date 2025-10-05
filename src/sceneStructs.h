#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

extern __device__ bool environmentMapEnabled;

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};


struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

struct Mesh {
    glm::vec3* vertices;
    glm::ivec3* indices;
    AABB bbox;
    int materialId;
	int vertexCount;
	int indexCount;
};


struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

	Mesh mesh;
	AABB bbox; // for mesh and cube
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    // subsurface controls

    bool hasSubsurface = false;
    float thickness = 0.0f;
    float distortion = 0.2f;
    float glow = 6.0f;
    float bssrdfScale = 3.0f;
    float ambient = 0.0f;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
	float lensRadius = 0.05f;
    float focalDistance = 1.0f;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
    std::string environmentMapFile;
};

struct PathSegment
{
    Ray ray;
    int pixelIndex;
    int remainingBounces;
    int materialId;
    glm::vec3 throughput;
    glm::vec3 radiance;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};

struct BVHNode {
    AABB bbox;
    int left;   // index of left child node, -1 if leaf
    int right;  // index of right child node, -1 if leaf
    int start;  // start index of triangles in this node
    int count;  // number of triangles in this node
};