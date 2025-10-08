#pragma once

#include "sceneStructs.h"
#include <vector>
#include <glm/glm.hpp>
#include <cstdint>

void BuildBVH();
void AllocBVH();
void FreeBVH();

extern std::vector<Tri> tri;
extern std::vector<unsigned int> triIdx;
extern std::vector<BVHNode> bvhNode;

extern __device__ Tri* d_tris;
extern __device__ unsigned int* d_triIdxs;
extern __device__ BVHNode* d_bvhNodes;