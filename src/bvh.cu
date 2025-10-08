#include "bvh.h"

__device__ Tri* d_tris = nullptr;
__device__ unsigned int* d_triIdxs = nullptr;
__device__ BVHNode* d_bvhNodes = nullptr;

