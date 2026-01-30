#include "cuda_utils.cuh"
#include "data_structures.h"
#include <cuda_runtime.h>

using namespace v3d;

// Marching cubes edge table and triangle table (abbreviated for space)
__constant__ int d_edgeTable[256];
__constant__ int d_triTable[256][16];

// Classify voxel for marching cubes
__global__ void classifyVoxelsKernel(
    const TSDFVoxel* voxel_grid,
    int* voxel_types,
    int* voxel_vertices,
    const VoxelGridConfig config,
    float iso_value
) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z * blockDim.z + threadIdx.z;
