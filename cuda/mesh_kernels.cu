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
    
    if (vx >= config.grid_dim_x - 1 || vy >= config.grid_dim_y - 1 || vz >= config.grid_dim_z - 1) return;
    
    int voxel_idx = vx + vy * config.grid_dim_x + vz * config.grid_dim_x * config.grid_dim_y;
    
    // Get 8 corner values
    float corners[8];
    int cube_index = 0;
    
    for (int i = 0; i < 8; i++) {
        int dx = (i & 1);
        int dy = (i & 2) >> 1;
        int dz = (i & 4) >> 2;
        
        int corner_idx = (vx + dx) + (vy + dy) * config.grid_dim_x + (vz + dz) * config.grid_dim_x * config.grid_dim_y;
        corners[i] = voxel_grid[corner_idx].tsdf;
        
        if (corners[i] < iso_value) {
            cube_index |= (1 << i);
        }
    }
    
    voxel_types[voxel_idx] = cube_index;
    
    // Count number of vertices this voxel will generate
    int num_vertices = 0;
    if (cube_index != 0 && cube_index != 255) {
        for (int i = 0; d_triTable[cube_index][i] != -1; i += 3) {
            num_vertices += 3;
        }
    }
    
    voxel_vertices[voxel_idx] = num_vertices;
}

// Generate mesh vertices and triangles
__global__ void generateMeshKernel(
    const TSDFVoxel* voxel_grid,
    const int* voxel_types,
    const int* voxel_offsets,
    ColoredPoint* vertices,
    Triangle* triangles,
    const VoxelGridConfig config,
    float iso_value
) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (vx >= config.grid_dim_x - 1 || vy >= config.grid_dim_y - 1 || vz >= config.grid_dim_z - 1) return;
    
    int voxel_idx = vx + vy * config.grid_dim_x + vz * config.grid_dim_x * config.grid_dim_y;
    int cube_index = voxel_types[voxel_idx];
    
    if (cube_index == 0 || cube_index == 255) return;
    
    // Get corner positions and values
    float corners[8];
    float3 positions[8];
    uint3 colors[8];
    
