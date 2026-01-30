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
    
    for (int i = 0; i < 8; i++) {
        int dx = (i & 1);
        int dy = (i & 2) >> 1;
        int dz = (i & 4) >> 2;
        
        int corner_idx = (vx + dx) + (vy + dy) * config.grid_dim_x + (vz + dz) * config.grid_dim_x * config.grid_dim_y;
        const TSDFVoxel& corner_voxel = voxel_grid[corner_idx];
        
        corners[i] = corner_voxel.tsdf;
        
        config.voxelToWorld(vx + dx, vy + dy, vz + dz, positions[i].x, positions[i].y, positions[i].z);
        
        colors[i] = make_uint3(corner_voxel.r, corner_voxel.g, corner_voxel.b);
    }
    
    // Interpolate edge vertices
    float3 edge_verts[12];
    uint3 edge_colors[12];
    
    for (int i = 0; i < 12; i++) {
        if (d_edgeTable[cube_index] & (1 << i)) {
            // Edge endpoints
            int v0 = i & 7;
            int v1 = (i + 1) & 7;
            if (i >= 8) {
                v0 = i - 8;
                v1 = v0 + 4;
            }
            
            // Linear interpolation
            float t = (iso_value - corners[v0]) / (corners[v1] - corners[v0]);
            t = fmaxf(0.0f, fminf(1.0f, t));
            
            edge_verts[i].x = positions[v0].x + t * (positions[v1].x - positions[v0].x);
            edge_verts[i].y = positions[v0].y + t * (positions[v1].y - positions[v0].y);
            edge_verts[i].z = positions[v0].z + t * (positions[v1].z - positions[v0].z);
            
            edge_colors[i].x = static_cast<uint8_t>(colors[v0].x + t * (colors[v1].x - colors[v0].x));
            edge_colors[i].y = static_cast<uint8_t>(colors[v0].y + t * (colors[v1].y - colors[v0].y));
            edge_colors[i].z = static_cast<uint8_t>(colors[v0].z + t * (colors[v1].z - colors[v0].z));
        }
    }
    
    // Generate triangles
    int base_vertex = voxel_offsets[voxel_idx];
    
    for (int i = 0; d_triTable[cube_index][i] != -1; i += 3) {
        int edge0 = d_triTable[cube_index][i];
        int edge1 = d_triTable[cube_index][i + 1];
        int edge2 = d_triTable[cube_index][i + 2];
        
        // Add vertices
        vertices[base_vertex + i].x = edge_verts[edge0].x;
        vertices[base_vertex + i].y = edge_verts[edge0].y;
        vertices[base_vertex + i].z = edge_verts[edge0].z;
        vertices[base_vertex + i].r = edge_colors[edge0].x;
        vertices[base_vertex + i].g = edge_colors[edge0].y;
        vertices[base_vertex + i].b = edge_colors[edge0].z;
        
        vertices[base_vertex + i + 1].x = edge_verts[edge1].x;
        vertices[base_vertex + i + 1].y = edge_verts[edge1].y;
        vertices[base_vertex + i + 1].z = edge_verts[edge1].z;
        vertices[base_vertex + i + 1].r = edge_colors[edge1].x;
        vertices[base_vertex + i + 1].g = edge_colors[edge1].y;
        vertices[base_vertex + i + 1].b = edge_colors[edge1].z;
        
        vertices[base_vertex + i + 2].x = edge_verts[edge2].x;
        vertices[base_vertex + i + 2].y = edge_verts[edge2].y;
        vertices[base_vertex + i + 2].z = edge_verts[edge2].z;
        vertices[base_vertex + i + 2].r = edge_colors[edge2].x;
        vertices[base_vertex + i + 2].g = edge_colors[edge2].y;
        vertices[base_vertex + i + 2].b = edge_colors[edge2].z;
        
        // Add triangle
        triangles[(base_vertex + i) / 3].v0 = base_vertex + i;
        triangles[(base_vertex + i) / 3].v1 = base_vertex + i + 1;
        triangles[(base_vertex + i) / 3].v2 = base_vertex + i + 2;
