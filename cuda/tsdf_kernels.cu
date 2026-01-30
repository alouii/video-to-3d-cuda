#include "cuda_utils.cuh"
#include "data_structures.h"
#include <cuda_runtime.h>

using namespace v3d;

// TSDF integration kernel
__global__ void integrateTSDFKernel(
    TSDFVoxel* voxel_grid,
    const float* depth_map,
    const uint8_t* rgb_image,
    const CameraIntrinsics intrinsics,
    const Pose pose,
    const VoxelGridConfig config,
    float truncation_distance,
    float max_weight,
    int width,
    int height
) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (vx >= config.grid_dim_x || vy >= config.grid_dim_y || vz >= config.grid_dim_z) return;
    
    // Compute world position of voxel center
    float world_x = config.min_x + (vx + 0.5f) * config.voxel_size;
    float world_y = config.min_y + (vy + 0.5f) * config.voxel_size;
    float world_z = config.min_z + (vz + 0.5f) * config.voxel_size;
    
    // Transform to camera space (inverse pose)
    // R^T * (p - t)
    float px = world_x - pose.translation[0];
    float py = world_y - pose.translation[1];
    float pz = world_z - pose.translation[2];
    
    float cam_x = pose.rotation[0] * px + pose.rotation[3] * py + pose.rotation[6] * pz;
    float cam_y = pose.rotation[1] * px + pose.rotation[4] * py + pose.rotation[7] * pz;
    float cam_z = pose.rotation[2] * px + pose.rotation[5] * py + pose.rotation[8] * pz;
    
    // Check if behind camera
    if (cam_z <= 0.0f) return;
    
    // Project to image plane
    int u = static_cast<int>(intrinsics.fx * cam_x / cam_z + intrinsics.cx);
    int v = static_cast<int>(intrinsics.fy * cam_y / cam_z + intrinsics.cy);
    
    // Check if inside image
    if (u < 0 || u >= width || v < 0 || v >= height) return;
    
    // Get depth measurement
    int pixel_idx = v * width + u;
    float measured_depth = depth_map[pixel_idx];
    
    // Check valid depth
    if (measured_depth <= 0.0f || isnan(measured_depth) || isinf(measured_depth)) return;
    
    // Compute SDF value
    float sdf = measured_depth - cam_z;
    
    // Truncate
    if (sdf < -truncation_distance) return;
    
    float tsdf = fminf(1.0f, sdf / truncation_distance);
    
    // Get voxel
    int voxel_idx = vx + vy * config.grid_dim_x + vz * config.grid_dim_x * config.grid_dim_y;
    TSDFVoxel& voxel = voxel_grid[voxel_idx];
    
    // Running average
    float old_weight = voxel.weight;
    float new_weight = fminf(old_weight + 1.0f, max_weight);
    
    voxel.tsdf = (voxel.tsdf * old_weight + tsdf) / new_weight;
    voxel.weight = new_weight;
    
    // Update color
    if (old_weight == 0.0f) {
        int rgb_idx = pixel_idx * 3;
        voxel.r = rgb_image[rgb_idx];
        voxel.g = rgb_image[rgb_idx + 1];
        voxel.b = rgb_image[rgb_idx + 2];
    } else {
        // Weighted average
        int rgb_idx = pixel_idx * 3;
        float inv_new_weight = 1.0f / new_weight;
        voxel.r = static_cast<uint8_t>((voxel.r * old_weight + rgb_image[rgb_idx]) * inv_new_weight);
        voxel.g = static_cast<uint8_t>((voxel.g * old_weight + rgb_image[rgb_idx + 1]) * inv_new_weight);
        voxel.b = static_cast<uint8_t>((voxel.b * old_weight + rgb_image[rgb_idx + 2]) * inv_new_weight);
    }
}

// Extract surface points from TSDF
__global__ void extractSurfacePointsKernel(
    const TSDFVoxel* voxel_grid,
    ColoredPoint* points,
    int* point_count,
    const VoxelGridConfig config,
    float weight_threshold
) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (vx >= config.grid_dim_x - 1 || vy >= config.grid_dim_y - 1 || vz >= config.grid_dim_z - 1) return;
    
    int voxel_idx = vx + vy * config.grid_dim_x + vz * config.grid_dim_x * config.grid_dim_y;
    const TSDFVoxel& voxel = voxel_grid[voxel_idx];
    
    // Check if this voxel contains a zero crossing
    if (voxel.weight < weight_threshold) return;
    if (fabsf(voxel.tsdf) > 0.5f) return;
    
    // Check neighbors for zero crossing
    bool has_zero_crossing = false;
    
    int neighbors[6] = {
        voxel_idx + 1,  // x+1
