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
