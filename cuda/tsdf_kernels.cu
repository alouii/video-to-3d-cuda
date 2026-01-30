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
    
