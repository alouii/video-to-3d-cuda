#include "cuda_utils.cuh"
#include "data_structures.h"
#include <cuda_runtime.h>

using namespace v3d;

// Kernel to convert depth map to point cloud
__global__ void depthToPointCloudKernel(
    const float* depth_map,
    const uint8_t* rgb_image,
    ColoredPoint* points,
    int* valid_count,
    const CameraIntrinsics intrinsics,
    const Pose pose,
    float min_depth,
    float max_depth,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float depth = depth_map[idx];
    
    // Filter invalid depths
    if (depth < min_depth || depth > max_depth || isnan(depth) || isinf(depth)) {
        return;
    }
    
    // Back-project to camera space
    float cam_x = (x - intrinsics.cx) * depth / intrinsics.fx;
    float cam_y = (y - intrinsics.cy) * depth / intrinsics.fy;
    float cam_z = depth;
    
    // Transform to world space
    float world_x = pose.rotation[0] * cam_x + pose.rotation[1] * cam_y + pose.rotation[2] * cam_z + pose.translation[0];
    float world_y = pose.rotation[3] * cam_x + pose.rotation[4] * cam_y + pose.rotation[5] * cam_z + pose.translation[1];
    float world_z = pose.rotation[6] * cam_x + pose.rotation[7] * cam_y + pose.rotation[8] * cam_z + pose.translation[2];
    
    // Get color
    int rgb_idx = idx * 3;
    uint8_t r = rgb_image[rgb_idx];
    uint8_t g = rgb_image[rgb_idx + 1];
    uint8_t b = rgb_image[rgb_idx + 2];
    
    // Atomic increment to get unique index
    int point_idx = atomicAdd(valid_count, 1);
    
    // Write point
    ColoredPoint& pt = points[point_idx];
    pt.x = world_x;
    pt.y = world_y;
    pt.z = world_z;
    pt.r = r;
    pt.g = g;
    pt.b = b;
    pt.confidence = 1.0f;
}

// Bilateral filtering for depth map smoothing
__global__ void bilateralFilterDepthKernel(
    const float* input_depth,
    float* output_depth,
    int width,
    int height,
    float sigma_space,
    float sigma_range
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int center_idx = y * width + x;
    float center_depth = input_depth[center_idx];
    
    if (center_depth <= 0.0f || isnan(center_depth)) {
        output_depth[center_idx] = center_depth;
        return;
    }
    
    int kernel_radius = 5;
    float sum_weights = 0.0f;
    float sum_values = 0.0f;
    
    for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
        for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            
