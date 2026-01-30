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
