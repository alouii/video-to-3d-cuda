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
