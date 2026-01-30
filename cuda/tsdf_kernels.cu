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
