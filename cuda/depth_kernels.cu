#include "cuda_utils.cuh"
#include <cuda_runtime.h>

// Semi-Global Matching (SGM) cost aggregation
__global__ void sgmCostAggregationKernel(
    const float* cost_volume,
    float* aggregated_cost,
    int width,
    int height,
    int max_disparity,
    float p1,
    float p2
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    int d = threadIdx.z + blockIdx.z * blockDim.z;
    
