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
    
    if (x >= width || d >= max_disparity) return;
    
    int idx = y * width * max_disparity + x * max_disparity + d;
    
    // Path aggregation (horizontal scanline)
    if (x > 0) {
        int prev_idx = y * width * max_disparity + (x - 1) * max_disparity;
        
        float min_prev = 1e9f;
        for (int pd = 0; pd < max_disparity; pd++) {
            min_prev = fminf(min_prev, aggregated_cost[prev_idx + pd]);
        }
        
        float prev_same = aggregated_cost[prev_idx + d];
        float prev_adj = 1e9f;
        if (d > 0) prev_adj = fminf(prev_adj, aggregated_cost[prev_idx + d - 1] + p1);
        if (d < max_disparity - 1) prev_adj = fminf(prev_adj, aggregated_cost[prev_idx + d + 1] + p1);
        
        float path_cost = fminf(prev_same, fminf(prev_adj, min_prev + p2));
        aggregated_cost[idx] = cost_volume[idx] + path_cost - min_prev;
