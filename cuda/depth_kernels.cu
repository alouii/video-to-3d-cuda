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
    } else {
        aggregated_cost[idx] = cost_volume[idx];
    }
}

// Winner-takes-all disparity selection
__global__ void selectDisparityKernel(
    const float* aggregated_cost,
    float* disparity_map,
    int width,
    int height,
    int max_disparity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int base_idx = y * width * max_disparity + x * max_disparity;
    
    float min_cost = 1e9f;
    int best_d = 0;
    
    for (int d = 0; d < max_disparity; d++) {
        float cost = aggregated_cost[base_idx + d];
        if (cost < min_cost) {
            min_cost = cost;
            best_d = d;
        }
    }
    
    // Subpixel refinement using parabola fitting
    float disparity = static_cast<float>(best_d);
    if (best_d > 0 && best_d < max_disparity - 1) {
        float c_prev = aggregated_cost[base_idx + best_d - 1];
        float c_curr = aggregated_cost[base_idx + best_d];
        float c_next = aggregated_cost[base_idx + best_d + 1];
        
        float denom = 2.0f * (c_prev - 2.0f * c_curr + c_next);
        if (fabsf(denom) > 1e-6f) {
            float offset = (c_prev - c_next) / denom;
            disparity += offset;
        }
    }
    
    disparity_map[y * width + x] = disparity;
}

// Compute matching cost (Census transform + Hamming distance)
__global__ void censusTransformKernel(
    const uint8_t* image,
    uint32_t* census,
    int width,
    int height,
    int window_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < window_size || x >= width - window_size ||
        y < window_size || y >= height - window_size) {
        return;
    }
    
    int center_idx = y * width + x;
    uint8_t center_val = image[center_idx];
    
    uint32_t census_val = 0;
    int bit = 0;
    
    for (int dy = -window_size; dy <= window_size; dy++) {
        for (int dx = -window_size; dx <= window_size; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = x + dx;
            int ny = y + dy;
            int neighbor_idx = ny * width + nx;
            
            if (image[neighbor_idx] >= center_val) {
                census_val |= (1 << bit);
            }
            bit++;
        }
    }
    
    census[center_idx] = census_val;
}

__global__ void computeMatchingCostKernel(
    const uint32_t* left_census,
    const uint32_t* right_census,
    float* cost_volume,
    int width,
    int height,
    int max_disparity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int left_idx = y * width + x;
    uint32_t left_val = left_census[left_idx];
    
    for (int d = 0; d < max_disparity && x - d >= 0; d++) {
        int right_idx = y * width + (x - d);
        uint32_t right_val = right_census[right_idx];
        
        // Hamming distance
        uint32_t xor_val = left_val ^ right_val;
        int hamming = __popc(xor_val);  // Population count
        
        int cost_idx = y * width * max_disparity + x * max_disparity + d;
        cost_volume[cost_idx] = static_cast<float>(hamming);
    }
}

// Median filter for disparity refinement
__global__ void medianFilterKernel(
    const float* input_disparity,
    float* output_disparity,
    int width,
    int height,
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < kernel_size || x >= width - kernel_size ||
        y < kernel_size || y >= height - kernel_size) {
        return;
    }
    
    int center_idx = y * width + x;
    
    // Collect values in window
    float values[25];  // Max 5x5 window
    int count = 0;
    
    for (int dy = -kernel_size; dy <= kernel_size; dy++) {
        for (int dx = -kernel_size; dx <= kernel_size; dx++) {
            int idx = (y + dy) * width + (x + dx);
            float val = input_disparity[idx];
            if (val > 0.0f) {
                values[count++] = val;
            }
        }
    }
    
    if (count == 0) {
        output_disparity[center_idx] = 0.0f;
        return;
    }
    
    // Bubble sort (small arrays)
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - i - 1; j++) {
            if (values[j] > values[j + 1]) {
                float temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
    
    output_disparity[center_idx] = values[count / 2];
}

// Left-right consistency check
__global__ void lrConsistencyCheckKernel(
    const float* left_disparity,
    const float* right_disparity,
    float* output_disparity,
    int width,
    int height,
    float threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float d_left = left_disparity[idx];
    
    int x_right = x - static_cast<int>(d_left + 0.5f);
    if (x_right < 0 || x_right >= width) {
        output_disparity[idx] = 0.0f;
        return;
    }
    
    int right_idx = y * width + x_right;
    float d_right = right_disparity[right_idx];
    
    if (fabsf(d_left - d_right) > threshold) {
        output_disparity[idx] = 0.0f;
    } else {
        output_disparity[idx] = d_left;
    }
}

// Convert disparity to depth
__global__ void disparityToDepthKernel(
    const float* disparity_map,
    float* depth_map,
    int width,
    int height,
    float baseline,
    float focal_length
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float disparity = disparity_map[idx];
