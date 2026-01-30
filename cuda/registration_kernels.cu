#include "cuda_utils.cuh"
#include "data_structures.h"
#include <cuda_runtime.h>

using namespace v3d;

// Find correspondences between two point clouds
__global__ void findCorrespondencesKernel(
    const ColoredPoint* source_points,
    const ColoredPoint* target_points,
    int* correspondences,
    float* distances,
    int num_source,
    int num_target,
    float max_distance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_source) return;
    
    const ColoredPoint& src = source_points[idx];
    
    float min_dist = max_distance;
    int best_match = -1;
    
    // Brute force nearest neighbor (in practice use KD-tree)
    for (int i = 0; i < num_target; i++) {
        const ColoredPoint& tgt = target_points[i];
        
        float dx = src.x - tgt.x;
        float dy = src.y - tgt.y;
        float dz = src.z - tgt.z;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        
        if (dist < min_dist) {
            min_dist = dist;
            best_match = i;
        }
    }
    
    correspondences[idx] = best_match;
    distances[idx] = min_dist;
}

// Transform point cloud with pose
__global__ void transformPointCloudKernel(
    const ColoredPoint* input_points,
    ColoredPoint* output_points,
    int num_points,
    const Pose pose
) {
