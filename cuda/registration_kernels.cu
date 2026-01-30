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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    const ColoredPoint& in_pt = input_points[idx];
    ColoredPoint& out_pt = output_points[idx];
    
    // Apply rotation and translation
    out_pt.x = pose.rotation[0] * in_pt.x + pose.rotation[1] * in_pt.y + pose.rotation[2] * in_pt.z + pose.translation[0];
    out_pt.y = pose.rotation[3] * in_pt.x + pose.rotation[4] * in_pt.y + pose.rotation[5] * in_pt.z + pose.translation[1];
    out_pt.z = pose.rotation[6] * in_pt.x + pose.rotation[7] * in_pt.y + pose.rotation[8] * in_pt.z + pose.translation[2];
    
    // Copy color
    out_pt.r = in_pt.r;
    out_pt.g = in_pt.g;
    out_pt.b = in_pt.b;
    
    // Transform normal
    out_pt.nx = pose.rotation[0] * in_pt.nx + pose.rotation[1] * in_pt.ny + pose.rotation[2] * in_pt.nz;
    out_pt.ny = pose.rotation[3] * in_pt.nx + pose.rotation[4] * in_pt.ny + pose.rotation[5] * in_pt.nz;
    out_pt.nz = pose.rotation[6] * in_pt.nx + pose.rotation[7] * in_pt.ny + pose.rotation[8] * in_pt.nz;
    
    out_pt.confidence = in_pt.confidence;
}

// Compute centroids for point clouds
__global__ void computeCentroidKernel(
    const ColoredPoint* points,
    float* centroid,
    int num_points
) {
    extern __shared__ float s_sum[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
    
    if (idx < num_points) {
        sum_x = points[idx].x;
        sum_y = points[idx].y;
        sum_z = points[idx].z;
    }
    
    // Reduce within block
