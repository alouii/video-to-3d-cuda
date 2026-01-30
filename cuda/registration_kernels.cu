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
    sum_x = cuda_utils::blockReduceSum(sum_x);
    sum_y = cuda_utils::blockReduceSum(sum_y);
    sum_z = cuda_utils::blockReduceSum(sum_z);
    
    if (tid == 0) {
        atomicAdd(&centroid[0], sum_x);
        atomicAdd(&centroid[1], sum_y);
        atomicAdd(&centroid[2], sum_z);
    }
}

// Compute cross-covariance matrix H for SVD-based ICP
__global__ void computeCovarianceKernel(
    const ColoredPoint* source_points,
    const ColoredPoint* target_points,
    const int* correspondences,
    const float* source_centroid,
    const float* target_centroid,
    float* covariance,  // 3x3 matrix (9 elements)
    int num_points
) {
    extern __shared__ float s_cov[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    for (int i = tid; i < 9; i += blockDim.x) {
        s_cov[i] = 0.0f;
    }
    __syncthreads();
    
    if (idx < num_points) {
        int corr_idx = correspondences[idx];
        if (corr_idx >= 0) {
            const ColoredPoint& src = source_points[idx];
            const ColoredPoint& tgt = target_points[corr_idx];
            
            // Centered points
            float src_x = src.x - source_centroid[0];
            float src_y = src.y - source_centroid[1];
            float src_z = src.z - source_centroid[2];
            
            float tgt_x = tgt.x - target_centroid[0];
            float tgt_y = tgt.y - target_centroid[1];
            float tgt_z = tgt.z - target_centroid[2];
            
            // Compute outer product contributions
            atomicAdd(&s_cov[0], src_x * tgt_x);
            atomicAdd(&s_cov[1], src_x * tgt_y);
            atomicAdd(&s_cov[2], src_x * tgt_z);
            atomicAdd(&s_cov[3], src_y * tgt_x);
            atomicAdd(&s_cov[4], src_y * tgt_y);
            atomicAdd(&s_cov[5], src_y * tgt_z);
            atomicAdd(&s_cov[6], src_z * tgt_x);
            atomicAdd(&s_cov[7], src_z * tgt_y);
            atomicAdd(&s_cov[8], src_z * tgt_z);
        }
    }
    __syncthreads();
    
    // Write to global memory
    if (tid < 9) {
        atomicAdd(&covariance[tid], s_cov[tid]);
    }
}

// Compute alignment error
