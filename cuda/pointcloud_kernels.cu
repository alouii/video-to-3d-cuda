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
            
            int neighbor_idx = ny * width + nx;
            float neighbor_depth = input_depth[neighbor_idx];
            
            if (neighbor_depth <= 0.0f || isnan(neighbor_depth)) continue;
            
            // Spatial weight
            float spatial_dist = sqrtf(dx * dx + dy * dy);
            float spatial_weight = expf(-(spatial_dist * spatial_dist) / (2.0f * sigma_space * sigma_space));
            
            // Range weight
            float range_dist = fabsf(neighbor_depth - center_depth);
            float range_weight = expf(-(range_dist * range_dist) / (2.0f * sigma_range * sigma_range));
            
            float weight = spatial_weight * range_weight;
            sum_weights += weight;
            sum_values += weight * neighbor_depth;
        }
    }
    
    output_depth[center_idx] = (sum_weights > 0.0f) ? (sum_values / sum_weights) : center_depth;
}

// Compute normals from depth map using cross product
__global__ void computeNormalsKernel(
    const float* depth_map,
    ColoredPoint* points,
    const CameraIntrinsics intrinsics,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width - 1 || y >= height - 1 || x == 0 || y == 0) return;
    
    int idx = y * width + x;
    float depth_center = depth_map[idx];
    
    if (depth_center <= 0.0f || isnan(depth_center)) return;
    
    // Get neighboring depths
    float depth_right = depth_map[y * width + (x + 1)];
    float depth_down = depth_map[(y + 1) * width + x];
    
    if (depth_right <= 0.0f || depth_down <= 0.0f) return;
    
    // Compute 3D positions
    float cx = (x - intrinsics.cx) * depth_center / intrinsics.fx;
    float cy = (y - intrinsics.cy) * depth_center / intrinsics.fy;
    float cz = depth_center;
    
    float rx = (x + 1 - intrinsics.cx) * depth_right / intrinsics.fx;
    float ry = (y - intrinsics.cy) * depth_right / intrinsics.fy;
    float rz = depth_right;
    
    float dx = (x - intrinsics.cx) * depth_down / intrinsics.fx;
    float dy = (y + 1 - intrinsics.cy) * depth_down / intrinsics.fy;
    float dz = depth_down;
    
    // Compute tangent vectors
    float tx = rx - cx;
    float ty = ry - cy;
    float tz = rz - cz;
    
    float bx = dx - cx;
    float by = dy - cy;
    float bz = dz - cz;
    
    // Cross product for normal
    float nx = ty * bz - tz * by;
    float ny = tz * bx - tx * bz;
    float nz = tx * by - ty * bx;
    
    // Normalize
    float length = sqrtf(nx * nx + ny * ny + nz * nz);
    if (length > 0.0f) {
        nx /= length;
        ny /= length;
        nz /= length;
    }
    
    // Ensure normal points towards camera
    if (nz > 0) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }
    
    // Store in point cloud
    points[idx].nx = nx;
    points[idx].ny = ny;
    points[idx].nz = nz;
}

// Statistical outlier removal
__global__ void statisticalOutlierRemovalKernel(
    const ColoredPoint* input_points,
    ColoredPoint* output_points,
    int* valid_mask,
    int num_points,
    int k_neighbors,
    float std_multiplier
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    const ColoredPoint& point = input_points[idx];
    
    // Compute mean distance to k nearest neighbors (simplified)
    float sum_distances = 0.0f;
    int count = 0;
    
    // Sample neighbors (in practice, use KD-tree)
    for (int i = max(0, idx - k_neighbors); i < min(num_points, idx + k_neighbors); i++) {
        if (i == idx) continue;
        
        const ColoredPoint& neighbor = input_points[i];
        float dx = point.x - neighbor.x;
        float dy = point.y - neighbor.y;
        float dz = point.z - neighbor.z;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        
        sum_distances += dist;
        count++;
    }
    
    if (count == 0) {
        valid_mask[idx] = 0;
        return;
    }
    
    float mean_distance = sum_distances / count;
    
    // Compute variance (simplified, in practice do two-pass)
    float variance = 0.0f;
    for (int i = max(0, idx - k_neighbors); i < min(num_points, idx + k_neighbors); i++) {
        if (i == idx) continue;
        
        const ColoredPoint& neighbor = input_points[i];
        float dx = point.x - neighbor.x;
        float dy = point.y - neighbor.y;
        float dz = point.z - neighbor.z;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        
        float diff = dist - mean_distance;
        variance += diff * diff;
    }
    variance /= count;
    
    float std_dev = sqrtf(variance);
    float threshold = mean_distance + std_multiplier * std_dev;
    
    // Check if outlier
    valid_mask[idx] = (mean_distance < threshold) ? 1 : 0;
    
    if (valid_mask[idx]) {
        output_points[idx] = point;
    }
}

// Voxel downsampling
__global__ void voxelDownsampleKernel(
    const ColoredPoint* input_points,
    ColoredPoint* output_points,
    int* voxel_counts,
    int num_points,
    float voxel_size,
    float min_x,
    float min_y,
    float min_z,
    int grid_dim_x,
    int grid_dim_y,
    int grid_dim_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    const ColoredPoint& point = input_points[idx];
    
    // Compute voxel indices
    int vx = static_cast<int>((point.x - min_x) / voxel_size);
    int vy = static_cast<int>((point.y - min_y) / voxel_size);
    int vz = static_cast<int>((point.z - min_z) / voxel_size);
    
    if (vx < 0 || vx >= grid_dim_x || vy < 0 || vy >= grid_dim_y || vz < 0 || vz >= grid_dim_z) {
        return;
    }
    
    int voxel_idx = vx + vy * grid_dim_x + vz * grid_dim_x * grid_dim_y;
    
    // Atomic add to accumulate points in voxel
    int count = atomicAdd(&voxel_counts[voxel_idx], 1);
    
    if (count == 0) {
        // First point in this voxel, store it
        output_points[voxel_idx] = point;
    } else {
        // Average with existing points
        ColoredPoint& avg_point = output_points[voxel_idx];
        float inv_count = 1.0f / (count + 1);
        
        avg_point.x = (avg_point.x * count + point.x) * inv_count;
        avg_point.y = (avg_point.y * count + point.y) * inv_count;
        avg_point.z = (avg_point.z * count + point.z) * inv_count;
        
        // Average colors (approximate)
        avg_point.r = static_cast<uint8_t>((avg_point.r * count + point.r) * inv_count);
        avg_point.g = static_cast<uint8_t>((avg_point.g * count + point.g) * inv_count);
        avg_point.b = static_cast<uint8_t>((avg_point.b * count + point.b) * inv_count);
    }
}

// C++ interface functions
extern "C" {

void launchDepthToPointCloud(
    const float* d_depth_map,
    const uint8_t* d_rgb_image,
    ColoredPoint* d_points,
    int* d_valid_count,
    const CameraIntrinsics& intrinsics,
    const Pose& pose,
    float min_depth,
    float max_depth,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    depthToPointCloudKernel<<<grid, block, 0, stream>>>(
        d_depth_map, d_rgb_image, d_points, d_valid_count,
        intrinsics, pose, min_depth, max_depth, width, height
    );
    
    CUDA_CHECK_LAST_ERROR();
}

void launchBilateralFilterDepth(
    const float* d_input_depth,
    float* d_output_depth,
    int width,
    int height,
    float sigma_space,
    float sigma_range,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    bilateralFilterDepthKernel<<<grid, block, 0, stream>>>(
        d_input_depth, d_output_depth, width, height, sigma_space, sigma_range
    );
    
    CUDA_CHECK_LAST_ERROR();
}

void launchComputeNormals(
    const float* d_depth_map,
    ColoredPoint* d_points,
    const CameraIntrinsics& intrinsics,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    computeNormalsKernel<<<grid, block, 0, stream>>>(
        d_depth_map, d_points, intrinsics, width, height
    );
    
    CUDA_CHECK_LAST_ERROR();
}

void launchStatisticalOutlierRemoval(
    const ColoredPoint* d_input_points,
    ColoredPoint* d_output_points,
    int* d_valid_mask,
    int num_points,
    int k_neighbors,
    float std_multiplier,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;
