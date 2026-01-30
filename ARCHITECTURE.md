# Architecture Documentation

## System Overview

The Video to 3D Point Cloud Reconstruction system is designed as a modular, GPU-accelerated pipeline that processes video streams into dense 3D reconstructions in real-time.

## Core Components

### 1. Video Capture (`video_capture.h/cpp`)

**Responsibility**: Multi-threaded frame acquisition

**Key Features**:
- Producer-consumer pattern with thread-safe queue
- Supports multiple input sources (files, cameras, streams)
- Configurable buffer size for smooth processing
- Hardware-accelerated decoding (NVDEC when available)

**Threading Model**:
```
Capture Thread → Frame Buffer → Processing Thread
     ↓
 cv::VideoCapture
```

### 2. Depth Estimation (`depth_estimator.h/cpp`)

**Responsibility**: Convert RGB images to depth maps

**Implementations**:
- **StereoDepthEstimator**: GPU-accelerated stereo matching
  - Census transform for robust matching
  - Semi-Global Matching (SGM) cost aggregation
  - Subpixel refinement for accuracy
  
- **MonocularDepthEstimator** (Integration Point):
  - TensorRT inference for MiDaS/DPT models
  - Real-time monocular depth prediction

**CUDA Kernels** (`depth_kernels.cu`):
- `censusTransformKernel`: Compute census transform
- `computeMatchingCostKernel`: Calculate matching costs
- `sgmCostAggregationKernel`: SGM path aggregation
- `selectDisparityKernel`: Winner-takes-all with subpixel
- `disparityToDepthKernel`: Convert disparity to metric depth

### 3. Point Cloud Generation (`point_cloud_generator.h/cpp`)

**Responsibility**: Transform depth + RGB into 3D colored points

**Processing Pipeline**:
1. Back-projection using camera intrinsics
2. Bilateral filtering for noise reduction
3. Normal estimation via surface gradients
4. Statistical outlier removal
5. Voxel downsampling for efficiency

**CUDA Kernels** (`pointcloud_kernels.cu`):
- `depthToPointCloudKernel`: Parallel back-projection
- `bilateralFilterDepthKernel`: Edge-preserving smoothing
- `computeNormalsKernel`: Surface normal calculation
- `statisticalOutlierRemovalKernel`: Outlier filtering
- `voxelDownsampleKernel`: Voxel grid downsampling

### 4. Registration (`registration.h/cpp`)

**Responsibility**: Align point clouds from different frames

**Algorithm**: Iterative Closest Point (ICP)

**Steps**:
1. Find correspondences (nearest neighbors)
2. Reject outliers based on distance threshold
3. Compute transformation (SVD-based)
4. Transform source point cloud
5. Repeat until convergence

**Optimizations**:
- GPU-accelerated correspondence finding
- Parallel covariance matrix computation
- Warp-level reductions for centroid calculation

**CUDA Kernels** (`registration_kernels.cu`):
- `findCorrespondencesKernel`: Brute-force NN search
- `computeCentroidKernel`: Parallel mean calculation
- `computeCovarianceKernel`: Cross-covariance matrix
- `transformPointCloudKernel`: Apply rigid transformation

### 5. TSDF Fusion (`tsdf_fusion.h/cpp`)

**Responsibility**: Volumetric integration of multiple depth maps

**Data Structure**: Truncated Signed Distance Function (TSDF)

**Voxel Grid**:
```cpp
struct TSDFVoxel {
    float tsdf;        // Distance to surface
    float weight;      // Integration confidence
    uint8_t r, g, b;   // Color
}
```

**Integration Process**:
1. For each voxel in grid:
   - Project to camera frame
   - Look up depth measurement
   - Compute signed distance
   - Update TSDF with weighted average

**CUDA Kernels** (`tsdf_kernels.cu`):
- `integrateTSDFKernel`: Volumetric fusion
- `extractSurfacePointsKernel`: Zero-crossing detection
- `raycastTSDFKernel`: Surface rendering

### 6. Mesh Generation (`mesh_generator.h/cpp`)

**Responsibility**: Extract triangulated mesh from TSDF

**Algorithm**: Marching Cubes

**Steps**:
1. Classify voxels (inside/outside surface)
2. Identify zero-crossing edges
3. Generate vertices via interpolation
4. Triangulate based on lookup table
5. Compute vertex normals

**CUDA Kernels** (`mesh_kernels.cu`):
