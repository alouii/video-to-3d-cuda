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
