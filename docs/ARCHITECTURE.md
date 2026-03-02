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
