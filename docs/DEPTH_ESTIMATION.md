# Depth Estimation Guide

## Overview

The depth estimation module provides two main implementations:

1. **StereoDepthEstimator**: GPU-accelerated stereo matching for stereo camera setups
2. **MonocularDepthEstimator**: Neural network-based depth from single images

## StereoDepthEstimator

### Basic Usage

```cpp
#include "depth_estimator.h"

// Setup camera intrinsics
v3d::CameraIntrinsics intrinsics;
intrinsics.fx = 525.0f;
intrinsics.fy = 525.0f;
intrinsics.cx = 319.5f;
intrinsics.cy = 239.5f;
intrinsics.width = 640;
intrinsics.height = 480;

// Create and initialize estimator
v3d::StereoDepthEstimator estimator;
estimator.setBaseline(0.12f);  // 12cm baseline
estimator.setMaxDisparity(128);
estimator.initialize(intrinsics);

// Load stereo images
cv::Mat left = cv::imread("left.png");
cv::Mat right = cv::imread("right.png");

// Estimate depth
v3d::DepthFrame depth_frame;
estimator.estimateDepthStereo(left, right, depth_frame);

// Depth is now on GPU at depth_frame.depth_map
// RGB is at depth_frame.rgb_image
```

### Advanced Configuration

```cpp
v3d::StereoDepthEstimator estimator;

// Configure stereo parameters
estimator.setBaseline(0.15f);           // 15cm baseline
estimator.setMaxDisparity(256);         // Search up to 256 pixels

// Enable/disable features
estimator.enableBilateralFiltering(true);  // Smooth depth
estimator.enableLeftRightCheck(true);      // Consistency check

// Fine-tune bilateral filter
estimator.setBilateralParams(
    2.0f,   // sigma_space (spatial smoothing)
    0.1f    // sigma_range (edge preservation)
);

// Initialize and process
estimator.initialize(intrinsics);
estimator.estimateDepthStereo(left, right, depth_frame);
```

### Performance Tuning

```cpp
// For speed (lower quality)
estimator.setMaxDisparity(64);             // Smaller search range
estimator.enableBilateralFiltering(false); // Skip filtering
estimator.enableLeftRightCheck(false);     // Skip consistency

// For quality (slower)
estimator.setMaxDisparity(256);            // Larger search range
estimator.enableBilateralFiltering(true);  // Enable filtering
estimator.setBilateralParams(3.0f, 0.05f); // Stronger filtering
```

## MonocularDepthEstimator

### Basic Usage

```cpp
#include "depth_estimator.h"

v3d::CameraIntrinsics intrinsics;
intrinsics.width = 640;
intrinsics.height = 480;

v3d::MonocularDepthEstimator estimator;
estimator.setModelPath("models/midas_v21.onnx");
estimator.initialize(intrinsics);

