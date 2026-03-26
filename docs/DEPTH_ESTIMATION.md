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

