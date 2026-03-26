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

