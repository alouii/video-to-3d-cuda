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

cv::Mat rgb_image = cv::imread("image.png");
v3d::DepthFrame depth_frame;
estimator.estimateDepth(rgb_image, depth_frame);
```

### Integration with Neural Networks

The MonocularDepthEstimator is designed to integrate with models like MiDaS or DPT.

#### TensorRT Integration (Example)

```cpp
class TensorRTDepthEstimator : public v3d::MonocularDepthEstimator {
private:
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
public:
    bool loadModel(const std::string& model_path) override {
        // Load TensorRT engine
        std::ifstream file(model_path, std::ios::binary);
        if (!file.good()) return false;
        
        // Deserialize engine
        // ... TensorRT loading code ...
        
        context_ = engine_->createExecutionContext();
        return context_ != nullptr;
    }
    
    bool estimateDepth(const cv::Mat& rgb, v3d::DepthFrame& depth) override {
        // Preprocess
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(384, 384));
        
        // Run inference
        void* bindings[] = {input_buffer_, output_buffer_};
        context_->executeV2(bindings);
        
        // Postprocess
        // ... copy output to depth_frame ...
        
        return true;
    }
};
```

## Depth Frame Usage

After depth estimation, you get a `DepthFrame` structure:

```cpp
v3d::DepthFrame depth_frame;
estimator.estimateDepth(image, depth_frame);

// Access depth data (on GPU)
float* d_depth = depth_frame.depth_map;   // Device pointer
uint8_t* d_rgb = depth_frame.rgb_image;   // Device pointer
int width = depth_frame.width;
int height = depth_frame.height;

// Download to CPU if needed
std::vector<float> depth_cpu(width * height);
cudaMemcpy(depth_cpu.data(), d_depth, 
           width * height * sizeof(float), 
           cudaMemcpyDeviceToHost);

// Visualize depth
cv::Mat depth_viz(height, width, CV_32F, depth_cpu.data());
cv::Mat depth_8u;
cv::normalize(depth_viz, depth_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
cv::applyColorMap(depth_8u, depth_8u, cv::COLORMAP_JET);
cv::imshow("Depth", depth_8u);
```

## Complete Example: Stereo Video Processing

```cpp
#include "depth_estimator.h"
#include <opencv2/opencv.hpp>

int main() {
    // Setup
    v3d::CameraIntrinsics intrinsics(525.0f, 525.0f, 319.5f, 239.5f, 640, 480);
    
    v3d::StereoDepthEstimator estimator;
