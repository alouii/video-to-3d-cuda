# Usage Examples

## Basic Examples

### 1. Process Video File

```bash
./video_to_3d -i sample_video.mp4 -o output.ply --visualize
```

### 2. Webcam Live Reconstruction

```bash
./video_to_3d --camera 0 --visualize --voxel-size 0.015
```

### 3. High-Quality Reconstruction

```bash
./video_to_3d -i input.mp4 \
    --voxel-size 0.005 \
    --mesh output_mesh.obj \
    --visualize
```

## Advanced Examples

### 4. Custom Camera Parameters

```bash
./video_to_3d -i video.mp4 \
    --fx 535.4 --fy 539.2 \
    --cx 320.1 --cy 247.6 \
    -o calibrated_output.ply
```

### 5. Fast Processing (Skip Frames)

```bash
./video_to_3d -i long_video.mp4 \
    --skip-frames 2 \
    --max-frames 300 \
    -o fast_output.ply
```

### 6. Memory-Constrained System

```bash
./video_to_3d -i input.mp4 \
    --voxel-size 0.02 \
    --no-fusion \
    -o lightweight_output.ply
```

## API Examples

### Example 1: Basic Pipeline

```cpp
#include "pipeline.h"

int main() {
    v3d::PipelineConfig config;
    config.video_source = "input.mp4";
    config.enable_visualization = true;
    
    v3d::VideoTo3DPipeline pipeline(config);
    pipeline.initialize();
    pipeline.processVideo();
    pipeline.exportPointCloud("output.ply");
    
    return 0;
}
```

### Example 2: Custom Processing

```cpp
#include "pipeline.h"

int main() {
    v3d::PipelineConfig config;
    config.use_camera = true;
    config.camera_id = 0;
    
    v3d::VideoTo3DPipeline pipeline(config);
    pipeline.initialize();
    
    while (pipeline.hasFrames()) {
        pipeline.processNextFrame();
        
        // Get intermediate results
        if (pipeline.reconstructionReady()) {
            auto pc = pipeline.getPointCloud();
            // Process point cloud...
        }
    }
    
    return 0;
}
```

### Example 3: Event-Driven Processing

```cpp
class MyPipeline {
public:
    void run() {
        v3d::VideoTo3DPipeline pipeline(config_);
        pipeline.initialize();
        
        while (!should_stop_) {
            if (paused_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            pipeline.processNextFrame();
        }
    }
    
    void pause() { paused_ = true; }
    void resume() { paused_ = false; }
    void stop() { should_stop_ = true; }
    
private:
    v3d::PipelineConfig config_;
    bool paused_ = false;
    bool should_stop_ = false;
};
```

## Integration Examples

### ROS Integration

```cpp
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include "pipeline.h"

class Video3DROS {
public:
    Video3DROS(ros::NodeHandle& nh) {
        pub_ = nh.advertise<sensor_msgs::PointCloud2>("pointcloud", 1);
        
        config_.video_source = "/dev/video0";
        pipeline_ = std::make_unique<v3d::VideoTo3DPipeline>(config_);
        pipeline_->initialize();
    }
    
    void spin() {
