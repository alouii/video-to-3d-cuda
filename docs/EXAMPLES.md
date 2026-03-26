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
    
