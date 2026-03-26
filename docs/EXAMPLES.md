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

