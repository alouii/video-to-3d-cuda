# Video to 3D Point Cloud Reconstruction (CUDA)

A high-performance, real-time 3D reconstruction system that converts video streams into dense 3D point clouds and meshes using CUDA-accelerated processing.

## Features

- **Real-time Processing**: Achieves 12+ FPS on 1080p video with consumer GPUs
- **CUDA Acceleration**: Optimized CUDA kernels for all compute-intensive operations
- **Multiple Depth Estimation Methods**:
  - Stereo matching (Semi-Global Matching)
  - Monocular depth estimation (integration ready for MiDaS/DPT)
- **Advanced Reconstruction**:
  - TSDF volumetric fusion
  - ICP registration
  - Marching Cubes mesh generation
- **Filtering & Enhancement**:
  - Bilateral depth filtering
  - Statistical outlier removal
  - Voxel downsampling
- **Multi-threaded Architecture**: Separate threads for capture, processing, and visualization
- **Flexible Input**: Supports video files, webcam, and RTSP streams

## System Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.5+ (Turing, Ampere, or newer)
- 4GB+ GPU memory (8GB+ recommended for large reconstructions)
- 8GB+ system RAM
- Multi-core CPU (4+ cores recommended)

### Software
- Ubuntu 20.04+ or Windows 10+ with WSL2
- CUDA Toolkit 11.0 or newer
- GCC 9+ or Clang 10+
- CMake 3.18+

## Dependencies

### Required
```bash
# CUDA Toolkit
sudo apt install nvidia-cuda-toolkit

# OpenCV with CUDA support
sudo apt install libopencv-dev

# Eigen3
sudo apt install libeigen3-dev

# OpenMP
sudo apt install libomp-dev
```

### Optional
```bash
# PCL for advanced visualization
sudo apt install libpcl-dev

# VTK for rendering
sudo apt install libvtk7-dev
```

## Building

```bash
# Clone repository
git clone https://github.com/alouii/video-to-3d-cuda.git
cd video-to-3d-cuda

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install (optional)
