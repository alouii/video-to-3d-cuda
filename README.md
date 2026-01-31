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
