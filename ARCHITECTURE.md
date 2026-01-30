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
