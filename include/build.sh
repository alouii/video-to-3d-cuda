#!/bin/bash

# Build script for Video to 3D CUDA project

set -e

echo "=== Building Video to 3D CUDA ==="

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    -DBUILD_TESTS=ON

# Build
