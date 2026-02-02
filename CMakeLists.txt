cmake_minimum_required(VERSION 3.18)
project(VideoTo3DCUDA LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fopenmp")

# Find packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Optional packages
find_package(PCL QUIET)
find_package(VTK QUIET)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

if(PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(-DUSE_PCL)
endif()

# CUDA kernels
set(CUDA_SOURCES
    cuda/depth_kernels.cu
    cuda/pointcloud_kernels.cu
    cuda/registration_kernels.cu
    cuda/tsdf_kernels.cu
    cuda/filtering_kernels.cu
    cuda/mesh_kernels.cu
    cuda/cuda_memory.cu
)

# C++ sources
