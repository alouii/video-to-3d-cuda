#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <memory>
#include <stdexcept>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(error)); \
        } \
    } while(0)

// CUDA kernel error checking
#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(error)); \
        } \
    } while(0)

namespace cuda_utils {

// Device function for atomic max with floats
__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Device function for atomic min with floats
__device__ __forceinline__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
