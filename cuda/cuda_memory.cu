#include "cuda_utils.cuh"
#include <cuda_runtime.h>

// Global memory pool instance
namespace cuda_utils {
    static CudaMemoryPool g_memory_pool;
}

extern "C" {

void* cudaAllocateFromPool(size_t size) {
