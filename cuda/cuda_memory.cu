#include "cuda_utils.cuh"
#include <cuda_runtime.h>

// Global memory pool instance
namespace cuda_utils {
    static CudaMemoryPool g_memory_pool;
}

extern "C" {

void* cudaAllocateFromPool(size_t size) {
    return cuda_utils::g_memory_pool.allocate(size);
}

void cudaDeallocateToPool(void* ptr) {
    cuda_utils::g_memory_pool.deallocate(ptr);
}

size_t cudaGetPoolMemoryUsage() {
    return cuda_utils::g_memory_pool.total_allocated();
}

