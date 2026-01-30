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
}

// Warp-level reduction sum
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction sum
__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// CUDA memory pool for efficient allocation
class CudaMemoryPool {
private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks_;
    size_t total_allocated_;
    
public:
    CudaMemoryPool() : total_allocated_(0) {}
    
    ~CudaMemoryPool() {
        for (auto& block : blocks_) {
            if (block.ptr) {
                cudaFree(block.ptr);
            }
        }
    }
    
    void* allocate(size_t size) {
        // Find free block of suitable size
        for (auto& block : blocks_) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }
        
        // Allocate new block
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        blocks_.push_back({ptr, size, true});
        total_allocated_ += size;
        return ptr;
    }
    
    void deallocate(void* ptr) {
        for (auto& block : blocks_) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }
    
    size_t total_allocated() const { return total_allocated_; }
};

// Pinned memory allocator for fast CPU-GPU transfers
template<typename T>
class PinnedMemoryAllocator {
public:
    using value_type = T;
    
    PinnedMemoryAllocator() = default;
    
    template<typename U>
    PinnedMemoryAllocator(const PinnedMemoryAllocator<U>&) {}
    
    T* allocate(std::size_t n) {
        T* ptr;
        CUDA_CHECK(cudaMallocHost(&ptr, n * sizeof(T)));
        return ptr;
    }
    
    void deallocate(T* ptr, std::size_t) {
        cudaFreeHost(ptr);
    }
};

// RAII wrapper for CUDA device memory
template<typename T>
class DeviceBuffer {
private:
    T* data_;
    size_t size_;
    
public:
    DeviceBuffer() : data_(nullptr), size_(0) {}
    
    explicit DeviceBuffer(size_t size) : size_(size) {
        CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
    }
    
    ~DeviceBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }
    
    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Delete copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    void resize(size_t new_size) {
        if (new_size != size_) {
            if (data_) cudaFree(data_);
            size_ = new_size;
            if (size_ > 0) {
                CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
            } else {
                data_ = nullptr;
            }
        }
    }
    
    void upload(const T* host_data, size_t count) {
        CUDA_CHECK(cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void download(T* host_data, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    void zero() {
        CUDA_CHECK(cudaMemset(data_, 0, size_ * sizeof(T)));
    }
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
};

