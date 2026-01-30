#include "cuda_utils.cuh"
#include "data_structures.h"
#include <cuda_runtime.h>

using namespace v3d;

// Find correspondences between two point clouds
__global__ void findCorrespondencesKernel(
    const ColoredPoint* source_points,
