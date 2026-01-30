#ifndef PIPELINE_H
#define PIPELINE_H

#include "data_structures.h"
#include "video_capture.h"
#include "cuda_utils.cuh"
#include <memory>
#include <vector>

namespace v3d {

// Forward declarations
class DepthEstimator;
class PointCloudGenerator;
class Registration;
class TSDFFusion;
