#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <cstdint>
#include <vector>
#include <Eigen/Dense>

namespace v3d {

// Camera intrinsic parameters
struct CameraIntrinsics {
    float fx, fy;  // Focal lengths
    float cx, cy;  // Principal point
