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
    int width, height;
    
    CameraIntrinsics() : fx(525.0f), fy(525.0f), cx(319.5f), cy(239.5f), 
                         width(640), height(480) {}
    
    CameraIntrinsics(float fx_, float fy_, float cx_, float cy_, int w, int h)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_), width(w), height(h) {}
    
    // Back-project pixel to 3D ray
    inline void backproject(int u, int v, float depth, float& x, float& y, float& z) const {
        z = depth;
        x = (u - cx) * depth / fx;
        y = (v - cy) * depth / fy;
    }
    
    // Project 3D point to pixel
    inline void project(float x, float y, float z, int& u, int& v) const {
        u = static_cast<int>(fx * x / z + cx);
        v = static_cast<int>(fy * y / z + cy);
