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
    }
};

// 6-DOF camera pose (rotation + translation)
struct Pose {
    float rotation[9];     // 3x3 rotation matrix (row-major)
    float translation[3];  // 3D translation vector
    double timestamp;      // Frame timestamp
    
    Pose() : timestamp(0.0) {
        // Identity
        rotation[0] = 1; rotation[1] = 0; rotation[2] = 0;
        rotation[3] = 0; rotation[4] = 1; rotation[5] = 0;
        rotation[6] = 0; rotation[7] = 0; rotation[8] = 1;
        translation[0] = 0; translation[1] = 0; translation[2] = 0;
    }
    
    // Convert to Eigen types
    Eigen::Matrix3f getRotationMatrix() const {
        Eigen::Matrix3f R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R(i, j) = rotation[i * 3 + j];
        return R;
    }
    
    Eigen::Vector3f getTranslationVector() const {
        return Eigen::Vector3f(translation[0], translation[1], translation[2]);
    }
    
    void setFromEigen(const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                rotation[i * 3 + j] = R(i, j);
        translation[0] = t(0);
        translation[1] = t(1);
        translation[2] = t(2);
    }
    
    // Transform point
    inline void transform(float x, float y, float z, float& out_x, float& out_y, float& out_z) const {
        out_x = rotation[0] * x + rotation[1] * y + rotation[2] * z + translation[0];
        out_y = rotation[3] * x + rotation[4] * y + rotation[5] * z + translation[1];
        out_z = rotation[6] * x + rotation[7] * y + rotation[8] * z + translation[2];
    }
};

// Colored 3D point with normal
struct ColoredPoint {
    float x, y, z;       // Position
    uint8_t r, g, b;     // Color
    float nx, ny, nz;    // Normal vector
    float confidence;    // Measurement confidence
    
    ColoredPoint() : x(0), y(0), z(0), r(0), g(0), b(0), 
                     nx(0), ny(0), nz(0), confidence(1.0f) {}
    
    ColoredPoint(float x_, float y_, float z_, uint8_t r_, uint8_t g_, uint8_t b_)
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_), 
          nx(0), ny(0), nz(0), confidence(1.0f) {}
};

// Depth frame with RGB and pose
struct DepthFrame {
    float* depth_map;      // Device pointer to H x W depth values
    uint8_t* rgb_image;    // Device pointer to H x W x 3 color
    int width, height;
    Pose camera_pose;
    double timestamp;
    bool on_device;
    
    DepthFrame() : depth_map(nullptr), rgb_image(nullptr), 
                   width(0), height(0), timestamp(0.0), on_device(false) {}
    
    ~DepthFrame() {
        // Note: Memory management handled by pipeline
    }
};

// Point cloud container
struct PointCloud {
    std::vector<ColoredPoint> points;
    bool has_normals;
    bool has_colors;
    
    PointCloud() : has_normals(false), has_colors(false) {}
    
    void clear() {
        points.clear();
        has_normals = false;
        has_colors = false;
    }
    
    size_t size() const {
        return points.size();
    }
    
    void reserve(size_t n) {
        points.reserve(n);
    }
    
    void add_point(float x, float y, float z, uint8_t r = 0, uint8_t g = 0, uint8_t b = 0) {
        points.emplace_back(x, y, z, r, g, b);
    }
};

