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

// Triangle mesh
struct Triangle {
    uint32_t v0, v1, v2;  // Vertex indices
};

struct Mesh {
    std::vector<ColoredPoint> vertices;
    std::vector<Triangle> triangles;
    bool has_normals;
    bool has_colors;
    
    Mesh() : has_normals(false), has_colors(false) {}
    
    void clear() {
        vertices.clear();
        triangles.clear();
        has_normals = false;
        has_colors = false;
    }
    
    size_t num_vertices() const { return vertices.size(); }
    size_t num_triangles() const { return triangles.size(); }
};

// TSDF voxel
struct TSDFVoxel {
    float tsdf;        // Truncated signed distance
    float weight;      // Integration weight
    uint8_t r, g, b;   // Color
    
    TSDFVoxel() : tsdf(1.0f), weight(0.0f), r(0), g(0), b(0) {}
};

// Voxel grid configuration
struct VoxelGridConfig {
    float voxel_size;           // Voxel size in meters
    float truncation_distance;  // TSDF truncation distance
    int grid_dim_x, grid_dim_y, grid_dim_z;
    float min_x, min_y, min_z;  // Volume bounds
    float max_x, max_y, max_z;
    
    VoxelGridConfig() 
        : voxel_size(0.01f), truncation_distance(0.05f),
          grid_dim_x(512), grid_dim_y(512), grid_dim_z(512),
          min_x(-2.5f), min_y(-2.5f), min_z(-2.5f),
          max_x(2.5f), max_y(2.5f), max_z(2.5f) {}
    
    int total_voxels() const {
        return grid_dim_x * grid_dim_y * grid_dim_z;
    }
    
    // Convert world coordinates to voxel indices
    inline bool worldToVoxel(float x, float y, float z, int& vx, int& vy, int& vz) const {
        vx = static_cast<int>((x - min_x) / voxel_size);
        vy = static_cast<int>((y - min_y) / voxel_size);
        vz = static_cast<int>((z - min_z) / voxel_size);
        return (vx >= 0 && vx < grid_dim_x &&
                vy >= 0 && vy < grid_dim_y &&
                vz >= 0 && vz < grid_dim_z);
    }
    
    // Convert voxel indices to world coordinates (center of voxel)
    inline void voxelToWorld(int vx, int vy, int vz, float& x, float& y, float& z) const {
        x = min_x + (vx + 0.5f) * voxel_size;
        y = min_y + (vy + 0.5f) * voxel_size;
        z = min_z + (vz + 0.5f) * voxel_size;
    }
};

// Pipeline configuration
struct PipelineConfig {
    // Video source
    std::string video_source;
    int camera_id;
    bool use_camera;
    
    // Processing
    bool use_depth_model;
    std::string depth_model_path;
    bool enable_fusion;
    bool enable_mesh;
    bool enable_visualization;
    
    // Camera parameters
    CameraIntrinsics intrinsics;
    
    // TSDF parameters
    VoxelGridConfig voxel_config;
    
    // Performance
    int max_frames;
    int skip_frames;
    int num_streams;
    
    // Output
    std::string output_dir;
    bool save_point_cloud;
    bool save_mesh;
    
    PipelineConfig() 
        : video_source(""), camera_id(0), use_camera(false),
          use_depth_model(true), depth_model_path("models/depth_model.onnx"),
          enable_fusion(true), enable_mesh(false), enable_visualization(true),
          max_frames(-1), skip_frames(0), num_streams(4),
          output_dir("output"), save_point_cloud(true), save_mesh(false) {}
};

// Statistics for performance monitoring
struct PerformanceStats {
    double frame_capture_time;
    double depth_estimation_time;
    double pointcloud_generation_time;
    double registration_time;
    double fusion_time;
    double visualization_time;
    double total_frame_time;
    
    int frames_processed;
    double fps;
    
    PerformanceStats() { reset(); }
    
    void reset() {
        frame_capture_time = 0.0;
        depth_estimation_time = 0.0;
        pointcloud_generation_time = 0.0;
        registration_time = 0.0;
        fusion_time = 0.0;
        visualization_time = 0.0;
        total_frame_time = 0.0;
