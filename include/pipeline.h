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
class MeshGenerator;
class Visualizer;

class VideoTo3DPipeline {
public:
    explicit VideoTo3DPipeline(const PipelineConfig& config);
    ~VideoTo3DPipeline();
    
    // Initialize pipeline
    bool initialize();
    
    // Process video
    void processVideo();
    
    // Process single frame
    bool processNextFrame();
    
    // Check if more frames available
    bool hasFrames() const;
    
    // Check if reconstruction is ready
    bool reconstructionReady() const;
    
    // Get results
    PointCloud getPointCloud() const;
    Mesh getMesh() const;
    
    // Export results
    bool exportPointCloud(const std::string& filename) const;
    bool exportMesh(const std::string& filename) const;
    
    // Statistics
    PerformanceStats getStatistics() const;
    void printStatistics() const;
    
    // Control
    void pause();
    void resume();
    void stop();
    
private:
    // Pipeline stages
    bool captureFrame();
    bool estimateDepth();
    bool generatePointCloud();
    bool registerPointCloud();
