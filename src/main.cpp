#include "pipeline.h"
#include "data_structures.h"
#include "cuda_utils.cuh"
#include <iostream>
#include <string>
#include <chrono>

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -i, --input <file>         Input video file\n"
              << "  -c, --camera <id>          Camera device ID (default: 0)\n"
              << "  -o, --output <file>        Output point cloud file (PLY format)\n"
              << "  --mesh <file>              Output mesh file (OBJ format)\n"
              << "  --visualize                Enable real-time visualization\n"
              << "  --no-fusion                Disable TSDF fusion\n"
              << "  --voxel-size <size>        Voxel size in meters (default: 0.01)\n"
              << "  --max-frames <n>           Maximum frames to process (default: all)\n"
              << "  --skip-frames <n>          Skip n frames between processing (default: 0)\n"
              << "  --fx <value>               Focal length X (default: 525.0)\n"
              << "  --fy <value>               Focal length Y (default: 525.0)\n"
              << "  --cx <value>               Principal point X (default: 319.5)\n"
              << "  --cy <value>               Principal point Y (default: 239.5)\n"
              << "  --help                     Display this help message\n"
              << std::endl;
}

bool parseArguments(int argc, char** argv, v3d::PipelineConfig& config) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            return false;
        }
        else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                config.video_source = argv[++i];
                config.use_camera = false;
            }
        }
        else if (arg == "-c" || arg == "--camera") {
            if (i + 1 < argc) {
                config.camera_id = std::stoi(argv[++i]);
                config.use_camera = true;
            }
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                config.output_dir = argv[++i];
                config.save_point_cloud = true;
            }
        }
        else if (arg == "--mesh") {
            if (i + 1 < argc) {
                config.output_dir = argv[++i];
                config.save_mesh = true;
                config.enable_mesh = true;
            }
        }
        else if (arg == "--visualize") {
            config.enable_visualization = true;
        }
        else if (arg == "--no-fusion") {
            config.enable_fusion = false;
        }
        else if (arg == "--voxel-size") {
            if (i + 1 < argc) {
                config.voxel_config.voxel_size = std::stof(argv[++i]);
            }
        }
        else if (arg == "--max-frames") {
            if (i + 1 < argc) {
                config.max_frames = std::stoi(argv[++i]);
            }
        }
        else if (arg == "--skip-frames") {
            if (i + 1 < argc) {
                config.skip_frames = std::stoi(argv[++i]);
            }
        }
        else if (arg == "--fx") {
            if (i + 1 < argc) {
                config.intrinsics.fx = std::stof(argv[++i]);
            }
        }
        else if (arg == "--fy") {
            if (i + 1 < argc) {
                config.intrinsics.fy = std::stof(argv[++i]);
            }
        }
        else if (arg == "--cx") {
            if (i + 1 < argc) {
                config.intrinsics.cx = std::stof(argv[++i]);
            }
        }
        else if (arg == "--cy") {
            if (i + 1 < argc) {
                config.intrinsics.cy = std::stof(argv[++i]);
            }
        }
    }
    
    return true;
}

int main(int argc, char** argv) {
    std::cout << "=== Video to 3D Point Cloud Reconstruction (CUDA) ===" << std::endl;
    std::cout << "Version 1.0\n" << std::endl;
    
    // Parse command line arguments
    v3d::PipelineConfig config;
    if (!parseArguments(argc, argv, config)) {
        printUsage(argv[0]);
        return 0;
    }
    
    // Validate configuration
    if (!config.use_camera && config.video_source.empty()) {
        std::cerr << "Error: No input source specified. Use -i or -c option." << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // Print CUDA device info
    cuda_utils::printDeviceInfo();
    
    // Create and initialize pipeline
    std::cout << "\nInitializing pipeline..." << std::endl;
    v3d::VideoTo3DPipeline pipeline(config);
    
    if (!pipeline.initialize()) {
        std::cerr << "Error: Failed to initialize pipeline." << std::endl;
        return 1;
    }
