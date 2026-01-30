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
