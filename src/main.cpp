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
