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
