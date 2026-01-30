#ifndef VIDEO_CAPTURE_H
#define VIDEO_CAPTURE_H

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>

namespace v3d {

struct VideoFrame {
    cv::Mat rgb_image;
    double timestamp;
    int frame_index;
    
    VideoFrame() : timestamp(0.0), frame_index(0) {}
