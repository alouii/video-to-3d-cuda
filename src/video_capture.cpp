#include "video_capture.h"
#include <iostream>

namespace v3d {

VideoCapture::VideoCapture()
    : max_buffer_size_(30)
    , running_(false)
    , stop_requested_(false)
    , width_(0)
    , height_(0)
    , fps_(0.0)
    , total_frames_(0)
    , current_frame_index_(0)
{
}

VideoCapture::~VideoCapture() {
    close();
}

bool VideoCapture::open(const std::string& source) {
    if (!capture_.open(source)) {
        std::cerr << "Error: Could not open video file: " << source << std::endl;
        return false;
    }
    
    width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = capture_.get(cv::CAP_PROP_FPS);
    total_frames_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "Opened video: " << width_ << "x" << height_ 
              << " @ " << fps_ << " FPS, " << total_frames_ << " frames" << std::endl;
    
    return true;
}

bool VideoCapture::open(int camera_id) {
    if (!capture_.open(camera_id)) {
        std::cerr << "Error: Could not open camera " << camera_id << std::endl;
        return false;
    }
    
    // Set camera properties
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture_.set(cv::CAP_PROP_FPS, 30);
    
    width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = capture_.get(cv::CAP_PROP_FPS);
    total_frames_ = -1;  // Unknown for camera
    
    std::cout << "Opened camera: " << width_ << "x" << height_ 
              << " @ " << fps_ << " FPS" << std::endl;
    
    return true;
}

void VideoCapture::close() {
    stop();
    
    if (capture_.isOpened()) {
        capture_.release();
    }
}

bool VideoCapture::isOpened() const {
    return capture_.isOpened();
}

void VideoCapture::start() {
    if (running_) {
        return;
