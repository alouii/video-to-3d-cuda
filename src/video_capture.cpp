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
    }
    
    stop_requested_ = false;
    running_ = true;
    
    capture_thread_ = std::thread(&VideoCapture::captureThread, this);
}

void VideoCapture::stop() {
    if (!running_) {
        return;
    }
    
    stop_requested_ = true;
    buffer_cv_.notify_all();
    
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    
    running_ = false;
}

bool VideoCapture::isRunning() const {
    return running_;
}

bool VideoCapture::getFrame(VideoFrame& frame) {
    std::unique_lock<std::mutex> lock(buffer_mutex_);
    
    // Wait for frame to be available
    buffer_cv_.wait(lock, [this] {
        return !frame_buffer_.empty() || stop_requested_;
    });
    
    if (frame_buffer_.empty()) {
        return false;
    }
    
    frame = frame_buffer_.front();
    frame_buffer_.pop();
    
    buffer_cv_.notify_all();
    
    return true;
}

void VideoCapture::captureThread() {
    while (!stop_requested_) {
        // Check if buffer is full
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            buffer_cv_.wait(lock, [this] {
                return frame_buffer_.size() < max_buffer_size_ || stop_requested_;
            });
            
            if (stop_requested_) {
                break;
            }
        }
        
        // Capture frame
        cv::Mat frame;
        if (!capture_.read(frame)) {
            std::cout << "End of video or capture error." << std::endl;
            stop_requested_ = true;
            buffer_cv_.notify_all();
            break;
        }
        
        // Create video frame
        VideoFrame video_frame;
        video_frame.rgb_image = frame.clone();
        video_frame.timestamp = capture_.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        video_frame.frame_index = current_frame_index_++;
        
        // Add to buffer
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            frame_buffer_.push(video_frame);
        }
        
        buffer_cv_.notify_all();
    }
    
    running_ = false;
}

int VideoCapture::getWidth() const {
    return width_;
}

int VideoCapture::getHeight() const {
    return height_;
}
