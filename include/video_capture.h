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
};

class VideoCapture {
public:
    VideoCapture();
    ~VideoCapture();
    
    // Open video source
    bool open(const std::string& source);
    bool open(int camera_id);
    
    // Close video source
    void close();
    
    // Check if opened
    bool isOpened() const;
    
    // Get next frame (blocks if queue is empty)
    bool getFrame(VideoFrame& frame);
    
    // Get video properties
    int getWidth() const;
    int getHeight() const;
    double getFPS() const;
    int getTotalFrames() const;
    
    // Buffer management
    void setBufferSize(size_t size);
    size_t getBufferSize() const;
    size_t getCurrentBufferCount() const;
    
    // Control
    void start();
    void stop();
    bool isRunning() const;
    
private:
    void captureThread();
    
    cv::VideoCapture capture_;
    
    std::queue<VideoFrame> frame_buffer_;
    size_t max_buffer_size_;
    
    std::thread capture_thread_;
    std::mutex buffer_mutex_;
    std::condition_variable buffer_cv_;
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
    
    int width_, height_;
    double fps_;
    int total_frames_;
    int current_frame_index_;
};

} // namespace v3d
