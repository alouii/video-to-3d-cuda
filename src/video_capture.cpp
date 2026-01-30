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
