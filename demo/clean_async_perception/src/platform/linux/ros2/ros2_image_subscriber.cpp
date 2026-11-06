#include "platform/linux/ros2/ros2_image_subscriber.hpp"

#include <opencv2/imgproc.hpp>

#include <limits>
#include <stdexcept>
#include <utility>

RosImageSubscriber::RosImageSubscriber(const std::string& imageTopic,
                                       const std::string& cameraInfoTopic,
                                       size_t queueCapacity)
    : queueCapacity_(queueCapacity) {
    if (!rclcpp::ok()) throw std::runtime_error("ROS2 is not initialized");
    if (imageTopic.empty() || cameraInfoTopic.empty() || queueCapacity_ == 0) {
        throw std::invalid_argument(
            "ROS image topics and queue capacity must be set");
    }
    node_ = std::make_shared<rclcpp::Node>("async_perception_image_receiver");
    const auto qos = rclcpp::SensorDataQoS();
    imageSubscription_ = node_->create_subscription<sensor_msgs::msg::Image>(
        imageTopic, qos,
        [this](const sensor_msgs::msg::Image::ConstSharedPtr message) {
            onImage(message);
        });
    cameraInfoSubscription_ =
        node_->create_subscription<sensor_msgs::msg::CameraInfo>(
            cameraInfoTopic, qos,
            [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
                onCameraInfo(message);
            });
    executor_.add_node(node_);
    executorThread_ = std::thread([this] { executor_.spin(); });
    RCLCPP_INFO(node_->get_logger(), "Receiving %s and %s", imageTopic.c_str(),
                cameraInfoTopic.c_str());
}

RosImageSubscriber::~RosImageSubscriber() {
    executor_.cancel();
    frameReady_.notify_all();
    if (executorThread_.joinable()) executorThread_.join();
    executor_.remove_node(node_);
}

void RosImageSubscriber::onImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& message) {
    if (!loggedFirstImage_) {
        RCLCPP_INFO(node_->get_logger(),
                    "First image: %ux%u encoding=%s step=%u bytes=%zu",
                    message->width, message->height, message->encoding.c_str(),
                    message->step, message->data.size());
        loggedFirstImage_ = true;
    }
    if (message->encoding != "rgb8") {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!warnedEncoding_) {
            RCLCPP_WARN(node_->get_logger(), "Expected rgb8, received %s",
                        message->encoding.c_str());
            warnedEncoding_ = true;
        }
        return;
    }
    const size_t width = message->width;
    const size_t height = message->height;
    if (width == 0 || height == 0 ||
        width > std::numeric_limits<size_t>::max() / 3) {
        return;
    }
    const size_t minStep = width * 3;
    if (message->step < minStep ||
        height > std::numeric_limits<size_t>::max() / message->step ||
        message->data.size() < height * message->step ||
        width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return;
    }

    const cv::Mat rgb(static_cast<int>(height), static_cast<int>(width),
                      CV_8UC3, const_cast<uint8_t*>(message->data.data()),
                      message->step);
    RosImageFrame frame;
    cv::cvtColor(rgb, frame.bgr, cv::COLOR_RGB2BGR);
    frame.frameId = message->header.frame_id;
    const auto& stamp = message->header.stamp;
    if (stamp.sec >= 0) {
        frame.captureTimestampMs =
            static_cast<uint64_t>(stamp.sec) * 1000 +
            static_cast<uint64_t>(stamp.nanosec) / 1000000;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frames_.size() >= queueCapacity_) frames_.pop_front();
        frames_.push_back(std::move(frame));
    }
    ++acceptedImages_;
    if (acceptedImages_ % 100 == 0) {
        RCLCPP_INFO(node_->get_logger(), "Accepted %lu RGB images",
                    static_cast<unsigned long>(acceptedImages_));
    }
    frameReady_.notify_one();
}

void RosImageSubscriber::onCameraInfo(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& message) {
    if (message->width == 0 || message->height == 0 || message->k[0] <= 0.0 ||
        message->k[4] <= 0.0) {
        return;
    }
    RosCameraCalibration calibration;
    calibration.width = static_cast<int>(message->width);
    calibration.height = static_cast<int>(message->height);
    calibration.fx = message->k[0];
    calibration.fy = message->k[4];
    calibration.cx = message->k[2];
    calibration.cy = message->k[5];
    std::lock_guard<std::mutex> lock(mutex_);
    calibration_ = calibration;
    calibrationFrameId_ = message->header.frame_id;
    hasCalibration_ = true;
}

bool RosImageSubscriber::waitForFrame(RosImageFrame* frame,
                                      std::chrono::milliseconds timeout) {
    if (!frame) return false;
    std::unique_lock<std::mutex> lock(mutex_);
    if (!frameReady_.wait_for(
            lock, timeout,
            [this] { return !frames_.empty() || !rclcpp::ok(); }) ||
        frames_.empty()) {
        return false;
    }
    *frame = std::move(frames_.back());
    frames_
        .clear();  // Freshness matters more than processing every camera frame.
    return true;
}

bool RosImageSubscriber::calibrationFor(
    const RosImageFrame& frame, RosCameraCalibration* calibration) const {
    if (!calibration) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hasCalibration_ || calibration_.width != frame.bgr.cols ||
        calibration_.height != frame.bgr.rows ||
        (!calibrationFrameId_.empty() && !frame.frameId.empty() &&
         calibrationFrameId_ != frame.frameId)) {
        return false;
    }
    *calibration = calibration_;
    return true;
}
