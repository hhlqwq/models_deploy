#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/core.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

struct RosImageFrame {
  cv::Mat bgr;
  uint64_t captureTimestampMs = 0;
  std::string frameId;
};

struct RosCameraCalibration {
  int width = 0;
  int height = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

class RosImageSubscriber final {
 public:
  RosImageSubscriber(const std::string& imageTopic,
                     const std::string& cameraInfoTopic, size_t queueCapacity);
  ~RosImageSubscriber();

  RosImageSubscriber(const RosImageSubscriber&) = delete;
  RosImageSubscriber& operator=(const RosImageSubscriber&) = delete;

  bool waitForFrame(RosImageFrame* frame, std::chrono::milliseconds timeout);
  bool calibrationFor(const RosImageFrame& frame,
                      RosCameraCalibration* calibration) const;

 private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr& message);
  void onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& message);

  const size_t queueCapacity_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSubscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoSubscription_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread executorThread_;

  mutable std::mutex mutex_;
  std::condition_variable frameReady_;
  std::deque<RosImageFrame> frames_;
  RosCameraCalibration calibration_;
  std::string calibrationFrameId_;
  bool hasCalibration_ = false;
  bool warnedEncoding_ = false;
  bool loggedFirstImage_ = false;
  uint64_t acceptedImages_ = 0;
};
