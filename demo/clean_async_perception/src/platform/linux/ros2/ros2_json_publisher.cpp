#include "platform/linux/ros2/ros2_json_publisher.hpp"

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {
constexpr char kNodeName[] = "async_perception_node";
constexpr char kTopicName[] = "/perception/result";
constexpr std::size_t kQueueDepth = 10;
}  // namespace

RosPublisher::RosPublisher(int argc, char** argv) {
  if (rclcpp::ok()) {
    throw std::runtime_error("ROS2 default context is already initialized");
  }
  rclcpp::init(argc, argv);
  ownsRuntime_ = true;
  node_ = std::make_shared<rclcpp::Node>(kNodeName);
  publisher_ = node_->create_publisher<std_msgs::msg::String>(kTopicName, kQueueDepth);
}

RosPublisher::~RosPublisher() {
  shutdown();
}

bool RosPublisher::publish(const std::string& json) noexcept {
  if (!publisher_ || !rclcpp::ok()) return false;
  try {
    std_msgs::msg::String message;
    message.data = json;
    publisher_->publish(message);
    return true;
  } catch (const std::exception& error) {
    std::cerr << "[ROS2] publish failed: " << error.what() << '\n';
  } catch (...) {
    std::cerr << "[ROS2] publish failed: unknown exception\n";
  }
  return false;
}

void RosPublisher::spinSome() noexcept {
  if (!node_ || !rclcpp::ok()) return;
  try {
    rclcpp::spin_some(node_);
  } catch (const std::exception& error) {
    std::cerr << "[ROS2] spin_some failed: " << error.what() << '\n';
  } catch (...) {
    std::cerr << "[ROS2] spin_some failed: unknown exception\n";
  }
}

void RosPublisher::shutdown() noexcept {
  publisher_.reset();
  node_.reset();
  if (ownsRuntime_) {
    if (rclcpp::ok()) rclcpp::shutdown();
    ownsRuntime_ = false;
  }
}
