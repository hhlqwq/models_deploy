#pragma once

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class RosPublisher final {
   public:
    RosPublisher(int argc, char** argv);
    ~RosPublisher();

    RosPublisher(const RosPublisher&) = delete;
    RosPublisher& operator=(const RosPublisher&) = delete;
    RosPublisher(RosPublisher&&) = delete;
    RosPublisher& operator=(RosPublisher&&) = delete;

    bool publish(const std::string& json) noexcept;
    void spinSome() noexcept;
    void shutdown() noexcept;

   private:
    bool ownsRuntime_{false};
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};
