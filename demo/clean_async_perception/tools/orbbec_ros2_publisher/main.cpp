#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Utils.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr char kImageTopic[] = "/camera/color/image_raw";
constexpr char kCameraInfoTopic[] = "/camera/color/camera_info";
constexpr char kFrameId[] = "camera_color_optical_frame";

std::shared_ptr<ob::Frame> ensureRgbFrame(
    const std::shared_ptr<ob::ColorFrame>& color,
    const std::shared_ptr<ob::FormatConvertFilter>& converter) {
    if (!color || !converter) return nullptr;
    switch (color->getFormat()) {
        case OB_FORMAT_RGB:
            return color;
        case OB_FORMAT_BGR:
            converter->setFormatConvertType(FORMAT_BGR_TO_RGB);
            break;
        case OB_FORMAT_MJPG:
            converter->setFormatConvertType(FORMAT_MJPG_TO_RGB);
            break;
        case OB_FORMAT_YUYV:
        case OB_FORMAT_YUY2:
        case OB_FORMAT_UYVY:
            converter->setFormatConvertType(FORMAT_YUYV_TO_RGB);
            break;
        case OB_FORMAT_NV12:
            converter->setFormatConvertType(FORMAT_NV12_TO_RGB);
            break;
        case OB_FORMAT_NV21:
            converter->setFormatConvertType(FORMAT_NV21_TO_RGB);
            break;
        case OB_FORMAT_I420:
            converter->setFormatConvertType(FORMAT_I420_TO_RGB);
            break;
        default:
            return nullptr;
    }
    return converter->process(color);
}

sensor_msgs::msg::CameraInfo makeCameraInfo(const OBCameraParam& params,
                                            uint32_t width, uint32_t height) {
    const auto& rgb = params.rgbIntrinsic;
    if (rgb.fx <= 0 || rgb.fy <= 0 || rgb.width <= 0 || rgb.height <= 0) {
        throw std::runtime_error("Orbbec returned invalid RGB intrinsics");
    }
    if (static_cast<uint64_t>(width) * rgb.height !=
        static_cast<uint64_t>(height) * rgb.width) {
        throw std::runtime_error(
            "RGB intrinsics aspect ratio differs from image");
    }
    const double scaleX = static_cast<double>(width) / rgb.width;
    const double scaleY = static_cast<double>(height) / rgb.height;
    sensor_msgs::msg::CameraInfo info;
    info.width = width;
    info.height = height;
    info.k = {rgb.fx * scaleX,
              0.0,
              rgb.cx * scaleX,
              0.0,
              rgb.fy * scaleY,
              rgb.cy * scaleY,
              0.0,
              0.0,
              1.0};
    info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    info.p = {info.k[0], 0.0, info.k[2], 0.0, 0.0, info.k[4],
              info.k[5], 0.0, 0.0,       0.0, 1.0, 0.0};
    // IPM currently uses K only. Distortion coefficients are intentionally
    // unspecified unless an exact ROS distortion-model mapping is established.
    return info;
}

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<rclcpp::Node>("orbbec_image_publisher");
        auto imagePublisher = node->create_publisher<sensor_msgs::msg::Image>(
            kImageTopic, rclcpp::SensorDataQoS());
        auto cameraInfoPublisher =
            node->create_publisher<sensor_msgs::msg::CameraInfo>(
                kCameraInfoTopic, rclcpp::SensorDataQoS());

        auto context = std::make_shared<ob::Context>();
        auto devices = context->queryDeviceList();
        if (!devices || devices->getCount() == 0) {
            throw std::runtime_error("No Orbbec camera found");
        }
        auto pipeline = std::make_shared<ob::Pipeline>(devices->getDevice(0));
        auto streamConfig = std::make_shared<ob::Config>();
        streamConfig->enableStream(OB_SENSOR_COLOR);
        pipeline->start(streamConfig);
        auto converter = std::make_shared<ob::FormatConvertFilter>();
        const auto cameraParams = pipeline->getCameraParam();

        RCLCPP_INFO(node->get_logger(), "Publishing %s and %s", kImageTopic,
                    kCameraInfoTopic);
        while (rclcpp::ok()) {
            std::shared_ptr<ob::FrameSet> frames;
            try {
                frames = pipeline->waitForFrames(2000);
            } catch (const ob::Error& error) {
                RCLCPP_WARN(node->get_logger(), "Orbbec waitForFrames: %s",
                            error.what());
                continue;
            }
            if (!frames) continue;
            auto rgbFrame = ensureRgbFrame(frames->colorFrame(), converter);
            if (!rgbFrame) continue;
            auto videoFrame = rgbFrame->as<ob::VideoFrame>();
            if (!videoFrame) continue;

            const auto width = static_cast<uint32_t>(videoFrame->getWidth());
            const auto height = static_cast<uint32_t>(videoFrame->getHeight());
            const size_t byteCount = static_cast<size_t>(width) * height * 3;
            if (width == 0 || height == 0 ||
                videoFrame->getDataSize() < byteCount) {
                RCLCPP_WARN(node->get_logger(), "Skipping invalid RGB frame");
                continue;
            }
            const auto stamp = node->now();
            sensor_msgs::msg::Image image;
            image.header.stamp = stamp;
            image.header.frame_id = kFrameId;
            image.height = height;
            image.width = width;
            image.encoding = "rgb8";
            image.is_bigendian = false;
            image.step = width * 3;
            image.data.resize(byteCount);
            std::memcpy(image.data.data(), videoFrame->getData(), byteCount);

            auto cameraInfo = makeCameraInfo(cameraParams, width, height);
            cameraInfo.header = image.header;
            cameraInfoPublisher->publish(cameraInfo);
            imagePublisher->publish(image);
        }
        pipeline->stop();
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "orbbec_ros2_publisher: " << error.what() << '\n';
        if (rclcpp::ok()) rclcpp::shutdown();
        return 1;
    }
}
