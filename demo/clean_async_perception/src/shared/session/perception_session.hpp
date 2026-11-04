// 会话层：Orbbec 取流、YOLO 异步推理、结果编码与 ROS2 发布。
#ifndef ASYNC_PERCEPTION_PERCEPTION_SESSION_HPP_
#define ASYNC_PERCEPTION_PERCEPTION_SESSION_HPP_

#include "shared/perception/async_inference_pipeline.hpp"
#include "shared/reporting/perception_result_protocol.hpp"

// -----------------------------------------------------------------------------
// Orbbec 会话配置
// -----------------------------------------------------------------------------
std::shared_ptr<ob::Config> buildPerceptionStreamConfig(
    const std::shared_ptr<ob::Device>& device) {
  auto config = std::make_shared<ob::Config>();
  if (device) {
    auto sensors = device->getSensorList();
    for (uint32_t index = 0; sensors && index < sensors->getCount(); ++index) {
      const auto sensor = sensors->getSensorType(index);
      if (sensor == OB_SENSOR_COLOR) {
        config->enableStream(OB_SENSOR_COLOR);
      } else if (sensor == OB_SENSOR_DEPTH) {
        config->enableStream(OB_SENSOR_DEPTH);
      }
    }
  }
  return config;
}

// -----------------------------------------------------------------------------
// RGB Frame 提取
// -----------------------------------------------------------------------------
std::shared_ptr<ob::Frame> ensureRgbFrame(
    const std::shared_ptr<ob::ColorFrame>& color,
    const std::shared_ptr<ob::FormatConvertFilter>& converter) {
  if (!color || !converter) return nullptr;
  switch (color->getFormat()) {
    case OB_FORMAT_RGB: return color;
    case OB_FORMAT_BGR: converter->setFormatConvertType(FORMAT_BGR_TO_RGB); break;
    case OB_FORMAT_MJPG: converter->setFormatConvertType(FORMAT_MJPG_TO_RGB); break;
    case OB_FORMAT_YUYV: case OB_FORMAT_YUY2: case OB_FORMAT_UYVY:
      converter->setFormatConvertType(FORMAT_YUYV_TO_RGB); break;
    case OB_FORMAT_NV12: converter->setFormatConvertType(FORMAT_NV12_TO_RGB); break;
    case OB_FORMAT_NV21: converter->setFormatConvertType(FORMAT_NV21_TO_RGB); break;
    case OB_FORMAT_I420: converter->setFormatConvertType(FORMAT_I420_TO_RGB); break;
    default: return nullptr;
  }
  return converter->process(color);
}

#endif  // ASYNC_PERCEPTION_PERCEPTION_SESSION_HPP_
