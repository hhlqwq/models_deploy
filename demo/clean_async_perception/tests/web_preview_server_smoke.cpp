#include "platform/linux/web_preview_server.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
  const int durationSeconds = argc > 1 ? std::max(1, std::stoi(argv[1])) : 3;
  WebPreviewServer server;
  std::string error;
  if (!server.start("0.0.0.0", 18080, 75, 10, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  for (uint64_t index = 1; index <= static_cast<uint64_t>(durationSeconds) * 10; ++index) {
    cv::Mat original(360, 480, CV_8UC3, cv::Scalar(42, 48, 55));
    const int objectX = 50 + static_cast<int>((index * 5) % 300);
    cv::rectangle(original, {0, 255, 480, 105}, cv::Scalar(62, 70, 76), cv::FILLED);
    cv::circle(original, {objectX + 45, 250}, 32, cv::Scalar(70, 145, 215), cv::FILLED,
               cv::LINE_AA);
    cv::putText(original, "ORIGINAL RGB", {18, 34}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(235, 235, 235), 2, cv::LINE_AA);

    cv::Mat overlay = original.clone();
    cv::Mat maskLayer = overlay.clone();
    cv::circle(maskLayer, {objectX + 45, 250}, 32, cv::Scalar(40, 220, 90), cv::FILLED,
               cv::LINE_AA);
    cv::addWeighted(maskLayer, 0.45, overlay, 0.55, 0.0, overlay);
    cv::rectangle(overlay, {objectX + 12, 217, 66, 66}, cv::Scalar(40, 255, 100), 2);
    cv::putText(overlay, "class-1 0.93", {objectX + 12, 207}, cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(40, 255, 100), 1, cv::LINE_AA);
    cv::putText(overlay, "INFERENCE OVERLAY", {18, 34}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(80, 230, 130), 2, cv::LINE_AA);

    cv::Mat frame;
    cv::hconcat(original, overlay, frame);
    WebPreviewMetrics metrics;
    metrics.frameCount = index;
    metrics.endToEndFps = 10.0;
    metrics.inferenceFps = 18.4;
    metrics.inferenceMs = 12.5 + static_cast<double>(index % 8) * 0.1;
    metrics.cpuPercent = 16.0 + static_cast<double>(index % 12) * 0.2;
    server.publish(frame, metrics);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  server.stop();
  return 0;
}
