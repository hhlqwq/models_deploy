#pragma once

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct WebPreviewMetrics {
    double preprocessMs{0.0};
    double inferenceFps{0.0};
    double inferenceMs{0.0};
    double fusionMs{0.0};
    double endToEndFps{0.0};
    double cpuPercent{0.0};
    double gpuPercent{0.0};
    bool gpuAvailable{false};
    uint64_t frameCount{0};
};

// Linux 部署端的自包含网页预览服务。感知线程只投递“最新帧”，JPEG 编码和网络
// 发送均在后台线程完成；浏览器慢或断开不会反向阻塞取流/推理 Pipeline。
class WebPreviewServer {
   public:
    WebPreviewServer() = default;
    WebPreviewServer(const WebPreviewServer&) = delete;
    WebPreviewServer& operator=(const WebPreviewServer&) = delete;
    ~WebPreviewServer();

    bool start(const std::string& bindAddress, uint16_t port, int jpegQuality,
               int maxFps, std::string* error);
    void publish(const cv::Mat& frame, const WebPreviewMetrics& metrics);
    void stop();

   private:
    void encodeLoop();
    void serveLoop();

    std::atomic_bool running_{false};
    int listenFd_{-1};
    uint16_t port_{0};
    int jpegQuality_{80};
    int maxFps_{10};
    std::thread encoderThread_;
    std::thread serverThread_;

    std::mutex pendingMutex_;
    std::condition_variable pendingCondition_;
    cv::Mat pendingFrame_;
    WebPreviewMetrics pendingMetrics_;
    std::chrono::steady_clock::time_point lastQueuedAt_{};

    std::mutex encodedMutex_;
    std::vector<unsigned char> encodedFrame_;
    WebPreviewMetrics encodedMetrics_;
    uint64_t encodedSequence_{0};
};
