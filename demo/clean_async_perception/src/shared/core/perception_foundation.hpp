// Shared infrastructure: SDK includes, logging, runtime state, performance metrics.
#ifndef ASYNC_PERCEPTION_SHARED_CORE_PERCEPTION_FOUNDATION_HPP_
#define ASYNC_PERCEPTION_SHARED_CORE_PERCEPTION_FOUNDATION_HPP_

#include <async_infer.h>
#include <async_infer_error.h>
#include <async_log.h>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Utils.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits.h>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// ============================================================================
// YOLO model input size
// ============================================================================
constexpr int kModelWidth = 640;
constexpr int kModelHeight = 640;
constexpr int kModelChannels = 3;
constexpr auto kInferenceTimeout = std::chrono::seconds(30);

// ============================================================================
// Letterbox geometry info
// ============================================================================
struct LetterboxInfo {
  float scale = 1.0f;
  int padX = 0;
  int padY = 0;
};

// ============================================================================
// Async log queue
// ============================================================================
struct LogRecord {
  int level{AIS_LOG_INFO};
  std::string message;
};

class AsyncLogQueue {
 public:
  void push(std::string message, int level = AIS_LOG_INFO) {
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.push_back({level, std::move(message)});
  }
  void drainTo(std::deque<LogRecord>* visible) {
    if (!visible) return;
    std::vector<LogRecord> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending.insert(pending.end(),
          std::make_move_iterator(messages_.begin()),
          std::make_move_iterator(messages_.end()));
      messages_.clear();
    }
    for (auto& m : pending) visible->push_back(std::move(m));
  }
 private:
  std::mutex mutex_;
  std::deque<LogRecord> messages_;
};

inline AsyncLogQueue gLogQueue;

inline void flushPendingLogsToFile() {
  std::deque<LogRecord> logs;
  gLogQueue.drainTo(&logs);
  if (logs.empty()) return;
  const auto logDir = std::filesystem::current_path() / "Log";
  std::filesystem::create_directories(logDir);
  std::ofstream out(logDir / "OrbbecSDK.log.txt", std::ios::app);
  for (const auto& r : logs) out << "[" << r.level << "] " << r.message << "\n";
}

// ============================================================================
// Wide / UTF-8 conversion helpers (Linux simplified)
// ============================================================================
inline std::wstring utf8ToWide(const std::string& s) {
  return std::wstring(s.begin(), s.end());
}
inline std::string wideToUtf8(const std::wstring& w) {
  return std::string(w.begin(), w.end());
}

// ============================================================================
// Run options
// ============================================================================
struct RunOptions {
  std::string bagPath;
  std::string outputDirectory;
  std::string liveDeviceUid;
  uint32_t maxFrames = 0;
  bool noDisplay = false;
};

// ============================================================================
// Rate counter for FPS / timing metrics
// ============================================================================
struct RateCounter {
  using Clock = std::chrono::steady_clock;
  void add(std::chrono::steady_clock::duration d) {
    std::lock_guard<std::mutex> lock(m_);
    if (samples_.size() >= 60) samples_.pop_front();
    samples_.push_back(d);
  }
  double fps() const {
    std::lock_guard<std::mutex> lock(m_);
    if (samples_.size() < 2) return 0.0;
    auto total = std::accumulate(samples_.begin(), samples_.end(),
                                 std::chrono::steady_clock::duration{});
    double sec = std::chrono::duration<double>(total).count()
                 / static_cast<double>(samples_.size());
    return sec > 0.0 ? 1.0 / sec : 0.0;
  }
  double milliseconds() const {
    std::lock_guard<std::mutex> lock(m_);
    if (samples_.empty()) return 0.0;
    auto total = std::accumulate(samples_.begin(), samples_.end(),
                                 std::chrono::steady_clock::duration{});
    return std::chrono::duration<double, std::milli>(total).count()
           / static_cast<double>(samples_.size());
  }
 private:
  mutable std::mutex m_;
  std::deque<std::chrono::steady_clock::duration> samples_;
};

// ============================================================================
// Runtime metrics
// ============================================================================
struct RuntimeMetrics {
  double preprocessMs = 0.0;
  double inferenceFps = 0.0;
  double inferenceMs = 0.0;
  double fusionMs = 0.0;
  double endToEndFps = 0.0;
  double endToEndMs = 0.0;
  uint64_t frameCount = 0;
  int progressPermille = 0;
  double cpuPercent = 0.0;
  bool gpuAvailable = false;
  double gpuPercent = 0.0;
};

// ============================================================================
// CPU usage sampler
// ============================================================================
struct CpuMeter {
  double sample() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - last_).count();
    double cpu_time = static_cast<double>(
        ru.ru_utime.tv_sec + ru.ru_stime.tv_sec +
        (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e6);
    last_ = now;
    double result = (elapsed > 0.0)
        ? std::clamp((cpu_time - lastCpu_) / elapsed * 100.0, 0.0, 100.0)
        : 0.0;
    lastCpu_ = cpu_time;
    return result;
  }
 private:
  std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
  double lastCpu_ = 0.0;
};

}  // namespace

#endif  // ASYNC_PERCEPTION_SHARED_CORE_PERCEPTION_FOUNDATION_HPP_