#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace voice_control {

// 无新增模型依赖的首版能量分句,不宣称是 Silero 或神经 VAD.
class SpeechSegmenter {
public:
    // 保存 200ms 前置音频,时长参数按 16k 样本数统一换算.
    SpeechSegmenter(float threshold, int silence_ms, int min_ms, int max_ms)
        : threshold_(threshold), silence_limit_(silence_ms * 16),
          minimum_(min_ms * 16), maximum_(max_ms * 16) {
        if (threshold <= 0 || threshold >= 1 || silence_ms < 100 || min_ms < 20 ||
            max_ms <= min_ms + silence_ms || max_ms > 30000) {
            throw std::runtime_error("音频分句参数无效.");
        }
    }

    // 根据帧 RMS 判断起止,超长语句整句丢弃直到静音,避免输出截断动作.
    bool Feed(const std::vector<float>& frame, std::vector<float>& utterance) {
        float power = 0;
        for (float value : frame) {
            power += value * value;
        }
        const bool speech = !frame.empty() && std::sqrt(power / frame.size()) >= threshold_;
        if (dropping_) {
            silence_ = speech ? 0 : silence_ + frame.size();
            if (silence_ >= silence_limit_) {
                dropping_ = false;
                silence_ = 0;
            }
            return false;
        }
        if (audio_.empty() && !speech) {
            prefix_.insert(prefix_.end(), frame.begin(), frame.end());
            while (prefix_.size() > 3200) {
                prefix_.pop_front();
            }
            return false;
        }
        if (audio_.empty()) {
            audio_.assign(prefix_.begin(), prefix_.end());
            prefix_.clear();
            std::cout << "[监听] 检测到声音." << std::endl;
        }
        audio_.insert(audio_.end(), frame.begin(), frame.end());
        speech_samples_ += speech ? frame.size() : 0;
        silence_ = speech ? 0 : silence_ + frame.size();
        if (audio_.size() >= maximum_) {
            std::cout << "[丢弃] 语句超长,等待静音后重新监听." << std::endl;
            audio_.clear();
            speech_samples_ = 0;
            silence_ = 0;
            dropping_ = true;
        } else if (silence_ >= silence_limit_) {
            const bool valid = speech_samples_ >= minimum_;
            if (valid) {
                utterance.swap(audio_);
            }
            audio_.clear();
            speech_samples_ = 0;
            silence_ = 0;
            return valid;
        }
        return false;
    }

private:
    float threshold_;
    std::size_t silence_limit_;
    std::size_t minimum_;
    std::size_t maximum_;
    std::size_t silence_ = 0;
    std::size_t speech_samples_ = 0;
    bool dropping_ = false;
    std::deque<float> prefix_;
    std::vector<float> audio_;
};

}  // 命名空间 voice_control
