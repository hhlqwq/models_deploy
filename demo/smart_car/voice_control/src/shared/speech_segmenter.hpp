#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace voice_control {

class SpeechSegmenter {
public:
    static constexpr std::size_t kFrameSamples = 512;
    struct Timing {
        double compute_ms = 0;
        double tail_audio_ms = 0;
    };

    // Silero v5.1.2 使用 16kHz、512 样本帧、64 样本上下文和循环状态.
    SpeechSegmenter(const std::string& model, float threshold,
                    int silence_ms, int min_ms, int max_ms)
        : threshold_(threshold), env_(ORT_LOGGING_LEVEL_WARNING, "silero_vad"),
          session_(env_, model.c_str(), CpuOptions()) {
        if (!(threshold > 0.15F && threshold < 1) || silence_ms < 100 ||
            silence_ms > 10000 || min_ms < 20 || min_ms > 10000 ||
            max_ms <= min_ms + silence_ms || max_ms > 30000) {
            throw std::runtime_error("Silero 分句参数无效.");
        }
        silence_limit_ = static_cast<std::size_t>(silence_ms) * 16;
        minimum_ = static_cast<std::size_t>(min_ms) * 16;
        maximum_ = static_cast<std::size_t>(max_ms) * 16;
        Ort::AllocatorWithDefaultOptions allocator;
        if (session_.GetInputCount() != 3 || session_.GetOutputCount() != 2) {
            throw std::runtime_error("Silero ONNX 输入输出数量不符.");
        }
        for (std::size_t i = 0; i < output_names_.size(); ++i) {
            output_names_[i] = session_.GetOutputNameAllocated(i, allocator).get();
        }
        std::cout << "[加载] Silero VAD,CPU Runtime " << Ort::GetVersionString() << std::endl;
    }

    // 文件之间清空循环状态,一个输入流内跨帧保留上下文.
    void Reset() {
        state_.fill(0);
        context_.fill(0);
        prefix_.clear();
        audio_.clear();
        speech_samples_ = 0;
        silence_ = 0;
        dropping_ = false;
        dropped_ = 0;
        timing_ = {};
    }

    // 最后一帧可不足 512 样本;补零只供 VAD 使用,不送入转录音频.
    bool Feed(const std::vector<float>& frame, std::vector<float>& utterance) {
        if (frame.empty() || frame.size() > kFrameSamples) {
            throw std::runtime_error("Silero 音频帧长度无效.");
        }
        const auto started = std::chrono::steady_clock::now();
        const auto probability = Probability(frame);
        timing_.compute_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        const bool active = !audio_.empty() || dropping_;
        const bool speech = probability >= (active ? threshold_ - 0.15F : threshold_);
        if (dropping_) {
            silence_ = speech ? 0 : silence_ + frame.size();
            if (silence_ >= silence_limit_) {
                dropping_ = false;
                silence_ = 0;
            }
            return false;
        }
        if (!active && !speech) {
            prefix_.insert(prefix_.end(), frame.begin(), frame.end());
            while (prefix_.size() > 3200) {
                prefix_.pop_front();
            }
            return false;
        }
        if (!active) {
            audio_.assign(prefix_.begin(), prefix_.end());
            prefix_.clear();
        }
        audio_.insert(audio_.end(), frame.begin(), frame.end());
        speech_samples_ += speech ? frame.size() : 0;
        silence_ = speech ? 0 : silence_ + frame.size();
        if (silence_ >= silence_limit_) {
            return Finish(utterance);
        }
        if (audio_.size() > maximum_) {
            std::cerr << "[丢弃] 单段语音超过 ASR 上限,等待静音后继续." << std::endl;
            ++dropped_;
            audio_.clear();
            speech_samples_ = 0;
            silence_ = 0;
            dropping_ = true;
        }
        return false;
    }

    // 文件结束时提交仍在积累的有效尾段,只保留少量尾部静音.
    bool Finish(std::vector<float>& utterance) {
        const bool valid = speech_samples_ >= minimum_ && !audio_.empty();
        if (valid) {
            audio_.resize(audio_.size() - (silence_ - std::min(silence_, std::size_t{1600})));
            if (audio_.size() <= maximum_) {
                timing_.tail_audio_ms = silence_ / 16.0;
                utterance.swap(audio_);
            } else {
                std::cerr << "[丢弃] 单段语音超过 ASR 上限." << std::endl;
                ++dropped_;
                audio_.clear();
                speech_samples_ = silence_ = 0;
                return false;
            }
        }
        audio_.clear();
        speech_samples_ = silence_ = 0;
        return valid;
    }

    // 供文件模式报告超长语句,避免把缺失结果当作完整处理.
    std::size_t Dropped() const { return dropped_; }

    // 取出自上次结果以来的 VAD 计时,包含期间的静音帧推理.
    Timing TakeTiming() {
        const auto timing = timing_;
        timing_ = {};
        return timing;
    }

private:
    // 只启用默认 CPU 提供者,限制推理线程以减少与语音链路争抢.
    static Ort::SessionOptions CpuOptions() {
        if (Ort::GetVersionString() != "1.20.1") {
            throw std::runtime_error("Silero 需使用配套 ONNX Runtime 1.20.1.");
        }
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        return options;
    }

    // 一次运行返回人声概率和下一帧的循环状态.
    float Probability(const std::vector<float>& frame) {
        std::array<float, 64 + kFrameSamples> input{};
        std::copy(context_.begin(), context_.end(), input.begin());
        std::copy(frame.begin(), frame.end(), input.begin() + context_.size());
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<std::int64_t, 2> input_shape{1, static_cast<std::int64_t>(input.size())};
        const std::array<std::int64_t, 3> state_shape{2, 1, 128};
        std::int64_t sample_rate = 16000;
        std::array<Ort::Value, 3> tensors{
            Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), input_shape.data(), input_shape.size()),
            Ort::Value::CreateTensor<float>(memory, state_.data(), state_.size(), state_shape.data(), state_shape.size()),
            Ort::Value::CreateTensor<std::int64_t>(memory, &sample_rate, 1, nullptr, 0),
        };
        const std::array<const char*, 3> names{"input", "state", "sr"};
        const std::array<const char*, 2> outputs{output_names_[0].c_str(), output_names_[1].c_str()};
        auto result = session_.Run(Ort::RunOptions{nullptr}, names.data(), tensors.data(), tensors.size(),
                                   outputs.data(), outputs.size());
        const auto probability_info = result[0].GetTensorTypeAndShapeInfo();
        const auto state_info = result[1].GetTensorTypeAndShapeInfo();
        if (probability_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            probability_info.GetElementCount() != 1 ||
            state_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            state_info.GetShape() != std::vector<std::int64_t>(state_shape.begin(), state_shape.end())) {
            throw std::runtime_error("Silero ONNX 输出契约不符.");
        }
        const float probability = result[0].GetTensorData<float>()[0];
        if (!std::isfinite(probability) || probability < 0 || probability > 1) {
            throw std::runtime_error("Silero 返回无效人声概率.");
        }
        std::copy_n(result[1].GetTensorData<float>(), state_.size(), state_.begin());
        std::copy(input.end() - context_.size(), input.end(), context_.begin());
        return probability;
    }

    float threshold_;
    Timing timing_;
    Ort::Env env_;
    Ort::Session session_;
    std::array<std::string, 2> output_names_;
    std::array<float, 256> state_{};
    std::array<float, 64> context_{};
    std::size_t silence_limit_ = 0;
    std::size_t minimum_ = 0;
    std::size_t maximum_ = 0;
    std::size_t silence_ = 0;
    std::size_t speech_samples_ = 0;
    std::size_t dropped_ = 0;
    bool dropping_ = false;
    std::deque<float> prefix_;
    std::vector<float> audio_;
};

}  // 命名空间 voice_control
