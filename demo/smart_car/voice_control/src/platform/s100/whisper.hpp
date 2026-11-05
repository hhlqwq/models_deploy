#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "platform/s100/hbm.hpp"

namespace voice_control {

class Whisper {
public:
    // 直接读取已有 tiny 资源,默认中文,保持英文原始规则文件不变.
    Whisper(const std::filesystem::path& root, const std::string& language)
        : encoder_((root / "tiny_encoder_w8a16.hbm").string(),
                   {{"mel", {1, 80, 3000}}}, {{"audio_features", {1, 1500, 384}}}),
          decoder_((root / "tiny_decoder_step_w8a16.hbm").string(), DecoderShapes(false), DecoderShapes(true)) {
        std::ifstream file(root / "decode_config.txt");
        std::string line;
        while (std::getline(file, line)) {
            const auto split = line.find('=');
            if (split != std::string::npos) {
                rules_[line.substr(0, split)] = line.substr(split + 1);
            }
        }
        if (rules_.at("cacheSize") != "256" || rules_.at("maxTokenLength") != "448") {
            throw std::runtime_error("Whisper 解码契约必须为 256 Cache/448 位置.");
        }
        initial_ = ParseIds(rules_.at("initial"));
        if (initial_.size() != 4 || initial_[0] != 50258 || initial_[2] != 50359 || initial_[3] != 50363) {
            throw std::runtime_error("Whisper 初始提示词不符.");
        }
        if (language != "zh" && language != "en") {
            throw std::runtime_error("当前仅接入 zh/en 语言提示.");
        }
        initial_[1] = language == "zh" ? 50260 : 50259;
        suppressed_ = ParseIds(rules_.at("suppress"));
        suppressed_first_ = ParseIds(rules_.at("suppress_first"));
        eot_ = std::stoi(rules_.at("eot"));
        max_generated_ = std::stoi(rules_.at("max_generated_tokens"));
        if (eot_ != 50257 || max_generated_ <= 0 || max_generated_ > 448 - static_cast<int>(initial_.size())) {
            throw std::runtime_error("Whisper EOS 或生成预算不符.");
        }
        filters_.resize(80 * 201);
        std::ifstream mel(root / "mel_filters_f32.bin", std::ios::binary);
        if (!mel.read(reinterpret_cast<char*>(filters_.data()), filters_.size() * sizeof(float)) || mel.peek() != EOF) {
            throw std::runtime_error("Mel 滤波器文件大小不符.");
        }
        std::ifstream vocab(root / "whisper/assets/multilingual.tiktoken");
        std::string encoded;
        int rank;
        while (vocab >> encoded >> rank) {
            ranks_.emplace(rank, Base64(encoded));
        }
        if (ranks_.size() != 50257) {
            throw std::runtime_error("Whisper 多语言词表不完整.");
        }
    }

    // 每句从空缓存开始,满 256 槽后按最后一维左移.
    std::string Transcribe(const std::vector<float>& pcm) {
        if (pcm.empty() || pcm.size() > 480000) {
            throw std::runtime_error("音频必须为 0 到 30 秒的 16k 单声道 PCM.");
        }
        std::cout << "[识别] 计算 Mel,运行 Whisper tiny." << std::endl;
        const auto features = encoder_.Run({{"mel", Mel(pcm)}}).at("audio_features");
        Hbm::Values feeds;
        for (const auto& item : DecoderShapes(false)) {
            std::size_t size = 1;
            for (int dimension : item.second) {
                size *= dimension;
            }
            feeds[item.first].resize(size);
        }
        feeds["audio_features"] = features;
        int token = initial_.front();
        int generated = 0;
        std::string text;
        bool ended = false;
        for (int position = 0; position < 448; ++position) {
            if (position >= 256) {
                for (auto& item : feeds) {
                    if (item.first.find("_cache_") == std::string::npos) {
                        continue;
                    }
                    for (std::size_t offset = 0; offset < item.second.size(); offset += 256) {
                        std::move(item.second.begin() + offset + 1,
                                  item.second.begin() + offset + 256, item.second.begin() + offset);
                        item.second[offset + 255] = 0;
                    }
                }
            }
            for (const auto* name : {"token_onehot", "position_weights", "cache_update_mask"}) {
                std::fill(feeds[name].begin(), feeds[name].end(), 0);
            }
            feeds["token_onehot"].at(token) = 1;
            feeds["position_weights"].at(position) = 1;
            const int slot = std::min(position, 255);
            feeds["cache_update_mask"][slot] = 1;
            auto& mask = feeds["attention_mask"];
            std::fill(mask.begin(), mask.end(), -100);
            std::fill(mask.begin(), mask.begin() + slot + 1, 0);
            auto outputs = decoder_.Run(feeds);
            for (auto& item : feeds) {
                if (item.first.find("_cache_") != std::string::npos) {
                    item.second = std::move(outputs.at(item.first.substr(0, item.first.size() - 3) + "_out"));
                }
            }
            if (position + 1 < static_cast<int>(initial_.size())) {
                token = initial_[position + 1];
                continue;
            }
            auto& logits = outputs.at("logits");
            for (int id : suppressed_) {
                logits.at(id) = -std::numeric_limits<float>::infinity();
            }
            if (!generated) {
                for (int id : suppressed_first_) {
                    logits.at(id) = -std::numeric_limits<float>::infinity();
                }
            }
            token = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (!std::isfinite(logits[token])) {
                throw std::runtime_error("Whisper 没有有效输出 Token.");
            }
            if (token == eot_) {
                ended = true;
                break;
            }
            if (ranks_.count(token)) {
                text += ranks_.at(token);
            }
            if (++generated % 32 == 0) {
                std::cout << "[识别] 已生成 " << generated << " Token." << std::endl;
            }
            if (generated >= max_generated_) {
                break;
            }
        }
        if (!ended) {
            throw std::runtime_error("Whisper 达到位置或生成上限,拒绝提交截断转录.");
        }
        return text;
    }

private:
    // 与已有 Python decoder.py 的名称及维度保持一致.
    static Hbm::Shapes DecoderShapes(bool output) {
        Hbm::Shapes shapes;
        if (output) {
            shapes["logits"] = {1, 51865};
        } else {
            shapes = {{"token_onehot", {1, 51865}}, {"audio_features", {1, 1500, 384}},
                      {"position_weights", {1, 448}}, {"cache_update_mask", {1, 1, 1, 256}},
                      {"attention_mask", {1, 1, 1, 256}}};
        }
        for (int layer = 0; layer < 4; ++layer) {
            for (const auto* kind : {"key", "value"}) {
                shapes[std::string(kind) + "_cache_" + std::to_string(layer) + (output ? "_out" : "_in")] = {6, 1, 64, 256};
            }
        }
        return shapes;
    }

    // 解析原始解码配置中的整数列表.
    static std::vector<int> ParseIds(const std::string& value) {
        std::stringstream stream(value);
        std::string item;
        std::vector<int> result;
        while (std::getline(stream, item, ',')) {
            if (!item.empty()) {
                result.push_back(std::stoi(item));
            }
        }
        return result;
    }

    // 解码已有 OpenAI tiktoken 表中的字节,不引入另一套 tokenizer.
    static std::string Base64(const std::string& encoded) {
        const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string result;
        unsigned value = 0;
        int bits = 0;
        for (char c : encoded) {
            if (c == '=') {
                break;
            }
            const auto index = alphabet.find(c);
            if (index == std::string::npos) {
                throw std::runtime_error("tiktoken 表含无效 Base64.");
            }
            value = (value << 6) | static_cast<unsigned>(index);
            bits += 6;
            if (bits >= 8) {
                bits -= 8;
                result.push_back(static_cast<char>((value >> bits) & 255));
            }
        }
        return result;
    }

    // 400 点混合基 FFT,仅依赖标准库,保持周期 Hann 和中心反射协议.
    static std::vector<std::complex<double>> Fft(const std::vector<std::complex<double>>& input) {
        const std::size_t n = input.size();
        if (n == 1) {
            return input;
        }
        const std::size_t radix = n % 2 == 0 ? 2 : 5;
        const std::size_t m = n / radix;
        std::vector<std::vector<std::complex<double>>> parts(radix);
        for (std::size_t r = 0; r < radix; ++r) {
            std::vector<std::complex<double>> sub(m);
            for (std::size_t j = 0; j < m; ++j) {
                sub[j] = input[j * radix + r];
            }
            parts[r] = Fft(sub);
        }
        std::vector<std::complex<double>> output(n);
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t r = 0; r < radix; ++r) {
                const double angle = -6.283185307179586 * static_cast<double>(r * k) / n;
                output[k] += parts[r][k % m] * std::complex<double>(std::cos(angle), std::sin(angle));
            }
        }
        return output;
    }

public:
    // 公开预处理以核对原实现;30 秒补齐、n_fft=400、hop=160、80 Mel.
    std::vector<float> Mel(const std::vector<float>& pcm) const {
        std::vector<float> audio(480000);
        std::copy(pcm.begin(), pcm.end(), audio.begin());
        std::vector<float> result(80 * 3000);
        float maximum = -std::numeric_limits<float>::infinity();
        std::array<float, 400> window{};
        for (int i = 0; i < 400; ++i) {
            window[i] = 0.5 - 0.5 * std::cos(6.283185307179586 * i / 400);
        }
        for (int frame = 0; frame < 3000; ++frame) {
            std::vector<std::complex<double>> values(400);
            for (int i = 0; i < 400; ++i) {
                int index = frame * 160 + i - 200;
                if (index < 0) {
                    index = -index;
                }
                if (index >= 480000) {
                    index = 959998 - index;
                }
                values[i] = audio[index] * window[i];
            }
            const auto spectrum = Fft(values);
            std::array<float, 201> power{};
            for (int bin = 0; bin < 201; ++bin) {
                power[bin] = static_cast<float>(std::norm(spectrum[bin]));
            }
            for (int band = 0; band < 80; ++band) {
                float energy = 0;
                for (int bin = 0; bin < 201; ++bin) {
                    energy += filters_[band * 201 + bin] * power[bin];
                }
                const float value = std::log10(std::max(energy, 1e-10F));
                result[band * 3000 + frame] = value;
                maximum = std::max(maximum, value);
            }
        }
        for (auto& value : result) {
            value = (std::max(value, maximum - 8) + 4) / 4;
        }
        return result;
    }

private:
    Hbm encoder_;
    Hbm decoder_;
    std::map<std::string, std::string> rules_;
    std::map<int, std::string> ranks_;
    std::vector<int> initial_;
    std::vector<int> suppressed_;
    std::vector<int> suppressed_first_;
    std::vector<float> filters_;
    int eot_ = 0;
    int max_generated_ = 0;
};

}  // 命名空间 voice_control
