#pragma once

#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#ifdef VOICE_INPUT_MIC
#include <alsa/asoundlib.h>
#endif

extern char** environ;

namespace voice_control {

// 信号处理只设置标记,旧 SDK 的在途生成完成后退出.
inline std::atomic<bool> stop_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free, "Signal flag must be lock-free");

// 不在信号处理函数内调用 SDK、分配内存或释放设备.
inline void HandleSignal(int) { stop_requested = 1; }

#ifdef VOICE_INPUT_MIC
class Microphone {
public:
    // 由 ALSA plug 设备完成采样格式协商,采集固定 16k 单声道 S16LE.
    explicit Microphone(const std::string& device) {
        const int code = snd_pcm_open(&pcm_, device.c_str(), SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK);
        if (code < 0) {
            throw std::runtime_error(std::string("打开麦克风失败: ") + snd_strerror(code));
        }
        const int setup = snd_pcm_set_params(pcm_, SND_PCM_FORMAT_S16_LE,
                                             SND_PCM_ACCESS_RW_INTERLEAVED, 1, 16000, 1, 100000);
        if (setup < 0) {
            snd_pcm_close(pcm_);
            pcm_ = nullptr;
            throw std::runtime_error(std::string("ALSA 参数设置失败: ") + snd_strerror(setup));
        }
    }

    // 只释放本应用打开的采集句柄.
    ~Microphone() {
        if (pcm_) {
            snd_pcm_close(pcm_);
        }
    }

    Microphone(const Microphone&) = delete;
    Microphone& operator=(const Microphone&) = delete;

    // 读取 20ms 音频,设备断开及溢出明确报错,避免拼接丢失的语句.
    bool Read(std::vector<float>& output) {
        std::array<std::int16_t, 320> samples{};
        std::size_t offset = 0;
        while (offset < samples.size() && !stop_requested) {
            const auto count = snd_pcm_readi(pcm_, samples.data() + offset, samples.size() - offset);
            if (count == -EAGAIN) {
                const int code = snd_pcm_wait(pcm_, 100);
                if (code < 0 && code != -EINTR) {
                    throw std::runtime_error(std::string("ALSA 等待失败: ") + snd_strerror(code));
                }
            } else if (count < 0) {
                throw std::runtime_error(std::string("ALSA 采集失败: ") + snd_strerror(count));
            } else {
                offset += static_cast<std::size_t>(count);
            }
        }
        if (offset != samples.size()) {
            return false;
        }
        output.resize(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            output[i] = samples[i] / 32768.0F;
        }
        return true;
    }

private:
    snd_pcm_t* pcm_ = nullptr;
};
#endif

// 使用已有 FFmpeg 解码本地常见音频容器,不调用外部模型推理程序.
inline std::vector<float> ReadAudioFile(const std::string& filename) {
    if (!std::filesystem::is_regular_file(filename)) {
        throw std::runtime_error("输入必须是存在的本地音频文件.");
    }
    int descriptors[2];
    if (pipe(descriptors) != 0) {
        throw std::runtime_error("无法创建音频解码管道.");
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, descriptors[0]);
    posix_spawn_file_actions_addclose(&actions, descriptors[1]);
    std::vector<std::string> args = {"ffmpeg", "-nostdin", "-v", "error", "-i", filename,
                                    "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"};
    std::vector<char*> argv;
    for (auto& arg : args) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    pid_t pid = -1;
    const int code = posix_spawnp(&pid, "ffmpeg", &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(descriptors[1]);
    if (code != 0) {
        close(descriptors[0]);
        throw std::runtime_error("无法启动已有 FFmpeg: " + std::to_string(code));
    }
    std::vector<char> bytes;
    std::array<char, 16384> buffer{};
    bool too_long = false;
    bool read_failed = false;
    while (!stop_requested) {
        pollfd descriptor{descriptors[0], POLLIN, 0};
        const int ready = poll(&descriptor, 1, 100);
        if (ready < 0 && errno != EINTR) {
            read_failed = true;
            break;
        }
        if (ready <= 0) {
            continue;
        }
        const auto size = read(descriptors[0], buffer.data(), buffer.size());
        if (size == 0) {
            break;
        }
        if (size < 0) {
            if (errno == EINTR) {
                continue;
            }
            read_failed = true;
            break;
        }
        if (bytes.size() + size > 480000 * sizeof(std::int16_t)) {
            too_long = true;
            break;
        }
        bytes.insert(bytes.end(), buffer.begin(), buffer.begin() + size);
    }
    close(descriptors[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            read_failed = true;
            break;
        }
    }
    if (too_long) {
        throw std::runtime_error("Whisper tiny 单片段上限为 30 秒,请提供较短片段;未截断识别.");
    }
    if (read_failed || !WIFEXITED(status) || WEXITSTATUS(status) != 0 || stop_requested ||
        bytes.empty() || bytes.size() % sizeof(std::int16_t) != 0) {
        throw std::runtime_error("音频解码失败、已中断或输入为空.");
    }
    std::vector<float> pcm(bytes.size() / sizeof(std::int16_t));
    // 与现有 Whisper 和 ALSA 一致,先固定 PCM16 再归一化.
    for (std::size_t i = 0; i < pcm.size(); ++i) {
        std::int16_t sample;
        std::memcpy(&sample, bytes.data() + i * sizeof(sample), sizeof(sample));
        pcm[i] = sample / 32768.0F;
    }
    return pcm;
}

}  // 命名空间 voice_control
