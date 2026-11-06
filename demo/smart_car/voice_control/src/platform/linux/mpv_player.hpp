#pragma once

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

#include "shared/command.hpp"

namespace voice_control {

class MpvPlayer {
public:
    // 空 socket 时只输出媒体意图,曲库中的路径相对配置文件.
    MpvPlayer(const nlohmann::json& config, const std::filesystem::path& config_root)
        : socket_path_(config.value("ipc_socket", std::string{})),
          device_(config.value("output_device", std::string{})) {
        if (!socket_path_.empty()) {
            socket_path_ = (config_root / socket_path_).lexically_normal().string();
        }
        for (const auto& track : config.value("tracks", std::vector<std::string>{})) {
            if (track.empty() || std::filesystem::path(track).extension() != ".mp3") {
                throw std::runtime_error("media.tracks 只能配置本地 .mp3 路径.");
            }
            tracks_.push_back((config_root / track).lexically_normal());
        }
    }

    // 只从曲库选文件,把 mpv 应答与识别意图分开报告.
    nlohmann::json Execute(const Command& command) const {
        if (socket_path_.empty()) {
            return {{"status", "intent_only"}};
        }
        try {
            const auto deadline = Clock::now() + std::chrono::milliseconds(500);
            if (command.code == "play") {
                if (!std::isfinite(command.value) || std::floor(command.value) != command.value ||
                    command.value < 1 || command.value > tracks_.size()) {
                    throw std::runtime_error("曲目编号不在已配置的曲库中.");
                }
                const auto& track = tracks_[static_cast<std::size_t>(command.value) - 1];
                if (!std::filesystem::is_regular_file(track)) {
                    throw std::runtime_error("预置 MP3 不存在: " + track.string());
                }
                if (!device_.empty()) {
                    Send({"set_property", "audio-device", device_}, deadline);
                }
                Send({"loadfile", track.string(), "replace"}, deadline);
                Send({"set_property", "pause", false}, deadline);
            } else if (command.code == "stop_playback") {
                Send({"stop"}, deadline);
            } else {
                Send({"set_property", "pause", command.code == "pause"}, deadline);
            }
            return {{"status", "accepted"}};
        } catch (const std::exception& error) {
            return {{"status", "error"}, {"error", error.what()}};
        }
    }

private:
    using Clock = std::chrono::steady_clock;

    struct Socket {
        int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);

        // 所有退出路径均释放本次 IPC 连接.
        ~Socket() {
            if (fd >= 0) {
                close(fd);
            }
        }
    };

    // 用同一个截止时刻限制整条 IPC 命令,避免播放器卡住推理线程.
    static void Wait(int fd, short events, Clock::time_point deadline) {
        pollfd descriptor{fd, events, 0};
        while (true) {
            const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now());
            if (remaining.count() <= 0) {
                throw std::runtime_error("mpv IPC 应答超时.");
            }
            const auto ready = poll(&descriptor, 1, static_cast<int>(remaining.count()));
            if (ready > 0 && (descriptor.revents & events)) {
                return;
            }
            if (ready < 0 && errno == EINTR) {
                continue;
            }
            if (ready == 0) {
                throw std::runtime_error("mpv IPC 应答超时.");
            }
            throw std::runtime_error("mpv IPC 连接中断.");
        }
    }

    // 发送固定格式的 JSON 命令,跳过异步事件并检查对应请求的应答.
    void Send(const nlohmann::json& command, Clock::time_point deadline) const {
        sockaddr_un address{};
        if (socket_path_.size() >= sizeof(address.sun_path)) {
            throw std::runtime_error("mpv IPC 路径过长.");
        }
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, socket_path_.c_str(), socket_path_.size() + 1);
        Socket connection;
        const int fd = connection.fd;
        if (fd < 0 || connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
            throw std::runtime_error("无法连接 mpv IPC: " + socket_path_);
        }
        const nlohmann::json request{{"command", command}, {"request_id", 1}};
        const auto payload = request.dump() + '\n';
        std::size_t offset = 0;
        while (offset < payload.size()) {
            Wait(fd, POLLOUT, deadline);
            const auto size = send(fd, payload.data() + offset, payload.size() - offset, MSG_NOSIGNAL);
            if (size < 0 && (errno == EINTR || errno == EAGAIN)) {
                continue;
            }
            if (size <= 0) {
                throw std::runtime_error("无法发送 mpv IPC 命令.");
            }
            offset += static_cast<std::size_t>(size);
        }
        std::string pending;
        while (pending.size() < 16384) {
            Wait(fd, POLLIN, deadline);
            char buffer[1024];
            const auto size = recv(fd, buffer, sizeof(buffer), 0);
            if (size < 0 && (errno == EINTR || errno == EAGAIN)) {
                continue;
            }
            if (size <= 0) {
                throw std::runtime_error("无法读取 mpv IPC 应答.");
            }
            pending.append(buffer, static_cast<std::size_t>(size));
            for (auto end = pending.find('\n'); end != std::string::npos; end = pending.find('\n')) {
                const auto reply = nlohmann::json::parse(pending.substr(0, end), nullptr, false);
                pending.erase(0, end + 1);
                if (!reply.is_object() || reply.value("request_id", 0) != 1) {
                    continue;
                }
                const auto error = reply.value("error", std::string{"应答缺少状态"});
                if (error != "success") {
                    throw std::runtime_error("mpv 拒绝命令: " + error);
                }
                return;
            }
        }
        throw std::runtime_error("mpv IPC 应答过长.");
    }

    std::string socket_path_;
    std::string device_;
    std::vector<std::filesystem::path> tracks_;
};

}  // 命名空间 voice_control
