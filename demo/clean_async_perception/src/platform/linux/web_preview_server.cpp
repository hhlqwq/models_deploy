#include "platform/linux/web_preview_server.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <sstream>
#include <vector>

namespace {

constexpr char kBoundary[] = "async-perception-frame";
constexpr size_t kMaxRequestBytes = 8192;

const char kIndexHtml[] = R"HTML(<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Async Perception Preview</title><style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}body{margin:0;background:#10151c;color:#e9f1fa}
header{display:flex;align-items:center;justify-content:space-between;padding:14px 20px;background:#17212c;border-bottom:1px solid #2b3a49}
h1{font-size:18px;margin:0}.state{color:#82d9a0}.wrap{padding:18px;max-width:1600px;margin:auto}img{display:block;width:100%;height:auto;background:#080b0f;border-radius:8px}
footer{color:#718397;font-size:12px;padding-top:12px}
</style></head><body><header><h1>舜宇扫地机器人 · 推理预览</h1><span id="state" class="state">连接中</span></header>
<main class="wrap"><img src="/stream.mjpg" alt="等待推理画面">
<footer>左侧为原始 RGB，右侧为实例/语义分割融合结果。服务仅提供只读预览。</footer></main>
<script>async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw 0;const s=await r.json();
document.getElementById('state').textContent=s.has_frame?'实时':'等待首帧';}catch(e){document.getElementById('state').textContent='状态接口断开';}}refresh();setInterval(refresh,2000);</script>
</body></html>)HTML";

bool sendAll(int fd, const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  size_t offset = 0;
  while (offset < size) {
    const ssize_t sent = ::send(fd, bytes + offset, size - offset, MSG_NOSIGNAL);
    if (sent > 0) {
      offset += static_cast<size_t>(sent);
      continue;
    }
    if (sent < 0 && errno == EINTR) continue;
    if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      pollfd writable{fd, POLLOUT, 0};
      if (::poll(&writable, 1, 200) > 0) continue;
    }
    return false;
  }
  return true;
}

bool sendText(int fd, const std::string& status, const std::string& contentType,
              const std::string& body) {
  std::ostringstream response;
  response << "HTTP/1.1 " << status << "\r\nContent-Type: " << contentType
           << "\r\nContent-Length: " << body.size()
           << "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
           << body;
  const auto data = response.str();
  return sendAll(fd, data.data(), data.size());
}

std::string makeStatusJson(const WebPreviewMetrics& metrics, bool hasFrame, uint64_t sequence) {
  std::ostringstream json;
  json << std::fixed << std::setprecision(3) << "{\"has_frame\":" << (hasFrame ? "true" : "false")
       << ",\"stream_sequence\":" << sequence << ",\"frame_count\":" << metrics.frameCount
       << ",\"end_to_end_fps\":" << metrics.endToEndFps
       << ",\"inference_fps\":" << metrics.inferenceFps
       << ",\"inference_ms\":" << metrics.inferenceMs
       << ",\"preprocess_ms\":" << metrics.preprocessMs << ",\"fusion_ms\":" << metrics.fusionMs
       << ",\"cpu_percent\":" << metrics.cpuPercent
       << ",\"gpu_available\":" << (metrics.gpuAvailable ? "true" : "false")
       << ",\"gpu_percent\":" << metrics.gpuPercent << '}';
  return json.str();
}

struct Client {
  int fd{-1};
  bool streaming{false};
  uint64_t lastSequence{0};
  std::string request;
};

void closeClient(Client* client) {
  if (client && client->fd >= 0) {
    ::shutdown(client->fd, SHUT_RDWR);
    ::close(client->fd);
    client->fd = -1;
  }
}

}  // namespace

WebPreviewServer::~WebPreviewServer() {
  stop();
}

bool WebPreviewServer::start(const std::string& bindAddress, uint16_t port, int jpegQuality,
                             int maxFps, std::string* error) {
  if (running_.load()) return true;
  const int socketFd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socketFd < 0) {
    if (error) *error = std::string("cannot create preview socket: ") + std::strerror(errno);
    return false;
  }
  int reuse = 1;
  ::setsockopt(socketFd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  const int flags = ::fcntl(socketFd, F_GETFL, 0);
  ::fcntl(socketFd, F_SETFL, flags | O_NONBLOCK);

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  if (::inet_pton(AF_INET, bindAddress.c_str(), &address.sin_addr) != 1) {
    if (error) *error = "web_preview.bind_address must be an IPv4 address";
    ::close(socketFd);
    return false;
  }
  if (::bind(socketFd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
      ::listen(socketFd, 8) != 0) {
    if (error)
      *error = std::string("cannot listen on ") + bindAddress + ':' + std::to_string(port) + ": " +
               std::strerror(errno);
    ::close(socketFd);
    return false;
  }

  listenFd_ = socketFd;
  port_ = port;
  jpegQuality_ = std::clamp(jpegQuality, 10, 100);
  maxFps_ = std::clamp(maxFps, 1, 30);
  running_.store(true);
  encoderThread_ = std::thread(&WebPreviewServer::encodeLoop, this);
  serverThread_ = std::thread(&WebPreviewServer::serveLoop, this);
  return true;
}

void WebPreviewServer::publish(const cv::Mat& frame, const WebPreviewMetrics& metrics) {
  if (!running_.load() || frame.empty()) return;
  const auto now = std::chrono::steady_clock::now();
  const auto interval = std::chrono::milliseconds(1000 / maxFps_);
  {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    if (lastQueuedAt_.time_since_epoch().count() != 0 && now - lastQueuedAt_ < interval) return;
    pendingFrame_ = frame.clone();
    pendingMetrics_ = metrics;
    lastQueuedAt_ = now;
  }
  pendingCondition_.notify_one();
}

void WebPreviewServer::stop() {
  if (!running_.exchange(false)) return;
  pendingCondition_.notify_all();
  if (listenFd_ >= 0) ::shutdown(listenFd_, SHUT_RDWR);
  if (encoderThread_.joinable()) encoderThread_.join();
  if (serverThread_.joinable()) serverThread_.join();
  if (listenFd_ >= 0) {
    ::close(listenFd_);
    listenFd_ = -1;
  }
}

void WebPreviewServer::encodeLoop() {
  while (running_.load()) {
    cv::Mat frame;
    WebPreviewMetrics metrics;
    {
      std::unique_lock<std::mutex> lock(pendingMutex_);
      pendingCondition_.wait(lock, [&] { return !running_.load() || !pendingFrame_.empty(); });
      if (!running_.load()) break;
      frame = std::move(pendingFrame_);
      metrics = pendingMetrics_;
    }
    std::vector<unsigned char> jpeg;
    if (!cv::imencode(".jpg", frame, jpeg, {cv::IMWRITE_JPEG_QUALITY, jpegQuality_})) continue;
    {
      std::lock_guard<std::mutex> lock(encodedMutex_);
      encodedFrame_ = std::move(jpeg);
      encodedMetrics_ = metrics;
      ++encodedSequence_;
    }
  }
}

void WebPreviewServer::serveLoop() {
  std::vector<Client> clients;
  while (running_.load()) {
    std::vector<pollfd> descriptors;
    descriptors.reserve(clients.size() + 1);
    descriptors.push_back({listenFd_, POLLIN, 0});
    for (const auto& client : clients)
      descriptors.push_back({client.fd, static_cast<short>(client.streaming ? 0 : POLLIN), 0});
    ::poll(descriptors.data(), descriptors.size(), 40);

    if (!descriptors.empty() && (descriptors[0].revents & POLLIN)) {
      while (true) {
        const int fd = ::accept(listenFd_, nullptr, nullptr);
        if (fd < 0) break;
        const int flags = ::fcntl(fd, F_GETFL, 0);
        ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        clients.push_back(Client{fd, false, 0, {}});
      }
    }

    for (size_t index = 0; index < clients.size(); ++index) {
      auto& client = clients[index];
      const short events = descriptors.size() > index + 1 ? descriptors[index + 1].revents : 0;
      if (events & (POLLERR | POLLHUP | POLLNVAL)) {
        closeClient(&client);
        continue;
      }
      if (!client.streaming && (events & POLLIN)) {
        char buffer[2048];
        const ssize_t count = ::recv(client.fd, buffer, sizeof(buffer), 0);
        if (count <= 0) {
          if (count == 0 || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR))
            closeClient(&client);
          continue;
        }
        client.request.append(buffer, static_cast<size_t>(count));
        if (client.request.size() > kMaxRequestBytes) {
          sendText(client.fd, "431 Request Header Fields Too Large", "text/plain",
                   "request too large\n");
          closeClient(&client);
          continue;
        }
        if (client.request.find("\r\n\r\n") == std::string::npos) continue;

        const auto firstSpace = client.request.find(' ');
        const auto secondSpace = firstSpace == std::string::npos
                                     ? std::string::npos
                                     : client.request.find(' ', firstSpace + 1);
        const std::string path =
            secondSpace == std::string::npos
                ? std::string()
                : client.request.substr(firstSpace + 1, secondSpace - firstSpace - 1);
        if (path == "/" || path == "/index.html") {
          sendText(client.fd, "200 OK", "text/html; charset=utf-8", kIndexHtml);
          closeClient(&client);
        } else if (path == "/api/status") {
          WebPreviewMetrics metrics;
          uint64_t sequence = 0;
          bool hasFrame = false;
          {
            std::lock_guard<std::mutex> lock(encodedMutex_);
            metrics = encodedMetrics_;
            sequence = encodedSequence_;
            hasFrame = !encodedFrame_.empty();
          }
          sendText(client.fd, "200 OK", "application/json",
                   makeStatusJson(metrics, hasFrame, sequence));
          closeClient(&client);
        } else if (path == "/stream.mjpg") {
          std::ostringstream header;
          header << "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary="
                 << kBoundary << "\r\nCache-Control: no-store, no-cache, must-revalidate\r\n"
                 << "Connection: close\r\n\r\n";
          const auto value = header.str();
          if (sendAll(client.fd, value.data(), value.size()))
            client.streaming = true;
          else
            closeClient(&client);
        } else {
          sendText(client.fd, "404 Not Found", "text/plain", "not found\n");
          closeClient(&client);
        }
      }
    }

    std::vector<unsigned char> jpeg;
    uint64_t sequence = 0;
    {
      std::lock_guard<std::mutex> lock(encodedMutex_);
      sequence = encodedSequence_;
      if (!encodedFrame_.empty()) jpeg = encodedFrame_;
    }
    if (!jpeg.empty()) {
      for (auto& client : clients) {
        if (client.fd < 0 || !client.streaming || client.lastSequence == sequence) continue;
        std::ostringstream partHeader;
        partHeader << "--" << kBoundary
                   << "\r\nContent-Type: image/jpeg\r\nContent-Length: " << jpeg.size()
                   << "\r\n\r\n";
        const auto header = partHeader.str();
        if (!sendAll(client.fd, header.data(), header.size()) ||
            !sendAll(client.fd, jpeg.data(), jpeg.size()) || !sendAll(client.fd, "\r\n", 2)) {
          closeClient(&client);
        } else {
          client.lastSequence = sequence;
        }
      }
    }
    clients.erase(std::remove_if(clients.begin(), clients.end(),
                                 [](const Client& client) { return client.fd < 0; }),
                  clients.end());
  }
  for (auto& client : clients) closeClient(&client);
}
