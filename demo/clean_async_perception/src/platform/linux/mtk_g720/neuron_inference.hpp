#pragma once

#include <neuron/api/RuntimeAPI.h>

#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>

namespace g720 {

// 保留实际 Runtime 调用和错误码, 便于定位模型加载及推理问题.
inline void check(int status, const char* operation) {
    if (status != 0) {
        throw std::runtime_error(std::string("Neuron ") + operation + ": " +
                                 std::to_string(status));
    }
}

// 绑定生成时记录的 NCHW 张量, 校验逻辑大小及末维对齐规则.
class Tensor {
   public:
    Tensor(const cv::FileNode& node, size_t bytes, size_t paddedBytes) {
        node["shape"] >> shape;
        const std::string dtype = node["dtype"];
        if (shape.size() < 2 || shape.size() > 4 || shape.front() != 1 ||
            std::any_of(shape.begin(), shape.end(),
                        [](int d) { return d <= 0 || d > 640; })) {
            throw std::runtime_error("Invalid Neuron tensor shape");
        }
        fp32_ = dtype == "float32";
        unsigned_ = dtype == "uint8";
        if (!fp32_ && !unsigned_ && dtype != "int8") {
            throw std::runtime_error("Unsupported Neuron tensor dtype: " +
                                     dtype);
        }
        if (!fp32_) {
            scale_ = static_cast<float>(node["scale"]);
            zero_ = static_cast<int>(node["zero_point"]);
            if (!std::isfinite(scale_) || scale_ <= 0 ||
                zero_ < (unsigned_ ? 0 : -128) ||
                zero_ > (unsigned_ ? 255 : 127)) {
                throw std::runtime_error(
                    "Invalid Neuron quantization parameters");
            }
        }
        elementBytes_ = fp32_ ? sizeof(float) : 1;
        width_ = static_cast<size_t>(shape.back());
        count_ = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                 std::multiplies<size_t>{});
        const size_t rows = count_ / width_;
        const size_t alignedWidth =
            ((width_ * elementBytes_ + 15) / 16 * 16) / elementBytes_;
        const size_t logicalBytes = count_ * elementBytes_;
        if ((bytes != logicalBytes && bytes != paddedBytes) ||
            (paddedBytes != logicalBytes &&
             paddedBytes != rows * alignedWidth * elementBytes_)) {
            throw std::runtime_error(
                "Neuron tensor bytes do not match the model contract");
        }
        stride_ = paddedBytes / rows / elementBytes_;
        storage.assign(paddedBytes, static_cast<uint8_t>(zero_));
    }

    // 按原生缓冲区步长写入逻辑输入, 补齐区域保留零点.
    void write(const std::vector<float>& values) {
        if (values.size() != count_) {
            throw std::runtime_error("Neuron input element count mismatch");
        }
        for (size_t i = 0; i < count_; ++i) {
            auto* address = storage.data() + offset(i);
            if (fp32_) {
                std::memcpy(address, &values[i], sizeof(float));
            } else {
                const float lo = unsigned_ ? 0.0f : -128.0f;
                const float hi = unsigned_ ? 255.0f : 127.0f;
                const auto quantized =
                    std::clamp(std::round(values[i] / scale_) + zero_, lo, hi);
                *address = static_cast<uint8_t>(static_cast<int>(quantized));
            }
        }
    }

    // 跳过物理补齐, 还原逻辑张量值.
    float at(size_t i) const {
        const auto* address = storage.data() + offset(i);
        if (fp32_) {
            float value;
            std::memcpy(&value, address, sizeof(value));
            return value;
        }
        const int value = *address;
        return ((unsigned_ || value < 128 ? value : value - 256) - zero_) *
               scale_;
    }

    std::vector<int> shape;
    std::vector<uint8_t> storage;

   private:
    // 把逻辑元素编号映射到实际缓冲区字节位置.
    size_t offset(size_t i) const {
        return (i / width_ * stride_ + i % width_) * elementBytes_;
    }

    size_t width_ = 0;
    size_t stride_ = 0;
    size_t count_ = 0;
    size_t elementBytes_ = 0;
    float scale_ = 1;
    int zero_ = 0;
    bool fp32_ = false;
    bool unsigned_ = false;
};

// 每个模型持有一个硬件 Runtime, 在工作线程中顺序调用.
class Model {
   public:
    Model(const std::filesystem::path& path, const cv::FileNode& contract,
          const std::vector<int>& inputShape,
          const std::vector<std::vector<int>>& expectedOutputs) {
        const EnvOptions options{kEnvOptHardware, Auto, 1, true, true};
        void* handle = nullptr;
        const int status = NeuronRuntime_create(&options, &handle);
        runtime_.reset(handle);
        check(status, "create");
        check(NeuronRuntime_loadNetworkFromFile(handle, path.c_str()),
              "loadNetworkFromFile");
        size_t inputs = 0;
        size_t outputs = 0;
        check(NeuronRuntime_getInputNumber(handle, &inputs), "getInputNumber");
        check(NeuronRuntime_getOutputNumber(handle, &outputs),
              "getOutputNumber");
        const auto outputNodes = contract["outputs"];
        if (inputs != 1 || outputs != expectedOutputs.size() ||
            !outputNodes.isSeq() || outputs != outputNodes.size()) {
            throw std::runtime_error("Neuron I/O count does not match: " +
                                     path.string());
        }
        input_.emplace(makeTensor(contract["input"], false, 0));
        if (input_->shape != inputShape) {
            throw std::runtime_error("Unexpected Neuron input shape: " +
                                     path.string());
        }
        auto remaining = expectedOutputs;
        for (const auto& node : outputNodes) {
            auto tensor = makeTensor(node, true, outputs_.size());
            const auto expected =
                std::find(remaining.begin(), remaining.end(), tensor.shape);
            if (expected == remaining.end()) {
                throw std::runtime_error(
                    "Unexpected or duplicate Neuron output shape: " +
                    path.string());
            }
            remaining.erase(expected);
            outputs_.push_back(std::move(tensor));
        }
    }

    // 绑定输入输出后执行同步 NPU 推理, 实际调用位于应用工作线程.
    void infer(const std::vector<float>& values) {
        input_->write(values);
        auto* handle = runtime_.get();
        constexpr BufferAttribute attribute{NON_ION_FD};
        check(NeuronRuntime_setInput(handle, 0, input_->storage.data(),
                                     input_->storage.size(), attribute),
              "setInput");
        for (size_t i = 0; i < outputs_.size(); ++i) {
            auto& storage = outputs_[i].storage;
            check(NeuronRuntime_setOutput(handle, i, storage.data(),
                                          storage.size(), attribute),
                  "setOutput");
        }
        check(NeuronRuntime_inference(handle), "inference");
    }

    // 按模型语义取张量, 支持编译器调整多个输出的顺序.
    const Tensor& output(const std::vector<int>& shape) const {
        return *std::find_if(
            outputs_.begin(), outputs_.end(),
            [&shape](const Tensor& tensor) { return tensor.shape == shape; });
    }

   private:
    // 从 SDK 获取实际字节数, 保留对 DLA 与元数据不匹配的检查.
    Tensor makeTensor(const cv::FileNode& node, bool output,
                      size_t index) const {
        auto* handle = runtime_.get();
        size_t bytes = 0;
        size_t padded = 0;
        const auto getSize =
            output ? NeuronRuntime_getOutputSize : NeuronRuntime_getInputSize;
        const auto getPadded = output ? NeuronRuntime_getOutputPaddedSize
                                      : NeuronRuntime_getInputPaddedSize;
        check(getSize(handle, index, &bytes), "get tensor size");
        check(getPadded(handle, index, &padded), "get padded tensor size");
        return Tensor(node, bytes, padded);
    }

    std::optional<Tensor> input_;
    std::vector<Tensor> outputs_;
    std::unique_ptr<void, decltype(&NeuronRuntime_release)> runtime_{
        nullptr, NeuronRuntime_release};
};

constexpr std::array<int, 4> kStrides{4, 8, 16, 32};

// 根据实测 ONNX 契约声明各尺度的框回归和分数张量.
inline std::vector<std::vector<int>> detectorShapes() {
    std::vector<std::vector<int>> shapes;
    for (const int stride : kStrides) {
        const int grid = kModelWidth / stride;
        shapes.push_back({1, 4, grid, grid});
        shapes.push_back({1, 6, grid, grid});
    }
    return shapes;
}

// 将 objectness 和类别 logits 转成概率.
inline float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

// 计算两个原图坐标框的 IoU.
inline float iou(const Detection::Box& a, const Detection::Box& b) {
    const float w = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float h = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float intersection = w * h;
    return intersection / ((a.x2 - a.x1) * (a.y2 - a.y1) +
                           (b.x2 - b.x1) * (b.y2 - b.y1) - intersection);
}

// 解码四个网格头, 恢复原图坐标, 执行按类别的 NMS.
inline YoloxResult detect(Model& model, const YoloxInput& image) {
    model.infer(image.nchw);
    std::vector<Detection> candidates;
    for (const int stride : kStrides) {
        const int grid = kModelWidth / stride;
        const size_t plane = grid * grid;
        const auto& boxes = model.output({1, 4, grid, grid});
        const auto& scores = model.output({1, 6, grid, grid});
        for (size_t index = 0; index < plane; ++index) {
            int label = 0;
            float best = scores.at(plane + index);
            for (int c = 1; c < 5; ++c) {
                const float score = scores.at((c + 1) * plane + index);
                if (score > best) {
                    best = score;
                    label = c;
                }
            }
            const float confidence = sigmoid(scores.at(index)) * sigmoid(best);
            if (!std::isfinite(confidence) || confidence < 0.5f) {
                continue;
            }
            const float cx = (boxes.at(index) + index % grid) * stride;
            const float cy = (boxes.at(plane + index) + index / grid) * stride;
            const float halfWidth =
                std::exp(boxes.at(2 * plane + index)) * stride / 2;
            const float halfHeight =
                std::exp(boxes.at(3 * plane + index)) * stride / 2;
            if (!std::isfinite(cx) || !std::isfinite(cy) ||
                !std::isfinite(halfWidth) || !std::isfinite(halfHeight)) {
                continue;
            }
            const auto& lb = image.letterbox;
            const Detection::Box box{
                std::clamp((cx - halfWidth - lb.padX) / lb.scale, 0.0f,
                           float(image.originalWidth)),
                std::clamp((cy - halfHeight - lb.padY) / lb.scale, 0.0f,
                           float(image.originalHeight)),
                std::clamp((cx + halfWidth - lb.padX) / lb.scale, 0.0f,
                           float(image.originalWidth)),
                std::clamp((cy + halfHeight - lb.padY) / lb.scale, 0.0f,
                           float(image.originalHeight))};
            if (box.x1 < box.x2 && box.y1 < box.y2) {
                candidates.push_back({box, label, confidence});
            }
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Detection& a, const Detection& b) {
                         return a.confidence > b.confidence;
                     });
    YoloxResult result;
    const size_t limit = std::min<size_t>(candidates.size(), 3000);
    for (size_t index = 0; index < limit && result.result.size() < 300;
         ++index) {
        const auto& candidate = candidates[index];
        if (std::none_of(result.result.begin(), result.result.end(),
                         [&](const Detection& kept) {
                             return kept.type_id == candidate.type_id &&
                                    iou(kept.bbox, candidate.bbox) > 0.45f;
                         })) {
            result.result.push_back(candidate);
        }
    }
    return result;
}

// 按地面分类训练链路裁剪、缩放及归一化, 输出 RGB NCHW.
inline std::vector<float> prepareClassification(const cv::Mat& bgr) {
    constexpr int side = 224;
    const int square = std::min(bgr.rows, bgr.cols);
    const cv::Rect crop((bgr.cols - square) / 2, (bgr.rows - square) / 2,
                        square, square);
    cv::Mat resized;
    cv::resize(bgr(crop), resized, {320, 320});
    cv::resize(resized, resized, {256, 256});
    constexpr std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    constexpr std::array<float, 3> stddev{0.229f, 0.224f, 0.225f};
    std::vector<float> input(3 * side * side);
    for (int row = 0; row < side; ++row) {
        const auto* pixels = resized.ptr<cv::Vec3b>(row + 16) + 16;
        for (int col = 0; col < side; ++col) {
            for (int channel = 0; channel < 3; ++channel) {
                input[(channel * side + row) * side + col] =
                    (pixels[col][2 - channel] / 255.0f - mean[channel]) /
                    stddev[channel];
            }
        }
    }
    return input;
}

// 单工作线程串行使用两个硬件实例, 避免同一 Runtime 并发访问.
class Worker {
   public:
    explicit Worker(const std::filesystem::path& directory)
        : contract_((directory / "tensors.json").string(),
                    cv::FileStorage::READ),
          detector_(directory / "yolox.dla", contract_["yolox"],
                    {1, 3, 640, 640}, detectorShapes()),
          classifier_(directory / "mobilenetv2.dla", contract_["mobilenetv2"],
                      {1, 3, 224, 224}, {{1, 2}}),
          thread_(&Worker::run, this) {}

    // 先停止接收新任务, 等在途 NPU 调用结束后再释放 Runtime.
    ~Worker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_one();
        thread_.join();
    }

    // 队列最多容纳一帧的两项待执行任务.
    std::shared_ptr<PendingInference> submit(InferenceTask type,
                                             const cv::Mat& bgr,
                                             std::string* error) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= 2 || bgr.empty() || bgr.type() != CV_8UC3) {
            if (error) *error = "Neuron queue full or invalid BGR frame";
            return nullptr;
        }
        auto pending = std::make_shared<PendingInference>();
        pending->sourceBgr = bgr;
        queue_.emplace_back(type, pending);
        wake_.notify_one();
        return pending;
    }

   private:
    // 取任务、执行并通知主循环, 任务自持图像直到执行结束.
    void run() {
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
            if (stopping_) {
                return;
            }
            auto [type, pending] = std::move(queue_.front());
            queue_.pop_front();
            lock.unlock();

            InferResult result;
            result.task_type = type;
            std::string error;
            try {
                if (type == TID_YOLOX) {
                    result.yolox = detect(
                        detector_, prepareYoloxInput(pending->sourceBgr));
                } else {
                    classifier_.infer(
                        prepareClassification(pending->sourceBgr));
                    const auto& logits = classifier_.output({1, 2});
                    const float a = logits.at(0);
                    const float b = logits.at(1);
                    if (!std::isfinite(a) || !std::isfinite(b)) {
                        throw std::runtime_error(
                            "MobileNetV2 returned nonfinite logits");
                    }
                    result.mobilenetv2 = {a >= b ? 0 : 1,
                                          sigmoid(std::abs(a - b))};
                }
            } catch (const std::exception& e) {
                error = e.what();
            }
            {
                std::lock_guard<std::mutex> done(pending->mutex);
                pending->result = std::move(result);
                pending->valid = error.empty();
                pending->error = std::move(error);
                pending->done = true;
            }
            pending->cv.notify_one();
        }
    }

    cv::FileStorage contract_;
    Model detector_;
    Model classifier_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::pair<InferenceTask, std::shared_ptr<PendingInference>>>
        queue_;
    bool stopping_ = false;
    std::thread thread_;
};

inline std::unique_ptr<Worker> worker;

}  // namespace g720

// 初始化两个业务模型及任务线程.
inline void initializeNeuronInference(const std::filesystem::path& directory) {
    g720::worker = std::make_unique<g720::Worker>(directory);
}

// 停止任务线程并释放 G720 模型.
inline void releasePerceptionInference() { g720::worker.reset(); }

// 提交五类检测任务.
inline std::shared_ptr<PendingInference> submitYolox(const cv::Mat& bgr,
                                                     std::string* error) {
    return g720::worker->submit(TID_YOLOX, bgr, error);
}

// 提交地面分类任务.
inline std::shared_ptr<PendingInference> submitMobilenetV2(const cv::Mat& bgr,
                                                           std::string* error) {
    return g720::worker->submit(TID_MOBILENETV2, bgr, error);
}
