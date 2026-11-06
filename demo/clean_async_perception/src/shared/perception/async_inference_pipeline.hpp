// Intel YOLOX 异步推理层：前处理、回调与结果收集。
#ifndef ASYNC_PERCEPTION_ASYNC_INFERENCE_PIPELINE_HPP_
#define ASYNC_PERCEPTION_ASYNC_INFERENCE_PIPELINE_HPP_

#include "shared/core/perception_foundation.hpp"

#ifdef ASYNC_PERCEPTION_MTK_G720
#include "platform/linux/mtk_g720/inference_types.hpp"
#endif

// -----------------------------------------------------------------------------
// 异步推理等待上下文
// -----------------------------------------------------------------------------
struct PendingInference {
    std::mutex mutex;
    std::condition_variable cv;
    InferResult result{};
    std::vector<float> modelInput;  // keep YOLOX buffer alive until callback
    cv::Mat sourceBgr;  // keep MobileNetV2 image alive until callback
    std::string error;
    bool valid = false;
    bool done = false;
};

#ifndef ASYNC_PERCEPTION_MTK_G720
// -----------------------------------------------------------------------------
// 异步任务完成通知
// -----------------------------------------------------------------------------
void inferenceCallback(const InferResult* result, bool valid, void* context) {
    // The callback owns a reference even if the caller stops waiting on timeout.
    std::unique_ptr<std::shared_ptr<PendingInference>> owner(
        static_cast<std::shared_ptr<PendingInference>*>(context));
    if (!owner || !*owner) return;
    const auto& pending = *owner;
    {
        std::lock_guard<std::mutex> lock(pending->mutex);
        if (result) pending->result = *result;
        pending->valid = valid;
        pending->done = true;
    }
    pending->cv.notify_one();
}

AsyncInferStatus submitRequest(
    const SubmitParams& request,
    const std::shared_ptr<PendingInference>& pending) {
    auto* owner = new std::shared_ptr<PendingInference>(pending);
    const auto status = async_infer_submit(&request, inferenceCallback, owner);
    if (status != AIS_OK) delete owner;
    return status;
}
#endif

// -----------------------------------------------------------------------------
// YOLOX 前处理：BGR + 左上角 letterbox(114) + FP32 NCHW [0,255]
// -----------------------------------------------------------------------------
struct YoloxInput {
    std::vector<float> nchw;  // FP32 NCHW buffer, ready for SDK
    LetterboxInfo letterbox;  // scale/pad for reference
    int originalWidth = 0;
    int originalHeight = 0;
};

YoloxInput prepareYoloxInput(const cv::Mat& sourceBgr, int fillValue = 114) {
    YoloxInput input;
    input.originalWidth = sourceBgr.cols;
    input.originalHeight = sourceBgr.rows;

    const float scale =
        std::min(static_cast<float>(kModelWidth) / sourceBgr.cols,
                 static_cast<float>(kModelHeight) / sourceBgr.rows);
    const int resizedWidth =
        std::max(1, static_cast<int>(sourceBgr.cols * scale));
    const int resizedHeight =
        std::max(1, static_cast<int>(sourceBgr.rows * scale));
    const int padX = 0;
    const int padY = 0;
    input.letterbox = {scale, padX, padY};

    cv::Mat canvas(kModelHeight, kModelWidth, CV_8UC3,
                   cv::Scalar(fillValue, fillValue, fillValue));
    cv::Mat resized;
    cv::resize(sourceBgr, resized, cv::Size(resizedWidth, resizedHeight), 0.0,
               0.0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(0, 0, resizedWidth, resizedHeight)));

    // 官方 YOLOX preproc 保持 OpenCV BGR 通道与 0..255 数值范围。
    cv::Mat floatImg;
    canvas.convertTo(floatImg, CV_32FC3);

    // Pack HWC → NCHW
    const int h = floatImg.rows, w = floatImg.cols, c = floatImg.channels();
    const size_t total = static_cast<size_t>(h) * w * c;
    input.nchw.resize(total);
    for (int row = 0; row < h; ++row) {
        const float* src = reinterpret_cast<const float*>(floatImg.ptr(row));
        for (int col = 0; col < w; ++col) {
            for (int ch = 0; ch < c; ++ch) {
                input.nchw[static_cast<size_t>(ch) * h * w + row * w + col] =
                    src[col * c + ch];
            }
        }
    }
    return input;
}

#ifdef ASYNC_PERCEPTION_MTK_G720
#include "platform/linux/mtk_g720/neuron_inference.hpp"
#else
// -----------------------------------------------------------------------------
// 异步提交 YOLOX 推理
// -----------------------------------------------------------------------------
std::shared_ptr<PendingInference> submitYolox(const cv::Mat& sourceBgr,
                                              std::string* error) {
    auto pending = std::make_shared<PendingInference>();
    auto input = prepareYoloxInput(sourceBgr);

    SubmitParams request{};
    request.task_type = TID_YOLOX;
    pending->modelInput = std::move(input.nchw);
    request.yolox_params.input_data = pending->modelInput.data();
    request.yolox_params.input_element_count = pending->modelInput.size();
    request.yolox_params.input_width = kModelWidth;
    request.yolox_params.input_height = kModelHeight;
    request.yolox_params.input_channels = kModelChannels;
    request.yolox_params.original_width = input.originalWidth;
    request.yolox_params.original_height = input.originalHeight;
    request.yolox_params.scale = input.letterbox.scale;
    request.yolox_params.pad_x = input.letterbox.padX;
    request.yolox_params.pad_y = input.letterbox.padY;

    const auto status = submitRequest(request, pending);
    if (status != AIS_OK) {
        if (error)
            *error = std::string("submit failed: ") +
                     async_infer_err_name(status) + " - " +
                     async_infer_err_desc(status);
        return nullptr;
    }
    return pending;
}

std::shared_ptr<PendingInference> submitMobilenetV2(const cv::Mat& sourceBgr,
                                                    std::string* error) {
    if (sourceBgr.empty() || sourceBgr.type() != CV_8UC3) {
        if (error) *error = "MobileNetV2 requires a nonempty BGR8 image";
        return nullptr;
    }

    auto pending = std::make_shared<PendingInference>();
    pending->sourceBgr = sourceBgr;

    SubmitParams request{};
    request.task_type = TID_MOBILENETV2;
    request.mobilenetv2_params.bgr_data = pending->sourceBgr.data;
    request.mobilenetv2_params.width = pending->sourceBgr.cols;
    request.mobilenetv2_params.height = pending->sourceBgr.rows;
    request.mobilenetv2_params.row_stride = pending->sourceBgr.step[0];

    const auto status = submitRequest(request, pending);
    if (status != AIS_OK) {
        if (error)
            *error = std::string("submit failed: ") +
                     async_infer_err_name(status) + " - " +
                     async_infer_err_desc(status);
        return nullptr;
    }
    return pending;
}

// 释放 Intel 异步推理 SDK.
inline void releasePerceptionInference() { async_infer_release(); }
#endif

bool waitInference(const std::shared_ptr<PendingInference>& pending,
                   std::string* error) {
    if (!pending) return false;
    std::unique_lock<std::mutex> lock(pending->mutex);
    if (!pending->cv.wait_for(lock, kInferenceTimeout,
                              [&] { return pending->done; })) {
        if (error) *error = "inference callback timed out";
        return false;
    }
    if (!pending->valid) {
        if (error)
            *error = pending->error.empty()
                         ? "inference returned invalid result"
                         : pending->error;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// 类别名称与可视化
// -----------------------------------------------------------------------------
std::string garbageClassName(int typeId) {
    switch (typeId) {
        case 0:
            return "liquid";
        case 1:
            return "debris";
        case 2:
            return "paper";
        case 3:
            return "cable";
        case 4:
            return "bottle";
        default:
            return "class-" + std::to_string(typeId);
    }
}

void drawYoloxDetections(cv::Mat* visual, const YoloxResult& yolox) {
    if (!visual) return;
    for (const auto& inst : yolox.result) {
        const int x1 = static_cast<int>(inst.bbox.x1);
        const int y1 = static_cast<int>(inst.bbox.y1);
        const int x2 = static_cast<int>(inst.bbox.x2);
        const int y2 = static_cast<int>(inst.bbox.y2);
        cv::rectangle(*visual, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 255, 0), 2);
        char buf[128];
        const std::string className = garbageClassName(inst.type_id);
        std::snprintf(buf, sizeof(buf), "%s %.2f", className.c_str(),
                      inst.confidence);
        cv::putText(*visual, buf, {x1, std::max(18, y1 - 5)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
    }
}

#endif  // ASYNC_PERCEPTION_ASYNC_INFERENCE_PIPELINE_HPP_
