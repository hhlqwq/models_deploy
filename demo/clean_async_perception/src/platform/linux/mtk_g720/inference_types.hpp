#pragma once

#include <vector>

// 应用结果所需字段, 对应检测和地面分类的公共消费接口.
enum InferenceTask { TID_YOLOX, TID_MOBILENETV2 };

struct Detection {
    struct Box {
        float x1;
        float y1;
        float x2;
        float y2;
    } bbox;
    int type_id;
    float confidence;
};

struct YoloxResult {
    std::vector<Detection> result;
};

struct InferResult {
    InferenceTask task_type{};
    YoloxResult yolox;
    struct {
        int class_id = -1;
        float confidence = 0.0f;
    } mobilenetv2;
};
