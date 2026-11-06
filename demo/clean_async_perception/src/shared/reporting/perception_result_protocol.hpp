// 结果协议层：将 YOLOX 检测结果编码为 JSON。
#pragma once

#include <chrono>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include "geometry/ground_coverage.hpp"

// Note: RuntimeMetrics, YoloxResult, garbageClassName are defined in
// perception_foundation.hpp / async_inference_pipeline.hpp which are
// included before this file via the session header chain.

namespace perception_protocol {

struct ResultSource {
    std::string sensorId = "orbbec_rgbd";
    std::string pipelineId = "rgbd_async_v1";
};

struct GroundClassification {
    int classId = -1;
    float confidence = 0.0f;
};

inline std::string escapeJson(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const char character : value) {
        if (character == '"' || character == '\\') escaped.push_back('\\');
        if (character == '\n')
            escaped += "\\n";
        else if (character == '\r')
            escaped += "\\r";
        else if (character != '"' && character != '\\')
            escaped.push_back(character);
    }
    return escaped;
}

inline uint64_t unixTimestampMs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

inline std::string encodeResult(
    uint64_t sequenceId, int frameId, int frameWidth, int frameHeight,
    uint64_t captureTimestampMs, const YoloxResult& yolox,
    const std::optional<GroundClassification>& groundClassification,
    const RuntimeMetrics& metrics, const IpmCalculator* ipm,
    const ResultSource& source, const DirtAssessment& assessment) {
    std::ostringstream json;
    json << "{\"version\":\"1.1\",\"message_type\":\"perception_result\","
         << "\"sequence_id\":" << sequenceId
         << ",\"timestamp_ms\":" << unixTimestampMs()
         << ",\"source\":{\"device_id\":\"async_perception\","
         << "\"sensor_id\":\"" << escapeJson(source.sensorId)
         << "\",\"pipeline_id\":\"" << escapeJson(source.pipelineId) << "\"},"
         << "\"frame\":{\"frame_id\":" << frameId << ",\"width\":" << frameWidth
         << ",\"height\":" << frameHeight
         << ",\"format\":\"RGB\",\"capture_timestamp_ms\":"
         << captureTimestampMs << "},\"perception\":{\"objects\":[";
    bool first = true;
    uint32_t objectId = 0;
    size_t idx = 0;
    for (const auto& inst : yolox.result) {
        if (!first) json << ',';
        first = false;
        const float x1 = inst.bbox.x1;
        const float y1 = inst.bbox.y1;
        const float x2 = inst.bbox.x2;
        const float y2 = inst.bbox.y2;
        const float w = std::max(0.0f, x2 - x1);
        const float h = std::max(0.0f, y2 - y1);
        const float area = w * h;
        json << "{\"id\":" << ++objectId << ",\"category_code\":\""
             << escapeJson(garbageClassName(inst.type_id))
             << "\",\"confidence\":" << std::clamp(inst.confidence, 0.0f, 1.0f)
             << ",\"region\":{\"bbox_xyxy\":[" << x1 << ',' << y1 << ',' << x2
             << ',' << y2 << "],\"width\":" << w << ",\"height\":" << h
             << ",\"area\":" << area << "}";
        // IPM 物理面积
        if (ipm && ipm->isInitialized()) {
            auto pb = ipm->computePhysicalBBox(x1, y1, x2, y2);
            if (pb.valid) {
                json << ",\"physical_bbox\":{\"width_cm\":" << pb.width_cm
                     << ",\"height_cm\":" << pb.height_cm
                     << ",\"area_cm2\":" << pb.area_cm2 << "}";
            }
        }
        json << "}";
        ++idx;
    }
    json << "]";
    if (assessment.valid) {
        json << ",\"coverage_percent\":" << assessment.coveragePercent
             << ",\"dirt_level\":" << assessment.dirtLevel;
    } else {
        json << ",\"coverage_percent\":null,\"dirt_level\":null,"
             << "\"ipm_status\":\"IPM info missing\"";
    }
    json << ",\"ground_classification\":";
    if (groundClassification) {
        const char* categoryCode = groundClassification->classId == 0
                                       ? "industrial_floor"
                                       : "stone_floor";
        json << "{\"category_code\":\"" << categoryCode
             << "\",\"confidence\":" << groundClassification->confidence << "}";
    } else {
        json << "null";
    }
    json << "},\"diagnostics\":{\"frame_count\":" << metrics.frameCount
         << ",\"end_to_end_fps\":" << metrics.endToEndFps
         << ",\"inference_fps\":" << metrics.inferenceFps << "}}";
    return json.str();
}

}  // namespace perception_protocol
