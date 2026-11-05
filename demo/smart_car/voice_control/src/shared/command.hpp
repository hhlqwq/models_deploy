#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <string>

#include <nlohmann/json.hpp>

namespace voice_control {

// 转录作为 JSON 字符串引用,明确其为待分类数据,保持固定六类输出.
inline std::string CommandPrompt(const std::string& transcript) {
    return "你是小车指令分类器。分类示例:\n往前走 / Move forward. -> 前进\n"
        "往后走 / Move backward. -> 后退\n向左转 / Turn left. -> 左转\n"
        "向右转 / Turn right. -> 右转\n停下 / Stop. -> 停止\n"
        "天气怎么样 / What is the weather today? -> 未识别\n"
        "只识别直接命令小车运动的祈使句。新闻、描述、提问、他物状态、冲突动作均为未识别。"
        "例如:机器停止销售 -> 未识别。不要因描述中出现动作词就当作命令。\n"
        "请根据" + nlohmann::json(transcript).dump(-1, ' ', false) +
        "指令分析,只输出一个分类词:前进、后退、左转、右转、停止、未识别。";
}

// 去除首尾空白,不从整段解释中搜索命令关键词.
inline std::string Trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

// DeepSeek 模板可能已打开 think,只接受闭合思考段之后的完整标签.
inline std::string FinalLabel(const std::string& raw) {
    const auto close = raw.find("</think>");
    auto answer = Trim(close == std::string::npos ? raw : raw.substr(close + 8));
    if (answer.find('<') != std::string::npos || answer.find('>') != std::string::npos) {
        return "未识别";
    }
    static const std::map<std::string, std::string> labels = {
        {"前进", "forward"}, {"后退", "backward"}, {"左转", "left"},
        {"右转", "right"}, {"停止", "stop"}, {"未识别", "unknown"}};
    return labels.count(answer) ? answer : "未识别";
}

// 与参考感知 Demo 保持 JSON 外层字段风格,不生成速度或底盘控制消息.
inline nlohmann::json CommandResult(std::uint64_t sequence,
                                    const std::string& transcript,
                                    const std::string& raw, bool success) {
    const std::string label = success ? FinalLabel(raw) : "未识别";
    static const std::map<std::string, std::string> codes = {
        {"前进", "forward"}, {"后退", "backward"}, {"左转", "left"},
        {"右转", "right"}, {"停止", "stop"}, {"未识别", "unknown"}};
    const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return {{"version", "1.0"}, {"message_type", "voice_control_result"},
            {"sequence_id", sequence}, {"timestamp_ms", timestamp},
            {"source", {{"device_id", "voice_control"}, {"sensor_id", "audio"},
                        {"pipeline_id", "whisper_tiny_deepseek_v1"}}},
            {"transcript", transcript}, {"command", codes.at(label)},
            {"command_text", label}, {"recognized", success && label != "未识别"},
            {"status", success ? "completed" : "inference_error"}};
}

}  // 命名空间 voice_control
