#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace voice_control {

struct Command {
    std::string code = "unknown";
    std::string label = "未识别";
    const char* parameter = nullptr;
    double value = 0;
    bool media = false;
};

// 将动作码、标签及默认参数统一维护在一张表内.
inline const std::map<std::string, Command>& MotionCommands() {
    static const std::map<std::string, Command> commands{
        {"前进", {"forward", "前进", "distance_m", 1}},
        {"后退", {"backward", "后退", "distance_m", 1}},
        {"左转", {"left", "左转", "angle_deg", 90}},
        {"右转", {"right", "右转", "angle_deg", 90}},
        {"转圈", {"spin", "转圈", "angle_deg", 360}},
        {"停止", {"stop", "停止"}},
    };
    return commands;
}

// 将转录作为待分类数据,只请求无显式参数的单一动作标签.
inline std::string CommandPrompt(const std::string& transcript) {
    return "你是小车指令分类器。只识别直接命令小车运动的单条祈使句。"
        "往前走 / Move forward -> 前进，往后走 / Move backward -> 后退，"
        "向左转 / Turn left -> 左转，向右转 / Turn right -> 右转，"
        "转个圈 -> 转圈，停下 / Stop -> 停止。新闻、描述、提问、否定、冲突动作、加速均为未识别。"
        "请根据" + nlohmann::json(transcript).dump(-1, ' ', false) +
        "指令分析,只输出一个分类词:前进、后退、左转、右转、转圈、停止、未识别。";
}

// 去除首尾空白,保留句中内容供整条匹配.
inline std::string Trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

// 检查完整的 UTF-8 前缀.
inline bool HasPrefix(const std::string& text, const std::string& prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

// 仅去除句尾标点或唤醒词后的分隔符,不改写句中指令.
inline std::string NormalizeCommand(std::string text, const std::string& wake_word,
                                    bool require_wake_word) {
    text = Trim(text);
    for (const std::string punctuation : {"。", "！", "!", "."}) {
        if (text.size() >= punctuation.size() &&
            text.compare(text.size() - punctuation.size(), punctuation.size(), punctuation) == 0) {
            text = Trim(text.substr(0, text.size() - punctuation.size()));
            break;
        }
    }
    if (wake_word.empty() || !HasPrefix(text, wake_word)) {
        return require_wake_word ? std::string{} : text;
    }
    text = Trim(text.substr(wake_word.size()));
    for (const std::string separator : {"，", ",", "：", ":"}) {
        if (HasPrefix(text, separator)) {
            return Trim(text.substr(separator.size()));
        }
    }
    return text;
}

// 解析常用口语数字,包括小数、半和中文个位至千位.
inline std::optional<double> ParseQuantity(const std::string& text) {
    static const std::regex decimal(R"([0-9]+(\.[0-9]+)?)");
    static const std::string digit = "(零|一|二|两|三|四|五|六|七|八|九)";
    static const std::regex chinese(
        "((" + digit + "千)?(" + digit + "百)?(" + digit + "?十)?" + digit +
        "?)(点(" + digit + ")+)?");
    static const std::map<std::string, int> digits{
        {"零", 0}, {"一", 1}, {"二", 2}, {"两", 2}, {"三", 3}, {"四", 4},
        {"五", 5}, {"六", 6}, {"七", 7}, {"八", 8}, {"九", 9},
    };
    if (text == "半") {
        return 0.5;
    }
    if (std::regex_match(text, decimal)) {
        try {
            return std::stod(text);
        } catch (const std::out_of_range&) {
            return std::nullopt;
        }
    }
    std::string chinese_text = text;
    const auto decimal_point = chinese_text.find("点");
    const auto integer = chinese_text.substr(0, decimal_point);
    const auto zero = chinese_text.find("零");
    static const std::regex ambiguous(R"((千|百)(一|二|两|三|四|五|六|七|八|九)$)");
    // 拒绝含义可能不同的简写,例如一百二.
    if (std::regex_search(integer, ambiguous)) {
        return std::nullopt;
    }
    // 中文整数中的占位零不参与计算.
    if (zero != std::string::npos && zero > 0 && zero < decimal_point) {
        static const std::regex placeholder(R"((千|百)零(一|二|两|三|四|五|六|七|八|九)$)");
        if (!std::regex_search(integer, placeholder)) {
            return std::nullopt;
        }
        chinese_text.erase(zero, std::string("零").size());
    }
    if (chinese_text.empty() || !std::regex_match(chinese_text, chinese)) {
        return std::nullopt;
    }
    double value = 0;
    double fraction = 0;
    int current = -1;
    for (std::size_t offset = 0; offset < chinese_text.size(); offset += 3) {
        const auto symbol = chinese_text.substr(offset, 3);
        const auto number = digits.find(symbol);
        if (symbol == "点") {
            value += current < 0 ? 0 : current;
            current = -1;
            fraction = 0.1;
        } else if (number != digits.end()) {
            if (fraction > 0) {
                value += number->second * fraction;
                fraction /= 10;
            } else {
                current = number->second;
            }
        } else {
            const int unit = symbol == "千" ? 1000 : symbol == "百" ? 100 : 10;
            value += (current < 0 ? 1 : current) * unit;
            current = -1;
        }
    }
    return value + (current < 0 ? 0 : current);
}

// 匹配显式单位,缺省参数仅由调用方在后缀为空时填充.
inline std::optional<double> QuantityWithUnit(const std::string& text, const std::string& unit,
                                               double multiplier = 1) {
    if (text.size() <= unit.size() ||
        text.compare(text.size() - unit.size(), unit.size(), unit) != 0) {
        return std::nullopt;
    }
    const auto number = ParseQuantity(Trim(text.substr(0, text.size() - unit.size())));
    if (!number || *number <= 0 || !std::isfinite(*number * multiplier)) {
        return std::nullopt;
    }
    return *number * multiplier;
}

// 返回本地动作或明确拒绝;仅无参数的其他表达允许交给 LLM 分类.
inline std::optional<Command> ParseCommand(const std::string& text) {
    if (text.empty()) {
        return Command{};
    }
    if (text == "停车" || text == "停" || text == "停下" || text == "停止") {
        return MotionCommands().at("停止");
    }
    if (text == "暂停音乐" || text == "暂停播放") {
        return Command{"pause", "暂停音乐", nullptr, 0, true};
    }
    if (text == "继续播放" || text == "继续音乐") {
        return Command{"resume", "继续播放", nullptr, 0, true};
    }
    if (text == "停止播放" || text == "停止音乐") {
        return Command{"stop_playback", "停止播放", nullptr, 0, true};
    }
    const std::string play_prefix = "播放第";
    if (HasPrefix(text, play_prefix)) {
        const auto track = QuantityWithUnit(text.substr(play_prefix.size()), "首");
        if (track && std::floor(*track) == *track && *track <= 9999) {
            return Command{"play", "播放", "track_number", *track, true};
        }
        return Command{};
    }

    for (const auto& [label, command] : MotionCommands()) {
        if (text == label) {
            return command;
        }
        if (command.parameter == nullptr) {
            continue;
        }
        const bool spin = command.code == "spin";
        const std::string prefix = spin && !HasPrefix(text, label) ? "转" : label;
        if (!HasPrefix(text, prefix)) {
            continue;
        }
        const auto quantity = Trim(text.substr(prefix.size()));
        if (spin && quantity == "个圈") {
            return command;
        }
        std::optional<double> value;
        if (command.code == "forward" || command.code == "backward") {
            value = QuantityWithUnit(quantity, "米");
            if (!value) {
                value = QuantityWithUnit(quantity, "厘米", 0.01);
            }
        } else {
            value = QuantityWithUnit(quantity, "度");
            if (!value) {
                value = QuantityWithUnit(quantity, "°");
            }
            if (spin && !value) {
                value = QuantityWithUnit(quantity, "圈", 360);
            }
        }
        if (!value) {
            return Command{};
        }
        auto result = command;
        result.value = *value;
        return result;
    }

    // 未支持的显式参数拒绝执行,避免模型忽略参数后套用默认值.
    static const std::regex parameter_hint(
        R"([0-9]|０|１|２|３|４|５|６|７|８|９|零|一|二|两|三|四|五|六|七|八|九|十|百|千|半|米|度|圈|°|几|多少|加速|播放|音乐|不要|别|不许)");
    if (std::regex_search(text, parameter_hint)) {
        return Command{};
    }
    return std::nullopt;
}

// 只接受闭合思考段之后的完整动作标签.
inline Command ModelCommand(const std::string& raw) {
    const auto close = raw.find("</think>");
    const auto answer = Trim(close == std::string::npos ? raw : raw.substr(close + 8));
    const auto command = MotionCommands().find(answer);
    return command == MotionCommands().end() ? Command{} : command->second;
}

// 保留小车结果外层字段,媒体意图使用独立消息类型.
inline nlohmann::json CommandResult(std::uint64_t sequence, const std::string& transcript,
                                    const Command& command, bool success) {
    const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    nlohmann::json result{
        {"version", "1.0"}, {"message_type", command.media ? "voice_media_intent" : "voice_control_result"},
        {"sequence_id", sequence}, {"timestamp_ms", timestamp},
        {"source", {{"device_id", "voice_control"}, {"sensor_id", "audio"},
                    {"pipeline_id", "whisper_tiny_deepseek_v1"}}},
        {"transcript", transcript}, {"command", command.code}, {"command_text", command.label},
        {"recognized", success && command.code != "unknown"},
        {"status", success ? "completed" : "inference_error"},
    };
    if (success && command.parameter) {
        result["parameters"] = {{command.parameter, command.value}};
    }
    return result;
}

}  // 命名空间 voice_control
