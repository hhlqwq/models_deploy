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

// 让模型理解完整自然语言,程序独立校验原文中的参数.
inline std::string CommandPrompt(const std::string& transcript) {
    return "你是小车语音意图分类器。根据整句识别一个用户要求的动作，理解口语和同义表达。"
        "往前挪两米->前进；退回去半米->后退；向左拐->左转；往右拐->右转；"
        "原地转两圈->转圈；把车停下->停止；放一下第二首歌->播放；"
        "音乐先暂停一下->暂停音乐；接着放->继续播放；别再放音乐了->停止播放。"
        "区分停车与停歌。闲聊、询问、转述、否定运动、冲突、多动作和不支持的要求归为未识别。"
        "不要输出数值、解释或标点，只能输出一个标签："
        "前进、后退、左转、右转、转圈、停止、播放、暂停音乐、继续播放、停止播放、未识别。"
        "用户转录：" + nlohmann::json(transcript).dump(-1, ' ', false);
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

// 只有带唤醒词的专用停车口令直接输出.普通停止由模型判断.
inline bool IsEmergencyStop(const std::string& transcript, const std::string& wake_word) {
    return NormalizeCommand(transcript, wake_word, true) == "停车";
}

// 判断原文是否还含有数量线索,防止识别失败后套用默认值.
inline bool HasQuantityHint(std::string text) {
    // “一下”用作语气词,例如“往前挪一下”.
    static const std::regex polite("一下");
    text = std::regex_replace(text, polite, "");
    static const std::regex hint(
        R"([0-9]|０|１|２|３|４|５|６|７|８|９|零|一|二|两|三|四|五|六|七|八|九|十|百|千|万|亿|半|米|度|°|几|多少)");
    return std::regex_search(text, hint);
}

// 从原文提取唯一的数量及单位,校验它是否适合模型选出的动作.
inline Command ApplyParameter(Command command, const std::string& transcript) {
    // 中文数字按 UTF-8 词项匹配,避免在单个汉字中切分字节.
    static const std::regex explicit_quantity(
        R"(([-+]?\s*(?:[0-9]+(?:\.[0-9]+)?|(?:零|一|二|两|三|四|五|六|七|八|九|十|百|千|万|亿|点|半)+))\s*(厘米|米|度|°|圈|首))");
    std::smatch match;
    if (!std::regex_search(transcript, match, explicit_quantity)) {
        return command.code == "play" || HasQuantityHint(transcript) ? Command{} : command;
    }
    const auto before = match.prefix().str();
    auto after = match.suffix().str();
    const auto unit = match[2].str();
    const bool has_extra_half = unit == "米" && HasPrefix(after, "半");
    if (has_extra_half) {
        after.erase(0, std::string("半").size());
    }
    static const std::regex invalid_prefix(R"((负|负数|−|－|\.)\s*$)");
    if (!command.parameter || HasQuantityHint(before) || HasQuantityHint(after) ||
        std::regex_search(before, invalid_prefix)) {
        return {};
    }
    const auto value = ParseQuantity(Trim(match[1].str()));
    if (!value || *value <= 0 || !std::isfinite(*value)) {
        return {};
    }

    if (has_extra_half && std::floor(*value) != *value) {
        return {};
    }
    const std::string parameter = command.parameter;
    if (parameter == "distance_m" && (unit == "米" || unit == "厘米")) {
        command.value = unit == "厘米" ? *value / 100 : *value + (has_extra_half ? 0.5 : 0);
    } else if (parameter == "angle_deg" && (unit == "度" || unit == "°")) {
        command.value = *value;
    } else if (parameter == "angle_deg" && unit == "圈" && std::isfinite(*value * 360)) {
        command.value = *value * 360;
    } else if (command.code == "play" && unit == "首" && std::floor(*value) == *value && *value <= 9999) {
        command.value = *value;
    } else {
        return {};
    }
    return command;
}

// 接受完整标签,由原始转录决定参数,保留媒体与小车的输出边界.
inline Command ModelCommand(const std::string& raw, const std::string& transcript) {
    const auto close = raw.find("</think>");
    const auto answer = Trim(close == std::string::npos ? raw : raw.substr(close + 8));
    const auto motion = MotionCommands().find(answer);
    if (motion != MotionCommands().end()) {
        return ApplyParameter(motion->second, transcript);
    }
    static const std::map<std::string, Command> media{
        {"播放", {"play", "播放", "track_number", 0, true}},
        {"暂停音乐", {"pause", "暂停音乐", nullptr, 0, true}},
        {"继续播放", {"resume", "继续播放", nullptr, 0, true}},
        {"停止播放", {"stop_playback", "停止播放", nullptr, 0, true}},
    };
    const auto result = media.find(answer);
    return result == media.end() ? Command{} : ApplyParameter(result->second, transcript);
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
