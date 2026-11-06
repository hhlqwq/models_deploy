#pragma once

#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "shared/command.hpp"

namespace voice_control {

using CommandPlan = std::vector<std::vector<Command>>;

// 识别明确的复合关系,避免单动作模式只发出句中的一部分.
inline bool HasCompoundRelation(const std::string& text) {
    static const std::regex relation(R"(然后|之后|同时|并且|一边|先.+再)");
    return std::regex_search(text, relation);
}

// 兼容短标签后端,只产生一个完整的单动作结果.
inline CommandPlan ParseSingleCommand(const std::string& raw, const std::string& transcript) {
    if (HasCompoundRelation(transcript)) {
        return {};
    }
    const auto close = raw.find("</think>");
    auto command = CommandFromLabel(
        Trim(close == std::string::npos ? raw : raw.substr(close + 8)), transcript);
    return command.code == "unknown" ? CommandPlan{} : CommandPlan{{std::move(command)}};
}

// 保留整段自然语言.模型负责拆动作、引用原文并标明顺序和并行关系.
inline std::string CommandPrompt(const std::string& transcript) {
    return "把语音命令转成 JSON 数组，只输出 JSON。外层按执行顺序排列，内层是同时发生的动作。"
        "每个动作是 [标签,原文片段]。标签只选：前进、后退、左转、右转、转圈、停止、播放、暂停音乐、继续播放、停止播放。"
        "原文片段按原顺序逐字引用，包含动作和参数，不得改写、遗漏动作或添加数值。"
        "「之后、再」表示先后，「同时」表示同一步。闲聊、否定、矛盾、无法确定关系输出 []。\n"
        "向前开三米 => [[[\"前进\",\"向前开三米\"]]]\n"
        "往后退再左转 => [[[\"后退\",\"往后退\"]],[[\"左转\",\"左转\"]]]\n"
        "往前挪同时放第二首歌 => [[[\"前进\",\"往前挪\"],[\"播放\",\"放第二首歌\"]]]\n"
        "先后退半米再右转然后前进三米同时放第二首歌 => "
        "[[[\"后退\",\"后退半米\"]],[[\"右转\",\"右转\"]],[[\"前进\",\"前进三米\"],[\"播放\",\"放第二首歌\"]]]\n"
        "天气如何 => []\n"
        "输入：" + nlohmann::json(transcript).dump(-1, ' ', false);
}

// 允许模型把完整答案放在一个 JSON 代码块内,拒绝前后附带的解释.
inline std::string PlanAnswer(const std::string& raw) {
    const auto close = raw.find("</think>");
    auto answer = Trim(close == std::string::npos ? raw : raw.substr(close + 8));
    const auto newline = answer.find('\n');
    if (newline != std::string::npos && answer.size() >= newline + 4 &&
        answer.compare(answer.size() - 3, 3, "```") == 0) {
        const auto header = Trim(answer.substr(0, newline));
        if (header == "```json" || header == "```") {
            answer = Trim(answer.substr(newline + 1, answer.size() - newline - 4));
        }
    }
    return answer;
}

// 检查相邻动作之间仅剩连接词或标点,避免漏掉未引用的指令.
inline bool IsPlanConnector(const std::string& gap) {
    static const std::regex connector(
        R"((\s|，|。|、|,|;|；|！|!|：|:|请|先|然后|之后|接着|随后|再|最后|的|时候|时|后|同时|并且|并|和|一边|边|吧|一下)*)");
    return std::regex_match(gap, connector);
}

// 校验模型生成的整份计划.单个动作无效时整份拒绝.
inline std::optional<CommandPlan> ParseCommandPlan(const std::string& raw, const std::string& transcript) {
    const auto answer = PlanAnswer(raw);
    if (answer.size() > 4096) {
        return std::nullopt;
    }
    const auto json = nlohmann::json::parse(answer, nullptr, false);
    if (!json.is_array() || json.size() > 8) {
        return std::nullopt;
    }
    CommandPlan plan;
    std::size_t offset = 0;
    static const std::regex sequence(R"(然后|之后|接着|随后|再|最后|后)");
    static const std::regex parallel(R"(同时|并且|并|和|一边|边)");
    for (const auto& step : json) {
        if (!step.is_array() || step.empty() || step.size() > 2) {
            return std::nullopt;
        }
        std::vector<Command> actions;
        for (const auto& item : step) {
            if (!item.is_array() || item.size() != 2 || !item[0].is_string() || !item[1].is_string()) {
                return std::nullopt;
            }
            const auto fragment = item[1].get<std::string>();
            const auto start = transcript.find(fragment, offset);
            if (fragment.empty() || start == std::string::npos ||
                HasCompoundRelation(fragment)) {
                return std::nullopt;
            }
            const auto gap = transcript.substr(offset, start - offset);
            if (!IsPlanConnector(gap) ||
                (!actions.empty() && std::regex_search(gap, sequence)) ||
                (actions.empty() && !plan.empty() && std::regex_search(gap, parallel))) {
                return std::nullopt;
            }
            auto command = CommandFromLabel(item[0].get<std::string>(), fragment);
            if (command.code == "unknown" || (!actions.empty() && actions.front().media == command.media)) {
                return std::nullopt;
            }
            actions.push_back(std::move(command));
            offset = start + fragment.size();
        }
        plan.push_back(std::move(actions));
    }
    if (!plan.empty() && !IsPlanConnector(transcript.substr(offset))) {
        return std::nullopt;
    }
    return plan;
}

// 保留单动作 JSON 兼容性.复合口令整份发送,不在语音模块调度.
inline nlohmann::json PlanResult(std::uint64_t sequence, const std::string& transcript,
                                 const CommandPlan& plan, bool success) {
    if (plan.empty()) {
        return CommandResult(sequence, transcript, {}, success);
    }
    if (plan.size() == 1 && plan.front().size() == 1) {
        return CommandResult(sequence, transcript, plan.front().front(), success);
    }
    auto result = CommandResult(sequence, transcript, {}, success);
    result["message_type"] = "voice_command_plan";
    result["recognized"] = true;
    result.erase("command");
    result.erase("command_text");
    auto steps = nlohmann::json::array();
    for (const auto& step : plan) {
        auto actions = nlohmann::json::array();
        for (const auto& command : step) {
            nlohmann::json action{{"command", command.code}};
            if (command.parameter) {
                action["parameters"] = {{command.parameter, command.value}};
            }
            actions.push_back(std::move(action));
        }
        steps.push_back(std::move(actions));
    }
    result["steps"] = std::move(steps);
    return result;
}

}  // 命名空间 voice_control
