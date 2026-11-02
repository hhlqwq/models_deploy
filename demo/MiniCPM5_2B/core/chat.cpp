// 维护终端会话与历史,模型加载和硬件操作由引擎接口负责.
#include "chat.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace minicpm {
namespace {

// 暂存分词解码末尾的替换字符,防止不完整中文字节提前显示.
std::string StableText(std::string text) {
    const std::string replacement = "\xef\xbf\xbd";
    while (text.size() >= replacement.size() &&
           text.compare(text.size() - replacement.size(), replacement.size(), replacement) == 0) {
        text.resize(text.size() - replacement.size());
    }
    return text;
}

// 只追加确定的文本前缀,不通过退格改写已经显示的回答.
void PrintDelta(const std::string& text, std::string& printed, std::ostream& output) {
    if (text.compare(0, printed.size(), printed) != 0) {
        throw std::runtime_error("分词解码改写了已输出前缀,本轮停止.");
    }
    output << text.substr(printed.size()) << std::flush;
    printed = text;
}

}  // 匿名命名空间.

// 历史达到容量后拒绝本轮输入,保留已有对话供用户选择重置.
void Chat(Engine& engine, size_t max_new_tokens, std::istream& input,
          std::ostream& output) {
    if (max_new_tokens == 0 || max_new_tokens >= engine.ContextSize()) {
        throw std::invalid_argument("生成上限必须大于 0 且小于上下文容量.");
    }
    std::vector<Message> history;
    output << "[就绪] 输入文本开始对话; /reset 清空历史; /exit 退出.\n";
    std::string line;
    while (output << "\n用户> " << std::flush, std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "/exit") break;
        if (line == "/reset") {
            history.clear();
            output << "[会话] 历史已清空,模型继续常驻.\n";
            continue;
        }
        if (line.find_first_not_of(" \t") == std::string::npos) continue;
        auto messages = history;
        messages.push_back({"user", line});
        const auto tokens = engine.Encode(messages);
        if (tokens.empty() || tokens.size() > engine.ContextSize() - max_new_tokens) {
            output << "[上下文不足] 输入 " << tokens.size() << " Token,需要预留 "
                   << max_new_tokens << ",容量 " << engine.ContextSize()
                   << ". 请缩短输入或执行 /reset,本轮未写入历史.\n";
            continue;
        }
        output << "[Prefill] " << tokens.size() << " Token.\n" << std::flush;
        std::string printed;
        bool labelled = false;
        auto result = engine.Generate(tokens, max_new_tokens,
            [&](const std::vector<int32_t>& generated) {
                if (!labelled) {
                    output << "助手> " << std::flush;
                    labelled = true;
                }
                PrintDelta(StableText(engine.Decode(generated)), printed, output);
            });
        const auto answer = StableText(engine.Decode(result.tokens));
        if (!labelled) output << "助手> ";
        PrintDelta(answer, printed, output);
        messages.push_back({"assistant", answer});
        history = std::move(messages);
        output << "\n[结束] " << result.stop_reason << ",生成 " << result.tokens.size()
               << " 个内容 Token;历史 " << history.size() / 2 << " 轮.\n";
    }
    output << "\n[退出] 释放本进程模型资源.\n";
}

}  // 命名空间 minicpm.
