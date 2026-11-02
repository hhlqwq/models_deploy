// 连续对话的公共类型与接口,不暴露 SDK 或第三方库类型.
#ifndef MINICPM5_CHAT_CORE_CHAT_H_
#define MINICPM5_CHAT_CORE_CHAT_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace minicpm {

struct Message {
    std::string role;
    std::string content;
};

struct Generation {
    std::vector<int32_t> tokens;
    std::string stop_reason;
};

class Engine {
 public:
    // 通过接口释放模型资源.
    virtual ~Engine() = default;
    // 将完整历史按模型模板编码,不复用上轮图缓存.
    virtual std::vector<int32_t> Encode(const std::vector<Message>& messages) = 0;
    // 解码生成文本,停止 Token 不属于展示内容.
    virtual std::string Decode(const std::vector<int32_t>& tokens) = 0;
    // 返回部署模型的实际静态上下文容量.
    virtual size_t ContextSize() const = 0;
    // 运行一次生成,每步通过回调提供当前完整 Token 序列.
    virtual Generation Generate(const std::vector<int32_t>& prompt, size_t limit,
        const std::function<void(const std::vector<int32_t>&)>& on_tokens) = 0;
};

// 终端多轮会话,只在成功生成后提交用户和助手消息.
void Chat(Engine& engine, size_t max_new_tokens, std::istream& input,
          std::ostream& output);

}  // 命名空间 minicpm.
#endif
