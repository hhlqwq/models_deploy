#pragma once

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>

#include <xlm.h>

namespace voice_control {

class DeepSeek {
public:
    struct Timing {
        double first_text_ms = -1;
        double total_ms = -1;
    };

    // 使用旧 SDK 的原始模型、tokenizer 和模板,不修改系统库.
    DeepSeek(const std::string& model, const std::string& tokenizer,
             const std::string& template_path, bool print_raw, bool thinking)
        : model_(model), tokenizer_(tokenizer), print_raw_(print_raw), thinking_(thinking) {
        std::ifstream stream(template_path);
        if (!stream) {
            throw std::runtime_error("无法读取官方聊天模板: " + template_path);
        }
        template_.assign(std::istreambuf_iterator<char>(stream), {});
        if (template_.empty()) {
            throw std::runtime_error("聊天模板为空.");
        }
        if (!thinking_) {
            // 旧 SDK 无独立开关,在内存模板中预填空思考段后直接生成答案.
            const std::string prefix = R"({{'<｜Assistant｜><think>\n'}})";
            const std::string direct = R"({{'<｜Assistant｜><think>\n\n</think>\n\n'}})";
            const auto position = template_.find(prefix);
            if (position == std::string::npos ||
                template_.find(prefix, position + prefix.size()) != std::string::npos) {
                throw std::runtime_error("聊天模板与已验证的 DeepSeek 直答前缀不匹配.");
            }
            template_.replace(position, prefix.size(), direct);
        }
        auto params = xlm_create_default_param();
        params.model_path = model_.c_str();
        params.token_config_path = tokenizer_.c_str();
        params.model_type = XLM_MODEL_TYPE_DEEPSEEK;
        params.k_cache_int8 = false;
        // 上下文采用旧库默认值;固定分类使用官方支持的贪心采样.
        params.sampling.temp = 0.0F;
        std::cerr << "[加载] DeepSeek,旧 SDK 1.0,thinking=" << (thinking_ ? "on" : "off") << std::endl;
        const int code = xlm_init(&params, Callback, &handle_);
        if (code != 0) {
            if (handle_) {
                xlm_destroy(&handle_);
            }
            throw std::runtime_error("xlm_init 失败: " + std::to_string(code));
        }
    }

    // 等同步调用结束后释放模型,不停止其他进程.
    ~DeepSeek() {
        if (handle_) {
            xlm_destroy(&handle_);
        }
    }

    DeepSeek(const DeepSeek&) = delete;
    DeepSeek& operator=(const DeepSeek&) = delete;

    // 下一次推理设置 new_chat,按官方示例清空 SDK 内部会话.
    void Reset() {
        new_chat_ = true;
        std::cerr << "[会话] 下轮开始新会话." << std::endl;
    }

    // 返回上一轮正常完成的原始文本,供固定标签解析而非关键词猜测.
    const std::string& LastText() const { return last_text_; }

    // 使用 SDK 内部多轮缓存,错误后要求新会话,不提交不完整回答.
    bool Ask(const std::string& text, Timing& timing) {
        last_text_.clear();
        xlm_lm_request_t request{};
        // 旧 SDK 示例使用固定槽 0,业务序号独立放在 ROS2 JSON 中.
        request.request_id = 0;
        request.type = XLM_INPUT_PROMPT;
        request.new_chat = new_chat_;
        request.prompt = text.c_str();
        request.chat_template = template_.c_str();
        request.infer_backend = XLM_INFER_BACKEND_BPU_ANY;
        xlm_input_t input{};
        input.request_num = 1;
        input.requests = &request;
        Reply reply;
        reply.print_raw = print_raw_;
        reply.started = std::chrono::steady_clock::now();
        std::cerr << "[生成] 分析语音指令." << std::endl;
        const int code = xlm_infer(handle_, &input, &reply);
        // 实际旧库可能在完成回调前返回,保持请求及 userdata 存活到终态.
        std::unique_lock<std::mutex> lock(reply.mutex);
        if (code == 0) {
            while (!reply.ended && !reply.failed) {
                if (reply.ready.wait_for(lock, std::chrono::seconds(10)) == std::cv_status::timeout) {
                    std::cerr << "[生成] 等待旧 SDK 完成回调." << std::endl;
                }
            }
        }
        std::cerr << std::endl;
        timing.first_text_ms = reply.first_text_ms;
        timing.total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - reply.started).count();
        if (code != 0 || reply.failed || !reply.ended) {
            new_chat_ = true;
            std::cerr << "[错误] 推理未正常结束,code=" << code
                      << ";下轮重置会话." << std::endl;
            return false;
        }
        if (!thinking_ && (reply.text.find("<think>") != std::string::npos ||
                          reply.text.find("</think>") != std::string::npos)) {
            new_chat_ = true;
            std::cerr << "[错误] 直答模式下模型生成了思考标记." << std::endl;
            return false;
        }
        new_chat_ = false;
        last_text_ = reply.text;
        return true;
    }

private:
    struct Reply {
        std::mutex mutex;
        std::condition_variable ready;
        bool failed = false;
        bool ended = false;
        std::string pending;
        std::string text;
        bool print_raw = false;
        double first_text_ms = -1;
        std::chrono::steady_clock::time_point started;
    };

    // 只刷新完整 UTF-8 字符,保留 SDK 分块末尾尚未到齐的字节.
    static void PrintUtf8(Reply& reply, const char* text) {
        if (text) {
            reply.pending += text;
        }
        std::size_t position = 0;
        while (position < reply.pending.size()) {
            const auto c = static_cast<unsigned char>(reply.pending[position]);
            std::size_t count = c < 0x80 ? 1 :
                (c >= 0xC2 && c <= 0xDF ? 2 :
                (c >= 0xE0 && c <= 0xEF ? 3 :
                (c >= 0xF0 && c <= 0xF4 ? 4 : 0)));
            if (!count) {
                throw std::runtime_error("SDK 输出无效 UTF-8 起始字节.");
            }
            if (position + count > reply.pending.size()) {
                break;
            }
            for (std::size_t i = 1; i < count; ++i) {
                if ((static_cast<unsigned char>(reply.pending[position + i]) & 0xC0) != 0x80) {
                    throw std::runtime_error("SDK 输出无效 UTF-8 后续字节.");
                }
            }
            position += count;
        }
        std::cerr << reply.pending.substr(0, position) << std::flush;
        reply.pending.erase(0, position);
    }

    // 遵循旧 SDK 示例: START/RUNNING 是增量,END 是完成终态.
    static void Callback(xlm_result_t* result, xlm_state_t state, void* userdata) {
        auto& reply = *static_cast<Reply*>(userdata);
        std::lock_guard<std::mutex> lock(reply.mutex);
        try {
            if (state == XLM_STATE_ERROR) {
                reply.failed = true;
            } else if (state == XLM_STATE_END) {
                reply.ended = true;
                reply.failed = reply.failed || !reply.pending.empty();
            } else if (result) {
                if (result->text) {
                    reply.text += result->text;
                    if (reply.first_text_ms < 0 && result->text[0]) {
                        reply.first_text_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - reply.started).count();
                    }
                }
                if (reply.print_raw) {
                    PrintUtf8(reply, result->text);
                }
            }
        } catch (const std::exception& error) {
            reply.failed = true;
            std::cerr << "[输出错误] " << error.what() << std::endl;
        }
        if (reply.ended || reply.failed) {
            reply.ready.notify_all();
        }
    }

    xlm_handle_t handle_ = nullptr;
    std::string model_;
    std::string tokenizer_;
    std::string template_;
    bool new_chat_ = true;
    std::string last_text_;
    bool print_raw_ = false;
    bool thinking_ = false;
};

}  // 命名空间 voice_control
