// 独立终端入口,只负责参数和平台引擎组装.
#include "core/chat.h"
#include "platforms/mtk/engine.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

// 打印参数说明,模型路径只由运行环境提供.
void Usage() {
    std::cout << "用法: minicpm5_chat --models <板端 models 目录> "
                 "--chat-template <官方 chat_template.jinja> [--max-new-tokens 128]\n"
                 "也可用 MINICPM5_MODELS 和 MINICPM5_CHAT_TEMPLATE 设置路径.\n";
}

}  // 匿名命名空间.

// 正常 EOF 与 /exit 均由 RAII 释放双 DLA,错误返回非零状态.
int main(int argc, char** argv) {
    try {
        const char* model_env = std::getenv("MINICPM5_MODELS");
        const char* template_env = std::getenv("MINICPM5_CHAT_TEMPLATE");
        std::filesystem::path models = model_env ? model_env : "";
        std::filesystem::path chat_template = template_env ? template_env : "";
        size_t limit = 128;
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--help") {
                Usage();
                return 0;
            }
            if (index + 1 >= argc) throw std::invalid_argument("参数缺少值: " + option);
            const std::string value = argv[++index];
            if (option == "--models") models = value;
            else if (option == "--chat-template") chat_template = value;
            else if (option == "--max-new-tokens") {
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
                    throw std::invalid_argument("生成上限必须是正整数.");
                }
                limit = std::stoull(value);
            } else throw std::invalid_argument("未知参数: " + option);
        }
        if (models.empty() || chat_template.empty()) {
            Usage();
            throw std::invalid_argument("必须指定板端模型资源和同版本聊天模板.");
        }
        auto engine = minicpm::LoadMtkEngine(models, chat_template);
        minicpm::Chat(*engine, limit, std::cin, std::cout);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\n[失败] " << error.what() << '\n';
        return 1;
    }
}
