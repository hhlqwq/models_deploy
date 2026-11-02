// 终端入口,解析运行参数并加载平台引擎.
#include "core/chat.h"
#include "platforms/mtk/engine.h"

#include <iostream>
#include <stdexcept>

namespace {

// 展示必要参数和两个可选参数.
void Usage() {
    std::cout << "用法: minicpm5_chat --models <模型目录> "
                 "[--chat-template chat_template.jinja] [--max-new-tokens 128]\n"
                 "/reset 清空历史; /exit 退出.\n";
}

}  // 匿名命名空间.

// 模型路径由启动参数指定,不读取环境变量或运行配置文件.
int main(int argc, char** argv) {
    try {
        std::filesystem::path models;
        std::filesystem::path chat_template = "chat_template.jinja";
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
                    throw std::invalid_argument("生成上限必须是 1 至 1023 的整数.");
                }
                const auto count = std::stoull(value);
                if (count == 0 || count >= 1024) {
                    throw std::invalid_argument("生成上限必须是 1 至 1023 的整数.");
                }
                limit = static_cast<size_t>(count);
            } else throw std::invalid_argument("未知参数: " + option);
        }
        if (models.empty() || chat_template.empty()) {
            Usage();
            throw std::invalid_argument("必须指定模型目录,聊天模板路径不能为空.");
        }
        auto engine = minicpm::LoadMtkEngine(models, chat_template);
        minicpm::Chat(*engine, limit, std::cin, std::cout);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\n[失败] " << error.what() << '\n';
        return 1;
    }
}
