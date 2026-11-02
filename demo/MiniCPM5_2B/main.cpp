// 独立终端入口,读取文件配置并组装平台引擎.
#include "core/chat.h"
#include "platforms/mtk/engine.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace {

// 所有运行参数保存在文件中,命令行只选择配置文件或显示帮助.
void Usage() {
    std::cout << "用法: minicpm5_chat [--config <config.json>] [--help]\n"
                 "默认读取可执行程序同目录的 config.json.\n";
}

// 模型与模板的相对路径以配置目录为基准,与启动目录无关.
std::filesystem::path ResourcePath(const nlohmann::json& config, const char* key,
                                   const std::filesystem::path& directory) {
    const auto value = config.at(key).get<std::string>();
    if (value.empty()) throw std::invalid_argument(std::string("配置不能为空: ") + key);
    const std::filesystem::path path(value);
    return (path.is_absolute() ? path : directory / path).lexically_normal();
}

}  // 匿名命名空间.

// 正常 EOF 与 /exit 均由 RAII 释放双 DLA,配置或运行失败返回非零状态.
int main(int argc, char** argv) {
    try {
        std::filesystem::path config_path;
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--help") {
                Usage();
                return 0;
            }
            if (option != "--config") throw std::invalid_argument("未知参数: " + option);
            if (index + 1 >= argc) throw std::invalid_argument("--config 缺少文件路径.");
            config_path = argv[++index];
        }
        if (config_path.empty()) config_path = minicpm::DefaultConfigPath();
        config_path = std::filesystem::absolute(config_path).lexically_normal();
        std::ifstream input(config_path);
        if (!input) {
            throw std::runtime_error("无法读取配置: " + config_path.string() +
                                     "; 请参照 config.example.json 创建 config.json.");
        }
        const auto config = nlohmann::json::parse(input);
        if (!config.is_object()) throw std::invalid_argument("配置必须是 JSON 对象.");
        const auto& runtime = config.at("runtime");
        if (!runtime.is_object()) throw std::invalid_argument("runtime 配置必须是 JSON 对象.");
        const auto& budget = runtime.at("max_new_tokens");
        if (!budget.is_number_unsigned() || budget.get<uint64_t>() == 0 ||
            budget.get<uint64_t>() >= 1024) {
            throw std::invalid_argument("max_new_tokens 必须是 1 至 1023 的整数.");
        }
        const size_t limit = budget.get<size_t>();
        const auto directory = config_path.parent_path();
        const auto models = ResourcePath(runtime, "models", directory);
        const auto chat_template = ResourcePath(runtime, "chat_template", directory);
        std::string library = runtime.value("runtime_library",
                                          "libneuronusdk_runtime.mtk.so.8");
        if (library.empty()) throw std::invalid_argument("runtime_library 不能为空.");
        const std::filesystem::path library_path(library);
        if (library_path.has_parent_path() && !library_path.is_absolute()) {
            library = (directory / library_path).lexically_normal().string();
        }
        std::cout << "[配置] " << config_path.string() << '\n' << std::flush;
        auto engine = minicpm::LoadMtkEngine(models, chat_template, library);
        minicpm::Chat(*engine, limit, std::cin, std::cout);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\n[失败] " << error.what() << '\n';
        return 1;
    }
}
