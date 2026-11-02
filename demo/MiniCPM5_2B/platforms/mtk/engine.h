// MTK 引擎的工厂声明,业务头文件不包含平台 SDK.
#ifndef MINICPM5_CHAT_MTK_ENGINE_H_
#define MINICPM5_CHAT_MTK_ENGINE_H_

#include "chat.h"
#include <filesystem>

namespace minicpm {

// 只读加载板端已有资源目录与同版本官方聊天模板.
std::unique_ptr<Engine> LoadMtkEngine(const std::filesystem::path& models,
    const std::filesystem::path& chat_template,
    const std::string& runtime_library = "libneuronusdk_runtime.mtk.so.8");

}  // 命名空间 minicpm.
#endif
