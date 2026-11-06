#pragma once

#include <cstddef>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace voice_control {

class SimplifiedChinese {
public:
    // 复用 S100 系统中的 OpenCC C 接口,启动时加载一次词典.
    SimplifiedChinese()
        : library_(dlopen("libopencc.so.1.1", RTLD_NOW | RTLD_LOCAL)) {
        if (!library_) {
            throw std::runtime_error(std::string("无法加载板端 OpenCC 运行库: ") + dlerror());
        }
        const auto open = Symbol<void* (*)(const char*)>("opencc_open");
        close_ = Symbol<int (*)(void*)>("opencc_close");
        convert_ = Symbol<char* (*)(void*, const char*, std::size_t)>("opencc_convert_utf8");
        free_ = Symbol<void (*)(char*)>("opencc_convert_utf8_free");
        const auto error = Symbol<const char* (*)()>("opencc_error");
        converter_ = open("/usr/share/opencc/t2s.json");
        if (converter_ == reinterpret_cast<void*>(-1)) {
            throw std::runtime_error(std::string("无法加载 OpenCC t2s 词典: ") + error());
        }
    }

    // 在共享库卸载前销毁转换器.
    ~SimplifiedChinese() {
        close_(converter_);
    }

    SimplifiedChinese(const SimplifiedChinese&) = delete;
    SimplifiedChinese& operator=(const SimplifiedChinese&) = delete;

    // 转换一段 UTF-8 文本,由 OpenCC 回收其分配的缓冲区.
    std::string Convert(const std::string& text) const {
        if (text.empty()) {
            return {};
        }
        const std::unique_ptr<char, decltype(free_)> converted(
            convert_(converter_, text.c_str(), text.size()), free_);
        if (!converted) {
            throw std::runtime_error("OpenCC 无法转换中文转录.");
        }
        return converted.get();
    }

private:
    struct CloseLibrary {
        // 释放本程序加载的共享库引用.
        void operator()(void* library) const {
            dlclose(library);
        }
    };

    // 只绑定官方 C 接口,交叉编译侧无需板端开发包.
    template <typename Function>
    Function Symbol(const char* name) const {
        auto* symbol = dlsym(library_.get(), name);
        if (!symbol) {
            throw std::runtime_error(std::string("OpenCC 缺少 C 接口: ") + name);
        }
        return reinterpret_cast<Function>(symbol);
    }

    std::unique_ptr<void, CloseLibrary> library_;
    int (*close_)(void*) = nullptr;
    char* (*convert_)(void*, const char*, std::size_t) = nullptr;
    void (*free_)(char*) = nullptr;
    void* converter_ = nullptr;
};

}  // 命名空间 voice_control
