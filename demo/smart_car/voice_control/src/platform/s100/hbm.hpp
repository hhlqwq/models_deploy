#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <hobot/dnn/hb_dnn.h>

namespace voice_control {

// 在同步 UCP 调用中维护 FP32 逻辑张量与实际 stride 的映射.
class Hbm {
public:
    using Shape = std::vector<int>;
    using Shapes = std::map<std::string, Shape>;
    using Values = std::map<std::string, std::vector<float>>;

    // 按模型实际名称、类型及形状验证现有 HBM 契约.
    Hbm(const std::string& path, const Shapes& inputs, const Shapes& outputs) {
        try {
            const char* filename = path.c_str();
            Check(hbDNNInitializeFromFiles(&packed_, &filename, 1), "加载 HBM");
            const char** names = nullptr;
            int count = 0;
            Check(hbDNNGetModelNameList(&names, &count, packed_), "模型列表");
            if (count != 1) {
                throw std::runtime_error("Whisper HBM 必须包含单个模型.");
            }
            Check(hbDNNGetModelHandle(&model_, packed_, names[0]), "模型句柄");
            Init(true, inputs, inputs_, input_names_);
            Init(false, outputs, outputs_, output_names_);
        } catch (...) {
            Release();
            throw;
        }
    }

    // 释放本实例持有的张量及模型.
    ~Hbm() { Release(); }

    Hbm(const Hbm&) = delete;
    Hbm& operator=(const Hbm&) = delete;

    // 一次推理复制全部逻辑输出,不让缓存引用 Runtime 内部可复用内存.
    Values Run(const Values& feeds) {
        if (feeds.size() != inputs_.size()) {
            throw std::runtime_error("HBM 输入数量不符.");
        }
        for (std::size_t i = 0; i < inputs_.size(); ++i) {
            Transfer(inputs_[i], feeds.at(input_names_[i]), nullptr);
            Check(hbUCPMemFlush(&inputs_[i].sysMem, HB_SYS_MEM_CACHE_CLEAN), "刷新输入");
        }
        hbUCPTaskHandle_t task = nullptr;
        try {
            Check(hbDNNInferV2(&task, outputs_.data(), inputs_.data(), model_), "创建推理");
            hbUCPSchedParam sched{};
            HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
            sched.backend = HB_UCP_BPU_CORE_ANY;
            Check(hbUCPSubmitTask(task, &sched), "提交推理");
            Check(hbUCPWaitTaskDone(task, 0), "等待推理");
            Check(hbUCPReleaseTask(task), "释放推理任务");
            task = nullptr;
        } catch (...) {
            if (task) {
                hbUCPReleaseTask(task);
            }
            throw;
        }
        Values result;
        for (std::size_t i = 0; i < outputs_.size(); ++i) {
            Check(hbUCPMemFlush(&outputs_[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE), "刷新输出");
            Transfer(outputs_[i], {}, &result[output_names_[i]]);
        }
        return result;
    }

private:
    // 官方非零返回码保留在错误中.
    static void Check(int code, const char* step) {
        if (code != 0) {
            throw std::runtime_error(std::string(step) + ": " + std::to_string(code));
        }
    }

    // 不假设张量连续,按公开字节 stride 逐元素映射.
    static void Transfer(hbDNNTensor& tensor, const std::vector<float>& input,
                         std::vector<float>* output) {
        const auto& p = tensor.properties;
        std::size_t count = 1;
        for (int i = 0; i < p.validShape.numDimensions; ++i) {
            count *= p.validShape.dimensionSize[i];
        }
        if (output) {
            output->resize(count);
        } else if (input.size() != count) {
            throw std::runtime_error("HBM 输入元素数不符.");
        }
        auto* bytes = static_cast<char*>(tensor.sysMem.virAddr);
        // 连续尾部一次复制,只在跨行时计算 stride,避免逐元素多次整数除法.
        std::size_t block = 1;
        int outer = p.validShape.numDimensions - 1;
        while (outer >= 0 && (p.validShape.dimensionSize[outer] == 1 ||
               p.stride[outer] == static_cast<std::int64_t>(block * sizeof(float)))) {
            block *= p.validShape.dimensionSize[outer];
            --outer;
        }
        for (std::size_t i = 0; i < count; i += block) {
            std::size_t index = i / block;
            std::size_t offset = 0;
            for (int d = outer; d >= 0; --d) {
                offset += (index % p.validShape.dimensionSize[d]) * p.stride[d];
                index /= p.validShape.dimensionSize[d];
            }
            if (offset + block * sizeof(float) > tensor.sysMem.memSize) {
                throw std::runtime_error("HBM stride 超出分配内存.");
            }
            if (output) {
                std::memcpy(output->data() + i, bytes + offset, block * sizeof(float));
            } else {
                std::memcpy(bytes + offset, input.data() + i, block * sizeof(float));
            }
        }
        if (output) {
            for (float value : *output) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("HBM 输出包含非有限值.");
                }
            }
        }
    }

    // 检查名称和形状,再分配按 Runtime 对齐的缓存.
    void Init(bool input, const Shapes& expected, std::vector<hbDNNTensor>& tensors,
              std::vector<std::string>& names) {
        int count = 0;
        Check(input ? hbDNNGetInputCount(&count, model_) : hbDNNGetOutputCount(&count, model_), "张量数量");
        if (count != static_cast<int>(expected.size())) {
            throw std::runtime_error("HBM 张量数量与契约不符.");
        }
        tensors.resize(count);
        for (int i = 0; i < count; ++i) {
            const char* name = nullptr;
            Check(input ? hbDNNGetInputName(&name, model_, i) : hbDNNGetOutputName(&name, model_, i), "张量名");
            names.emplace_back(name);
            auto& tensor = tensors[i];
            Check(input ? hbDNNGetInputTensorProperties(&tensor.properties, model_, i) :
                  hbDNNGetOutputTensorProperties(&tensor.properties, model_, i), "张量属性");
            const auto& p = tensor.properties;
            const Shape shape(p.validShape.dimensionSize,
                              p.validShape.dimensionSize + p.validShape.numDimensions);
            if (expected.at(name) != shape || p.tensorType != HB_DNN_TENSOR_TYPE_F32 || p.alignedByteSize <= 0) {
                throw std::runtime_error("HBM FP32 契约不符: " + std::string(name));
            }
            for (int d = 0; d < p.validShape.numDimensions; ++d) {
                if (p.stride[d] <= 0) {
                    throw std::runtime_error("HBM stride 必须为正.");
                }
            }
            Check(hbUCPMallocCached(&tensor.sysMem, p.alignedByteSize, 0), "分配张量");
            std::memset(tensor.sysMem.virAddr, 0, tensor.sysMem.memSize);
        }
    }

    // 构造失败和正常退出共用同一资源释放路径.
    void Release() {
        for (auto* group : {&inputs_, &outputs_}) {
            for (auto& tensor : *group) {
                if (tensor.sysMem.virAddr) {
                    hbUCPFree(&tensor.sysMem);
                    tensor.sysMem.virAddr = nullptr;
                }
            }
        }
        if (packed_) {
            hbDNNRelease(packed_);
            packed_ = nullptr;
        }
    }

    hbDNNPackedHandle_t packed_ = nullptr;
    hbDNNHandle_t model_ = nullptr;
    std::vector<hbDNNTensor> inputs_;
    std::vector<hbDNNTensor> outputs_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

}  // 命名空间 voice_control
