// 复用现有 INT16 静态图契约,直接调用 MTK SDK,不启动 Python 或外部 CLI.
#include "engine.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cnpy.h>
#include <minja/minja.hpp>
#include <neuron/api/RuntimeAPI.h>
#include <tokenizers_cpp.h>

namespace minicpm {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;

// 读取已有文本资源,读取失败保留具体文件位置.
std::string ReadText(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("无法读取资源: " + path.string());
    std::string result((std::istreambuf_iterator<char>(input)), {});
    if (input.bad()) throw std::runtime_error("资源读取失败: " + path.string());
    return result;
}

// 模型清单只能引用资源根目录内的文件,不能通过绝对路径或软链接越界.
fs::path Resource(const fs::path& root, const std::string& relative) {
    const fs::path name(relative);
    if (name.empty() || name.is_absolute()) throw std::runtime_error("资源路径必须为相对路径.");
    const auto path = fs::canonical(root / name);
    const auto local = fs::relative(path, fs::canonical(root));
    for (const auto& part : local) {
        if (part == "..") throw std::runtime_error("资源超出模型目录: " + relative);
    }
    if (!fs::is_regular_file(path)) throw std::runtime_error("资源不是文件: " + path.string());
    return path;
}

// 所有 SDK 错误均直接中止本进程的当前任务.
void Check(int status, const std::string& operation) {
    if (status != 0) throw std::runtime_error("NeuronRuntime " + operation +
                                            " 失败,错误码=" + std::to_string(status));
}

struct Spec {
    std::string name;
    std::vector<size_t> shape;
    float scale;
    int16_t zero;

    // 带溢出检查地计算逻辑元素数量.
    size_t Count() const {
        size_t count = 1;
        for (const auto dimension : shape) {
            if (!dimension || count > std::numeric_limits<size_t>::max() / dimension) {
                throw std::runtime_error("张量形状非法: " + name);
            }
            count *= dimension;
        }
        if (count > std::numeric_limits<size_t>::max() / sizeof(int16_t)) {
            throw std::runtime_error("张量字节数溢出: " + name);
        }
        return count;
    }
};

// 解析实际部署契约,不根据模型文件名猜测形状或量化尺度.
Spec ReadSpec(const Json& item) {
    if (item.at("dtype") != "INT16") throw std::runtime_error("本引擎只接受原生 INT16 I/O.");
    Spec spec;
    spec.name = item.at("name").get<std::string>();
    for (const auto& value : item.at("shape")) {
        const auto dimension = value.get<int64_t>();
        if (dimension <= 0) throw std::runtime_error("张量尺寸必须为正.");
        spec.shape.push_back(static_cast<size_t>(dimension));
    }
    spec.scale = item.at("scale").get<float>();
    const auto zero = item.at("zero_point").get<int64_t>();
    if (spec.shape.empty() || !std::isfinite(spec.scale) || spec.scale <= 0 ||
        zero < -32768 || zero > 32767) throw std::runtime_error("无效的 INT16 契约.");
    spec.zero = static_cast<int16_t>(zero);
    spec.Count();
    return spec;
}

struct Tensor {
    Spec spec;
    std::vector<int16_t> values;
};

class RuntimeApi {
 public:
    // 在目标板解析已有 SDK C ABI,不把新版板端 GLIBC 依赖带入交叉链接.
    explicit RuntimeApi(const std::string& library) {
        library_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library_) throw std::runtime_error(std::string("加载 MTK Runtime 失败: ") + dlerror());
        try {
            create = Symbol<decltype(create)>("NeuronRuntime_create");
            release = Symbol<decltype(release)>("NeuronRuntime_release");
            load = Symbol<decltype(load)>("NeuronRuntime_loadNetworkFromFile");
            input_count = Symbol<decltype(input_count)>("NeuronRuntime_getInputNumber");
            output_count = Symbol<decltype(output_count)>("NeuronRuntime_getOutputNumber");
            input_size = Symbol<decltype(input_size)>("NeuronRuntime_getInputPaddedSize");
            output_size = Symbol<decltype(output_size)>("NeuronRuntime_getOutputPaddedSize");
            set_input = Symbol<decltype(set_input)>("NeuronRuntime_setInput");
            set_output = Symbol<decltype(set_output)>("NeuronRuntime_setOutput");
            inference = Symbol<decltype(inference)>("NeuronRuntime_inference");
        } catch (...) {
            dlclose(library_);
            throw;
        }
    }

    // 两个 Graph 释放后才卸载本引擎持有的动态库引用.
    ~RuntimeApi() { dlclose(library_); }
    RuntimeApi(const RuntimeApi&) = delete;
    RuntimeApi& operator=(const RuntimeApi&) = delete;

    decltype(&NeuronRuntime_create) create = nullptr;
    decltype(&NeuronRuntime_release) release = nullptr;
    decltype(&NeuronRuntime_loadNetworkFromFile) load = nullptr;
    decltype(&NeuronRuntime_getInputNumber) input_count = nullptr;
    decltype(&NeuronRuntime_getOutputNumber) output_count = nullptr;
    decltype(&NeuronRuntime_getInputPaddedSize) input_size = nullptr;
    decltype(&NeuronRuntime_getOutputPaddedSize) output_size = nullptr;
    decltype(&NeuronRuntime_setInput) set_input = nullptr;
    decltype(&NeuronRuntime_setOutput) set_output = nullptr;
    decltype(&NeuronRuntime_inference) inference = nullptr;

 private:
    // 函数签名来自实际 SDK 头文件,缺少任何必要符号都直接失败.
    template <typename Function>
    Function Symbol(const char* name) {
        dlerror();
        void* symbol = dlsym(library_, name);
        const char* error = dlerror();
        if (error || !symbol) throw std::runtime_error(std::string("缺少 Runtime 符号: ") + name);
        return reinterpret_cast<Function>(symbol);
    }
    void* library_ = nullptr;
};

// 只有输入 mask 允许现有实现明确支持的末尾补齐.
Tensor Allocate(const Spec& spec, size_t bytes, bool input) {
    const size_t count = spec.Count();
    if (bytes != count * 2) {
        const size_t width = spec.shape.back();
        const size_t aligned = ((width * 2 + 15) / 16) * 16;
        if (!input || spec.name != "mask" || count != width || bytes != aligned) {
            throw std::runtime_error("SDK 原生字节数与契约不符: " + spec.name);
        }
    }
    Tensor tensor{spec, std::vector<int16_t>(bytes / 2, spec.zero)};
    if (bytes != count * 2) std::fill(tensor.values.begin() + count, tensor.values.end(), -32768);
    return tensor;
}

class Graph {
 public:
    // 创建硬件模型并按 SDK 实际大小分配缓冲区.
    Graph(std::shared_ptr<RuntimeApi> api, const fs::path& path, const Json& contract)
        : api_(std::move(api)) {
        EnvOptions options{};
        options.deviceKind = kEnvOptHardware;
        options.MDLACoreOption = Auto;
        options.CPUThreadNum = 1;
        options.suppressInputConversion = true;
        options.suppressOutputConversion = true;
        Check(api_->create(&options, &handle_), "create");
        try {
            Check(api_->load(handle_, path.c_str()), "load");
            size_t in_count = 0, out_count = 0;
            Check(api_->input_count(handle_, &in_count), "input count");
            Check(api_->output_count(handle_, &out_count), "output count");
            if (in_count != contract.at("inputs").size() || out_count != contract.at("outputs").size()) {
                throw std::runtime_error("SDK I/O 数量与清单不符.");
            }
            for (size_t kind = 0; kind < 2; ++kind) {
                auto& tensors = kind ? outputs : inputs;
                const auto& items = contract.at(kind ? "outputs" : "inputs");
                std::set<std::string> names;
                for (size_t index = 0; index < items.size(); ++index) {
                    size_t bytes = 0;
                    Check(kind ? api_->output_size(handle_, index, &bytes)
                               : api_->input_size(handle_, index, &bytes), "tensor size");
                    auto spec = ReadSpec(items.at(index));
                    if (!names.insert(spec.name).second) throw std::runtime_error("张量名重复.");
                    tensors.push_back(Allocate(spec, bytes, kind == 0));
                }
            }
        } catch (...) {
            api_->release(handle_);
            handle_ = nullptr;
            throw;
        }
    }

    // 正常退出和异常退出均释放模型,不影响其他进程.
    ~Graph() {
        if (handle_) api_->release(handle_);
    }
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    // 查找契约明确声明的输入.
    Tensor& Input(const std::string& name) { return Find(inputs, name); }
    // 查找契约明确声明的输出.
    Tensor& Output(const std::string& name) { return Find(outputs, name); }

    // 重置逻辑输入区域,保留 mask 的原生补齐哨兵.
    void Reset() {
        for (auto& tensor : inputs) {
            std::fill_n(tensor.values.begin(), tensor.spec.Count(), tensor.spec.zero);
        }
    }

    // 缓冲区保持常驻,每次调用重新绑定并同步执行.
    void Run() {
        const BufferAttribute attribute{NON_ION_FD};
        for (size_t index = 0; index < inputs.size(); ++index) {
            auto& values = inputs[index].values;
            Check(api_->set_input(handle_, index, values.data(), values.size() * 2, attribute), "setInput");
        }
        for (size_t index = 0; index < outputs.size(); ++index) {
            auto& values = outputs[index].values;
            Check(api_->set_output(handle_, index, values.data(), values.size() * 2, attribute), "setOutput");
        }
        Check(api_->inference(handle_), "inference");
    }

    std::vector<Tensor> inputs, outputs;

 private:
    // 不允许找不到张量时猜测索引或静默跳过.
    static Tensor& Find(std::vector<Tensor>& tensors, const std::string& name) {
        for (auto& tensor : tensors) if (tensor.spec.name == name) return tensor;
        throw std::runtime_error("缺少张量: " + name);
    }
    std::shared_ptr<RuntimeApi> api_;
    void* handle_ = nullptr;
};

// 按源目标尺度重新量化,采用与现有 NumPy 路线一致的最近偶数舍入.
int16_t Convert(int16_t value, const Spec& source, const Spec& target) {
    if (source.scale == target.scale && source.zero == target.zero) return value;
    const float real = (static_cast<float>(value) - source.zero) * source.scale;
    const float scaled = real / target.scale + target.zero;
    if (!std::isfinite(scaled)) throw std::runtime_error("重新量化产生非有限数值.");
    return static_cast<int16_t>(std::clamp(std::nearbyint(scaled), -32768.0f, 32767.0f));
}

// KV 坐标统一为两个前导维、时间维及通道维,拒绝其他布局.
void CheckCache(const Tensor& source, const Tensor& target) {
    const auto& a = source.spec.shape;
    const auto& b = target.spec.shape;
    if (a.size() != 4 || b.size() != 4 || a[0] != b[0] || a[1] != b[1] || a[3] != b[3]) {
        throw std::runtime_error("KV 布局不匹配.");
    }
}

// 将当前块追加到后缀有效 Cache,不会改变序列绝对位置.
void Append(const Tensor& source, Tensor& target) {
    CheckCache(source, target);
    const size_t incoming = source.spec.shape[2], capacity = target.spec.shape[2];
    const size_t width = target.spec.shape[3], groups = target.spec.shape[0] * target.spec.shape[1];
    if (incoming > capacity) throw std::runtime_error("KV 输出大于缓存容量.");
    for (size_t group = 0; group < groups; ++group) {
        auto* destination = target.values.data() + group * capacity * width;
        std::memmove(destination, destination + incoming * width, (capacity - incoming) * width * 2);
        for (size_t index = 0; index < incoming * width; ++index) {
            destination[(capacity - incoming) * width + index] =
                Convert(source.values[group * incoming * width + index], source.spec, target.spec);
        }
    }
}

// 末块只接收有效 Token,排除 Prefill 右侧填充值后初始化 Decode KV.
void Seed(const Tensor& previous, const Tensor& current, Tensor& target, size_t valid) {
    CheckCache(previous, target);
    CheckCache(current, target);
    const size_t old_length = previous.spec.shape[2], capacity = target.spec.shape[2];
    const size_t width = target.spec.shape[3], groups = target.spec.shape[0] * target.spec.shape[1];
    if (valid == 0 || valid > current.spec.shape[2]) throw std::runtime_error("末块有效长度非法.");
    const size_t total = old_length + valid, copied = std::min(total, capacity);
    std::fill_n(target.values.begin(), target.spec.Count(), target.spec.zero);
    for (size_t group = 0; group < groups; ++group) {
        for (size_t position = 0; position < copied; ++position) {
            const size_t from = total - copied + position;
            const auto& source = from < old_length ? previous : current;
            const size_t source_position = from < old_length ? from : from - old_length;
            for (size_t channel = 0; channel < width; ++channel) {
                const size_t source_index = (group * source.spec.shape[2] + source_position) * width + channel;
                const size_t target_index = (group * capacity + capacity - copied + position) * width + channel;
                target.values[target_index] = Convert(source.values[source_index], source.spec, target.spec);
            }
        }
    }
}

class Embedding {
 public:
    // 只读映射板端已有 embedding,不生成副本.
    Embedding(const fs::path& path, size_t bytes) : bytes_(bytes) {
        if (fs::file_size(path) != bytes) throw std::runtime_error("embedding 大小不匹配.");
        const int descriptor = open(path.c_str(), O_RDONLY);
        if (descriptor < 0) throw std::runtime_error("无法打开 embedding.");
        memory_ = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, descriptor, 0);
        close(descriptor);
        if (memory_ == MAP_FAILED) throw std::runtime_error("无法只读映射 embedding.");
    }
    // 映射随引擎退出释放.
    ~Embedding() { if (memory_ != MAP_FAILED) munmap(memory_, bytes_); }
    Embedding(const Embedding&) = delete;
    Embedding& operator=(const Embedding&) = delete;
    // 返回只读 INT16 元素.
    const int16_t* Data() const { return static_cast<const int16_t*>(memory_); }
 private:
    size_t bytes_;
    void* memory_ = MAP_FAILED;
};

class MtkEngine final : public Engine {
 public:
    // 检查资源后只加载一次双图、分词器和输入银行.
    MtkEngine(const fs::path& root, const fs::path& template_path,
              const std::string& runtime_library) {
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian) != 1 || std::fesetround(FE_TONEAREST) != 0) {
            throw std::runtime_error("需要小端平台与最近偶数浮点舍入.");
        }
        std::cout << "[加载] 读取部署契约与分词资源.\n" << std::flush;
        manifest_ = Json::parse(ReadText(Resource(root, "manifest.json")));
        if (manifest_.at("context") != 1024 || manifest_.at("prefill") != 128 ||
            manifest_.at("precision") != "asym4W_sym16A") {
            throw std::runtime_error("需要当前 MiniCPM5 W4A16 / 1024c / 128t 配套部署包.");
        }
        context_ = manifest_.at("context").get<size_t>();
        const auto& preparation = manifest_.at("preparation");
        bos_ = preparation.at("bos_token_id").get<int32_t>();
        eos_ = preparation.at("eos_token_id").get<int32_t>();
        for (const auto& token : preparation.at("stop_token_ids")) stops_.insert(token.get<int32_t>());
        if (stops_ != std::set<int32_t>{1, 130073}) throw std::runtime_error("停止 Token 与 MiniCPM5 约定不符.");
        template_ = ReadText(template_path);
        auto tokenizer_json = Json::parse(ReadText(CheckedResource(root, "tokenizer/tokenizer.json")));
        tokenizer_json["truncation"] = nullptr;
        tokenizer_json["padding"] = nullptr;
        for (const auto& token : tokenizer_json.at("added_tokens")) {
            if (token.at("special").get<bool>()) special_.insert(token.at("id").get<int32_t>());
        }
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tokenizer_json.dump());
        if (!tokenizer_ || tokenizer_->IdToToken(bos_).empty() || tokenizer_->IdToToken(eos_).empty()) {
            throw std::runtime_error("分词器缺少部署配置中的特殊 Token.");
        }
        bank_ = cnpy::npz_load(CheckedResource(root, "tokenizer/generation_inputs.npz").string());
        const auto& contracts = manifest_.at("graph_contracts");
        api_ = std::make_shared<RuntimeApi>(runtime_library);
        std::cout << "[加载] Prefill DLA.\n" << std::flush;
        prompt_ = std::make_unique<Graph>(api_, CheckedResource(root, manifest_.at("model_files").at("prompt").get<std::string>()), contracts.at("prompt"));
        std::cout << "[加载] Decode DLA.\n" << std::flush;
        decode_ = std::make_unique<Graph>(api_, CheckedResource(root, manifest_.at("model_files").at("decode").get<std::string>()), contracts.at("decode"));
        const auto& embeds = prompt_->Input("input_embeds").spec;
        const auto& logits = prompt_->outputs.at(0).spec;
        if (embeds.shape != std::vector<size_t>{1, 128, 2048} || logits.shape.size() != 3 ||
            logits.shape[0] != 1 || logits.shape[1] != 128 || logits.shape[2] < tokenizer_->GetVocabSize()) {
            throw std::runtime_error("Prefill 结构与模型不符.");
        }
        width_ = 2048;
        vocabulary_ = logits.shape[2];
        if (vocabulary_ > std::numeric_limits<size_t>::max() / width_ / 2 ||
            decode_->Input("input_embeds").spec.shape != std::vector<size_t>{1, 1, width_} ||
            decode_->outputs.at(0).spec.shape != std::vector<size_t>{1, 1, vocabulary_}) {
            throw std::runtime_error("Decode 或 embedding 结构与模型不符.");
        }
        embedding_ = std::make_unique<Embedding>(CheckedResource(root, "tokenizer/embedding_int16.bin"), vocabulary_ * width_ * 2);
        ValidateGraph(*prompt_, "prompt", 128);
        ValidateGraph(*decode_, "decode", 1);
        std::cout << "[加载完成] 模型常驻,上下文 1024 Token,每轮重建 KV.\n";
    }

    // 使用实际官方模板渲染历史,不硬编码角色分隔符.
    std::vector<int32_t> Encode(const std::vector<Message>& messages) override {
        Json history = Json::array();
        for (const auto& message : messages) history.push_back({{"role", message.role}, {"content", message.content}});
        // 在调用 Rust 分词 ABI 前拒绝非法 UTF-8,避免跨 ABI 发生 panic.
        history.dump();
        auto context = minja::Context::make(Json{
            {"messages", history}, {"bos_token", tokenizer_->IdToToken(bos_)},
            {"eos_token", tokenizer_->IdToToken(eos_)}, {"enable_thinking", false},
            {"add_generation_prompt", true}, {"tools", nullptr}});
        const auto tree = minja::Parser::parse(template_, {true, true, false});
        const auto tokens = tokenizer_->Encode(tree->render(context));
        if (std::count(tokens.begin(), tokens.end(), bos_) != 1) {
            throw std::runtime_error("模板编码必须恰好包含一个 BOS.");
        }
        return tokens;
    }

    // 按分词器声明跳过特殊 Token,与既有 Python 展示语义一致.
    std::string Decode(const std::vector<int32_t>& tokens) override {
        std::vector<int32_t> content;
        for (const auto token : tokens) if (!special_.count(token)) content.push_back(token);
        return content.empty() ? "" : tokenizer_->Decode(content);
    }

    // 静态模型容量来自已核对的部署清单.
    size_t ContextSize() const override { return context_; }

    // 每轮独立重置 KV,完整历史由输入消息重新编码后处理.
    Generation Generate(const std::vector<int32_t>& tokens, size_t limit,
        const std::function<void(const std::vector<int32_t>&)>& on_tokens) override {
        if (tokens.empty() || limit == 0 || limit > context_ || tokens.size() > context_ - limit) {
            throw std::runtime_error("生成预算超出模型上下文.");
        }
        for (const auto token : tokens) ValidateToken(token);
        prompt_->Reset();
        decode_->Reset();
        size_t valid = 0;
        for (size_t offset = 0; offset < tokens.size(); offset += 128) {
            valid = std::min<size_t>(128, tokens.size() - offset);
            Feed(*prompt_, "prompt", offset, tokens.data() + offset, valid);
            prompt_->Run();
            std::cout << "\n[Prefill] " << offset + valid << "/" << tokens.size() << "\n" << std::flush;
            if (offset + valid < tokens.size()) Transfer(*prompt_, *prompt_);
        }
        int32_t next = Next(*prompt_, valid - 1);
        for (auto& target : decode_->inputs) {
            if (target.spec.name.rfind("past_", 0) != 0) continue;
            const auto current_name = "curr_" + target.spec.name.substr(5);
            Seed(prompt_->Input(target.spec.name), prompt_->Output(current_name), target, valid);
        }
        Generation result;
        result.stop_reason = "generation_limit";
        for (size_t step = 0; step < limit; ++step) {
            if (stops_.count(next)) {
                result.stop_reason = "eos";
                break;
            }
            ValidateToken(next);
            if (tokenizer_->IdToToken(next).empty()) throw std::runtime_error("模型生成了未分配的 Token ID.");
            result.tokens.push_back(next);
            on_tokens(result.tokens);
            if (step + 1 == limit) break;
            Feed(*decode_, "decode", tokens.size() + step, &next, 1);
            decode_->Run();
            next = Next(*decode_, 0);
            Transfer(*decode_, *decode_);
        }
        return result;
    }

 private:
    // 核对文件登记与长度,不重复复制或输出整列哈希.
    fs::path CheckedResource(const fs::path& root, const std::string& relative) const {
        const auto path = Resource(root, relative);
        for (const auto& entry : manifest_.at("files")) {
            if (entry.at("name") == relative) {
                if (entry.at("bytes").get<uintmax_t>() != fs::file_size(path)) {
                    throw std::runtime_error("资源长度与清单不符: " + relative);
                }
                return path;
            }
        }
        throw std::runtime_error("清单未登记资源: " + relative);
    }

    // 检查输入银行和每组 KV 的形状及索引边界.
    void ValidateGraph(Graph& graph, const std::string& phase, size_t batch) {
        size_t caches = 0;
        for (const auto& input : graph.inputs) {
            const auto& name = input.spec.name;
            if (name == "mask" || name == "pos_emb") {
                const auto& bank = bank_.at(phase + "_" + name);
                auto shape = input.spec.shape;
                shape.insert(shape.begin(), context_ / batch);
                if (bank.shape != shape || bank.word_size != 2 || bank.fortran_order) {
                    throw std::runtime_error("mask/RoPE 银行与图契约不匹配.");
                }
            } else if (name.rfind("past_", 0) == 0) {
                auto& output = graph.Output("curr_" + name.substr(5));
                CheckCache(output, input);
                // 当前部署图保留完整 1024 槽 past,mask 还包含本批新增位置.
                if (input.spec.shape[2] != context_ || output.spec.shape[2] != batch) {
                    throw std::runtime_error("KV 时间维与静态图不符.");
                }
                ++caches;
            } else if (name != "input_embeds") throw std::runtime_error("未适配的输入: " + name);
        }
        if (caches != 84 || graph.outputs.size() != caches + 1) {
            throw std::runtime_error("MiniCPM5 需要 42 层的 Key/Value 缓存.");
        }
        graph.Input("mask");
        graph.Input("pos_emb");
    }

    // 明确拒绝非法输入 Token,防止 embedding 越界.
    void ValidateToken(int32_t token) const {
        if (token < 0 || static_cast<size_t>(token) >= vocabulary_) throw std::runtime_error("Token ID 超出词表.");
    }

    // 填充 embedding 与该绝对位置的 SDK mask/RoPE,不改变已有 KV.
    void Feed(Graph& graph, const std::string& phase, size_t position,
              const int32_t* tokens, size_t valid) {
        auto& input = graph.Input("input_embeds");
        const size_t batch = input.spec.shape[1];
        if (position >= context_ || position % batch || valid > batch) {
            throw std::runtime_error("静态图输入位置或长度非法.");
        }
        std::fill_n(input.values.begin(), input.spec.Count(), input.spec.zero);
        for (size_t row = 0; row < valid; ++row) {
            ValidateToken(tokens[row]);
            for (size_t column = 0; column < width_; ++column) {
                input.values[row * width_ + column] = Convert(
                    embedding_->Data()[static_cast<size_t>(tokens[row]) * width_ + column],
                    prompt_->Input("input_embeds").spec, input.spec);
            }
        }
        for (const std::string name : {"mask", "pos_emb"}) {
            auto& target = graph.Input(name);
            const auto& bank = bank_.at(phase + "_" + name);
            const size_t count = target.spec.Count();
            std::copy_n(bank.data<int16_t>() + (position / batch) * count, count, target.values.begin());
        }
    }

    // 第一输出为 logits,正尺度 INT16 的 argmax 与反量化后的排序一致.
    int32_t Next(Graph& graph, size_t position) const {
        const auto& logits = graph.outputs.at(0);
        if (position >= logits.spec.shape[1]) throw std::runtime_error("logits 位置越界.");
        const auto first = logits.values.begin() + position * vocabulary_;
        return static_cast<int32_t>(std::max_element(first, first + vocabulary_) - first);
    }

    // Prefill 与 Decode 均采用后缀缓存布局,按张量各自尺度转移.
    void Transfer(Graph& source, Graph& target) {
        for (auto& input : target.inputs) {
            if (input.spec.name.rfind("past_", 0) == 0) {
                Append(source.Output("curr_" + input.spec.name.substr(5)), input);
            }
        }
    }

    Json manifest_;
    std::string template_;
    size_t context_ = 0, width_ = 0, vocabulary_ = 0;
    int32_t bos_ = 0, eos_ = 0;
    std::set<int32_t> stops_, special_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    cnpy::npz_t bank_;
    std::shared_ptr<RuntimeApi> api_;
    std::unique_ptr<Graph> prompt_, decode_;
    std::unique_ptr<Embedding> embedding_;
};

}  // 匿名命名空间.

// 工厂使平台资源与会话业务分离,失败时由各资源的 RAII 释放已加载部分.
std::unique_ptr<Engine> LoadMtkEngine(const fs::path& models, const fs::path& chat_template,
                                    const std::string& runtime_library) {
    return std::make_unique<MtkEngine>(models, chat_template, runtime_library);
}

}  // 命名空间 minicpm.
