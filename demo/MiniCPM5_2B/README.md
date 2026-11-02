# MiniCPM5-2B C++ 连续对话 Demo

独立终端应用,使用 **CMake 一种构建入口**,不提供 Makefile,不依赖 ROS 2.直接读取开发板已经部署的 MiniCPM5-2B 模型资源,重写 C++ 推理和会话逻辑,不重新转换、量化、编译模型或生成模型副本.

**当前状态: 2026-11-02 已完成 AArch64 CMake 构建和 Genio 720 板端实际测试.七个固定用例的输入 Token、生成 Token、文本及停止原因全部与原生成器一致;连续对话、重置、超长输入恢复、中英文输出及退出均通过.** 实际范围见 [summary.json](results/summary.json),对话展示见 [dialogue.txt](examples/dialogue.txt).本次只验证功能与原实现一致性,未做全量精度、专项性能或 RSS 评价.

## 功能

- 启动时加载 Prefill/Decode 双 DLA,只读映射 embedding,分词器和 SDK mask/RoPE 银行常驻.
- 从终端连续读取输入,保留本进程中的用户/助手历史,按官方模板重新编码并执行 Prefill,随后 Greedy Decode.
- 增量打印完整 UTF-8 文本,显示加载、Prefill 进度和停止原因;达到生成上限时舍弃尚未完整解码的尾部字节.
- `/reset` 清空历史但不卸载模型;`/exit` 或终端 EOF 退出并释放资源.
- 按部署容量检查“完整模板后的历史输入 + 本轮生成预算”.当前为 1024 Token,默认生成上限 128 Token.超限时拒绝本轮,提示缩短输入或 `/reset`,不自动截掉用户内容或冒充无限上下文.
- 每轮重新构建 KV,保留多轮语义历史.不实现跨轮增量 KV、历史落盘、自动摘要、并发会话、性能评价或精度报告.

## 参考实现和兼容边界

模型配置与图契约参考 [原模型说明](../../models/gen_ai/llm/minicpm5_2b/README.md)、[部署适配记录](../../models/gen_ai/llm/minicpm5_2b/ADAPTATION.md).计算流程对齐 [现有 Python 生成器](../../tools/llm/generate.py)、[KV 转移实现](../../tools/llm/evaluate_board_quality.py)及 [SDK Runtime 调用](../../tools/runtime.py).没有将 Python 或 `llm_cmdline_tool` 作为后台推理进程.

当前只接受配套的 MiniCPM5-2B W4A16、1024 上下文、128 Token Prefill、单 Token Decode、42 层 KV 契约,停止 Token 为 1/130073.保留 INT16 量化、跨图重新量化和后缀有效 KV;最后一块 Prefill 的右侧填充不写入 Decode 有效缓存.未知布局和不匹配资源直接报错,不回退其他后端.

实际部署图的 Prefill 和 Decode 都有 1024 槽 past cache,当前块输出分别为 128/1 Token;mask 最后两维分别为 128×1152 和 1×1025.不能将 past 容量误算为上下文减去批长.引擎按 SDK 头文件声明的 C ABI 动态加载板端原有 Runtime,可通过本地环境 `MTK_RUNTIME_LIBRARY` 指定库位置,默认 `libneuronusdk_runtime.mtk.so.8`.构建不直接链接新板端 GLIBC 依赖,不覆盖系统库.

资源路径通过参数或本地环境指定,不修改原 `models/` 部署链路和共享 skill.不会检查大文件的完整哈希;会检查清单登记、文件长度、图契约及 SDK 原生缓冲区大小.这些检查不能代替资源完整性核验或真实运行.

## 板端资源

`--models` 指向**已有部署包的 models 目录**,其中必须有:

```text
models/
├── manifest.json
├── 1024c/                    # 清单 model_files 指定的两个 DLA.
└── tokenizer/
    ├── tokenizer.json
    ├── embedding_int16.bin
    └── generation_inputs.npz # 原部署包生成的 INT16 mask/RoPE.
```

额外通过 `--chat-template` 指向**同版本官方 `chat_template.jinja`**.现有打包实现只复制 tokenizer.json,不包含该模板;实施运行前需检查板端是否已有模板,没有时仅补充这一小份原始文本资源,无需移动或替换模型.不要使用固定 `en_demo.txt` 代替可处理历史的模板.

模板来源为 [官方模型资源](https://huggingface.co/openbmb/MiniCPM5-2B/tree/f97400052a43d642bbc6e9975e2397e3ae6a6b52),具体文件 `chat_template.jinja`.版本须与部署包的原始资源记录一致.本次已从现有 NAS 原始资源取得该小文件,放入板端独立 Demo 目录,未改动原模型包.引擎读取模板,使用 minja 渲染 `messages`、`bos_token`、`eos_token`、`enable_thinking=false` 和 `add_generation_prompt=true`,不会硬编码其他模型的角色分隔符.

## 已有依赖要求

依赖在配置阶段只查找,不自动下载或编译第三方项目.应提供与目标 AArch64 工具链和系统 ABI 匹配的已有库:

| 依赖 | 用途与来源 | CMake 设置 |
| --- | --- | --- |
| MTK Neuron SDK | 与当前 C ABI 匹配的 RuntimeAPI.h,运行时加载板端已有库 | 构建设置 `NEURON_INCLUDE_DIR`;运行时可设 `MTK_RUNTIME_LIBRARY` |
| nlohmann/json,至少 3.10 | 部署清单、分词资源和模板上下文;[官方项目](https://github.com/nlohmann/json) | `nlohmann_json_DIR` 或 `JSON_INCLUDE_DIR`,支持已有单头文件 |
| tokenizers-cpp | 加载部署包 HF tokenizer.json;[官方接口](https://github.com/mlc-ai/tokenizers-cpp/blob/v0.1.1/include/tokenizers_cpp.h) | `TOKENIZERS_INCLUDE_DIR`、`TOKENIZERS_LIBRARY`、`TOKENIZERS_C_LIBRARY` |
| minja | C++ Jinja 模板解释器;[官方项目](https://github.com/google/minja) | `MINJA_INCLUDE_DIR`,包含 `minja/minja.hpp` 的目录 |
| cnpy + zlib | 直接读取已有 generation_inputs.npz;[官方项目](https://github.com/rogersce/cnpy) | `CNPY_INCLUDE_DIR`、`CNPY_LIBRARY` 与 ZLIB 设置 |

本次使用 tokenizers-cpp v0.1.1 的 HF C++ 绑定与对应 Rust 静态库,分别为 `libtokenizers_cpp.a` 和 `libtokenizers_c.a`.Rust 工具链为 1.88.0,Rust tokenizers 固定为 **0.21.0**;不能让该 tag 自动升级到 0.21.4,否则原绑定的词表类型不匹配.已在 NAS 保留解析后的 Cargo.lock,重建时使用 `cargo build --locked`.不需要 SentencePiece 后端,也不以板端另一套 libhf-tokenizer 代替该接口.

下载、源码、Cargo 缓存及已编译依赖保存在用户指定 NAS 的 `G720/tools/cpp_llm/`,没有向系统安装.其中 `prepared/` 为已验证 AArch64 依赖前缀,`sources/` 保存分词器源码及锁文件,`archives/` 保存下载内容.具体源码地址为:

- JSON 3.11.3: [单头文件](https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp).
- minja: [本次实际使用的头文件](https://raw.githubusercontent.com/google/minja/021c2293c187789ef13d56c6cfd89c9b134fd80f/include/minja/minja.hpp).
- cnpy: [头文件](https://raw.githubusercontent.com/rogersce/cnpy/4e8810b1a8637695171ed346ce68f6984e585ef4/cnpy.h)、[实现](https://raw.githubusercontent.com/rogersce/cnpy/4e8810b1a8637695171ed346ce68f6984e585ef4/cnpy.cpp).
- zlib 1.3.1: [源码压缩包](https://zlib.net/fossils/zlib-1.3.1.tar.gz).
- tokenizers-cpp v0.1.1: [官方源码压缩包](https://codeload.github.com/mlc-ai/tokenizers-cpp/tar.gz/refs/tags/v0.1.1),Rust tokenizers 使用 [0.21.0 crate](https://static.crates.io/crates/tokenizers/tokenizers-0.21.0.crate).
- Rust: [rustup 1.28.2 安装器](https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init)、[Rust 1.88.0 发行清单](https://static.rust-lang.org/dist/channel-rust-1.88.0.toml).实际组件与 crate 下载地址记录在 NAS `download_sources.json`,不在本文堆叠校验值.

依赖与应用均在 Ubuntu Docker 中用 GCC 11.4.0 交叉构建,目标为 AArch64.本次任务容器复用了原容器实际使用的本地镜像,额外挂载指定 NAS tools 和任务临时目录,没有重启共享容器.各依赖许可证随 NAS 资源保存.minja 通过本次实际模型模板用例,不代表所有 Jinja 特性均已验证.

## 构建

在已配置的 Ubuntu Docker 环境中构建,提前从本地平台配置设置 `MINICPM5_CXX`、`MINICPM5_SYSROOT`、`MINICPM5_NEURON_INCLUDE` 和 `MINICPM5_TARGET_DEPS`.依赖前缀指向上述已验证的 NAS `prepared/` 目录在容器中的挂载位置.这些真实路径只放本地环境,不写进公开文件.

```bash
cmake -S demo/MiniCPM5_2B \
  -B /tmp/hailongcodex/2026-11-02/minicpm5_chat/build \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/demo/MiniCPM5_2B/cmake/genio_720.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEURON_INCLUDE_DIR="$MINICPM5_NEURON_INCLUDE" \
  -DCMAKE_PREFIX_PATH="$MINICPM5_TARGET_DEPS"

cmake --build /tmp/hailongcodex/2026-11-02/minicpm5_chat/build --parallel
```

没有标准安装布局的依赖可显式设置上表中的路径.所有库必须属于目标架构;只有构建工具在主机执行.也可以提供已有可信 CMake toolchain 文件替换仓库示例.不会写板端系统目录或升级 Runtime.

## 板端启动

板端独立 Demo 目录已部署可执行程序、同版本模板和本地 `.env`.程序复用系统原有 Runtime;分词、NPZ 和 zlib 本次采用静态库链接,无需向板端安装这些依赖.模型继续留在已有部署位置.

进入本地配置指定的 Demo 目录后执行:

```bash
. ./.env
./minicpm5_chat
```

也可以显式指定路径和本轮生成预算:

```bash
./minicpm5_chat --models "$MINICPM5_MODELS" \
  --chat-template "$MINICPM5_CHAT_TEMPLATE" --max-new-tokens 128
```

设置 `MINICPM5_MODELS` 与 `MINICPM5_CHAT_TEMPLATE` 后也可直接运行 `./minicpm5_chat`.这两个路径是必需运行输入,不在公开配置里写死.加载失败、模板不兼容、非法 Token 或 SDK 失败时返回非零退出码,不把不完整回答写入后续会话.

## 工程边界与板端验证

- `core/chat.*`: 纯 C++ 会话、历史、容量与输出逻辑,只依赖抽象 Engine.
- `platforms/mtk/engine.*`: SDK、INT16 张量、NPZ、分词器、模板渲染和 Linux 只读映射.平台依赖限制在后端,SDK 类型不进入会话接口.后续换平台新增引擎并在入口/CMake 选择,业务会话不变.
- `main.cpp`: 终端参数和平台工厂组装.本 Demo 不强制绑定小车工程的 ROS 2 构建.

2026-11-02 实测结果:

- 三条历史消息的 C++ 模板渲染为 188 字节,与原框架逐字一致.
- 七个真实 NPU 用例全部对齐: 短输入、多轮历史、中文、190 Token 跨块输入及精确的 128/129/256 Token 边界.输入 Token、生成 Token、文本和停止原因均一致;比较时统一排除末尾停止 Token.
- 实际终端连续两轮对话能回忆 Alice,过程中 Prefill/Decode 各只加载一次.执行 `/reset` 后历史重新计数.
- 1114 Token 输入超出当前预算时被拒绝,未写入历史;随后中文输入仍正常生成,没有 UTF-8 替换字符.
- `/exit`、EOF 和 `--help` 返回成功,缺失资源返回非零;最终安装目录再次生成 `OK`.安装产物与完成测试的二进制一致.

本次实际修复了两个实现/链接问题: past KV 容量的错误假设,以及旧 sysroot 直接链接板端新 GLIBC 库的失败.分词依赖同时固定到可编译的 0.21.0 组合.原框架参考加载 tokenizer 时使用了与现有部署相同的 BPE merges 内存序列化兼容,没有修改官方资源.

上述结果仅证明这套已有模型和当前板端环境中的功能与生成一致性.1024 Token 容量保持不变,不代表可无限保留历史;没有测量全量精度、独立吞吐、延迟或 RSS,也不沿用历史指标.
