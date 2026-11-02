# MiniCPM5-2B C++ 连续对话

Genio 720 / MT8189 终端对话 Demo,使用板端已有模型,CMake 构建.

## 配置

只使用一份 `config.json`,编译和运行共用.首次将 [config.example.json](config.example.json) 复制为 `config.json`,填写以下内容:

| 配置项 | 用途 |
| --- | --- |
| `build.platform` | 目标平台,当前为 `genio_720` |
| `build.type` | 构建类型,默认模板为 `Release` |
| `build.cxx` | Docker 内 AArch64 g++ 的绝对路径 |
| `build.sysroot` | 交叉编译器对应的 sysroot |
| `build.sdk_include` | 包含 `neuron/api/RuntimeAPI.h` 的 SDK include 目录 |
| `build.dependencies` | NAS `G720/tools/cpp_llm/prepared/` 在容器中的路径 |
| `runtime.models` | 板端已有模型资源目录 |
| `runtime.chat_template` | 同版本官方 `chat_template.jinja` 路径 |
| `runtime.max_new_tokens` | 每轮生成上限,默认模板为 128,有效范围 1～1023 |
| `runtime.runtime_library` | Runtime 库名或路径,默认模板为 `libneuronusdk_runtime.mtk.so.8` |

真实路径只写本地 `config.json`,该文件已被 Git 忽略.仓库中的 example 文件只用来复制填写.

## 编译

在 Ubuntu Docker 中,从仓库根目录执行,CMake 自动读取配置:

```bash
cmake -S demo/MiniCPM5_2B -B "/tmp/hailongcodex/$(date +%F)/minicpm5_chat/build"
cmake --build "/tmp/hailongcodex/$(date +%F)/minicpm5_chat/build" --parallel
```

需要 CMake ≥ 3.19、C++17 和已准备的 AArch64 依赖.产物为构建目录中的 `minicpm5_chat` 和同一份 `config.json`.依赖版本和下载地址见 NAS `G720/tools/cpp_llm/download_sources.json`.

## 运行

将程序、`config.json` 和聊天模板放到板端 Demo 目录,直接启动:

```bash
./minicpm5_chat
```

程序默认读取可执行文件旁的 `config.json`,只使用其中的 `runtime` 部分.模型、模板和带目录的 Runtime 相对路径均以配置文件所在目录为基准.`--config <文件>` 可选择另一份配置,`--help` 查看用法.

模型目录需有 `manifest.json`、清单指定的双 DLA,以及 `tokenizer/` 下的 `tokenizer.json`、`embedding_int16.bin`、`generation_inputs.npz`.

## 功能与限制

- 多轮历史对话、中英文增量输出,模型只加载一次.
- `/reset` 清空历史;`/exit` 或 EOF 退出.
- W4A16,上下文 1024 Token,Prefill 128 Token,Decode 每步 1 Token;Greedy 解码,关闭 thinking.
- 每轮重新 Prefill 历史,输入与生成预算之和不能超过 1024 Token.

本次统一配置改动未编译、未测试,由用户自行验证.[状态记录](results/summary.json)
