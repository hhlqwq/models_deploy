# MiniCPM5-2B C++ 连续对话

Genio 720 / MT8189 终端对话 Demo,使用板端已有模型,CMake 构建.

## 快速编译

在 Ubuntu Docker 中,从仓库根目录执行.先设置以下本地环境变量:

| 变量 | 内容 |
| --- | --- |
| `MINICPM5_CXX` | AArch64 g++ 的绝对路径 |
| `MINICPM5_SYSROOT` | 交叉编译器对应的目标 sysroot |
| `MINICPM5_NEURON_INCLUDE` | 包含 `neuron/api/RuntimeAPI.h` 的 SDK include 目录 |
| `MINICPM5_TARGET_DEPS` | NAS `G720/tools/cpp_llm/prepared/` 在容器中的挂载路径 |

```bash
BUILD_DIR="/tmp/hailongcodex/$(date +%F)/minicpm5_chat/build"

cmake -S demo/MiniCPM5_2B -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/demo/MiniCPM5_2B/cmake/genio_720.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEURON_INCLUDE_DIR="$MINICPM5_NEURON_INCLUDE" \
  -DCMAKE_PREFIX_PATH="$MINICPM5_TARGET_DEPS"

cmake --build "$BUILD_DIR" --parallel
```

产物: `$BUILD_DIR/minicpm5_chat`.

需要 CMake ≥ 3.18、C++17、Neuron SDK 头文件和 AArch64 依赖.已准备的依赖包括 JSON、minja、cnpy、zlib、tokenizers-cpp v0.1.1 及 Rust tokenizers 0.21.0 静态库.下载地址和版本记录位于 NAS `G720/tools/cpp_llm/download_sources.json`.

## 板端运行

进入已部署的 Demo 目录:

```bash
. ./.env
./minicpm5_chat
```

或显式传入参数:

```bash
./minicpm5_chat \
  --models "$MINICPM5_MODELS" \
  --chat-template "$MINICPM5_CHAT_TEMPLATE" \
  --max-new-tokens 128
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--models` | 已有模型资源目录 | 环境变量 `MINICPM5_MODELS` |
| `--chat-template` | 同版本官方 `chat_template.jinja` 路径 | 环境变量 `MINICPM5_CHAT_TEMPLATE` |
| `--max-new-tokens` | 每轮最多生成的 Token 数 | 128 |
| `--help` | 查看用法 | — |

模型目录需包含 `manifest.json`、清单指定的双 DLA,以及 `tokenizer/` 下的 `tokenizer.json`、`embedding_int16.bin`、`generation_inputs.npz`.运行时加载系统 `libneuronusdk_runtime.mtk.so.8`,可用 `MTK_RUNTIME_LIBRARY` 指定其他库路径.

## 功能与限制

- 多轮历史对话、中英文增量输出,模型只加载一次.
- `/reset` 清空历史;`/exit` 或 EOF 退出.
- 当前模型为 W4A16,上下文 **1024 Token**,Prefill 128 Token,Decode 每步 1 Token;使用 Greedy 解码并关闭 thinking.
- 每轮重新 Prefill 历史.输入与生成预算之和不能超过 1024 Token,超限时提示缩短输入或执行 `/reset`.

[板端测试结果](results/summary.json) · [实际对话示例](examples/dialogue.txt)
