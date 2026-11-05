# S100 语音控制 Demo

C++ 应用在 S100 上运行。FILE 处理完输入文件后退出，MIC 持续监听到 Ctrl+C。两者共用 Silero VAD → Whisper tiny → DeepSeek → 控制 JSON，结果输出到控制台，可作为 ROS2 String 的载荷。

**进度**：核心链路已开发。当前版本已整理应用入口、使用步骤和输出协议，FILE 与 MIC 均已通过 S100 交叉编译。上一版已在板端验证 FILE 静音、长文件多指令和文件末尾分句。新版输出待上板核对，真实 USB 麦克风、完整指令与噪声覆盖由用户验收，跨段会话历史仍待明确是否纳入六类控制场景。

## 1. 在已部署的板端运行

进入统一运行目录，按需执行一条命令。

```bash
cd /root/hailong.he/demo/voice_control

# FILE：读取音频文件，按检出的语音段输出零到多条结果。
./voice_control_file config.json /path/to/audio.wav

# MIC：连接 USB 麦克风并确认 ALSA 设备后启动，Ctrl+C 退出。
arecord -l
./voice_control_mic config.json
```

FILE 需要板端 PATH 中有 `ffmpeg`，支持其可解码的音频，程序分块转换为 16 kHz 单声道 PCM。纯静音时只有诊断日志。可以在 FILE 命令尾部追加文件路径，程序按参数顺序处理。MIC 的 `audio.device` 需与 `arecord -l` 查到的设备一致，例如 `plughw:1,0`。中文使用 `whisper.language=zh`，英文样例用 `en`。

程序将结果逐行写到 stdout，应用日志写到 stderr。需要结果文件时，在启动命令后加 `> results/run.jsonl`，终端仍可查看转录、进度和错误。

## 2. 从仓库重新构建和部署

1. 进入现有的 S100 Docker。首次使用时准备仓库的 `configs/platforms/s100.local.yaml`，填写平台所需的 `board.models_root`、`board.datasets_root` 和下列工具链路径。通过 `RDK_PLATFORM_CONFIG` 可指定已有的本地配置文件。

   ```yaml
   toolchain:
       voice_control:
           sdk_root: /path/to/llm-sdk-1.0
           board_sysroot: /path/to/s100-sysroot
           onnxruntime_root: /path/to/onnxruntime-aarch64
           cxx: /path/to/aarch64-linux-gnu-g++
   ```

   SDK、sysroot 和 ONNX Runtime 目录各提供匹配的 `include/` 和 `lib/`。MIC 构建还依赖目标 ALSA 头文件及链接库。Docker 中需要 Python 3 和 PyYAML 读取本地配置。
2. 在仓库根目录按需编译。脚本最后会打印二进制的完整路径，默认输出到 `/tmp/hailong/<当天日期>/voice_control/build/`。

   ```bash
   bash demo/smart_car/voice_control/build.sh FILE
   bash demo/smart_car/voice_control/build.sh MIC
   ```

3. 把新二进制复制到板端统一运行目录。已部署的板端可直接复用同目录的 `config.json`、`lib/` 和 `models/`。首次布置时，从 [配置示例](config.example.json) 创建 `config.json`，按其中的相对路径准备模型和配套文件。目标运行库放在 `lib/`，例如 `libonnxruntime.so.1`，板端固件库沿用 `/usr/hobot/lib`。切回第 1 节启动对应模式。

板端资源目录一览。

| 路径 | 内容 |
| --- | --- |
| `voice_control_file`、`voice_control_mic` | 两种编译模式的程序 |
| `config.json` | 模型路径、语言、设备、分句和队列设置 |
| `models/deepseek/` | 板端既有 DeepSeek-R1-Distill-Qwen-1.5B HBM、tokenizer 和原聊天模板，配套旧 LLM SDK 1.0 |
| `models/whisper_tiny/` | 本仓库部署的 Whisper tiny Encoder/Decoder HBM、Mel 滤波器、词表及解码配置 |
| `models/silero_vad/`、`lib/` | Silero VAD 5.1.2 ONNX、ONNX Runtime 1.20.1 aarch64 CPU 库和配套 SDK 库 |
| `examples/`、`results/` | 可选的测试音频与验证记录 |

模型路径相对配置文件所在目录解析。Silero 通过 ONNX Runtime CPU 推理，16 kHz 输入使用 512 样本帧和 64 样本上下文。[Silero 输入协议](https://github.com/snakers4/silero-vad/blob/v5.1.2/src/silero_vad/utils_vad.py) · [Silero 模型来源](https://pypi.org/project/silero-vad/5.1.2/) · [ONNX Runtime 库来源](https://pypi.org/project/onnxruntime/1.20.1/)

## 3. 控制 JSON 协议

每个进入识别流程的有效语音段产生一条 JSON。下面是 FILE 成功输出的字段示例，时间值为示意。

```json
{
    "version": "1.0",
    "message_type": "voice_control_result",
    "sequence_id": 1,
    "timestamp_ms": 1760000000000,
    "source": {
        "device_id": "voice_control",
        "sensor_id": "audio_file",
        "pipeline_id": "whisper_tiny_deepseek_v1"
    },
    "transcript": "请向前走",
    "command": "forward",
    "command_text": "前进",
    "recognized": true,
    "status": "completed",
    "since_audio_end_ms": 1344.0,
    "timing_ms": {
        "vad_compute": 30.0,
        "vad_tail_audio": 704.0,
        "queue": 0.0,
        "whisper_mel": 854.0,
        "whisper_encoder": 20.0,
        "whisper_decoder": 49.0,
        "deepseek_first_text": 355.0,
        "deepseek_total": 421.0
    }
}
```

- 用 `source.sensor_id` 区分 FILE 的 `audio_file` 和 MIC 的 `usb_microphone`。序号从每次进程启动的 1 开始，`timestamp_ms` 是 Unix 毫秒时间戳。
- 动作映射为前进 `forward`、后退 `backward`、左转 `left`、右转 `right`、停止 `stop`、未识别 `unknown`。只有 `status=completed` 且 `recognized=true` 时可取用动作。空转录或超限转录为 `ignored`，模型错误为 `inference_error`。有错误详情时附加 `error` 字段。无有效语音时不输出结果。
- `since_audio_end_ms` 从分句就绪计到 JSON 就绪前，包含队列等待和模型处理。VAD 推理累计耗时与尾静音音频时长分别记录，均在该边界之前。VAD 累计可包含上次结果之后的静音帧。未测阶段为 `null`。
- 本程序只输出载荷。旧版若按 `[结果] ` 前缀提取 JSON，更新后可直接逐行解析 stdout。

## 4. 代码入口与常用修改点

| 要改什么 | 入口 |
| --- | --- |
| 命令行、配置、FILE/MIC 主流程和退出 | [Linux 应用入口](src/platform/linux/application.cpp)，由 [main.cpp](src/main.cpp) 调用 |
| FILE 解码或 MIC 设备 | [Linux 音频输入](src/platform/linux/audio.hpp) |
| 语音分句、尾静音、最长语句 | [Silero 分句](src/shared/speech_segmenter.hpp)，优先修改 [运行配置](config.example.json) |
| 分类提示词、结果校验和动作映射 | [控制协议](src/shared/command.hpp) |
| 模型推理与计时 | [Whisper](src/platform/s100/whisper.hpp) · [DeepSeek](src/platform/s100/deepseek.hpp) |

FILE 同步逐段处理，保留顺序并在文件结束时处理最后一段。MIC 用采集线程和有界队列，队列满时丢弃新语音段并写日志，Ctrl+C 等待在途推理结束。超过单段长度上限的语音会丢弃，FILE 会报告超长段并返回失败。应用把当前段实际转录交给 DeepSeek，逐段重置上下文。默认 `llm.thinking=false`，通过在已验证的旧 SDK 模板前缀预填空思考段实现直答。模板不匹配时报错，模型重新生成思考标记时报告推理错误。可用 `llm.thinking=true` 对照原模板。[DeepSeek 模型说明](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B#usage-recommendations)

## 5. 验证记录与剩余工作

本次整理在既有 S100 Docker 中执行了构建脚本语法检查和 FILE、MIC 两种模式的交叉编译。两份产物均核对为 AArch64。本次验证范围限于构建，新版程序待部署。

2026-11-05 的板端 FILE 验证使用英文转录配置。纯静音输出零条控制结果。36.91 秒混合文件包含合成指令、31 秒静音和固定种子噪声，依序输出前进、右转、停止，最后一条没有尾静音。无关问句输出未识别。记录位于板端统一目录的 `results/silero_file_check.json` 和 `results/direct_file_check.json`。

| 指令 | 原模板分句后总耗时 | 直答分句后总耗时 | 直答 DeepSeek | Whisper Mel | Whisper Encoder | Whisper Decoder |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 前进 | 20.057 s | 1.344 s | 0.421 s | 0.854 s | 0.020 s | 0.049 s |
| 右转 | 9.769 s | 1.375 s | 0.452 s | 0.854 s | 0.020 s | 0.049 s |
| 停止 | 32.982 s | 1.304 s | 0.387 s | 0.855 s | 0.019 s | 0.042 s |

直答小样本中 CPU Mel 是最大单项开销，进程峰值 RSS 为 135 MiB。以上耗时从分句完成计起，真实 MIC 还包含采集和尾静音等待。

| 阶段 | 状态 |
| --- | --- |
| 资源、共同推理链路、FILE 分段 | 已接入并完成板端小样本验证 |
| MIC 持续采集、队列与退出 | 已实现，真实 USB 连续采集、繁忙和断开留给用户验收 |
| 应用入口、启动步骤、控制 JSON | 已整理，保留 `build.sh FILE\|MIC` 作为交叉编译入口 |
| 跨段会话历史 | 当前逐段独立分类。待明确是否需要理解上一条指令的上下文 |
| 完整验收 | 由用户执行，覆盖六类中文指令、噪声、否定、冲突和真实 USB 连续输入 |
