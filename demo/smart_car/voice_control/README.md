# S100 语音控制 Demo

C++ 应用在 S100 上运行。FILE 处理完输入文件后退出，MIC 持续监听到 Ctrl+C。两者共用 Silero VAD 和 Whisper tiny。转录交给本地口令规则或 DeepSeek，再输出小车控制 JSON 或独立的媒体意图 JSON，可作为 ROS2 String 的载荷。

**进度**：已实现运动参数、转圈、MIC 转录前缀检查和 mpv IPC 播放控制。实际麦克风、扬声器和曲库待接入，流式停车关键词检测待验证。当前方案使用原始麦克风输入，扬声器串音需在设备联调时观察。历史 FILE 板端验证见第 5 节。

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

MIC 的语音指令使用 `speech.wake_word` 前缀，默认「小车」，例如「小车，后退半米」「小车停车」「小车，播放第二首」。目前在 Whisper 转录后检查前缀，一句话包含一条命令。FILE 可使用有前缀或无前缀的录音，均只输出 JSON。

程序将结果逐行写到 stdout，应用日志写到 stderr。需要结果文件时，在启动命令后加 `> results/run.jsonl`，终端仍可查看转录、进度和错误。

### 启用 MIC 模式的音乐播放

1. 在 `config.json` 中填写板端实际 MP3 路径。曲目按数组顺序编号，路径和 IPC socket 都相对配置文件所在目录解析。 `output_device` 留空时使用 mpv 当前设备，接好扬声器后可用 `mpv --audio-device=help` 查看可用名称。

   ```json
   {
       "media": {
           "ipc_socket": "run/mpv.sock",
           "output_device": "",
           "tracks": ["audio/song1.mp3", "audio/song2.mp3"]
       }
   }
   ```

2. 在统一运行目录的另一个终端启动本地 mpv，再启动 MIC。socket 仅供本机访问。

   ```bash
   install -d -m 700 run
   mpv --no-config --no-video --idle=yes --input-ipc-server="$PWD/run/mpv.sock"
   ```

3. 说「小车，播放第一首」或「小车，停止播放」。播放时麦克风仍然持续采集。建议预置曲目避开唤醒词和完整口令，并在扬声器联调时核对误触发。 [mpv JSON IPC](https://mpv.io/manual/stable/#json-ipc)

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
    "parameters": {"distance_m": 1.0},
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
- 只有 `status=completed` 且 `recognized=true` 时可取用动作。空转录或超限转录为 `ignored`，模型错误为 `inference_error`。有错误详情时附加 `error` 字段。无有效语音时不输出结果。
- `since_audio_end_ms` 从分句就绪计到 JSON 就绪前，包含队列等待和模型处理。VAD 推理累计耗时与尾静音音频时长分别记录，均在该边界之前。VAD 累计可包含上次结果之后的静音帧。未测阶段为 `null`。
- 本程序只输出载荷。旧版若按 `[结果] ` 前缀提取 JSON，更新后可直接逐行解析 stdout。

| 动作 | 命令码 | 参数和默认值 | 带参数示例 |
| --- | --- | --- | --- |
| 前进 | `forward` | `distance_m`，默认 1 | 前进两米 |
| 后退 | `backward` | `distance_m`，默认 1 | 后退半米 |
| 左转 | `left` | `angle_deg`，默认 90 | 左转四十五度 |
| 右转 | `right` | `angle_deg`，默认 90 | 右转 30° |
| 转圈 | `spin` | `angle_deg`，默认 360 | 转两圈，输出 720° |
| 停止 | `stop` | 无 | 小车停车 |
| 未识别 | `unknown` | 无 | 前进两步 |

距离和角度使用正值。显式数值支持阿拉伯数字和常用中文数字，距离接受米、厘米，角度接受度、°，转圈另接受圈数。只有未指定参数时使用默认值。带显式参数的口令使用表中的动作写法，不完整、含糊或使用未支持单位时输出未识别。转圈方向由接收方约定。

媒体使用 `message_type=voice_media_intent`。当前支持「播放第 N 首」→ `play`（`parameters.track_number` 从 1 开始）、「暂停音乐」→ `pause`、「继续播放」→ `resume`、「停止播放」→ `stop_playback`。外层 `completed` 表示意图已识别。 `playback.status` 为 `accepted` 时表示 mpv 已应答，为 `error` 时附带错误信息。FILE 模式或 MIC 未配置 socket 时为 `intent_only`。

## 4. 代码入口与常用修改点

| 要改什么 | 入口 |
| --- | --- |
| 命令行、配置、FILE/MIC 主流程和退出 | [Linux 应用入口](src/platform/linux/application.cpp)，由 [main.cpp](src/main.cpp) 调用 |
| FILE 解码或 MIC 设备 | [Linux 音频输入](src/platform/linux/audio.hpp) |
| mpv 命令、曲库与 IPC 应答 | [播放器接口](src/platform/linux/mpv_player.hpp) |
| 语音分句、尾静音、最长语句 | [Silero 分句](src/shared/speech_segmenter.hpp)，优先修改 [运行配置](config.example.json) |
| 分类提示词、结果校验和动作映射 | [控制协议](src/shared/command.hpp) |
| 模型推理与计时 | [Whisper](src/platform/s100/whisper.hpp) · [DeepSeek](src/platform/s100/deepseek.hpp) |

FILE 同步逐段处理，保留顺序并在文件结束时处理最后一段。MIC 用采集线程和有界队列，队列满时丢弃新语音段并写日志，Ctrl+C 等待在途推理结束。超过单段长度上限的语音会丢弃，FILE 会报告超长段并返回失败。明确的停车口令在转录后直接输出，跳过 DeepSeek，仍经过分句与 Whisper。DeepSeek 只处理本地规则未覆盖的无参数表达，并逐段重置上下文。默认 `llm.thinking=false`，通过在已验证的旧 SDK 模板前缀预填空思考段实现直答。模板不匹配或模型重新生成思考标记时报告错误。可用 `llm.thinking=true` 对照原模板。[DeepSeek 模型说明](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B#usage-recommendations)

## 5. 验证记录与剩余工作

当前版本在既有 S100 Docker 中通过 FILE 和 MIC 交叉编译。模拟 mpv IPC 验证了播放、暂停、错误应答和超时。上一阶段用转录文本样例核对了默认参数、中文数字、无效参数、唤醒前缀、停车和媒体 JSON。新版待上板，实际语音识别和扬声器播放留待设备接入后核对。

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
| mpv、硬件音频与流式停车 | 已接入 mpv IPC，实际音频设备和曲库待配置；流式关键词检测待接入 |
| 完整验收 | 由用户执行，覆盖中文动作及参数、噪声、否定、冲突和真实 USB 连续输入 |
