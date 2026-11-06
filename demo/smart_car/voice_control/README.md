# S100 语音控制 Demo

C++ 应用在 S100 上运行。FILE 处理完输入文件后退出，MIC 持续监听到 Ctrl+C。两者共用 Silero VAD 和 Whisper tiny。中文转录统一为简体后，普通指令交给 DeepSeek 分类，程序校验参数，专用紧急停车直接输出。结果以小车控制 JSON 或媒体意图 JSON 表示，可作为 ROS2 String 的载荷。

**先看业务**：[架构图、流程图与数据示例](docs/flow.md)。这里可以追踪一条口令经过哪些模块、何时调用 DeepSeek，以及需求变更会影响哪里。

**进度**：已接入中文转录繁转简。普通口令由 DeepSeek 判断，专用紧急停车直出。已加入试验性复合指令模式和整份计划校验。下一阶段处理 LLM 耗时，语音前处理和流式转录的提速放在后面。

## 1. 在已部署的板端运行

### 中文录音验收

进入统一运行目录，选一个入口。两个入口显式启用试验性的复合指令模式，沿用 `config.json` 中的中文转录配置。

```bash
cd /root/hailong.he/demo/voice_control
./run.sh short
./run.sh replay
```

| 入口 | 输入和运行方式 | 看哪里 |
| --- | --- | --- |
| `short` | 一次加载模型，按文件名遍历 `examples/` 顶层音频，跳过 `old/` 和 `流式音频.m4a`。出错时继续检查其余文件。 | `source.file` 标明文件，最后汇总已处理、运行失败和输出条数；语义未识别通过每条 JSON 判断。 |
| `replay` | `流式音频.m4a` 按实际采样节奏逐帧送入与 USB 麦克风相同的分句和推理队列，正常 EOF 补交末段、排空队列后退出。 | `source.sensor_id=audio_replay`，终端每 30 秒打印输入进度，末尾汇总队列和超长丢段。 |

回放沿用 MIC 的「闪电」前缀规则；短文件允许省略前缀。两种验收入口都只输出 JSON，媒体结果显示 `playback.status=intent_only`。将命令末尾加 `> results/short.jsonl` 或 `> results/replay.jsonl` 即可保存结果，进度和诊断仍显示在终端。回放输入保持真实时间节奏，识别依旧是 Silero 分句后交给 Whisper。复合模式的 DeepSeek 语义准确率仍待验收。

### 直接启动程序

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

MIC 的语音指令使用 `speech.wake_word` 前缀，默认「闪电」，例如「闪电，后退半米」「闪电停车」「闪电，播放第二首」。已有配置里若显式写了其他词，以该配置为准。目前在 Whisper 转录后检查前缀，一句话包含一条命令。FILE 可使用有前缀或无前缀的录音，均只输出 JSON。

中文模式在启动时加载板端 `libopencc.so.1.1` 和 `/usr/share/opencc/t2s.json`。转换在 C++ 进程内完成，唤醒词、紧急停车、LLM 输入和参数校验使用同一份简体文本。运行库或词典缺失时，程序启动会给出错误。 [OpenCC t2s 配置](https://github.com/BYVoid/OpenCC/blob/ver.1.1.3/data/config/t2s.json)

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

3. 说「闪电，播放第一首」或「闪电，停止播放」。播放时麦克风仍然持续采集。建议预置曲目避开唤醒词和完整口令，并在扬声器联调时核对误触发。 [mpv JSON IPC](https://mpv.io/manual/stable/#json-ipc)

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

3. 把两个二进制和 [run.sh](run.sh) 复制到板端统一运行目录，赋予执行权限。已部署的板端可直接复用同目录的 `config.json`、`lib/` 和 `models/`。首次布置时，从 [配置示例](config.example.json) 创建 `config.json`，按其中的相对路径准备模型和配套文件。目标运行库放在 `lib/`，例如 `libonnxruntime.so.1`，板端固件库沿用 `/usr/hobot/lib`。切回第 1 节启动对应模式。

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
    "transcript": "請向前走",
    "normalized_transcript": "请向前走",
    "command": "forward",
    "decision_path": "deepseek",
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
- `decision_path` 标明本次路径，分别为 `deepseek`、`emergency_stop` 或 `not_classified`。
- `transcript` 保留 Whisper 的原始转录。中文转录完成转换时，`normalized_transcript` 保存去除唤醒前缀之前的简体文本。若两者不同，终端会额外打印一行「简体转录」。
- 文件输入的 `source.file` 记录原始文件名；离线文件和实时回放分别用 `audio_file` 和 `audio_replay` 表示来源。
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
| 停止 | `stop` | 无 | 闪电停车 |
| 未识别 | `unknown` | 无 | 前进两步 |

距离和角度使用正值。显式数值支持阿拉伯数字和常用中文数字，距离接受米、厘米，角度接受度、°，角度也可用圈数表示。只有未指定参数时使用默认值。普通指令的说法可以变化，例如「往前挪两米」和「前进两米」都先交给 DeepSeek。程序随后从原文校验参数，不完整、含糊或使用未支持单位时输出未识别。转圈方向由接收方约定。

媒体使用 `message_type=voice_media_intent`。当前支持「播放第 N 首」→ `play`（`parameters.track_number` 从 1 开始）、「暂停音乐」→ `pause`、「继续播放」→ `resume`、「停止播放」→ `stop_playback`。外层 `completed` 表示意图已识别。 `playback.status` 为 `accepted` 时表示 mpv 已应答，为 `error` 时附带错误信息。FILE 模式或 MIC 未配置 socket 时为 `intent_only`。

### 试验性复合指令

在测试配置里设置 `speech.command_mode=plan` 后，DeepSeek 处理整段转录，返回按顺序和并行关系分组的动作。程序逐条验证模型引用的原文、参数和组间关系，然后一次性写出计划。语音模块负责发送结果，控制层解释顺序并执行。单动作仍使用原有消息类型。只要有一个动作或引用未通过校验，整段就返回未识别，格式或校验失败带 `error=invalid_command_plan`。

下面是「往前走两米后左转，然后再走一米的同时播放第一首歌」的**期望消息摘要**。

```json
{
    "message_type": "voice_command_plan",
    "decision_path": "deepseek",
    "recognized": true,
    "status": "completed",
    "steps": [
        [{"command": "forward", "parameters": {"distance_m": 2}}],
        [{"command": "left", "parameters": {"angle_deg": 90}}],
        [
            {"command": "forward", "parameters": {"distance_m": 1}},
            {"command": "play", "parameters": {"track_number": 1}}
        ]
    ]
}
```

外层步骤有先后，同一步里的动作并行。当前校验上限是 8 步，每步最多一个运动和一个媒体动作。计划中的播放随消息一起发出；独立媒体口令仍按 MIC 的 mpv 配置处理。跨 VAD 分段续说、改口和复杂歧义属于待验证范围，测试时注意区分转录漏词和模型编排错误。

## 4. 代码入口与常用修改点

| 要改什么 | 入口 |
| --- | --- |
| 命令行、配置、FILE/MIC 主流程和退出 | [Linux 应用入口](src/platform/linux/application.cpp)，由 [main.cpp](src/main.cpp) 调用 |
| FILE 解码或 MIC 设备 | [Linux 音频输入](src/platform/linux/audio.hpp) |
| mpv 命令、曲库与 IPC 应答 | [播放器接口](src/platform/linux/mpv_player.hpp) |
| 语音分句、尾静音、最长语句 | [Silero 分句](src/shared/speech_segmenter.hpp)，优先修改 [运行配置](config.example.json) |
| 分类提示词、结果校验和动作映射 | [控制协议](src/shared/command.hpp) |
| 复合指令提示词、步骤校验和计划消息 | [复合计划](src/shared/command_plan.hpp) |
| 模型推理与计时 | [Whisper](src/platform/s100/whisper.hpp) · [DeepSeek](src/platform/s100/deepseek.hpp) |

FILE 同步逐段处理，保留顺序并在文件结束时处理最后一段。MIC 用采集线程和有界队列，队列满时丢弃新语音段并写日志，Ctrl+C 等待在途推理结束。超过单段长度上限的语音会丢弃，FILE 会报告超长段并返回失败。只有专用紧急口令「闪电停车」在转录后直接输出，仍经过分句与 Whisper。普通运动、停止和媒体口令均交给 DeepSeek 判意图，程序再校验标签与原文参数。模型逐段重置上下文。默认 `llm.thinking=false`，通过在已验证的旧 SDK 模板前缀预填空思考段实现直答。模板不匹配或模型重新生成思考标记时报告错误。可用 `llm.thinking=true` 对照原模板。[DeepSeek 模型说明](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B#usage-recommendations)

## 5. 验证记录与剩余工作

简体中文阶段已更新板端 FILE 和 MIC 程序。两种模式交叉编译通过。板端 OpenCC 文本探针验证了「閃電，往前走兩米」「閃電停車」和「閃電，向左轉同時暫停音樂」，分别覆盖唤醒词、距离、紧急停车及复合参数匹配。此记录为文字链路验证，用户录音仍留待验收。

中文验收入口已部署到板端。两个程序通过 S100 交叉编译，板端短检查覆盖中文及带空格的文件名筛选、跳过历史归档、遇到缺失文件后继续处理，以及 0.24 秒临时静音按节奏回放后正常退出。用户提供的 15 段短录音和长录音留给正式验收。

复合指令阶段通过 FILE/MIC 交叉编译及本地断言。断言覆盖三步含并行的正确计划、原文被改写、缺少曲目、错分先后与并行及整体拒绝。2026-11-06 的板端 DeepSeek 文本探针中，试验模式识别了单条「前进两米」（约 0.94 秒），并返回空计划处理「天气怎么样」（约 0.65 秒）。目标复合口令约 2.66 秒返回，但把前进与左转放进同一步，并改写了原文片段，校验后输出未识别。保留 `single` 为默认，复合模式尚未通过板端语义验收。

普通口令分流已纠正。快速文本断言覆盖了运动参数、同义表达、无效参数和紧急停车条件。板端使用既有 DeepSeek 跑了四条转录文本探针。「前进两米」和「往前挪两米」分别在约 0.41、0.39 秒内返回「前进」，最终参数均为 2 米。「把车停下」复述原话，「放一下第二首歌」生成了约 7.66 秒的解释，两条都因返回格式不符而输出未识别。预填空思考段仍不能保证模型每次只给标签。当前记录是文本探针，普通停止和媒体分类仍需改进，实际语音识别和扬声器播放待设备联调。

2026-11-05 的 `0f37254` 做了关闭思考的板端对照，当时三个样本都经过 DeepSeek。2026-11-06 的 `820ff58` 增加了普通口令的本地分流，本版恢复普通口令先进入模型。两次变动的耗时数据应分别看。

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
| 复合语义 | 已实现试验模式、关系校验和消息协议；现用 DeepSeek 未通过目标复合口令探针 |
| mpv、硬件音频与流式停车 | 已接入 mpv IPC，实际音频设备和曲库待配置；流式关键词检测待接入 |
| 流式转录 | 当前先分句再转录；候选是 sherpa-onnx 中文流式 Zipformer 14M INT8。后续用同一批录音对比，统计说完到 JSON 输出的耗时。[候选模型说明](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html) |
| 完整验收 | 由用户执行，覆盖中文动作及参数、噪声、否定、冲突和真实 USB 连续输入 |

### 中文样本判定要点

验收使用板端 `examples/` 顶层中文录音。以实际说话内容和 JSON 中的 `transcript` 为准，文件名仅用于定位。历史英文合成样本放在 `old/`，不参与批量验收。

| 场景 | 样本举例 | 期望与当前边界 |
| --- | --- | --- |
| 单动作 | 往前挪两米、向左转一下、原地转一个圈、暂停音乐 | 核对动作码与参数。转写正确但输出未识别时，检查模型和参数校验。 |
| 普通停止 | 闪电停止 | 走 DeepSeek，期望 `stop`。只有「闪电停车」属于紧急直出。 |
| 先后与并行 | 先向前再向右、向前一米同时暂停音乐 | 核对 `steps` 的顺序与并行组。若实际表达是向右平移，当前动作表没有对应命令。 |
| 改口 | 向左……不对，向右转 | 检查是否误发左转。复合模式允许拒绝整段，改口语义仍待改进。 |
| 超出动作表 | 上一首、下一首、加快一点速度、随便走走 | 当前应输出未识别。不能把上一首、下一首猜成某个固定曲目，也不能把加速或任意方向猜成前进。 |
| 实时回放 | `流式音频.m4a` | 核对每段输出、排队时间和丢段计数。样本中的停顿可能把相邻口令分成多段。 |
