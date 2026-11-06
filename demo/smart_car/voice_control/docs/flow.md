# S100 语音控制 Demo｜架构、流程与数据

这份文档描述仓库当前的业务。运行、配置和验证进度见 [README](../README.md)。

## 1. 从使用者看，这个 Demo 做什么

- **MIC 模式**持续听 USB 麦克风。说「闪电，后退半米」，程序输出一条小车命令 JSON，由外部接收方决定怎么使用。说「闪电，播放第二首」，如果配置了本地 mpv，程序就请求播放器播放预置的第二首 MP3，同时输出一条媒体结果 JSON。
- **FILE 模式**用已有录音验证识别和分类。一段录音可以有零条、一条或多条命令，程序按语音段顺序输出，读完退出。可以用它验证「播放第二首」的识别结果，实际播放入口只在 MIC 模式启用。
- **实时回放**用长录音替代 USB 麦克风，按实际时间输入 PCM。它与 MIC 共用分句和有界队列，便于发现慢推理造成的排队和丢段，结果带 `source.sensor_id=audio_replay`。
- **对外边界**是控制台 JSON 和本地 mpv IPC。运动命令到 JSON 为止，底盘、ROS 2 发布和动作执行由外部系统处理。

## 2. 架构图｜哪些模块互相依赖

```mermaid
flowchart LR
    subgraph Inputs["音频入口 · FILE 与 MIC"]
        Mic["MIC · USB 麦克风 / ALSA"]
        Replay["MIC 回放 · 音频文件 / 实时节奏"]
        File["FILE · 音频文件 / ffmpeg"]
    end
    subgraph Speech["语音处理"]
        Vad["Silero VAD · CPU 分句"]
        Asr["Whisper tiny · CPU Mel + BPU 转录"]
    end
    subgraph Decision["指令处理"]
        Gate["前缀检查 · 紧急停车分流"]
        Validate["动作和原文参数校验"]
        Llm["DeepSeek · 单动作判断或复合计划"]
    end
    subgraph Outputs["输出"]
        Json["stdout · 一段一行 JSON"]
        Player["本机 mpv · Unix socket"]
        Speaker["板端扬声器"]
    end

    Mic -->|"16 kHz 单声道 PCM"| Vad
    Replay -->|"每 32 ms 输入一帧 PCM"| Vad
    File -->|"16 kHz 单声道 PCM"| Vad
    Vad -->|"完整语音段"| Asr
    Asr -->|"原始转录"| Text["OpenCC · 中文转简体"]
    Text -->|"统一字形后的文本"| Gate
    Gate -->|"普通口令"| Llm
    Gate -->|"紧急停车或入口拒绝"| Json
    Llm -->|"动作标签或分组计划"| Validate
    Validate -->|"小车命令、媒体意图或未识别"| Json
    Validate -->|"MIC 媒体意图且已配置 IPC"| Player
    Player -->|"IPC 应答"| Json
    Player -->|"预置 MP3"| Speaker
```

前缀检查通过后，普通运动、普通停止和媒体口令均由 DeepSeek 理解。「闪电停车」是专用紧急直出口令。MIC 调用了 mpv 时，IPC 应答会先并入同一条媒体结果，再输出。标准输出承载结果，标准错误承载日志。[应用入口](../src/platform/linux/application.cpp) · [配置示例](../config.example.json)

## 3. 流程图｜收到一段语音后怎么分流

```mermaid
flowchart TD
    Audio["持续采集或分块读取音频"] --> Segment{"Silero 产出有效语音段？含 FILE 末尾收尾"}
    Segment -->|"否"| More["继续收音或读文件；FILE 结束则退出"]
    Segment -->|"是"| Queue["MIC 进有界队列；FILE 同步处理"]
    Queue --> Whisper["计算 Mel，Whisper 生成文字"]
    Whisper --> Valid{"转录文本有效？"}
    Whisper -->|"推理失败"| Error["输出 inference_error"]
    Valid -->|"空或超限"| Ignored["输出 ignored"]
    Valid -->|"有效"| Simplify["中文转简体"]
    Simplify --> Wake{"MIC 前缀有效，且有命令正文？"}
    Wake -->|"否"| Unknown["输出 unknown"]
    Wake -->|"是，或 FILE 有正文"| Emergency{"完整转录是闪电停车？"}
    Emergency -->|"是"| Stop["直接输出 stop"]
    Emergency -->|"否"| DeepSeek["把普通口令正文交给 DeepSeek"]
    DeepSeek -->|"返回动作标签或分组计划"| Verify{"动作、原文参数和关系是否有效？"}
    Verify -->|"有效运动或普通停止"| Motion["输出小车命令和适用参数"]
    Verify -->|"有效媒体意图"| Playback{"MIC 已配置 mpv socket？"}
    Verify -->|"有效复合计划"| Plan["一次发送 voice_command_plan"]
    Verify -->|"无效"| Unknown
    Playback -->|"是"| IPC["查曲库或设置播放状态，记录 mpv 应答"]
    Playback -->|"否"| Intent["仅保留媒体意图"]
    IPC --> Media["输出媒体结果 JSON"]
    Intent --> Media
    DeepSeek -->|"推理失败"| Error
```

| 输入模式 | 从开始到结束 |
| --- | --- |
| FILE | ffmpeg 分块解码，Silero 检出一段就同步识别一段。按文件及语音段顺序输出，到文件末尾还会提交没有尾静音的最后一段。纯静音可以零输出。 |
| MIC | 采集线程持续读取、分句并入队，识别线程按顺序消费。队列满时丢弃新完成的语音段并记录日志。Ctrl+C 停止采集并等待在途处理。 |
| 实时回放 | 用文件解码帧替换 MIC 采集，按 16 kHz 样本时钟喂入同一队列。正常 EOF 补交尾段并排空队列，记录队列及超长丢段，媒体仅输出意图。 |

「闪电」目前由 **Whisper 转录后的文本规则**检查。专用「闪电停车」通过检查后直接发 `stop`，仍经过 Silero、队列和 Whisper。普通「闪电，停下」仍走 DeepSeek。实时音频上的关键词快路径属于待办。扬声器播放时 MIC 继续收音，当前没有回声消除。

## 4. 数据图｜拿一句话走一遍

以 MIC 听到「闪电，往前挪两米」为例。图中文字是示意转录，JSON 只摘出业务字段，完整字段和状态见 [JSON 协议](../README.md#3-控制-json-协议)。

```mermaid
sequenceDiagram
    participant A as 麦克风
    participant V as Silero
    participant W as Whisper
    participant C as 指令分流
    participant L as DeepSeek
    participant P as 参数校验
    participant O as 控制台

    A->>V: 16 kHz 单声道 PCM，逐帧输入
    V->>W: 完整语音段，通过 MIC 队列
    W->>C: 转录为「闪电，往前挪两米」
    C->>L: 去掉闪电前缀，提交「往前挪两米」
    L->>P: 动作标签「前进」
    Note over P: 从原文校验「两米」，转换成 2 m
    P->>O: forward，distance_m 为 2
```

| 处理位置 | 输入 → 输出 | 在哪里改 |
| --- | --- | --- |
| 音频入口 | 麦克风采样或音频文件 → 16 kHz 单声道 PCM。FILE 解码后的数据按块进入分句。 | [Linux 音频输入](../src/platform/linux/audio.hpp) |
| Silero | 每次 512 个新采样点，带上历史上下文和循环状态 → 人声概率 → 根据尾静音及段长规则生成 PCM 语音段。 | [分句规则](../src/shared/speech_segmenter.hpp) · [audio、vad 配置](../config.example.json) |
| Whisper | 语音段 → CPU 计算 Mel → BPU Encoder 和 Decoder → Token 序列 → 中文或英文转录。原始转录保留在结果的 `transcript`。 | [Whisper 实现](../src/platform/s100/whisper.hpp) |
| 繁转简 | 中文原始转录 → 进程内 OpenCC → `normalized_transcript`。例如「閃電，往前走兩米」变成「闪电，往前走两米」。 | [简体转换](../src/platform/linux/simplified_chinese.hpp) |
| 指令分流 | 转换后的文本 → 去掉前缀及分隔符 → 紧急停车直出，或把普通口令正文完整交给 DeepSeek。 | [动作、参数和媒体规则](../src/shared/command.hpp) |
| DeepSeek | 普通口令正文 → 分类提示词 → 模型动作标签。默认直答由模板控制。 | [分类提示词](../src/shared/command.hpp) · [DeepSeek 接入](../src/platform/s100/deepseek.hpp) |
| 参数校验 | 模型动作标签 + 原始口令正文 → 校验单位、唯一数量及默认值 → 可输出动作或 `unknown`。 | [参数规则](../src/shared/command.hpp) |
| 试验性复合计划 | 同一语音段 → 模型引用原文并拆动作 → 程序校验各步骤和并行关系 → 一条 `voice_command_plan`。发送后继续收音。 | [计划解析](../src/shared/command_plan.hpp) · [启用方式及协议](../README.md#试验性复合指令) |
| 媒体播放 | `track_number` → 配置中的本地 MP3 路径 → mpv IPC 命令 → `playback.status`。 | [mpv 接入](../src/platform/linux/mpv_player.hpp) |
| 结果输出 | 动作或媒体意图 + 转录 + 来源 + 状态 + 计时 → 一行 JSON。 | [应用入口](../src/platform/linux/application.cpp) · [JSON 编码](../src/shared/command.hpp) |

### 几个结果对照

| 转录或输入 | 处理路径 | 结果摘要 |
| --- | --- | --- |
| 「闪电，往前挪两米」 | DeepSeek 判断动作，程序从原文校验距离 | `forward`，`parameters.distance_m=2` |
| 「闪电，右转」 | DeepSeek 判断动作，用户未指定角度 | `right`，`parameters.angle_deg=90` |
| 「闪电，往前走两步」 | 先交给 DeepSeek，参数校验发现单位不支持 | `unknown`，无运动参数 |
| 「闪电停车」 | Whisper 后的紧急直出规则 | `stop`，无运动参数 |
| 「闪电，往前走」 | DeepSeek 返回「前进」时 | `forward`，`parameters.distance_m=1` |
| 「闪电，放一下第二首歌」 | DeepSeek 判断播放，程序校验编号；MIC 按配置发 IPC | `voice_media_intent`，`command=play`，`parameters.track_number=2` |
| 「闪电，别再放音乐了」 | DeepSeek 区分停歌和停车 | `voice_media_intent`，`command=stop_playback` |
| MIC 听到「前进」 | 缺少前缀 | `unknown` |
| FILE 输入全静音 | Silero 未提交语音段 | 零条结果 JSON |

下面两行分别摘取运动和媒体结果的业务字段。第二行以 FILE 输入为例。

```jsonl
{"message_type":"voice_control_result","transcript":"闪电，往前挪两米","decision_path":"deepseek","command":"forward","parameters":{"distance_m":2},"recognized":true,"status":"completed"}
{"message_type":"voice_media_intent","transcript":"闪电，播放第二首","decision_path":"deepseek","command":"play","parameters":{"track_number":2},"recognized":true,"status":"completed","playback":{"status":"intent_only"}}
```

媒体结果的外层 `completed` 表示识别到了意图。`playback.status=accepted` 表示 mpv 接受了命令，`error` 表示播放器拒绝、连接失败或超时等异常。FILE 和未配置播放器的 MIC 返回 `intent_only`。

## 5. 需求变了，先看哪里

| 需求变化 | 先确认业务约定 | 主要影响的位置 |
| --- | --- | --- |
| 增加小车动作或单位 | 口令、命令码、单位、默认值和无效参数行为 | DeepSeek 标签、参数校验和 JSON 协议 |
| 更换 LLM | 哪些表达进入模型，允许返回哪些标签 | DeepSeek 接入、模型配置和输出校验。专用紧急停车走独立分支 |
| 停车再提速或加入流式唤醒 | 是音频阶段检测还是转录后判断，是否需要抢占普通队列 | MIC 采集和分流入口。当前停车仍等待分句和转录 |
| 换麦克风、扬声器或曲库 | 采集设备、mpv 输出设备和曲目编号 | 运行配置、ALSA 输入和 mpv IPC |
| 以后接 ROS 2 或其他应用 | 接收方采用的消息类型、动作码和参数语义 | stdout JSON 契约及外部适配层 |

排查一条命令时，先看 `transcript` 是否正确，再看 `decision_path` 和参数规则。`deepseek` 表示已走模型，`emergency_stop` 表示专用紧急直出，`not_classified` 表示入口没有进入分类。耗时字段缺失时仍可用该字段区分处理路径。判断语音整体快慢时，还要单独看句尾等待，因为 `since_audio_end_ms` 从分句就绪后才开始计时。

### 已实现与待联调

FILE 和 MIC 均已交叉编译，早期 FILE 在板端验证过分句和模型链路。运动参数、媒体意图和 mpv IPC 分别做过文本或模拟验证。真实 USB 麦克风、实际扬声器和曲库仍待设备联调，流式停车和可切换 LLM 尚未接入。具体版本与实测结果见 [验证记录](../README.md#5-验证记录与剩余工作)。
