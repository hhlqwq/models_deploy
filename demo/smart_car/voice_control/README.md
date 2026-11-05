# S100 语音控制 Demo

本文是整体计划与实际进度入口.更新时间: 2026-11-05.

**当前任务暂停,等待用户审阅整体框架.不再新增代码修改、构建、下载或测试.下面区分用户已经确认的目标和实现中采用、仍需审阅的取舍,不将功能通路跑通等同于整体验收通过.**

最新方案: **C++ + 板端现有 LLM SDK 1.0 + DeepSeek-R1-Distill-Qwen-1.5B + 已部署 Whisper tiny**.不继续原始 Qwen3 权重适配、W4 量化或 LLM 2.0 环境构建.参考 [clean_async_perception](../../clean_async_perception/README.md) 的构建和平台划分方式.

## 整体框架与职责

用户已确认的交付目标:

- 使用 C++ 和板端现有 DeepSeek-1.5B、旧 SDK、Whisper tiny.
- 编译时选择两种测试方式: USB 麦克风完整链路,或输入本地语音片段文件.
- 将 Whisper 转录填入“请根据 xxxx 指令分析”的提示词,由 LLM 输出前进、后退、左转、右转、停止、未识别中的一个结果.
- 将结果编码为便于 ROS2 消息承载的 JSON,当前只输出控制台.
- 板端程序、资源和运行数据统一到用户指定目录;真实根路径记录在本地 board.voice_control_root 配置.复用一个 Docker 环境.

```text
                       build.sh FILE / MIC
                                |
                       初始化配置与本地资源
                                |
                 +--------------+---------------+
                 |                              |
              FILE 模式                      MIC 模式
           本地语音片段文件                 外接 USB 麦克风
                 |                              |
          FFmpeg 解码/重采样                 ALSA 持续采集
                 |                              |
          16k 单声道 PCM16                 16k 单声道 PCM16
                 |                              |
                 |                         语音起止分句
                 |                              |
                 |                         有界语句队列
                 +--------------+---------------+
                                |
                     静音/输入长度检查
                                |
                      Whisper tiny C++
              Mel -> Encoder HBM -> Decoder HBM
                                |
                        实际语音转录 xxxx
                                |
                       转录文本有效性检查
                                |
                 构造分类提示词,包含实际 xxxx
                                |
                 DeepSeek-1.5B / 旧 SDK xlm
                                |
                   分离思考内容与最终回答
                                |
                      六类完整标签校验
                                |
                 command + 中文结果 + 转录
                                |
                       编码 JSON 并打印
```

| 模块 | 当前文件 | 职责 |
| --- | --- | --- |
| 构建入口 | build.sh | 读取本地工具路径,选择 FILE/MIC,调用已有官方 aarch64 编译器 |
| 应用编排 | src/main.cpp | 配置与生命周期、两种输入路径、模型顺序调用、结果打印 |
| Linux 音频 | src/platform/linux/audio.hpp | FILE 调用 FFmpeg 解码;MIC 通过 ALSA 采集;统一 PCM16 后归一化 |
| 分句 | src/shared/speech_segmenter.hpp | 当前为能量阈值分句,管理前置音频、静音结束与长句丢弃 |
| Whisper | src/platform/s100/whisper.hpp | 复用现成 tiny 资源,执行 Mel、Encoder、逐 Token Decoder 和滑动缓存 |
| HBM 接口 | src/platform/s100/hbm.hpp | 通过官方 DNN/UCP API 检查张量契约、管理内存和同步推理 |
| LLM 接口 | src/platform/s100/deepseek.hpp | 通过旧 SDK xlm 加载模型并接收生成回调 |
| 指令协议 | src/shared/command.hpp | 构造提示词、校验最终六类结果、编码 JSON;不以关键词代替 LLM 分类 |
| 配置 | config.example.json | 公开参数模板;板端 config.json 保存实际资源相对路径、语言和音频参数 |

FILE 按文件顺序处理,每个有效片段输出一条最终 JSON.MIC 的采集分句线程持续收音,模型处理线程从有界队列取完整语句,Whisper 与 DeepSeek 顺序执行;队列满时当前实现丢弃新语句并提示.程序保持模型加载,当前没有语音打断或在途 LLM 取消能力.

### 相对原设想的变化与待审阅取舍

| 项目 | 当前实现/状态 | 确认边界 |
| --- | --- | --- |
| 原始 Qwen3 适配 | 改为使用现成 DeepSeek 和旧 SDK | 用户已明确确认 |
| Whisper small | 改为已有 Whisper tiny | 用户已明确确认 |
| Python Demo | 改为 C++ | 用户已明确确认 |
| ROS2 发布 | 当前仅打印 ROS2 可承载的 JSON | 用户已明确要求先输出控制台 |
| Silero VAD | 当前替换为 RMS 能量分句 | 实现中的取舍,未单独获得用户确认;噪声下效果未验收 |
| 多轮对话 | 当前逐条重置 LLM 会话,独立分类连续指令 | 从原有多轮问答目标作出的调整,未单独确认是否还需保留上下文 |
| 参考工程构建方式 | 当前 build.sh 直接使用交叉编译器,未沿用 CMake | 因现有 Docker 缺少 CMake 作出的取舍,可在框架审阅时调整 |
| 分类质量 | 提示词变更曾导致其他类别退化 | 尚不能宣称整体可靠;不同版本结果不能混作一次验收 |

暂停后的待办顺序: 先审阅以上框架和取舍,确认后再确定最终实现,核对同一版本的文件测试结果,最后在 USB 麦克风可用时完成真实采集和噪声验证.

### 阶段提交与同步

按用户要求,每完成一个可核验的小阶段,先更新本文的实际进度、验证范围和缺失项,再以中文规范提交信息提交并推送代码,核对远端提交与本地一致.未完成的验收保持未完成,不等全部 Demo 完成才同步.

本次同步阶段为 C++ 双输入代码、已有板端资源接入、文件通路及整体框架文档.代码同步不代表麦克风实测通过,也不代表上表待审阅取舍已获得确认.当前开发仍暂停.

## 两种测试方式

通过编译入口参数 build.sh FILE 或 build.sh MIC 选择测试方式,不在运行时切换.

```text
FILE: 本地语音片段 -> FFmpeg 解码为 16k 单声道 -> Whisper tiny
MIC:  USB 麦克风 -> ALSA 16k 单声道 -> 能量分句 -> 有界队列 -> Whisper tiny
    -> 转录文本检查 -> DeepSeek 六类指令分析 -> JSON -> 控制台
```

结果仅允许: 前进、后退、左转、右转、停止、未识别.模型提示词包含实际转录和同义指令示例,不通过本地关键词匹配替代 LLM.先剥离 DeepSeek 已闭合的思考段,再对最终回答做完整标签匹配;解释性文字或其他回答记为未识别.

按用户最新要求,JSON 保留便于 ROS2 String 承载的格式,本轮仅输出到控制台,不链接 ROS2、不发布话题、不执行底盘动作.每条控制指令独立重置 SDK 会话,模型保持加载,避免前一条动作污染后续判断;当前范围是连续控制指令分析,不再是开放式问答.

FILE 使用板端已有 FFmpeg,支持其已启用的 WAV、FLAC、MP3 等音频格式,自动混单声道并重采样.Whisper tiny 单片段上限 30 秒,超限明确报错,不截断输出动作.MIC 持续监听,采集和推理分线程,队列满时丢弃新语句并提示,过长语句整句丢弃直至静音.停止信号只退出本应用,旧 SDK 无取消接口,在途生成完成后释放资源.

当前未找到 Silero/ONNX Runtime,首版用可配置 RMS 能量分句,不称为 Silero VAD;噪声下的可靠性待真实设备验证.不新增大模型下载,不升级系统或覆盖系统库.

## 当前进度

| 工作项 | 状态 | 证据及下一步 |
| --- | --- | --- |
| 资源选择与范围 | 已确认 | 使用已有 DeepSeek、旧 SDK 和 Whisper tiny;FILE/MIC 编译参数选择,控制台输出 JSON |
| 现成资源核查 | 已完成 | 板端存在约 1.8GB DeepSeek HBM、tokenizer、模板、旧 SDK 库和 Whisper tiny 双 HBM/滤波器/词表/解码规则 |
| 官方旧 SDK 例程 | 单轮实测通过 | 既有官方多轮例程成功加载和回答,不需要 LLM 2.0 系统升级 |
| C++ LLM 接入 | 连续三次推理通过 | 已按官方固定请求槽 0 调用;第一版提示词仅 1/3 分类正确,已调整提示词,新版分类待复测 |
| LLM 性能观察 | 仅示例范围 | 第一版三次端到端约 11-15 秒,不是定长 benchmark;回调未提供可用 Token 指标,不填 0 冒充实测.思考输出影响耗时 |
| C++ Whisper tiny | 已完成文件通路及单样本预处理核对 | 已执行英语与中文音频;同一 AISHELL 样本的 C++/原 Python Mel 比较 max_abs=0、RMSE=0、CosSim=1,仅代表该输入的预处理核对 |
| FILE/MIC 和 JSON | 双模式编译通过,FILE 已执行 | WAV/FLAC 文件到 JSON 已跑通;MIC 尚无真实声卡验证,不能视作完整通过 |
| Docker | 已复用现有唯一环境构建 | 正式 build.sh FILE/MIC 均已执行并生成 aarch64 程序,未安装另一套环境 |
| 板端统一部署目录 | 已整理并运行 | 应用、配置、模型、SDK 非系统依赖和音频已统一;原资源保留,大模型硬链接复用 |
| USB 麦克风 | 未检测到 | 最近 arecord -l 仍为 no soundcards found,真实麦克风验收未完成 |
| 固定语音到控制 JSON | 通路已验证,整体质量未验收 | 曾有六条英文合成片段 6/6 匹配;后续提示词修改出现类别退化,不复用旧版本成绩代表最终版本 |
| 新版六类分类及噪声检查 | 暂停,最终结果待核对 | 最近一轮同版本结果尚未在本文完成汇总;中文实录控制命令、真实 USB 与噪声验证仍未完成 |

## 构建与运行

在本地 configs/platforms/s100.local.yaml 的 toolchain.voice_control 填写 sdk_root、board_sysroot 和 cxx,分别指向旧 SDK 1.0 头文件/库、匹配板端头文件/库和已有官方 aarch64 编译器.真实路径不提交 Git.进入唯一 S100 Docker 后直接构建,不重新编译模型,不需要 CMake 或新增容器.显式覆盖编译器使用 VOICE_CXX,避免镜像自带 CXX=g++ 误选 x86 编译器.

```bash
bash build.sh FILE
bash build.sh MIC
```

模式默认 FILE;构建入口检查编译器目标为 aarch64,不自动安装依赖.默认输出到日期临时目录的 voice_control/build/,也可用 VOICE_BUILD_DIR 指定本次目录.输出分别为 voice_control_file 和 voice_control_mic.

板端统一目录由本地配置指定,下文用 VOICE_DEMO_ROOT 代指,不在公开文档写内部路径.交付布局:

```text
${VOICE_DEMO_ROOT}/
  voice_control_file
  voice_control_mic
  config.json
  lib/                          SDK 非系统依赖
  models/deepseek/               HBM、tokenizer 和官方聊天模板
  models/whisper_tiny/           双 HBM、Mel、词表和原始解码配置
  examples/                     测试语音
  results/                      必要实测结果
```

config.example.json 是公开参数模板,实际 config.json 位于板端统一目录.资源相对 config.json 解析,无需改动代码.编译选出的程序部署后分别命名为 voice_control_file 和 voice_control_mic.

```bash
cd "${VOICE_DEMO_ROOT}"
./voice_control_file config.json examples/input.wav
./voice_control_mic config.json
```

这些是待验证的交付命令,真实通过状态以进度表为准.USB 稳定设备名和分句阈值放 config.json.运行时不下载或安装资源.

## JSON 协议

下面为格式示例,不是实际模型结果:

```json
{
    "version": "1.0",
    "message_type": "voice_control_result",
    "sequence_id": 1,
    "timestamp_ms": 0,
    "source": {
        "device_id": "voice_control",
        "sensor_id": "audio",
        "pipeline_id": "whisper_tiny_deepseek_v1"
    },
    "transcript": "往前走",
    "command": "forward",
    "command_text": "前进",
    "recognized": true,
    "status": "completed",
    "since_audio_end_ms": 0
}
```

| 中文结果 | command |
| --- | --- |
| 前进 | forward |
| 后退 | backward |
| 左转 | left |
| 右转 | right |
| 停止 | stop |
| 未识别 | unknown |

模型或 ASR 错误为 status=inference_error、command=unknown,包含必要错误信息.未识别与停止分开,不把错误转换为动作.since_audio_end_ms 在 MIC 下从分句完成开始,包含队列等待;FILE 下从解码完成开始,不包含文件解码时间.模型调用耗时、首文本耗时与进程峰值 RSS 单独打印,不当作纯 BPU 时间.首文本可能是 DeepSeek 思考 Token,不代表控制指令已就绪.RSS 不包含全部 BPU/驱动内存.

## 验收计划

1. FILE/MIC 在同一 Docker 编译和链接,检查目标为 aarch64.
2. 核实统一部署目录和原资源均完好,程序只读取配套模型及系统 Runtime.
3. 使用已有固定音频验证 C++ Whisper,与原有实现比较转录,再验证完整文件到 JSON.
4. 小范围验证六类、同义指令、否定/冲突、无关语音、空/损坏/超长文件及多次请求.只记录真实样本和结果,失败不掩盖.
5. USB 接入后验证持续监听、短句、静音、噪声、繁忙、断开和退出.缺设备时保持未完成,不以音频文件替代麦克风验收.

原始 Qwen 适配、W4/W8 对比及其 MMLU/PPL 不再是本轮任务.不复用厂商 benchmark 或旧模型成绩作为 Demo 验收.本次无原始权重转换、系统升级、重启、其他程序终止或新增大模型下载.
