# Whisper-Tiny

面向 RDK S100 的多语言语音识别模型,使用已有 OpenAI Whisper `tiny.pt` 和配套 FP32 ONNX.输入为 16 kHz 单声道音频,板端通过官方 Python `hbm_runtime` 执行 Encoder 与单步 Decoder 两个 HBM.模型来源见 [模型卡](model_card.md),资源位置见 [模型资源](models/README.md).

| 项目 | 当前配置 |
| --- | --- |
| 模型 | OpenAI Whisper-Tiny,多语言权重,约 39M 参数 |
| 目标平台 | RDK S100 / nash-e |
| 量化 | W8A16 配方,常量仿射层使用 INT8 权重、INT16 激活 |
| 公开 I/O | FP32,由 HBM 处理与内部量化张量之间的转换 |
| Encoder 输入 / 输出 | Log-Mel `[1,80,3000]` / audio_features `[1,1500,384]` |
| Decoder | 每步 1 Token,KV Cache 为 256,提示与生成合计上限 448 Token |
| 解码 | 英语贪心转写,使用官方禁止 Token 和英文文本规范化规则 |

## 执行范围

`deploy/compile.sh` 在编译服务器完成量化、编译、固定子集浮点评估及上传.登录板端后执行 `compile.sh`,对相同的 50 条音频计算 WER 和精度损失,对其中三条比较 Encoder 输出的 CosSim、RMSE、NRMSE,保存 `results/summary_test.json` 及展示示例.

`deploy/evaluate.sh` 独立使用已有模型.编译服务器生成 LibriSpeech test-clean 全部 2620 条音频的浮点参考并上传,板端执行 `evaluate.sh` 计算同范围 WER、性能和进程峰值 RSS,保存 `results/summary.json`.

## 第一步 编译并上传

模型、数据集和部署位置由 [config.yaml](config.yaml) 及本地 S100 平台配置管理.在已配置的 S100 编译容器内,进入本模型目录执行:

```bash
bash deploy/compile.sh
```

入口复用已有的 `tiny_encoder.onnx`、`tiny_decoder_step.onnx` 和配套预处理、解码与文本资源.分别量化编译两个 HBM,核对实际位宽及执行位置,运行固定 50 条子集的浮点推理,上传模型、同子集精度参考、三条数值参考及必要音频.

上传完成后,登录 S100,在本模型部署目录执行:

```bash
bash compile.sh
```

板端保存子集结果到 `results/summary_test.json`,生成 `examples/output/transcripts.json`.对应的三条原始音频位于 `examples/input/`.

## 第二步 独立评价

在编译容器的本模型目录执行:

```bash
bash deploy/evaluate.sh
```

该入口对已有 FP32 ONNX 计算完整 test-clean 的浮点参考,上传参考与评价输入.随后登录 S100,在本模型部署目录执行:

```bash
bash evaluate.sh
```

板端使用已有 HBM 完成同范围评价并生成 `results/summary.json`.性能记录包括 Encoder 与 Decoder 的同步 HBM 调用平均耗时、端到端 RTF 和推理进程峰值 RSS.同步 HBM 调用包含 Runtime 绑定等开销;RSS 表示进程内存,纯 BPU 设备耗时及 BPU 专用内存需另行测量.

## 数据与精度评测

数据集为 [LibriSpeech test-clean](https://www.openslr.org/12/),保留官方 FLAC 音频及 `*.trans.txt`,共 2620 条.校准使用按路径排序的前 8 条音频;compile 子集排除这 8 条后固定间隔抽取 50 条,并保留可用的仓库语音示例.完整 test-clean 中包含校准音频,全量结果需按这一范围解释.

两端使用相同音频、16 kHz 单声道 PCM16、30 秒截补齐、80 维 Log-Mel、滑动 Cache 和贪心解码规则.文本使用 OpenAI Whisper `EnglishTextNormalizer` 规范化,WER 为整个评价范围的词级编辑距离总和除以参考词数总和.

### 子集精度结果

| 指标 | 浮点 ONNX | S100 W8A16 |
| --- | ---: | ---: |
| 音频数 | 50 | 50 |
| 参考词数 | 1128 | 1128 |
| 词级编辑错误数 | 83 | 83 |
| WER | 7.3582% | 7.3582% |
| WER 损失 | — | +0.0000 个百分点 |
| 正常遇到 EOT | — | 50/50 |

WER 越低越好,精度损失为 `100 × (BPU WER - ONNX WER)`,正值表示变差.该子集的 WER 相同,当前采用 W8A16.完整 2620 条精度、正式性能和 RSS 尚待评价.实际子集结果见 [summary_test.json](results/summary_test.json).

### 输出误差统计

compile 对三条固定音频比较 Encoder 原始 `audio_features` 输出,包含两端预处理与模型执行差异.NRMSE 为 `100 × RMSE / RMS(浮点参考)`.

| 音频标识 | CosSim | RMSE | NRMSE |
| --- | ---: | ---: | ---: |
| 1089-134686-0008 | 0.996148 | 0.112743 | 8.8131% |
| 1089-134691-0023 | 0.997257 | 0.096478 | 7.4406% |
| 1188-133604-0000 | 0.996579 | 0.108644 | 8.3119% |

少量数值参考采用 NPZ,保存在主机本地过程目录,上传至板端 `build/reference/encoder_reference.npz`.数值比较仅在 compile 执行,任务精度与数值误差共同写入 `summary_test.json`.

## 效果示例

以下音频选自相同的 50 条子集,识别文本来自真实 S100 W8A16 推理.原始音频可播放或下载,完整的参考、浮点和板端转写保存在 [transcripts.json](examples/output/transcripts.json).

```text
examples/
├── input/sample_1.flac
├── input/sample_2.flac
├── input/sample_3.flac
└── output/transcripts.json
```

### 示例 1

[播放或下载音频](examples/input/sample_1.flac)

音频标识为 `1089-134686-0008`.

参考文本为:

> THE CHAOS IN WHICH HIS ARDOUR EXTINGUISHED ITSELF WAS A COLD INDIFFERENT KNOWLEDGE OF HIMSELF

板端识别为:

> The chaos in which his ardor extinguished itself was a cold in different knowledge of himself.

### 示例 2

[播放或下载音频](examples/input/sample_2.flac)

音频标识为 `1089-134691-0023`.

参考文本为:

> IT WAS A PAIN TO SEE THEM AND A SWORD LIKE PAIN TO SEE THE SIGNS OF ADOLESCENCE THAT MADE REPELLENT THEIR PITIABLE NAKEDNESS

板端识别为:

> It was a pain to see them and a sword-like pain to see the signs of adolescence that made repellent their pitiful nakedness.

### 示例 3

[播放或下载音频](examples/input/sample_3.flac)

音频标识为 `1188-133604-0000`.

参考文本为:

> YOU WILL FIND ME CONTINUALLY SPEAKING OF FOUR MEN TITIAN HOLBEIN TURNER AND TINTORET IN ALMOST THE SAME TERMS

板端识别为:

> You will find me continually speaking of foreman, Titian, Holbein, Turner, and Tintarat, and almost the same terms.

## 板端部署结构

部署目录由本地 S100 配置指定,布局为:

```text
部署目录/
├── compile.sh
├── evaluate.sh
├── config.json
├── tiny_encoder_w8a16.hbm
├── tiny_decoder_step_w8a16.hbm
├── board/
│   ├── evaluate.py
│   ├── decoder.py
│   └── shared.py
├── build/reference/             # 精度参考、数值参考及运行所需文本资源.
├── examples/
│   ├── input/                   # 三条展示音频.
│   └── output/transcripts.json
└── results/
    ├── summary_test.json        # compile 子集报告.
    └── summary.json             # evaluate 全量报告.
```

源代码中的 `deploy/host/` 负责校准、编译、浮点参考及上传,`deploy/board/` 保存推理、音频前后处理和解码模块.主机从本模型的 `deploy/board/` 复用相同预处理与解码协议.资源与依赖来源见 [source_url.txt](models/source_url.txt).

## 量化参数与编译命令

参数见 [config.yaml](config.yaml),实现见 [compile.py](deploy/host/compile.py),板端推理见 [evaluate.py](deploy/board/evaluate.py).

### 量化设置

| 参数 | 当前设置 |
| --- | --- |
| 工具链 | S100/S600 OpenExplorer 3.7.0,`hb_compile` |
| 编译目标 | `nash-e` |
| 精度配方 | W8A16 PTQ,`calibration_type: max` |
| 主计算激活 | `quant_config.model_config.all_node_type: int16` |
| 常量权重 | 按原图 Conv/MatMul/Gemm 节点设置 `quant_config.node_config.<节点名>.input1: int8` |
| 校准音频 | test-clean 按路径排序前 8 条,Encoder 每条 1 次 |
| Decoder 校准 | 每条最多 32 步真实浮点自回归状态,遇 EOT 结束;当前产物使用 187 组状态 |
| 输入校准 | 每路 `float32`,`preprocess_on: false` |
| 公开输入 | `featuremap`,`no_preprocess`,逐输入按原图填写静态形状 |
| 静态解码 | 每步 1 Token,Cache 为 256,总长度上限 448 |
| 编译参数 | `compile_mode: latency`,`core_num: 1`,`optimize_level: O2` |

实际 PTQ 图确认 Encoder 26 层、Decoder 43 层常量权重仿射计算为 INT16 激活 / INT8 权重.动态注意力矩阵乘两路输入为 INT16 / INT8,Encoder 8 个、Decoder 16 个.公开 I/O 为 FP32.编译器 advice 分别记录 362、637 个 BPU 操作,这些数量包含降低和融合操作,不同于原图层数.

### 特殊 ONNX 导出处理

复用 [MTK Whisper 导出实现](../../../../mtk_models/audio/stt/whisper_tiny/deploy/host/export_onnx.py) 生成的 FP32 ONNX.原始权重、ONNX、Mel 滤波器及解码配置均来自仓库已有资源.

LayerNorm 保留原始参数和 epsilon,按行中心化并除以该行最大绝对值,缩放下限为 1;epsilon 同步除以缩放值平方.该改写保持浮点数学等价,避免量化时小方差在较大的平方张量量程中失真.

Encoder 固定输入 `[1,80,3000]`,输出 `[1,1500,384]`.Decoder 将 KV Cache 显式作为输入输出,使用 Token One-Hot、448 维位置权重和 Cache Mask 替代动态索引与更新.Cache 满后左移,保留最近 256 个 Token.浮点参考、校准和板端共用该静态解码协议.

### 完整编译命令

`deploy/host/compile.py` 按真实 ONNX 的输入名称、静态形状和常量权重节点生成两份完整 `compile.yaml`,其中包含每路校准目录及全部权重节点的位宽配置.在执行 compile 入口准备好配置与校准输入后,实际编译命令为:

```bash
WORK="/tmp/hailongcodex/$(date +%F)/s100/whisper_tiny"
hb_compile --config "$WORK/w8a16/encoder/compile.yaml"
hb_compile --config "$WORK/w8a16/decoder_step/compile.yaml"
```

编译产物分别为 `tiny_encoder_w8a16.hbm` 和 `tiny_decoder_step_w8a16.hbm`,实际位宽和执行位置核对后发布到 NAS 模型资源目录.原始权重放 `original/`,配套资源与审计记录放 `build/`,具体布局见 [模型资源](models/README.md).
