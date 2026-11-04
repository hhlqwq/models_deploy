# Whisper-Tiny · RDK S100

使用仓库已有 OpenAI Whisper 多语言 `tiny.pt` 和配套 FP32 ONNX,通过 S100 官方 `hb_compile` 编译,板端通过 Python `hbm_runtime` 识别语音.本次采用 W8A16 配方,完成 50 条非校准音频的真实板端测试.

## 本次验证

2026-11-04,LibriSpeech test-clean 排除全部 8 条校准音频后固定间隔选择 50 条,保留可用的仓库语音示例.两端使用同一组原始音频、解码规则和 OpenAI 英文文本规范化,按整个子集的词级编辑距离计算 WER.

| 项目 | 浮点 ONNX | S100 W8A16 |
| --- | ---: | ---: |
| 音频数 | 50 | 50 |
| 参考词数 | 1128 | 1128 |
| 词级编辑错误数 | 83 | 83 |
| WER | 7.3582% | 7.3582% |
| WER 损失 | — | +0.0000 个百分点 |
| 正常遇到 EOT | — | 50/50 |

WER 越低越好,损失为 `100 × (BPU WER - ONNX WER)`.相同 WER 不代表每条转写文本相同.该子集未出现整体精度退化,本次保留 W8A16.当前结果覆盖 50 条子集;全量 2620 条、性能和峰值 RSS 尚未执行.

Encoder 原始 `audio_features` 输出的三条数值对照如下,包含两端音频预处理及模型执行差异.NRMSE 使用 `100 × RMSE / RMS(reference)`.

| 音频标识 | CosSim | RMSE | NRMSE |
| --- | ---: | ---: | ---: |
| 1089-134686-0008 | 0.996148 | 0.112743 | 8.8131% |
| 1089-134691-0023 | 0.997257 | 0.096478 | 7.4406% |
| 1188-133604-0000 | 0.996579 | 0.108644 | 8.3119% |

实测结果见 [summary_test.json](results/summary_test.json),三条转写见 [transcripts.json](examples/output/transcripts.json).

## 输入、导出与解码

使用已有多语言 Tiny 权重,约 39M 参数,本次固定英语转写.输入为 16 kHz 单声道 PCM16,截取或补齐至 30 秒,按 Whisper 规则计算 80 维 Log-Mel,形状为 `[1,80,3000]`,公开模型输入输出为 FP32.

既有 ONNX 与 MTK 资源中的对应文件逐文件一致.复用了已验证的特殊导出处理:

- LayerNorm 按行中心化并以最大绝对值缩放,epsilon 同步缩放,保持浮点数学等价,避免量化时平方张量的动态范围过大.
- Token 和绝对位置使用 one-hot 输入,图内矩阵乘完成 embedding 与位置选择.
- Decoder 单步处理 1 Token,KV Cache 固定 256,满后左移并保留最近 256 个 Token.提示和生成合计上限 448 Token,主机参考与板端使用同一贪心解码及禁止 Token 规则.

当前复用 Encoder 与单步 Decoder 两个 ONNX 并生成两个 HBM.地瓜官方 Whisper-medium 示例使用编码器及包含 prefill/decode 的解码器 HBM,通过 OELLM 执行;当前 Tiny 使用已具备的静态单步图和官方 Python Runtime.模型版本、接口及量化配置见 [模型卡](model_card.md).

## 量化参数与完整编译命令

| 参数 | 当前值 |
| --- | --- |
| 工具链 | S100/S600 OpenExplorer 3.7.0,`hb_compile` |
| 编译目标 | `nash-e` |
| 校准方法 | PTQ,`calibration_type: max` |
| 主计算激活 | `quant_config.model_config.all_node_type: int16` |
| 常量 Conv/MatMul/Gemm 权重 | 按原图节点设置 `quant_config.node_config.<节点名>.input1: int8` |
| 校准数据 | test-clean 排序前 8 条真实音频 |
| Decoder 校准 | 每条最多 32 步真实浮点自回归状态,遇 EOT 结束;本次共 187 组状态 |
| 输入校准类型 | 各输入 `float32`,`preprocess_on: false` |
| 公开输入设置 | `featuremap`,`no_preprocess`,按原图逐输入填写静态形状 |
| 编译参数 | `compile_mode: latency`,`core_num: 1`,`optimize_level: O2` |

实际 PTQ 图审计确认 Encoder 26 层、Decoder 43 层常量权重仿射计算为 INT16 激活 / INT8 权重.动态注意力矩阵乘为 INT16 / INT8 两路输入,Encoder 8 个、Decoder 16 个.公开 I/O 为 FP32.编译器 advice 分别记录 362、637 个 BPU 操作,无 CPU backend 操作;这些数量包含降低和融合操作,不等于原图层数.

主机 `deploy/host/compile.py` 根据真实 ONNX 的输入、形状和节点名称生成两份完整 `compile.yaml`,包含各输入的真实校准目录以及全部权重节点配置.实际调用命令为:

```bash
WORK="/tmp/hailongcodex/$(date +%F)/s100/whisper_tiny"
hb_compile --config "$WORK/w8a16/encoder/compile.yaml"
hb_compile --config "$WORK/w8a16/decoder_step/compile.yaml"
```

配置准备、真实输入校准、编译、实际位宽审计、同子集浮点评估及上传统一由以下入口完成:

```bash
bash rdk_models/audio/stt/whisper_tiny/deploy/compile.sh
```

命令在已配置的 S100 编译容器、仓库根目录执行.已有模型根目录需具备 [资源说明](models/README.md) 中的 ONNX 和配套预处理/文本资源.入口使用已有文件,每个模型的 NAS 根目录和板端位置来自本地 S100 平台配置.

## 板端子集与独立全量评价

主机 compile 上传模型、50 条音频、配套浮点精度与三条 Encoder 数值参考后,登录板端,在本模型部署目录执行:

```bash
bash compile.sh
```

生成 `results/summary_test.json` 和 `examples/output/transcripts.json`.板端数据集为本地配置指定的公共数据集目录,模型 HBM 放本模型根目录,Python 程序放 `board/`,共享解码/音频模块放根目录,必要参考及文本资源放 `build/reference/`.

正式评价在编译容器使用现有 ONNX 执行:

```bash
bash rdk_models/audio/stt/whisper_tiny/deploy/evaluate.sh
```

完成 2620 条浮点参考及输入上传后,在板端部署目录执行:

```bash
bash evaluate.sh
```

板端计算同范围 WER,记录同步 HBM 调用的 Encoder/Decoder 平均耗时、端到端 RTF 和推理进程峰值 RSS,发布 `results/summary.json`.HBM 同步调用包含 Runtime 绑定等开销,不能当作纯 BPU 设备耗时.RSS 是进程内存,不代表 BPU 专用内存.全量阶段使用已有产物,具体入口见 [evaluate.sh](deploy/evaluate.sh).

子集音频与校准互斥;完整 test-clean 中包含这些校准音频,全量报告需按这一范围解释.英文规范化复用 OpenAI Whisper 原版规则及固定版本的 more-itertools 源码;必要文本资源随包提供,不要求进入环境后安装依赖.

## 来源与参考

- 模型源码和 MIT 许可见 [OpenAI Whisper](https://github.com/openai/whisper/tree/v20250625).
- 音频与转录见 [LibriSpeech / OpenSLR 12](https://www.openslr.org/12/).
- 地瓜运行组织参考见 [官方 Whisper-medium 示例](https://developer.d-robotics.cc/case_doc/basic/speech_to_text).
- 具体权重和依赖来源记录在 [source_url.txt](models/source_url.txt).
