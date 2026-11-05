# Whisper-Tiny

多语言语音识别。输入为 16 kHz 单声道音频，板端运行 Encoder 和 Decoder 两个 DLA。模型来源见[模型卡](model_card.md)。

当前交付为 W8A16: 主计算权重 INT8、激活和原生 I/O 为对称 INT16.2026-10-27 已完成修复后的编译和小样本实测,历史浮点模型指标不代表当前实现.

当前 Converter 不直接量化 LayerNorm.导出时保留官方参数和 epsilon,先对每行中心化并除以该行绝对值最大值(下限为 1),再计算平方、均值和 RSQRT;epsilon 同时除以缩放值的平方,保持浮点数学等价.这避免少数大幅值行占满整个 INT16 平方张量的量程,使其他行的小方差舍入为零.22 个 LayerNorm 的等价检查覆盖普通、小方差、常量和大幅值输入;两个 TFLite 均通过严格审计,仅有 INT8、INT16、INT32 张量,Encoder/Decoder 分别核实 26/43 个 W8A16 仿射层.注意力 BatchMatMul 的第二输入要求对称 INT16,因此启用 `use_symmetric_quantization`,不使用非对称 INT16 或 CPU 桥接.

### 归一化修复的小样本诊断

正式 `deploy/compile.sh` 已完成修复后的双 DLA 编译及上传,板端 `compile.sh` 已完成 5 条通路检查和示例生成.随后单独在 LibriSpeech test-clean 固定选择 50 条音频,排除全部 8 条校准音频,用相同预处理、滑动 Cache 和文本规范化对照 ONNX/NPU:

| 项目 | 实测 |
| --- | --- |
| ONNX 浮点 WER | 5.81% |
| W8A16 NPU WER | 5.52% |
| NPU WER 减 ONNX WER | -0.29 个百分点 |
| 正常遇到 EOT | 50/50 |

WER 越低越好.本次参考文本共 1032 个词,NPU 共 57 个编辑错误(替换 41、删除 6、插入 10).小样本略优于浮点不代表全量会改善.修复前同批音频的 NPU WER 为 182.27%;逐层定位发现 Encoder 第四个注意力块的归一化首次发生严重误差,SDK 量化图软件执行同样复现.改为逐行缩放后,同一条非校准音频的真实 NPU Encoder 输出余弦从约 -0.0609 提高到 0.99883,NRMSE 从约 170.30% 降到 4.84%.

修复后的诊断预测、脚本和 `comparison.json` 保存在日期临时目录的 `whisper_w8a16_fixed50/`,未将此子集结果混入历史全量报告.正式精度、性能和 RSS 见下文 G720 实测结果。

部署配置固定为 `cacheSize: 256`、`maxTokenLength: 448`.Decoder 每步输入 1 Token,使用 448 个绝对位置,Cache 满后左移并保留最近 256 个 Token.这是滑动窗口解码,不是保存 448 Token 完整历史的注意力;ONNX 参考、量化校准和板端使用同一实现.448 为提示词与生成文本的总 Token 上限,达到上限时记录停止原因,不宣称已遇到 EOT.

DLA 分别命名为 `tiny_encoder_w8a16.dla` 和 `tiny_decoder_w8a16_1t256c.dla`.转换后从实际 TFLite 提取 `runtime_contract.json`,记录 I/O 索引、逻辑形状、INT16 scale/zero point 和精度审计.板端按契约分配补齐存储,跨图特征与 KV Cache 先反量化再按目标尺度量化,不将 INT16 内存解释为 FP16.

校准默认使用 LibriSpeech test-clean 排序前 8 条音频,Decoder 每条最多 32 步真实浮点自回归状态,遇到 EOT 提前结束.校准与正式全量评价存在重叠,契约和新报告明确记录;本次 50 条诊断已排除全部校准音频.

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与上传后结束.随后登录开发板,在部署目录执行 `bash compile.sh`,做少量通路检查并保存 README 示例到 `examples/output/`.该阶段计算并打印子集任务精度,不发布正式报告,不统计性能和峰值 RSS.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.板端预测、日志及内部指标统一位于本模型部署目录的 build/work/,每次开始推理时覆盖上次中间结果.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

## 第一步: 编译并上传

模型、数据集和部署路径见 [config.yaml](config.yaml),工具链与 NAS 根目录按[仓库说明](../../../../README.md)配置.

在编译主机的本模型目录运行:

```bash
bash deploy/compile.sh
```

转换、量化、双 DLA 编译和上传在当前容器内完成;正式参考独立由 evaluate.sh 准备.

双 DLA 编译限定 `mdla5.3`,禁止桥接并使用原生 INT16 输入输出.不支持的算子或无法满足 W8A16 审计时直接失败,不退回浮点模型继续交付.旧 128/200 Cache 浮点 DLA 与新契约不兼容,需要重新执行本模型 compile.sh 转换并上传,不能仅替换板端 Python 脚本.

## 第二步: 独立评价

上传完成后,请登录开发板,在部署目录执行 `bash compile.sh` 完成通路检查和示例生成.正式评价前,按[板端规范](../../../../docs/genio_720.md)准备同协议参考及必要输入,在已部署目录执行:

```bash
bash evaluate.sh
```

精度、性能和峰值 RSS 统一保存到 results/summary.json,板端过程数据位于本模型部署目录的 build/work/,每次开始推理时覆盖,不重新量化或编译.

## 数据与精度评测

参考评价使用 `config.yaml` 指定的数据集,与板端保持相同样本、标注和预后处理协议.

数据目录要求: LibriSpeech test-clean: 保留官方目录下的 FLAC 音频和 *.trans.txt,2620 条音频.

精度变化以百分点表示,正数为改善,负数为下降.

Docker 需要 ONNX Runtime、NumPy、OpenCV、tqdm、ffmpeg 和 OpenAI Whisper。Whisper 仅提供音频处理、分词及文本规范化,模型推理由 ONNX Runtime 执行。
仓库 Docker 镜像包含 ffmpeg; 现有容器若缺少该依赖,需在容器中执行 `apt-get update && apt-get install -y --no-install-recommends ffmpeg`.脚本会在模型转换前检查该依赖.

## G720 实测结果

正式评价覆盖 LibriSpeech test-clean。主指标计分 2,620 条音频，运行编号 `20261103_090940_1944759`。[正式报告](results/summary.json)

| 指标 | 浮点参考 | G720 | 损失 |
| --- | ---: | ---: | ---: | ---: |
| WER | 7.558% | 11.200% | +3.642 个百分点 |

板端性能为 NPU 平均 311.436 ms，P95 556.279 ms，Python 进程峰值 RSS 为 87.29 MiB。
NPU 耗时为每条音频的 Encoder 与全部 Decoder 调用之和。校准音频与正式评价有重叠。

编译子集主指标计分 50 条音频，运行编号 `20261030_080700_1724563`。[子集报告](results/summary_test.json)

| 指标 | 浮点参考 | G720 | 损失 |
| --- | ---: | ---: | ---: | ---: |
| WER | 6.625% | 6.108% | -0.518 个百分点 |

损失为正表示退化。子集报告中的输出误差仅覆盖下表所列调用。

| 输出 | 调用次数 | CosSim | RMSE | NRMSE |
| --- | ---: | ---: | ---: | ---: |
| `output` | 5 | 0.998900 | 0.0610516 | 4.694% |

## 效果示例

提供两位说话人的短音频,输入选自 [LibriSpeech test-clean](https://www.openslr.org/12/),对应输出来自 2026-10-27 修复后的 W8A16 板端 compile.sh 少量样本检查.

输入已放在 `examples/input/`,编译时从配置的数据集自动生成来源与样本清单,随部署上传,不纳入 Git.

compile.sh 复用固定少量样本的板端预测,生成展示文件到部署目录的 `examples/output/`,不依赖全量评价.
输出目录保留一个 [transcripts.md](examples/output/transcripts.md),集中展示两条音频的实际板端识别结果。

```text
examples/
├── input/sample_1.wav
├── input/sample_2.wav
└── output/transcripts.md
```

### 示例 1

[播放或下载音频](examples/input/sample_1.wav)

参考文本: HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE

板端识别: He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fat and sauce.

### 示例 2

[播放或下载音频](examples/input/sample_2.wav)

参考文本: YOU WILL FIND ME CONTINUALLY SPEAKING OF FOUR MEN TITIAN HOLBEIN TURNER AND TINTORET IN ALMOST THE SAME TERMS

板端识别: You will find me continually speaking of foreman, Titian, Holbein, Turner, and Tintarat, and almost the same terms.

输出文件: [examples/output/transcripts.md](examples/output/transcripts.md),包含上述两条音频的参考文本与实际板端识别文本。

## 板端部署结构

第一步上传到 `BOARD_DEPLOY_DIR` 后的布局统一为:

```text
部署目录/
├── compile.sh
├── evaluate.sh          # 独立精度、性能与 RSS 评价.
├── config.json          # 不含连接地址的模型配置.
├── models/              # 模型与推理所需参数.
├── board/               # 板端程序、评测代码及必要依赖.
├── examples/            # 少量输入、来源清单和展示输出.
└── results/summary.json  # 唯一正式评价报告.
```

评价结果只保存在部署目录的 `results/summary.json`.板端预测、日志和中间指标统一保存在本模型部署目录的 `build/work/`,每次开始推理时覆盖;展示示例保存在 `examples/output/`.取回唯一报告后据实更新 README.
## 文件结构

- `deploy/compile.sh`: 编译部署与展示示例入口;精度、性能和 RSS 使用独立的 `deploy/evaluate.sh`.
- `deploy/host/`: 编译主机使用的导出、转换与辅助工具.
- `deploy/board/`: 板端程序源码、预处理和评测代码.
- `models/`: 原始权重、转换产物与来源说明; 必要的上游源码放在 `models/upstream/`.
- `examples/`: 少量固定输入与实际板端效果输出.
- `results/summary.json`: 上传后的最新测试汇总.

### 输出误差统计

compile.sh 独立对固定少量输入比较浮点与 NPU 输出,在终端显示 CosSim、RMSE、NRMSE 和实际覆盖范围.普通模型最多选择 5 次输入,不减少原有 compile 通路检查样本数;LLM 比较示例首个 Prefill 块的有效位置,BEVFusion 比较单帧的 NPU 子图.这些结果用于数值检查,不能代替任务精度.全量 evaluate.sh 只计算任务精度、性能和 RSS,不保存或比较全量浮点张量.

参考为普通 NPZ 数组文件,保存在编译机器本地日期临时目录并上传板端 build/reference/output_reference.npz,由下次 compile 覆盖,不使用数据库,不写入 NAS.板端中间结果仍使用本模型 build/work/.NRMSE 定义为 100 × RMSE / RMS(浮点参考),多输出分别统计.详见[输出误差统计](../../../../docs/output_metrics.md).

### compile 子集精度检查

compile.sh 使用 50 条 LibriSpeech 音频,复用正式评价协议.主机执行子集浮点推理,板端执行同一子集 NPU 推理,在终端打印精度和损失,并保存 results/summary_test.json.正的损失表示 NPU 变差,负值表示改善;Top-1、AP、AR 等使用浮点减 NPU 的百分点,WER 使用 NPU 减浮点的百分点,PPL 使用增量及相对增幅.

普通样本跨完整划分按固定间隔抽取,保留展示及数值检查的必要输入;LFW、ESC-50 和 MMLU 保持折、类别和学科覆盖.同一子集两端通过样本指纹核对;MMLU 固定 seed=17 分层抽样.该报告不代表全量成绩,不改写 results/summary.json.中间数据覆盖保存在本模型 build/work/,CosSim、RMSE、NRMSE 仍只比较固定少量输入.详见[输出误差统计](../../../../docs/output_metrics.md).

## 量化参数与编译命令

配置见 [config.yaml](config.yaml),转换与编译实现见 [convert_model.py](deploy/host/convert_model.py)、[compile_model.py](deploy/host/compile_model.py).

### 量化设置

| 参数 | 当前设置 |
| --- | --- |
| 转换器 | MTK `OnnxConverter`,分别转换 `tiny_encoder.onnx` 和 `tiny_decoder_step.onnx` |
| 精度 | `precision=w8a16`,INT8 权重、对称 INT16 激活及原生 I/O,逐输出通道权重量化 |
| 校准 | LibriSpeech test-clean 按路径排序前 8 条真实音频;Encoder 每条 1 次,Decoder 每条最多 32 步真实浮点自回归状态,遇 EOT 提前结束 |
| 静态配置 | `cacheSize=256`,`maxTokenLength=448`,Decoder 每次 1 Token,满 Cache 后使用滑动窗口 |
| 注意力掩码 | 导出图的 `MASK_NEGATIVE=-100.0` |
| 对称性 | `allow_asym16_quantization=False`,`use_symmetric_quantization=True`;满足注意力 BatchMatMul 的 INT16 要求 |
| 位宽 | `quantize=True`,`default_quantization_bitwidth=16`,`default_weights_quantization_bitwidth=8`,`input_quantization_bitwidths=16` |
| 其他属性 | `use_per_output_channel_quantization=True`,`append_output_dequantize_ops=False` |
| 审计与尺度 | 逐个仿射层核对 INT16/INT8/INT16,拒绝浮点回退;实际 I/O 顺序、形状和尺度写入 `build/runtime_contract.json` |

校准语音与 test-clean 全量评价存在重叠,报告必须保留此说明.8 条音频和最多 32 步的量化校准范围不同于任务精度子集.

### 特殊 ONNX 导出处理

[export_onnx.py](deploy/host/export_onnx.py) 关闭 SDPA,使用官方 MatMul/Softmax 分支;将 22 个 LayerNorm 改写为逐行缩放的中心化、方差和 RSQRT,epsilon 同步除以缩放值平方,保留原始权重和数学语义,避免直接 LayerNorm 融合及整数小方差失真.改写函数对普通、小方差、常量及大幅值输入执行 `atol=rtol=2e-5` 检查.

Encoder 固定 Mel 输入 `[1,80,3000]`,输出 `[1,1500,384]`.Decoder 拆成显式 KV Cache 的单步图,用 Token One-Hot、448 维位置权重和 Cache Mask 替代动态索引/更新,将 Cache 作为输入输出,固定长度 256.两图使用 opset 17,浮点参考、校准和板端共享该静态解码协议.

### 完整 NCC 编译命令

在仓库根目录的已配置编译容器中,分别编译已有的两个 TFLite:

```bash
source tools/storage.sh mtk_models/audio/stt/whisper_tiny
env LD_LIBRARY_PATH="${NCC_ROOT}/lib" \
    "${NCC_ROOT}/bin/ncc-tflite" \
    --arch=mdla5.3 --suppress-output --disallow-bridge --suppress-input \
    "${MODEL_ASSETS_DIR}/build/tiny_encoder_w8a16.tflite" \
    -o "${MODEL_ASSETS_DIR}/tiny_encoder_w8a16.dla"

env LD_LIBRARY_PATH="${NCC_ROOT}/lib" \
    "${NCC_ROOT}/bin/ncc-tflite" \
    --arch=mdla5.3 --suppress-output --disallow-bridge --suppress-input \
    "${MODEL_ASSETS_DIR}/build/tiny_decoder_w8a16_1t256c.tflite" \
    -o "${MODEL_ASSETS_DIR}/tiny_decoder_w8a16_1t256c.dla"
```
