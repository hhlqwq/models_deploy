# ResNet101

采用 TorchVision `v0.15.1` 的 `ResNet101_Weights.IMAGENET1K_V2`,用于 ImageNet-1K 的 1000 类图像分类.当前已实现离线导出、INT8 PTQ、G720 编译上传和 Python 板端流程;编译与硬件验证状态见下文.正式来源为 [pytorch/vision](https://github.com/pytorch/vision/tree/v0.15.1),Qualcomm 页面仅作交付形式参考.

## 权重与数据

请自行下载官方权重 [resnet101-cd907fc2.pth](https://download.pytorch.org/models/resnet101-cd907fc2.pth),约 170.5 MiB,放入本地 `models_root/perception/image_classification/resnet101/`.脚本使用 `weights=None` 创建网络,只加载该本地文件,不会自动下载权重或安装依赖.完整来源见 [models/source_url.txt](models/source_url.txt).

模型参数见 [config.yaml](config.yaml).`models_root`、`dataset_root` 和平台连接信息由仓库本地存储及平台配置提供,公开文件不保存实际地址.默认数据布局为:

```text
dataset_root/ImageNet/
├── ILSVRC2012_img_val/          也可使用 val/,校准路径需在模型配置中对应调整.
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ...
│   └── ILSVRC2012_val_00050000.JPEG
└── val_labels_0based.txt
```

标签按上述图片编号排列,每行一个 TorchVision 零基类别 ID,共 50000 行;也接受从 1 开始的连续行号加类别 ID 两列.必须由 ImageNet devkit 的类别信息映射到 TorchVision 类别顺序,不能直接将 devkit 原始标签减 1.数据来自 [ILSVRC2012 官方入口](https://image-net.org/challenges/LSVRC/2012/),遵守其访问和使用条件.

默认校准按文件名排序跳过 1000 张,取之后的 100 张.这是复用验证集的 PTQ 校准,与正式 50000 张验证集有 100 张重叠,不是独立校准集.报告保留这项重叠说明,分类精度只评价完整验证集 Top-1.若改用独立校准集,须同步调整配置,重新编译后再评价.

## 张量与预处理

| 项目 | 契约 |
| --- | --- |
| 官方模型 | TorchVision ResNet101 V1.5 架构,`IMAGENET1K_V2` 权重,44,549,160 参数 |
| 图像预处理 | PIL RGB,短边缩放至 232,双线性插值,按 TorchVision 规则中心裁剪 224 |
| ONNX 输入 | `image`,FP32 NCHW `[1,3,224,224]`,范围 `[0,1]` |
| 图内归一化 | mean=`[0.485,0.456,0.406]`,std=`[0.229,0.224,0.225]` |
| ONNX 输出 | `logits`,FP32 `[1,1000]`,不包含 Softmax |
| 量化 | W8A8 INT8 PTQ,逐输出通道权重量化;偏置等张量不宣称为 INT8 |
| 原生 I/O | 从实际 TFLite 提取逐张量 INT8 scale/zero point;不符合 dtype 或形状时停止 |
| DLA 布局 | 要求连续存储或最后一维补齐到 16 字节,运行时核对 SDK 实际缓冲区大小 |
| 类别排序 | Top-5 按分数降序;相同分数按类别 ID 升序,不展示伪造概率 |

校准、ONNX 参考和板端共用 `deploy/board/resnet101.py` 的预处理.归一化已在图内,外部不可重复执行.转换与板端报告分别记录实际 Pillow 版本,预处理兼容性通过同输入对比检查.模型摘要仅保留必要版本绑定,不在终端打印成串校验值.

## 编译并上传

在已有 Ubuntu 编译容器的仓库根目录执行:

```bash
bash mtk_models/perception/image_classification/resnet101/deploy/compile.sh
```

入口顺序导出 FP32 ONNX、用 100 张图片校准为 INT8 TFLite、通过公共 NCC 配方编译 MDLA 5.3 DLA,最后上传部署包.使用 `--suppress-output --disallow-bridge`,禁止 CPU bridge.原始权重、ONNX、DLA 和必要参数保存在模型资源目录,转换中间文件使用该目录的 `build/`.编译过程不运行全量 ONNX 参考、精度评分、测速或 RSS 统计.

主机上传后结束,只提示板端路径,不通过 SSH 自动启动板端推理.请登录开发板,进入终端提示的部署目录执行:

```bash
bash compile.sh
```

板端 compile 读取 ImageNet 固定 1000 张子集评分,另从部署包三张示例生成展示;常驻加载 DLA 后通过 Python 直接调用 MTK SDK 动态库执行 NPU 推理,生成 `examples/output/sample_1_classification.jpg` 至 `sample_3_classification.jpg`.展示图片来自本次板端预测,不会复用 ViT 的结果.子集精度、损失及数值误差写入 `results/summary_test.json`.

主机依赖现有容器中的 PyTorch 2.0.0、TorchVision 0.15.1、ONNX、MTK Converter、NCC、NumPy、Pillow 和 tqdm.板端需要 Python 3.10+、NumPy、Pillow 9.1+、用于结果绘图的 OpenCV,以及匹配板端架构与系统的 Neuron Runtime SDK 8 动态库.代码不部署或覆盖系统动态库.

## 独立评价

在已有编译容器执行:

```bash
bash mtk_models/perception/image_classification/resnet101/deploy/evaluate.sh
```

只对已导出的 `resnet101.onnx` 计算全量 ImageNet Top-1,参考保存为本地日期过程目录的 `onnx_reference.json`.入口随后校验已有板端模型版本,自动将参考上传到 `build/reference/full.json`,不上传或替换模型,也不自动启动板端任务.确认板端已有完整验证集和标签后,在部署目录执行:

```bash
bash evaluate.sh
```

板端执行 5 次预热和 50000 张验证图推理,严格核对图片顺序、数量、标签和模型版本,完整后将 Top-1、相对同协议 FP32 的百分点变化、性能和 RSS 发布到唯一 `results/summary.json`.缺参考时保留绝对精度,差值为空并标记 `partial`;存在不匹配参考时直接报错.评价不触发模型导出、量化、编译或重新部署.

全量 evaluate.sh 在主机评分完成后自动上传同协议参考,无需手动上传命令.板端过程数据统一位于 build/work/,子集及全量报告分别为 summary_test.json、summary.json.

摘要沿用已有 ViT 分类模型的字段: `metric`、`board_accuracy`、`reference_accuracy`、`accuracy_change_percentage_points`、`npu_mean_ms`、`peak_rss_mib` 和 `model_size`.分类指标只有 Top-1,精度使用 0-1 比例,文件大小使用十进制 MB.完整预处理参数留在模型契约和本页说明中,不在摘要里重复 mean/std、原生缓冲区大小或辅助子集精度;运行时契约和已有预测格式保持不变.

- 纯 NPU 时间: SDK 同步 `inference` 调用,不含 I/O 绑定与前后处理,报告均值和 P95.
- 端到端时间: 图片读取、预处理、量化、I/O 绑定、NPU 和 Top-5,不含模型加载、预热、预测写盘及评分.
- 峰值 RSS: Python 推理进程全生命周期的高水位,不是设备全部驱动或 NPU 内存.
- 板端过程数据: 本模型部署目录的 `build/work/`,每次开始推理时覆盖,只保存预测和评分所需的推理元数据.

单独运行 `board/infer.py` 默认只检查三张示例;`board/evaluate.py` 默认全量,只评分当前 `build/work/` 中同模型的全量推理结果,缺失时停止,不会补跑推理.两个主要入口无需参数.

## 当前状态

2026-10-23 已完成官方权重转换、编译、部署、三图通路检查及 ImageNet val2012 全量 50000 张的 FP32 ONNX 与 Python NPU 评价.正式报告见 [results/summary.json](results/summary.json),运行编号为 `20261023_041137_1275282`.后续补齐参考与精简报告只读取已有预测,保留原运行编号和全部主要实测值,未重新执行模型.

| 全量指标 | 本次结果 |
| --- | ---: |
| ONNX FP32 Top-1 | **81.898%** |
| NPU INT8 Top-1 | **81.544%** |
| NPU 相对 ONNX 变化 | **下降 0.354 个百分点** |
| 纯 NPU 平均耗时 | **5.157 ms** |
| 纯 NPU P95 | **5.308 ms** |
| 端到端平均耗时 | **22.666 ms** |
| Python 推理进程峰值 RSS | **243.74 MiB** |

上述精度采用同一 ONNX、标签映射、全量样本及预处理协议.耗时来自 5 次预热后的 50000 次 batch=1 实际调用;RSS 为推理进程高水位,不等同于设备全部驱动或 NPU 内存.默认校准与这次验证集重叠 100 张,没有将辅助子集指标混入完整验证集结果.

| 产物 / 检查 | 本次实际结果 |
| --- | --- |
| 原始权重 | 178,814,045 bytes |
| FP32 ONNX | 178,039,369 bytes |
| INT8 TFLite | 46,865,112 bytes |
| G720 DLA | 45,377,343 bytes,约 45.38 MB |
| 官方变换对比 | 三张图片的归一化输入与 TorchVision 官方 V2 变换完全一致 |
| FP32 转换对比 | 三张图片的 ONNX 与 PyTorch logits 最大绝对差小于 `6e-6`,Top-1 一致 |
| 主机 / 板端预处理 | Pillow 11.3.0 / 10.3.0,三张输入张量逐字节一致 |
| 板端通路 | Python 3.12.13,实际 SDK NPU 推理 3/3 完成,原生 INT8 I/O 契约检查通过 |

## 实际分类示例

以下图片来自编译阶段的板端真实预测,输入为 ImageNet 验证集前三张图片,保留当时的 Top-5 类别排序展示.正式分类评价只报告 Top-1,以本页全量结果为准.第三张图片的 NPU Top-1 与 FP32 Top-1 不同,这项样例变化没有被删改.

![海蛇的板端 Top-5](examples/output/sample_1_classification.jpg)

![雪山场景的板端 Top-5](examples/output/sample_2_classification.jpg)

![牧羊犬的板端 Top-5](examples/output/sample_3_classification.jpg)

[TorchVision 官方文档](https://docs.pytorch.org/vision/0.15/models/generated/torchvision.models.resnet101.html)报告该 V2 权重 ImageNet-1K Top-1 为 81.886%;它仅作选型参考,本页精度变化使用本仓库实际测得的 ONNX 81.898% 计算.G5100 尚未适配或验证.

### 输出误差统计

compile.sh 独立对固定少量输入比较浮点与 NPU 输出,在终端显示 CosSim、RMSE、NRMSE 和实际覆盖范围.普通模型最多选择 5 次输入,不减少原有 compile 通路检查样本数;LLM 比较示例首个 Prefill 块的有效位置,BEVFusion 比较单帧的 NPU 子图.这些结果用于数值检查,不能代替任务精度.全量 evaluate.sh 只计算任务精度、性能和 RSS,不保存或比较全量浮点张量.

参考为普通 NPZ 数组文件,保存在编译机器本地日期临时目录并上传板端 build/reference/output_reference.npz,由下次 compile 覆盖,不使用数据库,不写入 NAS.板端中间结果仍使用本模型 build/work/.NRMSE 定义为 100 × RMSE / RMS(浮点参考),多输出分别统计.详见[输出误差统计](../../../../docs/output_metrics.md).

### compile 子集精度检查

compile.sh 使用 1000 张 ImageNet 验证图片,复用正式评价协议.主机执行子集浮点推理,板端执行同一子集 NPU 推理,在终端打印精度和损失,并保存 results/summary_test.json.正的损失表示 NPU 变差,负值表示改善;Top-1、AP、AR 等使用浮点减 NPU 的百分点,WER 使用 NPU 减浮点的百分点,PPL 使用增量及相对增幅.

普通样本跨完整划分按固定间隔抽取,保留展示及数值检查的必要输入;LFW、ESC-50 和 MMLU 保持折、类别和学科覆盖.同一子集两端通过样本指纹核对;MMLU 固定 seed=17 分层抽样.该报告不代表全量成绩,不改写 results/summary.json.中间数据覆盖保存在本模型 build/work/,CosSim、RMSE、NRMSE 仍只比较固定少量输入.详见[输出误差统计](../../../../docs/output_metrics.md).

## 量化参数与编译命令

以下为当前 [config.yaml](config.yaml)、[量化实现](deploy/host/convert_int8.py) 和 [编译实现](deploy/host/compile_model.py) 中的配置.未显式覆盖的 Converter 属性沿用当前 SDK 默认值,实际 scale/zero point 由校准产物读取,不能写死为通用常量.

### 量化设置

| 参数 | 当前设置 |
| --- | --- |
| 转换入口 | `deploy/host/convert_int8.py`,MTK `OnnxConverter` |
| 精度 | W8A8 PTQ,逐输出通道权重量化 |
| 校准 | ImageNet 验证图片按文件名排序,`calibration_offset=1000`,取 `calibration_samples=100` 张 |
| 输入 | FP32 `[1,3,224,224]`,RGB / 255;mean/std 在 ONNX 图内,外部不重复归一化 |
| 显式设置 | `quantize=True`,`use_per_output_channel_quantization=True`,`append_output_dequantize_ops=False` |
| 量化结果 | `build/model_int8.tflite`;实际 INT8 I/O 的 scale/zero point 写入 `build/quantization.json` |

### 完整 NCC 编译命令

在仓库根目录的已配置编译容器中,对已生成的 TFLite 执行.路径通过本地配置加载,命令不触发重新量化:

```bash
source tools/storage.sh mtk_models/perception/image_classification/resnet101
env LD_LIBRARY_PATH="${NCC_ROOT}/lib" \
    "${NCC_ROOT}/bin/ncc-tflite" \
    --arch=mdla5.3 --suppress-output --disallow-bridge \
    "${MODEL_ASSETS_DIR}/build/model_int8.tflite" \
    -o "${MODEL_ASSETS_DIR}/model_int8.dla"
```
