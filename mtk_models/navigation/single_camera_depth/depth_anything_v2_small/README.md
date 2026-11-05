# Depth Anything V2 Small

单目相对深度估计，输入为 518×518 RGB 图像。输出只表达单张图内的相对结构，不能直接用于米制测距。来源见[模型卡](model_card.md),资源下载地址见 [source_url.txt](models/source_url.txt)。

量化方式: INT8 训练后量化 (PTQ),采用逐输出通道权重量化.

板端 compile 保留固定 100 张图片用于快速观察模型效果.中间深度图保存为模型原始的 518x518 FP32 输出,单张约 1 MiB,评分和展示时再双线性插值到原图尺寸.路径遵循全部模型共同的 work/ 覆盖规则,不单独设置临时目录.

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

转换、编译、参考评价和上传均在当前容器内完成。

## 第二步: 独立评价

上传完成后,请登录开发板,在部署目录执行 `bash compile.sh` 完成通路检查和示例生成.正式评价前,按[板端规范](../../../../docs/genio_720.md)准备同协议参考及必要输入,在已部署目录执行:

```bash
bash evaluate.sh
```

精度、性能和峰值 RSS 统一保存到 results/summary.json,板端过程数据位于本模型部署目录的 build/work/,每次开始推理时覆盖,不重新量化或编译.

## 数据与精度评测

参考评价使用 `config.yaml` 指定的数据集,与板端保持相同样本、标注和预后处理协议.

数据目录要求: DA-2K: images/ 和 annotations.json,1033 张图片、2068 个点对.

精度变化以百分点表示,正数为改善,负数为下降.

Docker 需要 ONNX Runtime、NumPy、OpenCV 和 tqdm。

## G720 实测结果

正式评价覆盖 DA-2K。主指标计分 2,068 个标注点对，运行编号 `20261103_090139_1919879`。[正式报告](results/summary.json)

| 指标 | 浮点参考 | G720 | 损失 |
| --- | ---: | ---: | ---: | ---: |
| DA-2K 点对准确率 | 94.826% | 85.783% | +9.043 个百分点 |

板端性能为 NPU 平均 133.478 ms，P95 133.937 ms，Python 进程峰值 RSS 为 243.45 MiB。

编译子集主指标计分 195 个标注点对，运行编号 `20261030_080608_1721215`。[子集报告](results/summary_test.json)

| 指标 | 浮点参考 | G720 | 损失 |
| --- | ---: | ---: | ---: | ---: |
| DA-2K 点对准确率 | 92.821% | 80.513% | +12.308 个百分点 |

损失为正表示退化。子集报告中的输出误差仅覆盖下表所列调用。

| 输出 | 调用次数 | CosSim | RMSE | NRMSE |
| --- | ---: | ---: | ---: | ---: |
| `output` | 5 | 0.965085 | 1.06951 | 37.396% |

## 效果示例

原图与板端相对深度图并排,暖色表示较近,冷色表示较远.

输入已放在 `examples/input/`,编译时从配置的数据集自动生成来源与样本清单,随部署上传,不纳入 Git.

compile.sh 复用固定少量样本的板端预测,生成展示文件到部署目录的 `examples/output/`,不依赖全量评价.
以下三张输出来自本次全量测试的板端预测,保存在 `examples/output/`.示例目录仅保留三张输入和三张对应输出.

### 示例 1: 室内场景

![室内场景板端相对深度输出](examples/output/sample_1_depth.jpg)

### 示例 2: 室外街景

![室外街景板端相对深度输出](examples/output/sample_2_depth.jpg)

### 示例 3: 玻璃建筑

![玻璃建筑板端相对深度输出](examples/output/sample_3_depth.jpg)

## 板端部署结构

第一步上传到 `BOARD_DEPLOY_DIR` 后的布局统一为:

```text
部署目录/
├── compile.sh
├── evaluate.sh          # 独立精度、性能与 RSS 评价.
├── config.json
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
- `models/`: 公开来源说明.实际资源位于 NAS 对应模型资源目录: `original/` 保存原始权重与 `upstream/` 离线官方源码,`build/` 保存转换中间产物,最外层只保存导出的 ONNX 和编译后的 DLA.导出直接加载离线源码,不依赖 Git 命令或 `.git` 目录;源码版本见 `models/source_url.txt`.
- `examples/`: 少量固定输入与实际板端效果输出.
- `results/summary.json`: 上传后的最新测试汇总.

### 输出误差统计

compile.sh 独立对固定少量输入比较浮点与 NPU 输出,在终端显示 CosSim、RMSE、NRMSE 和实际覆盖范围.普通模型最多选择 5 次输入,不减少原有 compile 通路检查样本数;LLM 比较示例首个 Prefill 块的有效位置,BEVFusion 比较单帧的 NPU 子图.这些结果用于数值检查,不能代替任务精度.全量 evaluate.sh 只计算任务精度、性能和 RSS,不保存或比较全量浮点张量.

参考为普通 NPZ 数组文件,保存在编译机器本地日期临时目录并上传板端 build/reference/output_reference.npz,由下次 compile 覆盖,不使用数据库,不写入 NAS.板端中间结果仍使用本模型 build/work/.NRMSE 定义为 100 × RMSE / RMS(浮点参考),多输出分别统计.详见[输出误差统计](../../../../docs/output_metrics.md).

### compile 子集精度检查

compile.sh 使用 100 张 DA-2K 图片,复用正式评价协议.主机执行子集浮点推理,板端执行同一子集 NPU 推理,在终端打印精度和损失,并保存 results/summary_test.json.正的损失表示 NPU 变差,负值表示改善;Top-1、AP、AR 等使用浮点减 NPU 的百分点,WER 使用 NPU 减浮点的百分点,PPL 使用增量及相对增幅.

普通样本跨完整划分按固定间隔抽取,保留展示及数值检查的必要输入;LFW、ESC-50 和 MMLU 保持折、类别和学科覆盖.同一子集两端通过样本指纹核对;MMLU 固定 seed=17 分层抽样.该报告不代表全量成绩,不改写 results/summary.json.中间数据覆盖保存在本模型 build/work/,CosSim、RMSE、NRMSE 仍只比较固定少量输入.详见[输出误差统计](../../../../docs/output_metrics.md).

## 量化参数与编译命令

以下为当前 [config.yaml](config.yaml)、[量化实现](deploy/host/convert_int8.py) 和 [编译实现](deploy/host/compile_model.py) 中的配置.未显式覆盖的 Converter 属性沿用当前 SDK 默认值,实际 scale/zero point 由校准产物读取,不能写死为通用常量.

### 量化设置

| 参数 | 当前设置 |
| --- | --- |
| 转换入口 | `deploy/host/convert_int8.py`,MTK `OnnxConverter` |
| 精度 | W8A8 PTQ,逐输出通道权重量化 |
| 校准 | COCO val2017 图片按文件名排序取前 16 张;任务精度仍使用 DA-2K |
| 输入 | FP32 `[1,3,518,518]`,复用 `depth_utils.preprocess` 的图像缩放和归一化 |
| 显式设置 | `quantize=True`,`use_per_output_channel_quantization=True`,`append_output_dequantize_ops=False` |
| 量化输入/结果 | `depth_anything_v2_vits.onnx` → `build/model_int8.tflite`,实际 I/O 契约保存到 `build/model_int8.json` |

### 特殊 ONNX 导出处理

[export_model.py](deploy/host/export_model.py) 复用原始 ViT-S 权重,针对固定 518×518 输入预先插值位置编码,再把注意力替换为最多四维的 Q/K/V、MatMul 和 Softmax.保留原始 QKV/投影权重,以相同输入检查改写前后输出,最大绝对差须不超过 `1e-4`.随后导出固定 opset 17 图.只读取离线官方源码文件,不依赖 Git 工作目录.

### 完整 NCC 编译命令

在仓库根目录的已配置编译容器中,对已生成的 TFLite 执行.路径通过本地配置加载,命令不触发重新量化:

```bash
source tools/storage.sh mtk_models/navigation/single_camera_depth/depth_anything_v2_small
env LD_LIBRARY_PATH="${NCC_ROOT}/lib" \
    "${NCC_ROOT}/bin/ncc-tflite" \
    --arch=mdla5.3 --suppress-output --disallow-bridge --suppress-input \
    "${MODEL_ASSETS_DIR}/build/model_int8.tflite" \
    -o "${MODEL_ASSETS_DIR}/model_int8.dla"
```

Depth 的最多 5 次输出数值对比按 DA-2K 相对图片路径配对,两端各自使用原有预处理,不传输输入数组或要求逐字节一致.原始图片、DLA 和张量契约仍需一致.CosSim/RMSE/NRMSE 包含预处理及模型执行差异,不等同于纯量化误差;100 张子集任务精度和全量评价保持原有协议.
