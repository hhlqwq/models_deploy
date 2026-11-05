# FastSAM-s

类别无关实例分割模型,输入为 640×640 RGB 图像.采用 INT8 训练后量化 (PTQ),使用逐输出通道权重量化,部署目标为 Genio 720.

模型来源见[模型卡](model_card.md),权重与依赖来源见 [source_url.txt](models/source_url.txt).

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与上传后结束.随后登录开发板,在部署目录执行 `bash compile.sh`,做少量通路检查并保存 README 示例到 `examples/output/`.该阶段计算并打印子集任务精度,不发布正式报告,不统计性能和峰值 RSS.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.板端预测、日志及内部指标统一位于本模型部署目录的 build/work/,每次开始推理时覆盖上次中间结果.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

## 配置与运行

- [config.yaml](config.yaml): 模型、校准数据、输出和板端部署目录.
- [G720 平台配置](../../../../configs/platforms/genio_720.yaml): 工具链和开发板连接信息.
- `configs/storage.local.json`: 模型和数据集的本地存储根目录,准备方式见[仓库说明](../../../../README.md#存储规则).

在项目根目录进入编译容器:

```bash
bash docker/mtk_g720/enter_container.sh
```

进入容器后,在本模型目录执行:

```bash
bash deploy/compile.sh
```

上传完成后,请登录开发板,在部署目录执行 `bash compile.sh` 完成通路检查与示例生成.正式评价时按[板端规范](../../../../docs/genio_720.md)准备参考及必要输入,再登录开发板的部署目录执行:

```bash
bash evaluate.sh
```

两个入口均无需参数.需要逐步执行时,查看 [deploy/compile.sh](deploy/compile.sh).

板端预处理、NPU 调用和掩码后处理统一使用 Python,不再单独构建 C++ 程序.G5100 尚无对应 SDK 与开发板验证,不能直接复用 G720 的 DLA 或运行库.

## 数据与评测

使用 COCO val2017 全量 5000 张图片,目录包含 `images/` 和 `annotations/instances_val2017.json`.INT8 校准默认按文件名排序选取 100 张图片.

评测采用类别无关实例分割协议,核心指标为 `segm AR@100`.ONNX 与板端使用相同协议,精度变化以百分点表示.详细定义见 [accuracy_protocol.json](deploy/board/accuracy_protocol.json).

## G720 实测结果

正式评价覆盖 COCO val2017。主指标计分 5,000 张图片，运行编号 `20261103_082451_1918740`。[正式报告](results/summary.json)

| 指标 | 浮点参考 | G720 | 损失 |
| --- | ---: | ---: | ---: | ---: |
| segm AR@100 | 39.010% | 37.570% | +1.440 个百分点 |

板端性能为 NPU 平均 14.537 ms，P95 14.592 ms，Python 进程峰值 RSS 为 177.04 MiB。

编译子集主指标计分 300 张图片，运行编号 `20261030_080352_1720392`。[子集报告](results/summary_test.json)

| 指标 | 浮点参考 | G720 | 损失 |
| --- | ---: | ---: | ---: | ---: |
| segm AR@100 | 38.153% | 36.387% | +1.766 个百分点 |

损失为正表示退化。子集报告中的输出误差仅覆盖下表所列调用。

| 输出 | 调用次数 | CosSim | RMSE | NRMSE |
| --- | ---: | ---: | ---: | ---: |
| `box_8` | 5 | 0.953049 | 1.05274 | 30.417% |
| `score_8` | 5 | 0.959462 | 4.4653 | 30.299% |
| `coefficient_8` | 5 | 0.959908 | 0.230138 | 28.142% |
| `box_16` | 5 | 0.942828 | 0.99721 | 33.611% |
| `score_16` | 5 | 0.953721 | 4.33453 | 32.181% |
| `coefficient_16` | 5 | 0.930904 | 0.363132 | 36.640% |
| `box_32` | 5 | 0.931159 | 1.08638 | 37.464% |
| `score_32` | 5 | 0.921175 | 4.82909 | 43.418% |
| `coefficient_32` | 5 | 0.864638 | 0.516318 | 51.463% |
| `prototype` | 5 | 0.932993 | 0.449901 | 36.436% |

## 效果示例

以下图片叠加了已有全量测试的板端实例分割结果,展示分数阈值为 0.25.

### 示例 1: 室内场景

![室内场景输入](examples/input/sample_1.jpg)

![室内场景分割输出](examples/output/sample_1_segmentation.jpg)

### 示例 2: 熊

![熊输入](examples/input/sample_2.jpg)

![熊分割输出](examples/output/sample_2_segmentation.jpg)

### 示例 3: 滑雪场景

![滑雪场景输入](examples/input/sample_3.jpg)

![滑雪场景分割输出](examples/output/sample_3_segmentation.jpg)

示例图片来源: 示例 1: [原图](http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg) / [图片许可](http://creativecommons.org/licenses/by-nc/2.0/); 示例 2: [原图](http://farm8.staticflickr.com/7434/9138147604_c6225224b8_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/); 示例 3: [原图](http://farm8.staticflickr.com/7015/6795644157_f019453ae7_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/).

## 资源与结果

模型资源保存在 `models_root/navigation/segmentation/fastsam/`,原始权重、ONNX 和 DLA 位于该目录根层,其余产物保存在 `build/`.文件用途见[资源布局说明](models/README.md).

评价结果只保存在部署目录的 `results/summary.json`.板端预测、日志和中间指标统一保存在本模型部署目录的 `build/work/`,每次开始推理时覆盖;展示示例保存在 `examples/output/`.取回唯一报告后据实更新 README.

### 输出误差统计

compile.sh 独立对固定少量输入比较浮点与 NPU 输出,在终端显示 CosSim、RMSE、NRMSE 和实际覆盖范围.普通模型最多选择 5 次输入,不减少原有 compile 通路检查样本数;LLM 比较示例首个 Prefill 块的有效位置,BEVFusion 比较单帧的 NPU 子图.这些结果用于数值检查,不能代替任务精度.全量 evaluate.sh 只计算任务精度、性能和 RSS,不保存或比较全量浮点张量.

参考为普通 NPZ 数组文件,保存在编译机器本地日期临时目录并上传板端 build/reference/output_reference.npz,由下次 compile 覆盖,不使用数据库,不写入 NAS.板端中间结果仍使用本模型 build/work/.NRMSE 定义为 100 × RMSE / RMS(浮点参考),多输出分别统计.详见[输出误差统计](../../../../docs/output_metrics.md).

### compile 子集精度检查

compile.sh 使用 300 张 COCO 验证图片,复用正式评价协议.主机执行子集浮点推理,板端执行同一子集 NPU 推理,在终端打印精度和损失,并保存 results/summary_test.json.正的损失表示 NPU 变差,负值表示改善;Top-1、AP、AR 等使用浮点减 NPU 的百分点,WER 使用 NPU 减浮点的百分点,PPL 使用增量及相对增幅.

普通样本跨完整划分按固定间隔抽取,保留展示及数值检查的必要输入;LFW、ESC-50 和 MMLU 保持折、类别和学科覆盖.同一子集两端通过样本指纹核对;MMLU 固定 seed=17 分层抽样.该报告不代表全量成绩,不改写 results/summary.json.中间数据覆盖保存在本模型 build/work/,CosSim、RMSE、NRMSE 仍只比较固定少量输入.详见[输出误差统计](../../../../docs/output_metrics.md).

## 量化参数与编译命令

以下为当前 [config.yaml](config.yaml)、[量化实现](deploy/host/convert_int8.py) 和 [编译实现](deploy/host/compile_model.py) 中的配置.未显式覆盖的 Converter 属性沿用当前 SDK 默认值,实际 scale/zero point 由校准产物读取,不能写死为通用常量.

### 量化设置

| 参数 | 当前设置 |
| --- | --- |
| 转换入口 | `deploy/host/convert_int8.py`,MTK `OnnxConverter` |
| 精度 | W8A8 PTQ,逐输出通道权重量化 |
| 校准 | COCO val2017 图片按文件名排序取前 100 张 |
| 输入 | FP32 `[1,3,640,640]`,复用板端 letterbox、RGB / 255 预处理 |
| 显式设置 | `quantize=True`,`use_per_output_channel_quantization=True`,`append_output_dequantize_ops=False` |
| 量化输入/结果 | `FastSAM-s.onnx` → `build/model_int8.tflite`,输出量化尺度和索引保存到 `build/model_int8.json` |

### 特殊 ONNX 导出处理

[export_model.py](deploy/host/export_model.py) 将三个尺度的框回归、类别、mask 系数头分别输出,再加 prototype 输出,共十个输出.保留原始权重,DFL、NMS 及实例 mask 组合放到 CPU 后处理.对旧 Upsample 补 `recompute_scale_factor=None`,固定 batch=1、opset 13;导出前检查原始头解码与官方前向、prototype 数值一致性.

### 完整 NCC 编译命令

在仓库根目录的已配置编译容器中,对已生成的 TFLite 执行.路径通过本地配置加载,命令不触发重新量化:

```bash
source tools/storage.sh mtk_models/navigation/segmentation/fastsam
env LD_LIBRARY_PATH="${NCC_ROOT}/lib" \
    "${NCC_ROOT}/bin/ncc-tflite" \
    --arch=mdla5.3 --suppress-output --disallow-bridge \
    "${MODEL_ASSETS_DIR}/build/model_int8.tflite" \
    -o "${MODEL_ASSETS_DIR}/model_int8.dla"
```
