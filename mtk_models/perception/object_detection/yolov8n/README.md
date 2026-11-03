# YOLOv8n / Genio 720

> Python 推理代码已迁移,尚未做板端验证.下文报告和示例属于迁移前的 C++/原生 CLI 实现,不能作为当前 Python 实现的精度、性能或 RSS.

COCO 80 类目标检测,固定输入 `1x3x640x640` RGB,INT8 PTQ,目标为
MT8189 / MDLA 5.3.

## 执行范围

先在已有编译容器执行 `bash deploy/compile.sh`,完成转换、量化、编译与上传后结束.随后登录开发板,在部署目录执行 `bash compile.sh`,做少量通路检查并保存 README 示例到 `examples/output/`.该阶段计算并打印子集任务精度,不发布正式报告,不统计性能和峰值 RSS.

正式评价使用独立的 `evaluate.sh`: 容器端只计算已有浮点模型的同协议参考及必要评价输入,资源单独放入已有部署包后,板端运行 `bash evaluate.sh`,统计精度、性能和推理进程峰值 RSS.正式结果只写入 `results/summary.json`,不重新量化或编译.板端预测、日志及内部指标统一位于本模型部署目录的 build/work/,每次开始推理时覆盖上次中间结果.详见[板端规范](../../../../docs/genio_720.md).

下文已归档结果保留原始模型版本和评价范围,本次入口调整不代表重新实测.

## 官方来源

- 实现: https://github.com/ultralytics/ultralytics/tree/v8.0.111
- 安装版本: `ultralytics==8.0.111`,复用现有 Docker 包,不自动升级.
- 权重: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
- 权重 SHA256: `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95`.
- 数据集: https://cocodataset.org/#download ,使用现有 COCO val2017.
- Qualcomm 仅作为交付结构参考,本模型不使用其预导出产物.

## 两步运行

将官方权重离线放入外部模型资源目录的 `yolov8n.pt`,脚本会验证哈希,不在线下载模型和依赖.

通过 `bash docker/mtk/enter_container.sh` 进入编译容器,在项目根目录执行:

```bash
bash mtk_models/perception/object_detection/yolov8n/deploy/compile.sh
```

流程使用 NeuroPilot SDK 8.0.11、MTK Converter 8.16.0,选取 100 张固定排序的 COCO 图片校准,完成 ONNX 导出、INT8 转换、DLA 编译和上传后结束,板端示例由用户登录开发板执行 `bash compile.sh` 生成.模型产物位于配置的外部资源目录,中间结果位于构建目录,仓库 `results/` 保留最终 [summary.json](results/summary.json).

上传完成后,请登录开发板,在部署目录执行 `bash compile.sh` 完成通路检查与示例生成.正式评价时按[板端规范](../../../../docs/genio_720.md)准备参考及必要输入,再登录开发板的部署目录执行:

```bash
bash evaluate.sh
```

compile.sh 使用固定 100 张真实 NPU 输入检查通路并生成检测示例,不计算正式指标.evaluate.sh 对 COCO val2017 全量 5000 张评价,排除 20 次预热,将精度、性能与峰值 RSS 统一发布至 `results/summary.json`.

## 适配与评测协议

三个尺度分别导出 64 通道框分布和 80 通道分类 logits,共六个输出.
DFL、Sigmoid、类别内 NMS 和原图坐标还原在 CPU 执行.
量化参数及输出索引从本次 TFLite 解析,通过 `runtime_config.csv` 部署;
Runtime 检查缓冲区尺寸,支持 NCHW 行宽按 16 对齐.
每次推理重新登记输入和全部输出,避免复用陈旧缓冲区.

PyTorch、ONNX 和板端使用相同 letterbox、单最佳类别、`conf=0.001`,
`IoU=0.6`,`max_det=300`.该协议用于本项目三后端对比,
不直接等同于 Ultralytics 官方公开指标的复现协议.
公开示例取自已有 COCO 数据,原图地址和许可证在编译时从 COCO 标注自动读取,随部署清单上传.展示阈值为 0.25.

## 迁移前结果

运行 ID: `20261008_yolov8n_full_v1`,三个后端均使用 COCO val2017 全量
5000 张图片,主机与板端标注 SHA256 相同.

| 后端 | mAP@0.5:0.95 |
| --- | ---: |
| PyTorch FP32 | **36.6444%** |
| ONNX FP32 | **36.6260%** |
| MTK NPU INT8 | **35.3510%** |

最终精度和 NPU 性能汇总见 [summary.json](results/summary.json),
三后端详细实测指标和环境信息保留在本 README.

ONNX 相对 PyTorch 变化为 **-0.0184 个百分点**,INT8 相对本次 ONNX
变化为 **-1.2750 个百分点**.单图原始头解码与官方前向的最大绝对误差为
`8.312659338116646e-5`,六个 ONNX 输出已通过数值一致性校验.

| 板端指标 | 平均 | P95 |
| --- | ---: | ---: |
| 预处理 | 13.236 ms | 16.963 ms |
| 独立 NPU 推理 | **6.283 ms** | 6.381 ms |
| CPU 后处理 | 3.233 ms | 4.669 ms |
| 端到端 | **22.925 ms** | **27.088 ms** |

推理进程峰值 RSS 为 **31.324 MiB**.NPU 耗时只统计常驻模型的
`NeuronRuntime_inference` 调用,排除 20 次预热;端到端从图片读取开始,
包含预处理、IO 登记、推理和后处理,不包含预测文件写入、COCO 汇总或画图.
完整分位数保存在下述板端原始运行目录的 `work/timing_summary_current_run.json`.

运行环境为 Rity Demo 26.0-release / scarthgap / Linux 6.6.137,
Neuron Runtime 8.2.16,CPU governor 为 `schedutil`.主机使用 Torch 2.0.0+cu118,
ONNX 1.13.1,ONNX Runtime 1.18.0,ONNX 精度基准使用 CPU EP.
Genio 5100 尚未执行,上述数字只对应 Genio 720.

模型、张量契约和板端程序哈希保存在板端原始运行目录的 `SHA256SUMS`.
本次 DLA 大小为 `3,527,145` 字节,板端模型与编译主机产物哈希一致.
本次 DLA SHA256 为 `a009dea16d53fa64dbfb98fe3156afcd2dc8028a62ef69b865060e6ec37ab2c2`.
全部预测、图片覆盖清单、逐图耗时、日志和效果图保留在:

```text
<板端部署目录>/results/20261008_yolov8n_full_v1/
```

以上目录属于历史实测现场.新流程将板端过程文件保存在本模型部署目录的 `build/work/`,每次开始推理时覆盖,正式报告统一更新 `results/summary.json`.

## 真实板端效果

展示阈值为 0.25,图像取自 COCO val2017,来源清单在编译时生成并随部署上传.
示例渲染和 COCO 评价均读取本次过程目录 `work/predictions.json` 中的 JSON 数组预测,不生成或依赖 `predictions.jsonl`.已有少量推理结果时,可在板端单独执行 `MTK_EVAL_SCOPE=subset python3 board/render_examples.py --run-id <运行编号>` 补生成示例,无需重新推理.
效果由上述全量 NPU 预测生成,标签排布仅用于提高可读性.

![室内检测](examples/output/sample_1_detections.jpg)

![熊检测](examples/output/sample_2_detections.jpg)

![滑雪场景检测](examples/output/sample_3_detections.jpg)

示例图片来源: 示例 1: [原图](http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg) / [图片许可](http://creativecommons.org/licenses/by-nc/2.0/); 示例 2: [原图](http://farm8.staticflickr.com/7434/9138147604_c6225224b8_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/); 示例 3: [原图](http://farm8.staticflickr.com/7015/6795644157_f019453ae7_z.jpg) / [图片许可](http://creativecommons.org/licenses/by/2.0/).

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
| 量化结果 | `build/model_int8.tflite`;各输出头的实际量化尺度和索引保存到 `build/model_int8.json` |

### 特殊 ONNX 导出处理

[export_model.py](deploy/host/export_model.py) 使用固定 Ultralytics 8.0.111 和原始权重,将 Detect 前向改成三个尺度的框回归/类别原始头,共六个输出,不把 DFL 解码和 NMS 编入图中.对旧 Upsample 序列化缺失的 `recompute_scale_factor` 补 `None`,保持旧默认语义.固定 opset 13、batch=1,分别检查 CPU 解码与官方前向、ONNX Runtime 与导出头的数值一致性.

### 完整 NCC 编译命令

在仓库根目录的已配置编译容器中,对已生成的 TFLite 执行.路径通过本地配置加载,命令不触发重新量化:

```bash
source tools/storage.sh mtk_models/perception/object_detection/yolov8n
env LD_LIBRARY_PATH="${NCC_ROOT}/lib" \
    "${NCC_ROOT}/bin/ncc-tflite" \
    --arch=mdla5.3 --suppress-output --disallow-bridge \
    "${MODEL_ASSETS_DIR}/build/model_int8.tflite" \
    -o "${MODEL_ASSETS_DIR}/model_int8.dla"
```
