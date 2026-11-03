# 04 校准量化与 DLA 编译

## 1. 阶段目标

使用固定模型、固定预处理和代表性校准数据生成 INT8 TFLite,再面向 Genio 720 的 MDLA 5.3 编译 DLA.

## 2. 产物流

```text
yolov5s.pt
  → yolov5s.torchscript
  → model_fp32.onnx
  → model_int8.tflite
  → model_int8.dla
```

ONNX 用于 FP32 基线和结构检查；本项目的 MTK INT8 PTQ 输入是 TorchScript.

## 3. 执行位置

以下命令在编译容器中执行：

```bash
cd "${MTK_MODELS_ROOT}/mtk_models/perception/object_detection/yolov5s"
bash ./deploy/prepare.sh
```

## 4. 转换步骤

`convert.sh` 依次完成：

1. 验证容器内工具链.
2. 使用锁定 YOLOv5 源码导出 TorchScript.
3. 导出 FP32 ONNX.
4. 调用 `convert_int8.py` 执行 INT8 PTQ.
5. 确认产物非空并记录生成路径.

校准图片来自完整 COCO val2017 图片目录,但校准过程和正式精度评测是两个概念.校准数据只用于确定量化范围,不能把校准结果当作正式 COCO mAP.

## 5. 校准检查

- 图像来源和许可明确.
- 样本覆盖常见场景、尺度、目标密度和亮度.
- 预处理与推理输入保持一致.
- RGB/BGR、NCHW/NHWC、缩放范围和量化参数没有混淆.
- 数据读取顺序可复现.
- 记录校准数量和抽样方式.

## 6. MT8189 的 NCC 命令

本项目实际使用：

```bash
ncc-tflite \
    --arch=mdla5.3 \
    --suppress-output \
    --disallow-bridge \
    model_int8.tflite \
    -o model_int8.dla
```

参数解释：

| 参数 | 作用 |
| --- | --- |
| `--arch=mdla5.3` | 指定 Genio 720/MT8189 的 MDLA 目标 |
| `--suppress-output` | 抑制默认输出数据转换桥,暴露 MDLA 原生输出 |
| `--disallow-bridge` | 禁止编译器引入目标板不支持的桥接路径 |

抑制输出桥后,YOLOv5s 的 80、40、20 宽度检测头可能按 16 元素对齐为 80、48、32.板端程序必须按 metadata 和项目已验证规则解交织,不能假设输出连续无 padding.

## 7. 编译失败怎么查

1. 保存完整 NCC 命令和完整日志.
2. 从日志定位失败算子编号、算子名、shape、rank 和 dtype.
3. 在 NP8/G720 支持算子表核对约束.
4. 确认使用的是 NP8 对应 NCC,而不是 NP6 编译器.
5. 如属于硬件不支持,优先考虑修改源模型、等价改写或裁剪子图.
6. 不要只因 Converter 成功就认定 NPU 一定支持.

## 8. 通过标准

- 四级模型产物均非空.
- 转换日志没有未解释的关键错误.
- NCC 目标明确为 MDLA 5.3.
- 编译日志没有未解释的硬件算子失败或桥接目标.
- DLA 必须继续经过目标板加载验证,不能在此阶段宣布完整交付.

## 9. MTK 官网参考

- [NeuroPilot Converter Tool](https://genio.mediatek.com/doc/iot-aihub/ai_hub/ai-workflow/model_converter/neuropilot_converter_tool.html)
- [Neuron Compiler and Runtime](https://genio.mediatek.com/doc/iot-aihub/ai_hub/ai-workflow/neuron-sdk.html)
- [G720 TFLite Supported Operations](https://neuropilot.mediatek.com/sphinx/g720/html/l1_supported_operations/l2_supported_operations/supported_operations.html)
- [Handling Unsupported OP and Models](https://genio.mediatek.com/doc/iot-aihub/ai_hub/ai-workflow/troubleshooting.html)
- [Example: Pruning the Model](https://genio.mediatek.com/doc/iot-aihub/ai_hub/ai-workflow/troubleshooting/prune_model.html)

MTK 的部分通用 YOLOv5s 页面展示 MDLA 3.0 命令.该命令不能覆盖平台资源页对 Genio 720/NP8/MDLA 5.3 的定义,本项目以实际 MT8189 编译和板端加载结果为准.
