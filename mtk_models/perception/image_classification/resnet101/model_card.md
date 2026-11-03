# ResNet101 模型卡

## 官方来源

- 项目: [pytorch/vision v0.15.1](https://github.com/pytorch/vision/tree/v0.15.1).
- 构造器: `torchvision.models.resnet101`,TorchVision ResNet V1.5,不修改网络结构.
- 权重: `ResNet101_Weights.IMAGENET1K_V2`,文件 `resnet101-cd907fc2.pth`.
- 下载: [PyTorch 官方权重](https://download.pytorch.org/models/resnet101-cd907fc2.pth).
- 源码许可: [TorchVision BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.15.1/LICENSE).
- 预训练权重与数据集的使用条件见 [TorchVision 提示](https://docs.pytorch.org/vision/0.15/models.html)及 ImageNet 原始条款,源码许可不替代数据集使用条件.

从本地官方原始 state_dict 自行导出 ONNX,再做 W8A8 PTQ 和 G720 DLA 编译,不使用第三方预导出模型.官方权重版本、固定 224 输入、短边 232 的 V2 预处理与 TorchVision 类别顺序共同构成模型身份.

## 规格与评价

输入为 RGB FP32 `[1,3,224,224]`、范围 `[0,1]`,图内包含 ImageNet mean/std 归一化;输出为 1000 类 logits.原生量化 I/O 必须为逐张量 INT8,实际 scale/zero point 从转换图提取,缓冲区大小在 SDK Runtime 中核对.

正式评价使用 ImageNet ILSVRC2012 validation 全量 50000 张,单中心裁剪,分类指标只有 Top-1.默认从该验证集取 100 张用于校准,完整评价如实记录这 100 张的重叠.不将三张展示输入当作正式精度.

官方 V2 浮点 Top-1 81.886%,仅用于选型.本仓库已完成同协议全量实测: ONNX Top-1 81.898%,NPU Top-1 81.544%,下降 0.354 个百分点.预热后的纯 NPU 平均耗时 5.157 ms,Python 推理进程峰值 RSS 243.74 MiB.本次报告复用原运行记录补齐参考,未重新执行模型,详见 [README](README.md)与 [报告](results/summary.json).

## 交付参考与限制

[Qualcomm ResNet101](https://aihub.qualcomm.com/iot/models/resnet101)仅作为分类模型交付及结果展示参考,其平台产物和性能不作为 MTK 移植来源或结果.当前代码面向 G720 / MDLA 5.3,G5100 未验证.量化可产生类别排序变化,须通过独立评价确认影响.
