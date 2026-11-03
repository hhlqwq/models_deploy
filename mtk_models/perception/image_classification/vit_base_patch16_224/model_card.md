# ViT-Base Patch16 224 模型卡

## 来源

- 官方项目：<https://github.com/pytorch/vision>，版本 `v0.15.1`。
- 模型构造器：`torchvision.models.vit_b_16`。
- 权重枚举：`ViT_B_16_Weights.IMAGENET1K_V1`。
- 官方权重：<https://download.pytorch.org/models/vit_b_16-c867db91.pth>。
- 许可证：TorchVision BSD-3-Clause。

本项目从官方权重自行导出 ONNX，再转换为 INT8 TFLite 和 DLA。Qualcomm 页面仅用于交付形式参考，不作为正式模型来源。

## 规格

输入为 NCHW 224×224 RGB,输出为 ImageNet-1K 的 1000 类 logits。评测使用 ILSVRC2012 val 全量 50000 张图片,核心精度为 Top-1。
