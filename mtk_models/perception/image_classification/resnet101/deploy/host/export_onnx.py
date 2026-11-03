"""使用本地 TorchVision 官方 ResNet101 V2 权重离线导出 ONNX."""

from pathlib import Path
import sys

import onnx
import torch
from torch import nn
import torchvision
from torchvision.models import resnet101

from tools.deploy import arguments
from tools.files import require_file, sha256

# 主机和板端使用同一份模型契约,不复制预处理常量.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from resnet101 import PROTOCOL, TORCHVISION_VERSION


class NormalizedResNet(nn.Module):
    """在官方 ResNet101 图前添加 ImageNet 归一化."""

    def __init__(self, model: nn.Module) -> None:
        """复用已加载权重,将 mean/std 作为常量导出."""
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(PROTOCOL["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(PROTOCOL["std"]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """将 RGB `[0,1]` 输入归一化并输出 1000 类 logits."""
        return self.model((image - self.mean) / self.std)


def main() -> None:
    """严格加载本地权重,导出固定 batch=1 的 FP32 ONNX."""
    ctx = arguments(__file__)
    if torchvision.__version__.split("+")[0] != TORCHVISION_VERSION:
        raise RuntimeError(f"需要 TorchVision {TORCHVISION_VERSION},请使用已有编译容器.")
    weights = require_file(ctx.config["model"]["weights"])
    if weights.name != "resnet101-cd907fc2.pth":
        raise ValueError("需要官方 IMAGENET1K_V2 权重 resnet101-cd907fc2.pth.")
    if not sha256(weights).startswith("cd907fc2"):
        raise ValueError("本地权重与官方 V2 文件不匹配,请核对下载是否完整.")
    print(f"[导出] 从本地加载官方权重: {weights.name}.", flush=True)
    model = resnet101(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model = NormalizedResNet(model.eval()).eval()
    destination = ctx.output / "resnet101.onnx"
    torch.onnx.export(
        model, torch.zeros(1, 3, 224, 224), str(destination),
        export_params=True, opset_version=13, do_constant_folding=True,
        input_names=["image"], output_names=["logits"],
    )
    onnx.checker.check_model(onnx.load(str(destination)))
    print(f"[导出完成] {destination}; 无动态维度,无 Softmax.", flush=True)


if __name__ == "__main__":
    main()
