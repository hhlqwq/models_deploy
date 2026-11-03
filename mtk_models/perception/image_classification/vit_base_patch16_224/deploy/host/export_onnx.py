"""从 TorchVision 官方 ViT-B/16 权重离线导出 ONNX."""

import argparse
from pathlib import Path

import onnx
import torch
from torch import nn
from torch.nn import functional as functional
import torchvision
from torchvision.models import vit_b_16

from tools.deploy import configure_parser
from tools.storage import storage_roots


EXPECTED_TORCHVISION_VERSION = "0.15.1"
IMAGE_SIZE = 224
PATCH_SIZE = 16
TOKEN_COUNT = (IMAGE_SIZE // PATCH_SIZE) ** 2 + 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ExportableSelfAttention(nn.Module):
    """使用最高四维张量实现固定输入 ViT 自注意力."""

    def __init__(self, attention: nn.MultiheadAttention) -> None:
        """复用官方 MultiheadAttention 的全部投影参数.

        Args:
            attention: TorchVision ViT 中的原始自注意力层.

        Raises:
            ValueError: 注意力配置不符合当前固定 ViT-B/16 图.
        """
        super().__init__()
        if not attention.batch_first or attention.in_proj_weight is None:
            raise ValueError("ViT 自注意力必须使用 batch_first 和合并 QKV 权重.")
        if attention.embed_dim % attention.num_heads:
            raise ValueError("ViT embed_dim 必须能够整除 num_heads.")
        self.embed_dim = attention.embed_dim
        self.num_heads = attention.num_heads
        self.head_dim = attention.embed_dim // attention.num_heads
        self.in_proj_weight = attention.in_proj_weight
        self.in_proj_bias = attention.in_proj_bias
        self.out_proj = attention.out_proj

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask=None,
            need_weights: bool = False,
            attn_mask=None,
            average_attn_weights: bool = True,
            is_causal: bool = False) -> tuple[torch.Tensor, None]:
        """执行固定 batch、token 数且无掩码的多头自注意力.

        Args:
            query: `[1,197,768]` 的输入张量.
            key: 自注意力 Key,必须与 Query 相同.
            value: 自注意力 Value,必须与 Query 相同.
            key_padding_mask: 必须为空.
            need_weights: 当前编码器必须不请求注意力权重.
            attn_mask: 必须为空.
            average_attn_weights: 保留接口兼容性,当前不使用.
            is_causal: 必须为 False.

        Returns:
            注意力输出和空权重.

        Raises:
            ValueError: 调用参数不符合固定推理图约束.
        """
        del key, value, average_attn_weights
        if key_padding_mask is not None or attn_mask is not None:
            raise ValueError("ViT 固定推理图不接受 attention mask.")
        if need_weights or is_causal:
            raise ValueError("ViT 固定推理图不输出权重且不使用 causal mask.")
        projected = functional.linear(
            query, self.in_proj_weight, self.in_proj_bias)
        query_projection, key_projection, value_projection = projected.chunk(
            3, dim=-1)
        query_heads = query_projection.reshape(
            1, TOKEN_COUNT, self.num_heads, self.head_dim).transpose(1, 2)
        key_heads = key_projection.reshape(
            1, TOKEN_COUNT, self.num_heads, self.head_dim).transpose(1, 2)
        value_heads = value_projection.reshape(
            1, TOKEN_COUNT, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(
            query_heads, key_heads.transpose(-2, -1)) * self.head_dim ** -0.5
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, value_heads)
        merged = attended.transpose(1, 2).reshape(
            1, TOKEN_COUNT, self.embed_dim)
        return self.out_proj(merged), None


def replace_self_attention(model: nn.Module) -> None:
    """将 12 层原始注意力替换为 MTK 支持的四维等价实现."""
    for layer in model.encoder.layers:
        layer.self_attention = ExportableSelfAttention(
            layer.self_attention)


class NormalizedViT(nn.Module):
    """为 TorchVision ViT 增加固定的 ImageNet 输入归一化."""

    def __init__(self, model: nn.Module) -> None:
        """初始化导出包装器.

        Args:
            model: 已加载官方权重的 TorchVision ViT 模型.
        """
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """对 NCHW RGB `[0,1]` 输入归一化并输出 ImageNet logits."""
        return self.model((image - self.mean) / self.std)


def load_model(weights_path: Path, approximate_gelu: bool) -> nn.Module:
    """加载官方权重并构造导出模型.

    Args:
        weights_path: TorchVision 官方权重路径.
        approximate_gelu: 是否将 GELU 改为标准 tanh 近似.

    Returns:
        包含输入归一化的推理模型.
    """
    installed_version = torchvision.__version__.split("+")[0]
    if installed_version != EXPECTED_TORCHVISION_VERSION:
        raise RuntimeError(
            "TorchVision 版本不匹配: "
            f"expected={EXPECTED_TORCHVISION_VERSION}, "
            f"actual={installed_version}")
    model = vit_b_16(weights=None)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    if approximate_gelu:
        for module in model.modules():
            if isinstance(module, nn.GELU):
                module.approximate = "tanh"
    model.eval()
    generator = torch.Generator().manual_seed(20260910)
    validation_input = torch.rand(
        1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    with torch.inference_mode():
        reference_output = model(validation_input)
    replace_self_attention(model)
    with torch.inference_mode():
        rewritten_output = model(validation_input)
    max_abs = float((reference_output - rewritten_output).abs().max())
    if max_abs > 1e-4:
        raise ValueError(
            f"ViT 四维注意力改写偏差超限: max_abs={max_abs:.8g}")
    print(f"[VERIFY] ViT 四维注意力改写: max_abs={max_abs:.8g}")
    return NormalizedViT(model).eval()


def export_model(
        weights_path: Path,
        output_path: Path,
        approximate_gelu: bool) -> None:
    """导出固定输入 ViT ONNX 并执行结构检查.

    Args:
        weights_path: TorchVision 官方权重路径.
        output_path: ONNX 输出路径.
        approximate_gelu: 是否使用 tanh GELU.
    """
    model = load_model(weights_path, approximate_gelu)
    dummy_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
    )
    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)
    print(f"[OK] ViT ONNX 已导出: {output_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数.

    Returns:
        已解析参数.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--compatible-output", type=Path, required=True)
    configure_parser(parser, __file__, {
        "weights": lambda c: Path(c.config["model"]["weights"]),
        "reference_output": lambda c: c.output / "vit_b_16.onnx",
        "compatible_output": lambda c: c.output / "vit_b_16_mtk_compatible.onnx",
    })
    return parser.parse_args()


def main() -> None:
    """依次导出精确基线和 MTK 兼容候选模型."""
    args = parse_args()
    print("[1/2] 导出官方精确 GELU 基线 ONNX.")
    export_model(args.weights, args.reference_output, False)
    print("[2/2] 导出 tanh GELU 的 MTK 兼容候选 ONNX.")
    export_model(args.weights, args.compatible_output, True)


if __name__ == "__main__":
    main()
