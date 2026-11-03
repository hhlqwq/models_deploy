"""从固定的官方 Depth Anything V2 Small 权重导出静态 ONNX。"""

import argparse
from pathlib import Path
import sys

import torch

from tools.deploy import configure_parser


class ExportableAttention(torch.nn.Module):
    """将官方 QKV 权重用于最多四维的固定自注意力计算。"""

    def __init__(self, original: torch.nn.Module) -> None:
        """复用官方注意力层参数，不修改权重内容。"""
        super().__init__()
        self.qkv = original.qkv
        self.proj = original.proj
        self.num_heads = original.num_heads
        self.scale = original.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """用四维 Q、K、V 张量执行多头自注意力。"""
        batch, tokens, channels = x.shape
        projected = self.qkv(x)
        query, key, value = projected.chunk(3, dim=-1)
        head_size = channels // self.num_heads
        query = query.reshape(batch, tokens, self.num_heads, head_size)
        key = key.reshape(batch, tokens, self.num_heads, head_size)
        value = value.reshape(batch, tokens, self.num_heads, head_size)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        merged = attended.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(merged)


def export_model(upstream: Path, weights: Path, output: Path, size: int) -> None:
    """检查离线源码文件,加载权重并导出固定方形输入的深度图."""
    if size <= 0 or size % 14:
        raise ValueError("输入边长必须是 14 的正整数倍。")
    required = (
        "LICENSE", "depth_anything_v2/dpt.py", "depth_anything_v2/dinov2.py",
        "depth_anything_v2/dinov2_layers/__init__.py",
        "depth_anything_v2/util/blocks.py",
    )
    missing = [name for name in required if not (upstream / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"离线官方源码不完整: {upstream}; 缺少: {', '.join(missing)}")

    sys.path.insert(0, str(upstream))
    # 上游源码目录由参数指定并校验后加载,不能在解析参数前导入.
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.rand((1, 3, size, size),
                        generator=torch.Generator().manual_seed(20260924))
    patch_count = (size // 14) ** 2
    token = torch.zeros((1, patch_count + 1, model.pretrained.embed_dim))
    with torch.no_grad():
        baseline = model(sample)
        position = model.pretrained.interpolate_pos_encoding(token, size, size)
        model.pretrained.pos_embed = torch.nn.Parameter(position)
        for block in model.pretrained.blocks:
            block.attn = ExportableAttention(block.attn)
        revised = model(sample)
    max_difference = (baseline - revised).abs().max().item()
    if max_difference > 1e-4:
        raise ValueError(f"固定位置编码与原模型输出不一致: {max_difference}")
    print(f"[EXPORT] 固定位置编码和四维注意力，PyTorch 最大差异 {max_difference:.8g}。",
          flush=True)
    print(f"[EXPORT] 固定输入 1x3x{size}x{size}，输出相对深度图。", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            model, sample, str(output), opset_version=17,
            input_names=["image"], output_names=["relative_depth"],
            do_constant_folding=True)
    print(f"[OK] ONNX: {output}", flush=True)


def main() -> None:
    """读取命令行参数并执行导出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=518)
    configure_parser(parser, __file__, {
        "upstream": lambda c: c.output / "original/upstream",
        "weights": lambda c: Path(c.config["model"]["weights"]),
        "output": lambda c: c.output / "depth_anything_v2_vits.onnx",
    })
    args = parser.parse_args()
    export_model(args.upstream, args.weights, args.output, args.size)


if __name__ == "__main__":
    main()
