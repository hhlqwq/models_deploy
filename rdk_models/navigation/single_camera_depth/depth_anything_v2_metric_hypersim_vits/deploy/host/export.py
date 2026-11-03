"""从固定作者源码和已有室内权重导出 S100 静态 ONNX."""

import sys
from types import MethodType

import torch
import onnx

from context import context


def fixed_position(encoder, value, height, width):
    """对已固定的矩形输入直接使用预先插值的位置编码."""
    return encoder.pos_embed


class StaticAttention(torch.nn.Module):
    """使用原 QKV 参数执行四维注意力,减少导出图的维度变换."""

    def __init__(self, original):
        """复用官方注意力参数,不改变训练权重."""
        super().__init__()
        self.qkv = original.qkv
        self.proj = original.proj
        self.heads = original.num_heads
        self.scale = original.scale

    def forward(self, value):
        """在固定输入契约下计算多头注意力."""
        batch, tokens, channels = value.shape
        query, key, content = self.qkv(value).chunk(3, dim=-1)
        query = query.reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        key = key.reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        content = content.reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        output = torch.matmul(torch.softmax(scores, dim=-1), content)
        return self.proj(output.transpose(1, 2).reshape(batch, tokens, channels))


def main():
    """加载度量深度模型,核对静态图改写并导出 FP32 ONNX."""
    ctx = context()
    required = ctx.source / "depth_anything_v2/dpt.py"
    if not required.is_file():
        raise FileNotFoundError("NAS tools 中缺少已固定版本的 metric_depth 官方源码.")
    sys.path.insert(0, str(ctx.source))
    from depth_anything_v2.dpt import DepthAnythingV2

    settings = ctx.config["model"]
    model = DepthAnythingV2(
        encoder=settings["encoder"], features=settings["features"],
        out_channels=settings["out_channels"], max_depth=settings["max_depth"])
    weights = ctx.output / settings["weights"]
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    shape = ctx.config["export"]["input_shape"]
    height, width = shape[-2:]
    sample = torch.randn(shape, generator=torch.Generator().manual_seed(17))
    print("[导出 1/2] 固定位置编码与注意力图,检查改写前后输出.", flush=True)
    with torch.no_grad():
        original = model(sample)
        tokens = torch.zeros((1, height // 14 * (width // 14) + 1, model.pretrained.embed_dim))
        position = model.pretrained.interpolate_pos_encoding(tokens, height, width)
        model.pretrained.pos_embed = torch.nn.Parameter(position)
        model.pretrained.interpolate_pos_encoding = MethodType(fixed_position, model.pretrained)
        for block in model.pretrained.blocks:
            block.attn = StaticAttention(block.attn)
        revised = model(sample)
        if not torch.isfinite(revised).all() or (original - revised).abs().max().item() > 1e-4:
            raise ValueError("静态图改写与原模型不一致,停止导出.")
        ctx.output.mkdir(parents=True, exist_ok=True)
        print("[导出 2/2] 写出静态 FP32 米制深度 ONNX.", flush=True)
        output = ctx.output / "depth_anything_v2_metric_hypersim_vits.onnx"
        torch.onnx.export(
            model, sample, str(output),
            input_names=["image"], output_names=["metric_depth"],
            opset_version=ctx.config["export"]["opset_version"],
            do_constant_folding=True, dynamo=False)
    graph = onnx.load(str(output))
    # IR 是模型容器格式版本,与算子集 opset 版本分别设置.
    target_ir = ctx.config["export"]["ir_version"]
    if graph.ir_version > target_ir:
        raise ValueError("导出器产生了更高 IR,不能仅改版本号降级模型.")
    graph.ir_version = target_ir
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    print(f"[导出完成] IR={graph.ir_version},opset={graph.opset_import[0].version},文件={output}", flush=True)


if __name__ == "__main__":
    main()
