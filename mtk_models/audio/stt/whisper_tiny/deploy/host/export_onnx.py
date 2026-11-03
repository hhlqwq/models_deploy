"""从 OpenAI 官方 tiny.pt 导出固定 Shape Encoder 与 Decoder-Step ONNX."""

import argparse
from importlib import metadata
import shutil
from pathlib import Path
from typing import Any

from torch import nn
import numpy as np
import torch
import whisper
from whisper.model import MultiHeadAttention
from whisper.tokenizer import get_tokenizer

from tools.deploy import configure_parser
from tools.storage import storage_roots


MAX_DECODE_TOKENS = 256
MASK_NEGATIVE = -100.0


class ExportLayerNorm(nn.Module):
    """按行缩放后计算 LayerNorm,避免 INT16 平方张量丢失小方差."""

    def __init__(self, original):
        """保留官方仿射参数和 epsilon,不重新训练或修改权重."""
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.eps = original.eps

    def forward(self, tensor):
        """保持 epsilon 等价,将每行中心化数值限制在 [-1, 1] 再平方."""
        values = tensor.float()
        centered = values - values.mean(dim=-1, keepdim=True)
        # 大幅值行不能占满整个平方张量的量程,否则小方差行会被量化成零.
        magnitude = centered.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
        scaled = centered / magnitude
        variance = (scaled * scaled).mean(dim=-1, keepdim=True)
        epsilon = self.eps / (magnitude * magnitude)
        normalized = scaled * torch.rsqrt(variance + epsilon)
        return (normalized * self.weight + self.bias).to(tensor.dtype)


def rewrite_layer_norm(module):
    """递归替换官方 LayerNorm,逐层检查等价性,避免浮点算子融合回退."""
    count = 0
    generator = torch.Generator().manual_seed(17)
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            replacement = ExportLayerNorm(child)
            values = torch.randn(2, 3, child.normalized_shape[-1], generator=generator)
            # 同时覆盖小方差、常量行和真实 Encoder 中出现的大幅值特征.
            values[0, 0] *= 0.01
            values[0, 1] = 0
            values[1, 0, 0] = 250
            with torch.inference_mode():
                expected, actual = child(values), replacement(values)
            difference = float((expected - actual).abs().max())
            if not torch.allclose(expected, actual, atol=2e-5, rtol=2e-5):
                raise ValueError(f"LayerNorm 等价改写误差超限: {name}, max_abs={difference}.")
            setattr(module, name, replacement)
            count += 1
        else:
            count += rewrite_layer_norm(child)
    return count


class EncoderWrapper(nn.Module):
    """封装 Whisper Encoder,固定返回音频特征."""

    def __init__(self, encoder: nn.Module) -> None:
        """初始化 Encoder 包装器.

        Args:
            encoder: OpenAI Whisper Encoder 模块.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """运行固定 30 秒 Log-Mel Encoder.

        Args:
            mel: `[1, 80, 3000]` FP32 Log-Mel.

        Returns:
            `[1, 1500, n_audio_state]` 音频特征.
        """
        return self.encoder(mel)


class DecoderStepWrapper(nn.Module):
    """将 OpenAI Decoder 改写为固定 KV Cache 的单 Token 图."""

    def __init__(self, decoder: nn.Module, max_tokens: int) -> None:
        """初始化 Decoder-Step 包装器.

        Args:
            decoder: OpenAI Whisper TextDecoder 模块.
            max_tokens: 静态 KV Cache Token 上限.
        """
        super().__init__()
        self.decoder = decoder
        self.max_tokens = max_tokens

    @staticmethod
    def _project_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        """将 `[B, T, C]` 投影结果拆为多头布局.

        Args:
            tensor: 注意力投影张量.
            heads: 注意力头数.

        Returns:
            `[B, H, T, D]` 张量.
        """
        batch, tokens, channels = tensor.shape
        return tensor.reshape(batch, tokens, heads, channels // heads).permute(
            0, 2, 1, 3)

    def _self_attention(
            self, attention: nn.Module, hidden: torch.Tensor,
            past_key: torch.Tensor, past_value: torch.Tensor,
            cache_update_mask: torch.Tensor,
            attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
        """执行带固定长度 KV Cache 的自注意力.

        Args:
            attention: OpenAI MultiHeadAttention 模块.
            hidden: 当前 Token 隐状态.
            past_key: 固定长度历史 Key Cache.
            past_value: 固定长度历史 Value Cache.
            cache_update_mask: KV Cache 当前写入位置的 One-Hot Mask.
            attention_mask: 屏蔽未写入 Cache 的加法 Mask.

        Returns:
            注意力输出、更新后的 Key Cache 和 Value Cache.
        """
        heads = attention.n_head
        query = self._project_heads(
            attention.query(hidden), heads).permute(1, 0, 2, 3)
        key_new = self._project_heads(
            attention.key(hidden), heads).permute(1, 0, 3, 2)
        value_new = self._project_heads(
            attention.value(hidden), heads).permute(1, 0, 3, 2)
        key = (past_key * (1.0 - cache_update_mask) +
               key_new * cache_update_mask)
        value = (past_value * (1.0 - cache_update_mask) +
                 value_new * cache_update_mask)
        scale = (query.shape[-1]) ** -0.25
        scores = (query * scale) @ (key * scale)
        scores = scores + attention_mask
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        context = probabilities @ value.transpose(-1, -2)
        context = context.permute(1, 2, 0, 3).flatten(start_dim=2)
        return attention.out(context), key, value

    def _cross_attention(
            self, attention: nn.Module, hidden: torch.Tensor,
            audio_features: torch.Tensor) -> torch.Tensor:
        """执行当前 Token 到 Encoder 特征的交叉注意力.

        Args:
            attention: OpenAI MultiHeadAttention 模块.
            hidden: 当前 Token 隐状态.
            audio_features: Encoder 输出.

        Returns:
            交叉注意力输出.
        """
        heads = attention.n_head
        query = self._project_heads(attention.query(hidden), heads)
        key = self._project_heads(attention.key(audio_features), heads)
        value = self._project_heads(attention.value(audio_features), heads)
        scale = (query.shape[-1]) ** -0.25
        scores = (query * scale) @ (key * scale).transpose(-1, -2)
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        context = probabilities @ value
        context = context.permute(0, 2, 1, 3).flatten(start_dim=2)
        return attention.out(context)

    def forward(
            self, token_onehot: torch.Tensor, audio_features: torch.Tensor,
            position_weights: torch.Tensor, cache_update_mask: torch.Tensor,
            attention_mask: torch.Tensor,
            *kv_caches: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """运行一个 Decoder Token 并返回 Logits 和完整新 Cache.

        Args:
            token_onehot: `[1, vocab]` FP32 Token One-Hot.
            audio_features: `[1, 1500, n_audio_state]` Encoder 输出.
            position_weights: `[1, n_text_ctx]` 位置 One-Hot 权重.
            cache_update_mask: `[1, 1, 1, M]` KV Cache 写入 Mask.
            attention_mask: `[1, 1, 1, M]` 加法注意力 Mask.
            *kv_caches: 每层一对 `[H, 1, D, M]` Key/Value Cache.

        Returns:
            `[1, vocab]` Logits 与逐层更新后的固定 KV Cache.
        """
        position = position_weights @ self.decoder.positional_embedding
        token_embedding = token_onehot @ self.decoder.token_embedding.weight
        hidden = token_embedding.unsqueeze(1) + position.unsqueeze(1)
        next_caches = []
        for index, block in enumerate(self.decoder.blocks):
            residual = hidden
            attention_output, key, value = self._self_attention(
                block.attn, block.attn_ln(hidden), kv_caches[index * 2],
                kv_caches[index * 2 + 1], cache_update_mask, attention_mask)
            hidden = residual + attention_output
            residual = hidden
            hidden = residual + self._cross_attention(
                block.cross_attn, block.cross_attn_ln(hidden), audio_features)
            hidden = hidden + block.mlp(block.mlp_ln(hidden))
            next_caches.extend((key, value))
        hidden = self.decoder.ln(hidden)
        logits = hidden @ self.decoder.token_embedding.weight.transpose(0, 1)
        return (logits[:, 0, :].float(), *next_caches)


def load_model(weights: Path) -> Any:
    """从已校验的本地 checkpoint 加载 OpenAI Whisper.

    Args:
        weights: `tiny.pt` 路径.

    Returns:
        CPU FP32 Eval 模式 Whisper 模型.
    """

    # Torch 2.0 ONNX 不支持 aten::scaled_dot_product_attention,强制使用
    # OpenAI 官方的 MatMul/Softmax 等价分支,数值由后续三方对齐验证.
    MultiHeadAttention.use_sdpa = False
    model = whisper.load_model(str(weights), device="cpu")
    model = model.float().eval()
    count = rewrite_layer_norm(model)
    if count != 22:
        raise ValueError(f"Whisper-Tiny 应有 22 个 LayerNorm,实际替换 {count} 个.")
    print(f"[归一化] {count} 个 LayerNorm 等价改写检查通过,使用逐行缩放的整数算子路线.", flush=True)
    return model


def export_models(weights: Path, output_dir: Path,
                  max_tokens: int) -> None:
    """导出 Encoder 与 Decoder-Step ONNX.

    Args:
        weights: OpenAI 官方权重路径.
        output_dir: ONNX 输出目录.
        max_tokens: 静态 KV Cache 长度.
    """
    if max_tokens != 256:
        raise ValueError("当前交付固定 cacheSize=256,Decoder 绝对位置上限为 448.")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[1/3] 加载 OpenAI Whisper-Tiny FP32 模型.")
    model = load_model(weights)
    dimensions = model.dims
    if (dimensions.n_text_ctx, dimensions.n_text_state, dimensions.n_text_head,
            dimensions.n_text_layer, dimensions.n_vocab) != (448, 384, 6, 4, 51865):
        raise ValueError(f"Whisper-Tiny 官方维度不符: {dimensions}.")
    encoder = EncoderWrapper(model.encoder).eval()
    decoder = DecoderStepWrapper(model.decoder, max_tokens).eval()
    mel = torch.zeros(1, dimensions.n_mels, 3000, dtype=torch.float32)

    print("[2/3] 导出固定 30 秒 Encoder ONNX.")
    encoder_path = output_dir / "tiny_encoder.onnx"
    torch.onnx.export(
        encoder, mel, str(encoder_path), input_names=["mel"],
        output_names=["audio_features"], opset_version=17,
        do_constant_folding=True)

    print("[3/3] 导出固定 KV Cache Decoder-Step ONNX.")
    with torch.no_grad():
        audio_features = encoder(mel)
    head_dim = dimensions.n_text_state // dimensions.n_text_head
    kv_caches = []
    for _ in range(dimensions.n_text_layer):
        kv_caches.extend((
            torch.zeros(dimensions.n_text_head, 1, head_dim, max_tokens),
            torch.zeros(dimensions.n_text_head, 1, head_dim, max_tokens),
        ))
    token_onehot = torch.zeros(1, dimensions.n_vocab, dtype=torch.float32)
    token_onehot[0, 0] = 1.0
    position_weights = torch.zeros(
        1, dimensions.n_text_ctx, dtype=torch.float32)
    position_weights[0, 0] = 1.0
    cache_update_mask = torch.zeros(1, 1, 1, max_tokens)
    cache_update_mask[..., 0] = 1.0
    attention_mask = torch.full(
        (1, 1, 1, max_tokens), MASK_NEGATIVE, dtype=torch.float32)
    attention_mask[..., 0] = 0.0
    decoder_path = output_dir / "tiny_decoder_step.onnx"
    torch.onnx.export(
        decoder,
        (token_onehot, audio_features, position_weights, cache_update_mask,
         attention_mask, *kv_caches),
        str(decoder_path),
        input_names=[
            "token_onehot", "audio_features", "position_weights",
            "cache_update_mask", "attention_mask",
            *[
                f"{cache_type}_cache_{layer}_in"
                for layer in range(dimensions.n_text_layer)
                for cache_type in ("key", "value")
            ],
        ],
        output_names=[
            "logits",
            *[
                f"{cache_type}_cache_{layer}_out"
                for layer in range(dimensions.n_text_layer)
                for cache_type in ("key", "value")
            ],
        ],
        opset_version=17,
        do_constant_folding=True)
    print(f"[OK] Encoder: {encoder_path}")
    print(f"[OK] Decoder-Step: {decoder_path}")


SOURCE_FILES = (
    "tokenizer.py",
    "assets/multilingual.tiktoken",
    "normalizers/__init__.py",
    "normalizers/basic.py",
    "normalizers/english.py",
    "normalizers/english.json",
)


def export_assets(output_dir: Path) -> None:
    """导出官方 80 Mel 滤波器和英文 Greedy 解码规则,不运行推理."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filters = whisper.audio.mel_filters(torch.device("cpu"), 80)
    if tuple(filters.shape) != (80, 201):
        raise ValueError(f"Whisper Mel 滤波器形状异常: {filters.shape}.")
    np.asarray(filters, dtype="<f4").tofile(output_dir / "mel_filters_f32.bin")

    tokenizer = get_tokenizer(multilingual=True, language="en",
                              task="transcribe")
    suppressed = set(tokenizer.non_speech_tokens)
    suppressed.update((tokenizer.transcribe, tokenizer.translate,
                       tokenizer.sot, tokenizer.sot_prev, tokenizer.sot_lm))
    if tokenizer.no_speech is not None:
        suppressed.add(tokenizer.no_speech)
    suppress_first = set(tokenizer.encode(" "))
    suppress_first.add(tokenizer.eot)
    join = lambda values: ",".join(map(str, values))
    config = "\n".join((
        f"initial={join(tokenizer.sot_sequence_including_notimestamps)}",
        f"suppress={join(sorted(suppressed))}",
        f"suppress_first={join(sorted(suppress_first))}",
        f"eot={tokenizer.eot}",
        "cacheSize=256",
        "maxTokenLength=448",
        f"max_generated_tokens={448 - len(tokenizer.sot_sequence_including_notimestamps)}",
    )) + "\n"
    (output_dir / "decode_config.txt").write_text(
        config, encoding="utf-8")
    print(f"[OK] Whisper 静态资源: {output_dir}.")


def export_vendor(output_dir: Path) -> None:
    """复制固定版本的原版文本组件。"""
    version = metadata.version("openai-whisper")
    if version != "20250625":
        raise ValueError(f"OpenAI Whisper 版本不符: {version}。")
    source_root = Path(whisper.__file__).resolve().parent
    package_root = output_dir / "whisper"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    for relative in SOURCE_FILES:
        source = source_root / relative
        destination = package_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[EXPORT] {relative}", flush=True)


def parse_args() -> argparse.Namespace:
    """解析命令行参数.

    Returns:
        命令行参数.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=MAX_DECODE_TOKENS)
    configure_parser(parser, __file__, {
        "weights": lambda c: Path(c.config["model"]["weights"]),
        "output_dir": lambda c: c.output,
        "max_tokens": lambda c: c.config["conversion"]["cacheSize"],
    })
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    export_models(arguments.weights, arguments.output_dir,
                  arguments.max_tokens)
    export_assets(arguments.work_dir)
    export_vendor(arguments.work_dir / "whisper_eval_vendor")
