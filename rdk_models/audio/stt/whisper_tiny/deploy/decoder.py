"""主机参考、量化校准和板端共用 Whisper 单 Token 缓存协议."""

import numpy as np


CACHE_SIZE = 256
MAX_TOKEN_LENGTH = 448
VOCAB_SIZE = 51865
CACHE_NAMES = [f"{kind}_cache_{layer}_in"
               for layer in range(4) for kind in ("key", "value")]
INPUT_SHAPES = {
    "token_onehot": (1, VOCAB_SIZE),
    "audio_features": (1, 1500, 384),
    "position_weights": (1, MAX_TOKEN_LENGTH),
    "cache_update_mask": (1, 1, 1, CACHE_SIZE),
    "attention_mask": (1, 1, 1, CACHE_SIZE),
    **{name: (6, 1, 64, CACHE_SIZE) for name in CACHE_NAMES},
}
OUTPUT_SHAPES = {"logits": (1, VOCAB_SIZE),
                 **{name.removesuffix("_in") + "_out": shape
                    for name, shape in INPUT_SHAPES.items() if name in CACHE_NAMES}}


def feeds_for_step(token, position, features, caches):
    """使用绝对位置编码,Cache 满后左移一格,保留最近 256 个 Token."""
    if not 0 <= position < MAX_TOKEN_LENGTH or not 0 <= token < VOCAB_SIZE:
        raise ValueError(f"Decoder Token 或位置越界: token={token}, position={position}.")
    slot = min(position, CACHE_SIZE - 1)
    shifted = {name: (np.concatenate((value[..., 1:],
                                     np.zeros_like(value[..., :1])), axis=-1)
                      if position >= CACHE_SIZE else value)
               for name, value in caches.items()}
    feeds = {name: np.zeros(shape, np.float32) for name, shape in INPUT_SHAPES.items()
             if name not in CACHE_NAMES}
    feeds["audio_features"] = features
    feeds["token_onehot"][0, token] = 1
    feeds["position_weights"][0, position] = 1
    feeds["cache_update_mask"][..., slot] = 1
    feeds["attention_mask"].fill(-100)
    feeds["attention_mask"][..., :slot + 1] = 0
    return {**feeds, **shifted}


def next_token(logits, config, first=False):
    """应用同一组官方禁止 Token 和首 Token 过滤规则后贪心选词."""
    values = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
    for key in (("suppress", "suppress_first") if first else ("suppress",)):
        indices = [int(value) for value in config[key].split(",") if value]
        values[indices] = -np.inf
    token = int(values.argmax())
    if not np.isfinite(values[token]):
        raise ValueError("Decoder 没有有效 Token.")
    return token
