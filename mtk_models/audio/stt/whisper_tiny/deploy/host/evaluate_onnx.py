"""读取模型配置,直接调用公共 ONNX 精度评测函数."""

import os
from tools.accuracy.evaluate_onnx import evaluate_model
from tools.deploy import arguments, reference_path


def main():
    """使用本模型的固定 ONNX 和数据协议,不再启动中转 Python 进程."""
    ctx = arguments(__file__)
    evaluate_model(model="whisper_tiny", onnx=ctx.output / "tiny_encoder.onnx",
                   dataset_root=ctx.dataset, output=reference_path(ctx),
                   decoder_onnx=ctx.output / "tiny_decoder_step.onnx")


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
