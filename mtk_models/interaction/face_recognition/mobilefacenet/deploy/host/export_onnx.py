"""从锁定的 MobileFaceNet PyTorch 权重导出 ONNX。"""

import argparse
import sys
from pathlib import Path

import torch


from tools.storage import model_storage

SOURCE_ROOT = model_storage(Path(__file__).resolve().parents[2]) / "original/upstream"
sys.path.insert(0, str(SOURCE_ROOT))

from mobilefacenet import MobileFaceNet  # noqa: E402

from tools.deploy import configure_parser
from tools.storage import storage_roots


def export_model(weights: Path, output: Path) -> None:
    """加载原始权重并导出固定输入的特征提取图。"""
    model = MobileFaceNet().eval()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    sample = torch.zeros((1, 3, 112, 112), dtype=torch.float32)
    with torch.no_grad():
        embedding = model(sample)
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(f"原始输出结构异常: {tuple(embedding.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, sample, str(output), opset_version=13,
                      input_names=["face"], output_names=["embedding"],
                      do_constant_folding=True)
    print(f"[OK] PyTorch 输出: {tuple(embedding.shape)}")
    print(f"[OK] ONNX: {output}")


def main() -> None:
    """解析参数并执行导出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    configure_parser(parser, __file__, {
        "weights": lambda c: Path(c.config["model"]["weights"]),
        "output": lambda c: c.output / "mobilefacenet.onnx",
    })
    args = parser.parse_args()
    export_model(args.weights, args.output)


if __name__ == "__main__":
    main()
