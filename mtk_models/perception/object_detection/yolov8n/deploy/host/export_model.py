"""离线导出 YOLOv8n 原始检测头,保留官方前向结果用于一致性检查."""

import argparse
import sys
import copy
import json
import types
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics

# 复用板端预处理,保持校准与推理输入一致.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from yolov8_utils import OUTPUT_NAMES, OUTPUT_SHAPES, decode_heads, preprocess

from tools.deploy import configure_parser
from tools.storage import storage_roots


def raw_forward(self, features):
    """导出六个原始检测头,DFL 和 NMS 留在 CPU."""
    outputs = []
    for index in range(3):
        outputs.extend((self.cv2[index](features[index]),
                        self.cv3[index](features[index])))


    return tuple(outputs)


def load_model(weights):
    """只加载已在本地放置的可信官方权重,禁止自动下载与依赖安装."""
    if ultralytics.__version__ != "8.0.111":
        raise RuntimeError(f"要求 ultralytics==8.0.111,实际 {ultralytics.__version__}.")
    if not weights.is_file():
        raise FileNotFoundError(f"请离线放置官方权重: {weights}")
    checkpoint = torch.load(str(weights), map_location="cpu")
    model = checkpoint.get("ema") or checkpoint["model"]
    model = model.float().cpu().eval()
    # 兼容早期 PyTorch 序列化的 Upsample; None 保持旧版默认缩放语义.
    for module in model.modules():
        if isinstance(module, torch.nn.Upsample) and not hasattr(
                module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    head = model.model[-1]
    if (head.nc, head.reg_max, head.nl) != (80, 16, 3):
        raise ValueError("权重不是预期的 YOLOv8n 检测结构.")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if not 3_000_000 < parameters < 3_500_000:
        raise ValueError(f"要求 YOLOv8n,参数量不匹配: {parameters}.")
    if list(head.stride.tolist()) != [8, 16, 32]:
        raise ValueError("检测头 stride 不匹配.")
    head.export = False
    return model


def main():
    """检查前向等价并导出固定 opset 13 模型."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    ctx = configure_parser(parser, __file__, {
        "weights": lambda c: Path(c.config["model"]["weights"]),
        "image": lambda c: c.calibration / "000000000139.jpg",
        "output_dir": lambda c: c.output,
    })
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[1/4] 离线加载官方 YOLOv8n 权重.", flush=True)
    original = load_model(args.weights)
    raw_model = copy.deepcopy(original)
    raw_model.model[-1].forward = types.MethodType(raw_forward, raw_model.model[-1])
    tensor, geometry = preprocess(cv2.imread(str(args.image)))
    with torch.no_grad():
        reference = original(torch.from_numpy(tensor))
        outputs = raw_model(torch.from_numpy(tensor))
    arrays = [value.numpy() for value in outputs]
    boxes, scores = decode_heads(arrays)
    xywh = np.concatenate(((boxes[:, :2] + boxes[:, 2:]) / 2,
                           boxes[:, 2:] - boxes[:, :2]), axis=1)
    decoded = np.concatenate((xywh, scores), axis=1).T[None]
    expected = reference[0].numpy()
    np.testing.assert_allclose(decoded, expected, atol=2e-4, rtol=2e-4)

    print("[2/4] 原始头 CPU 解码与官方前向一致,导出 ONNX.", flush=True)
    onnx_path = args.output_dir / "yolov8n.onnx"
    torch.onnx.export(raw_model, torch.from_numpy(tensor), str(onnx_path),
                      input_names=["images"], output_names=OUTPUT_NAMES,
                      opset_version=13, do_constant_folding=True)
    exported = onnx.load(str(onnx_path))
    onnx.checker.check_model(exported)
    shapes = [[dim.dim_value for dim in item.type.tensor_type.shape.dim]
              for item in exported.graph.output]
    if shapes != OUTPUT_SHAPES:
        raise ValueError(f"ONNX 输出形状错误: {shapes}.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx_path), options,
                                   providers=["CPUExecutionProvider"])
    actual = session.run(OUTPUT_NAMES, {"images": tensor})
    for expected_head, actual_head in zip(arrays, actual):
        np.testing.assert_allclose(actual_head, expected_head, atol=2e-4, rtol=2e-4)
    print("[3/4] 保存 PyTorch 输入与输出基线.", flush=True)
    np.savez(ctx.work / "pytorch_reference.npz", images=tensor,
             **dict(zip(OUTPUT_NAMES, arrays)))
    manifest = {
        "weights": str(args.weights),
        "implementation": "ultralytics/ultralytics@v8.0.111",
        "torch": torch.__version__, "onnx": onnx.__version__,
        "image": str(args.image), "geometry": geometry,
        "output_names": OUTPUT_NAMES, "output_shapes": OUTPUT_SHAPES,
        "raw_decode_max_absolute_error": float(np.max(np.abs(decoded - expected))),
        "onnx_runtime_verified": True,
    }
    (ctx.work / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[4/4] 导出和 ONNX 数值校验完成,NPU 验证仍需单独执行.", flush=True)


if __name__ == "__main__":
    main()
