"""S100 少量通路检查与示例,不生成精度、性能或 RSS 报告."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np

from runtime import Runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import fingerprint, preprocess, restore_depth, score_depth


def main():
    """只运行部署包内固定示例,按统一米制色阶保存输入和结果."""
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if fingerprint(root / config["artifact"]) != config["identity"]["hbm_sha256"]:
        raise ValueError("板端 HBM 与部署配置不一致.")
    engine = Runtime(root / config["artifact"], config["export"]["input_shape"])
    output = root / "examples/output"
    output.mkdir(parents=True, exist_ok=True)
    inputs = [root / "examples/input" / f"sample_{index:04d}.png"
              for index in config["compile_sample_ids"]]
    if not inputs:
        raise FileNotFoundError("缺少固定示例输入,请重新准备部署包.")
    reference_path = root / config["compile_reference"]
    if fingerprint(reference_path) != config["compile_reference_sha256"]:
        raise ValueError("小批量浮点参考与部署包不一致.")
    reference = np.load(reference_path)
    if reference["ids"].tolist() != config["compile_sample_ids"]:
        reference.close()
        raise ValueError("小批量参考样本编号不一致.")
    comparisons = []
    for index, path in enumerate(inputs, 1):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"示例图片不可读: {path.name}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw = engine.run(preprocess(image, config["export"]["input_shape"]))
        expected = reference["depth"][index - 1]
        if raw.shape != expected.shape:
            raise ValueError("BPU 与 ONNX 输出形状不一致.")
        actual_vector = raw.astype(np.float64).ravel()
        expected_vector = expected.astype(np.float64).ravel()
        norm = np.linalg.norm(actual_vector) * np.linalg.norm(expected_vector)
        magnitude = np.sqrt(np.mean(expected_vector ** 2))
        if norm <= 0 or magnitude <= 0:
            raise ValueError("深度输出范数为零,不能计算有效数值误差.")
        rmse = float(np.sqrt(np.mean((actual_vector - expected_vector) ** 2)))
        cosine = float(np.dot(actual_vector, expected_vector) / norm)
        depth = restore_depth(raw, *image.shape[:2])
        baseline = restore_depth(expected, *image.shape[:2])
        truth = reference["truth"][index - 1]
        npu_metrics = score_depth(depth, truth, config["evaluation"])
        onnx_metrics = score_depth(baseline, truth, config["evaluation"])
        comparisons.append((npu_metrics, onnx_metrics))
        print(f"[小批量数值] {path.stem}: CosSim={cosine:.8f},RMSE={rmse:.6f} 米,"
              f"NRMSE={rmse / magnitude:.6f}", flush=True)
        scale = np.clip(depth / config["model"]["max_depth"], 0, 1)
        color = cv2.applyColorMap(np.round(scale * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        destination = output / f"{path.stem}_depth.jpg"
        if not cv2.imwrite(str(destination), np.concatenate([bgr, color], axis=1)):
            raise OSError("示例保存失败.")
        print(f"[通路与示例] {index}/{len(inputs)}: {destination.name},色阶固定 0-20 米.", flush=True)
    reference.close()
    for key in comparisons[0][0]:
        npu = float(np.mean([row[0][key] for row in comparisons]))
        onnx = float(np.mean([row[1][key] for row in comparisons]))
        print(f"[三张样本指标] {key}: ONNX={onnx:.6f},BPU={npu:.6f},差值(BPU-ONNX)={npu-onnx:+.6f}", flush=True)
    print("[小批量完成] 仅三个固定样本,不代表正式全量精度;未统计性能或 RSS.", flush=True)


if __name__ == "__main__":
    main()
