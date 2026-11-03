"""S100 固定子集任务精度、少量数值对比与展示示例."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from runtime import Runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import (NYUDataset, fingerprint, preprocess, restore_depth,
                   sample_identity, save_json, score_depth)


def main():
    """完整运行选定百张样本,验证覆盖后发布唯一子集报告."""
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if fingerprint(root / config["artifact"]) != config["identity"]["hbm_sha256"]:
        raise ValueError("板端 HBM 与部署配置不一致.")
    dataset = NYUDataset(config["dataset_directory"], config["evaluation"], split="subset")
    ids = [int(index) + 1 for index in dataset.test]
    if ids != config["compile_sample_ids"]:
        raise ValueError("两端子集覆盖不一致.")
    if sample_identity(dataset, dataset.test) != config["compile_dataset_identity"]:
        raise ValueError("板端子集 RGB/深度内容与浮点参考不一致.")
    path = root / config["compile_reference"]
    if fingerprint(path) != config["compile_reference_sha256"]:
        raise ValueError("少量数值参考身份不一致.")
    reference = np.load(path, allow_pickle=False)
    if reference["ids"].tolist() != config["numerical_sample_ids"]:
        raise ValueError("数值参考覆盖不一致.")
    engine = Runtime(root / config["artifact"], config["export"]["input_shape"])
    output = root / "examples/output"
    output.mkdir(parents=True, exist_ok=True)
    metrics, numerical = [], []
    try:
        for step, index in enumerate(dataset.test, 1):
            image, truth = dataset.read(int(index))
            raw = engine.run(preprocess(image, config["export"]["input_shape"]))
            depth = restore_depth(raw, *image.shape[:2])
            metrics.append(score_depth(depth, truth, config["evaluation"]))
            sample_id = int(index) + 1
            if sample_id in config["numerical_sample_ids"]:
                position = config["numerical_sample_ids"].index(sample_id)
                expected = reference["depth"][position]
                if raw.shape != expected.shape:
                    raise ValueError("HBM 与 ONNX 输出形状不一致.")
                actual = raw.astype(np.float64).ravel()
                baseline = expected.astype(np.float64).ravel()
                norm = np.linalg.norm(actual) * np.linalg.norm(baseline)
                magnitude = np.sqrt(np.mean(baseline ** 2))
                if norm <= 0 or magnitude <= 0:
                    raise ValueError("输出范数无效,停止数值评分.")
                rmse = float(np.sqrt(np.mean((actual - baseline) ** 2)))
                row = {"sample_id": sample_id,
                       "cosine_similarity": float(np.dot(actual, baseline) / norm),
                       "rmse_m": rmse, "nrmse_percent": float(100 * rmse / magnitude)}
                numerical.append(row)
                print(f"[输出数值] NYU {sample_id}: CosSim={row['cosine_similarity']:.8f},"
                      f"RMSE={rmse:.6f} 米,NRMSE={row['nrmse_percent']:.4f}%", flush=True)
                scale = np.clip(depth / config["model"]["max_depth"], 0, 1)
                color = cv2.applyColorMap(np.round(scale * 255).astype(np.uint8),
                                         cv2.COLORMAP_INFERNO)
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                target = output / f"sample_{sample_id:04d}_depth.jpg"
                if not cv2.imwrite(str(target), np.concatenate([bgr, color], axis=1)):
                    raise OSError("示例保存失败.")
            if step % 10 == 0 or step == len(ids):
                print(f"[HBM 子集精度] {step}/{len(ids)}", flush=True)
    finally:
        reference.close()
        dataset.close()
    if len(metrics) != config["evaluation"]["compile_samples"] or len(numerical) != 3:
        raise ValueError("子集任务或数值对比覆盖不完整,不发布报告.")
    accuracy = {}
    for key, baseline in config["compile_onnx_metrics"].items():
        actual = float(np.mean([row[key] for row in metrics]))
        higher = key.startswith("delta")
        loss = baseline - actual if higher else actual - baseline
        accuracy[key] = {"fp32_accuracy": baseline, "npu_accuracy": actual,
                         "higher_is_better": higher,
                         "accuracy_loss": loss,
                         "unit": "m" if key == "rmse_m" else "dimensionless"}
        if higher:
            accuracy[key]["accuracy_loss_percentage_points"] = 100 * loss
        print(f"[子集精度] {key}: ONNX={baseline:.6f},HBM={actual:.6f},"
              f"损失={loss:+.6f}", flush=True)
    report = {
        "model": root.name, "platform": "s100", "evaluation_scope": "subset",
        "status": "complete", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "samples": len(metrics), "sample_unit": "image", "sample_ids": ids,
        "sampling": "fixed_task_subset_v2", "identity": config["identity"],
        "dataset_identity": config["compile_dataset_identity"],
        "quantization": config["quantization"], "protocol": config["evaluation"],
        "aggregation": "mean_per_image", "accuracy": accuracy,
        "loss_definition": "正值表示 HBM 相对 ONNX 精度下降;误差指标取 HBM-ONNX,delta 指标取 ONNX-HBM",
        "numerical_comparison": {"samples": numerical,
                                 "scope": "原始输出,NRMSE=100*RMSE/RMS(ONNX)"},
        "missing": ["未运行全量654张评价、性能或RSS测试"],
    }
    save_json(root / "results/summary_test.json", report)
    print("[子集完成] 100 张任务精度及三张数值误差已保存至 results/summary_test.json.", flush=True)


if __name__ == "__main__":
    main()
