"""在 S100 执行同协议精度、Runtime 调用性能与进程峰值 RSS."""

from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np

from runtime import Runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import NYUDataset, fingerprint, preprocess, restore_depth, save_json, score_depth


def main():
    """核对模型和评价资源,完整评分后只发布唯一 summary.json."""
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    reference = json.loads((root / "build/reference/reference.json").read_text(encoding="utf-8"))
    if (reference["identity"] != config["identity"]
            or reference["evaluation"] != config["evaluation"]
            or reference["export"] != config["export"]):
        raise ValueError("浮点参考与部署包版本或协议不一致.")
    if fingerprint(root / config["artifact"]) != config["identity"]["hbm_sha256"]:
        raise ValueError("板端 HBM 与部署配置不一致.")
    dataset_root = root / config["dataset_directory"]
    print("[评价 1/3] 检查原始数据与参考身份.", flush=True)
    dataset = NYUDataset(dataset_root, config["evaluation"], split="test")
    if dataset.identity() != reference["dataset_identity"]:
        dataset.close()
        raise ValueError("板端 PNG/NPY 数据与浮点参考不一致.")
    if [int(index) + 1 for index in dataset.test] != reference["sample_ids"]:
        dataset.close()
        raise ValueError("板端评价样本与浮点参考覆盖不一致.")
    engine = Runtime(root / config["artifact"], config["export"]["input_shape"])
    metrics = []
    try:
        for step, index in enumerate(dataset.test, 1):
            image, truth = dataset.read(int(index))
            prediction = engine.run(preprocess(image, config["export"]["input_shape"]))
            restored = restore_depth(prediction, *image.shape[:2])
            metrics.append(score_depth(restored, truth, config["evaluation"]))
            if step % 25 == 0 or step == len(dataset.test):
                print(f"[评价 2/3] NPU 精度 {step}/{len(dataset.test)}", flush=True)
        image, _ = dataset.read(int(dataset.test[0]))
    finally:
        dataset.close()
    tensor = preprocess(image, config["export"]["input_shape"])
    for _ in range(config["evaluation"]["warmup"]):
        engine.run(tensor)
    calls, total = [], []
    iterations = config["evaluation"]["iterations"]
    for index in range(iterations):
        begin = perf_counter()
        current = preprocess(image, config["export"]["input_shape"])
        start = perf_counter()
        prediction = engine.run(current)
        end = perf_counter()
        restore_depth(prediction, *image.shape[:2])
        total.append((perf_counter() - begin) * 1000)
        calls.append((end - start) * 1000)
        if (index + 1) % 10 == 0 or index + 1 == iterations:
            print(f"[评价 3/3] 性能 {index + 1}/{iterations}", flush=True)
    accuracy = {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}
    summary = {
        "status": "partial", "platform": "s100", "march": "nash-e",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "identity": config["identity"], "dataset_identity": reference["dataset_identity"],
        "scope": {"dataset": "NYU_Depth_V2", "split": "test", "samples": len(metrics), "sample_unit": "image"},
        "protocol": config["evaluation"],
        "accuracy": {
            "aggregation": "mean_per_image", "npu": accuracy, "onnx": reference["metrics"],
            "difference_npu_minus_onnx": {key: accuracy[key] - reference["metrics"][key] for key in accuracy},
        },
        "performance": {
            "warmup": config["evaluation"]["warmup"], "iterations": iterations,
            "runtime_call_mean_ms": float(np.mean(calls)),
            "runtime_call_p95_ms": float(np.percentile(calls, 95)),
            "end_to_end_mean_ms": float(np.mean(total)),
            "runtime_scope": "同步 Python Runtime 调用,含绑定与输出转换,不等于纯 BPU 设备耗时",
            "end_to_end_scope": "预处理+Runtime+深度尺寸恢复,不含图片读取与展示保存",
        },
        "memory": {"peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   "scope": "整个 Python 评价进程,含模型、数据读取与前后处理;不等于驱动总内存"},
        "missing": ["纯 BPU 设备计时尚未采集"],
    }
    save_json(root / "results/summary.json", summary)
    print("[评价完成] 654 帧同协议结果与性能/RSS 已保存至 results/summary.json.", flush=True)


if __name__ == "__main__":
    main()
