"""独立全量 3D 检测评价,只发布一个 summary.json."""


import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from dataset import load_frame
from pipeline import Pipeline
from scoring import format_predictions, score_predictions
from tools.board import context
from tools.evaluation import is_test, select_samples
from tools.runtime import peak_rss_kib
from tools.evaluation import accuracy_result, record_selection


def identity(root):
    """绑定完整浮点图,稀疏参数和 Python 协议,拒绝混用旧参考."""
    digest = hashlib.sha256()
    root = Path(root)
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    graphs = {f"bevfusion-det_{name}.onnx" for name in contract["stages"]}
    graphs.update(item["file"] for item in contract["stages"].values()
                  if item["backend"] != "mtk_npu")
    paths = [root / "contract.json", *[root / name for name in sorted(graphs)],
             root / "sparse_weights.npz", root / "geometry.npz"]
    paths.extend(Path(__file__).with_name(name) for name in ("geometry.py", "sparse.py", "dataset.py", "scoring.py"))
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def evaluate_dataset(models, dataset, samples, predictions_path, floating=False):
    """预热后逐帧推理,只在调用方明确执行评价时采集耗时及 RSS."""
    from tqdm import tqdm

    pipeline = Pipeline(models, floating=floating, verbose=False)
    predictions, elapsed, npu_elapsed = {}, [], []
    try:
        if not is_test():
            print("[评价预热] 一帧完整链路,不计入性能统计.", flush=True)
            pipeline(load_frame(dataset, samples[0]))
        for frame in tqdm(samples, desc="BEVFusion FP32" if floating else "BEVFusion CPU+NPU"):
            boxes, scores, labels, milliseconds, npu = pipeline(load_frame(dataset, frame), measure=not is_test())
            predictions[frame["token"]] = format_predictions(frame, boxes, scores, labels)
            elapsed.append(milliseconds)
            npu_elapsed.append(npu)
    finally:
        pipeline.close()
    rss = None if is_test() else peak_rss_kib() / 1024
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    if is_test():
        return predictions, None, None, None
    return predictions, float(np.mean(elapsed)), float(np.mean(npu_elapsed)), rss


def validate_manifest(manifest, data, subset):
    """核对约定划分及完整覆盖,正式评价固定使用 mini_val 的 81 帧."""
    version, split = data["version"], data["split"]
    counts = {("v1.0-mini", "mini_val"): 81, ("v1.0-trainval", "val"): 6019}
    if not subset and (version, split) != ("v1.0-mini", "mini_val"):
        raise ValueError("本模型正式评价约定为 v1.0-mini/mini_val 的 81 帧,2 个场景.")
    if (version, split) not in counts:
        raise ValueError("不支持的 nuScenes 评价划分.")
    expected = min(100, counts[(version, split)]) if subset else 81
    frames = manifest["frames"]
    if (manifest["version"], manifest["split"], len(frames)) != (version, split, expected):
        raise ValueError(f"评价清单与配置不符,要求 {version}/{split} 的 {expected} 帧.")
    if len({frame["token"] for frame in frames}) != expected:
        raise ValueError("评价样本重复.")
    return expected


def main():
    """缺少数据,评分依赖或匹配参考时如实停止或记录缺失项."""
    # 在长时间推理之前检查评分依赖,避免到最后才失败.
    import nuscenes.eval.detection.algo
    from tqdm import tqdm

    ctx = context(__file__, new_run=True)
    manifest_path = ctx.root / "build/reference" / ("compile_samples.json" if is_test() else "full_samples.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["frames"]
    expected_count = validate_manifest(manifest, ctx.config["data"], is_test())
    tokens = [frame["token"] for frame in samples]
    record_selection(tokens)
    ground_truth = ctx.root / "build/reference" / ("compile_ground_truth.json" if is_test() else "full_ground_truth.json")
    if len(set(tokens)) != expected_count or not ground_truth.is_file():
        raise ValueError("完整标注缺失或评价样本重复.")
    for frame in tqdm(samples, desc="评价输入预检"):
        names = [frame["lidar"], *(item["path"] for item in frame["sweeps"]),
                 *(item["path"] for item in frame["cameras"])]
        if any(not (ctx.dataset / name).is_file() for name in names):
            raise FileNotFoundError(f"样本资源不完整: {frame['token']}.")
    models = ctx.root / "models"
    samples = select_samples(samples, "mit_bevfusion", key=lambda frame: frame["token"])
    tokens = [frame["token"] for frame in samples]
    record_selection(tokens)
    model_identity = identity(models)
    reference_path = ctx.root / "build/reference/full.json"
    reference = None
    if reference_path.is_file() and not is_test():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if reference["identity"] != model_identity or reference["sample_tokens"] != tokens:
            raise ValueError("已有 FP32 参考与当前模型或完整样本不匹配.")
    predictions, elapsed, npu, rss = evaluate_dataset(
        models, ctx.dataset, samples, ctx.result / "predictions.json")
    accuracy = score_predictions(ground_truth, predictions)
    if is_test():
        report = {"identity": model_identity, "sample_tokens": tokens, "accuracy": accuracy,
                  "evaluation": accuracy_result("mit_bevfusion", "npu", accuracy, len(tokens), "frame")}
        (ctx.work / "compile_accuracy.json").write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return
    files = [models / item["file"] for item in json.loads((models / "contract.json").read_text())["stages"].values()]
    files.extend(models / name for name in ("sparse_weights.npz", "geometry.npz"))
    report = {"model": "MIT BEVFusion camera+lidar swint_v0p075", "run_id": ctx.run_id,
              "status": "complete" if reference else "partial", "platform": "Genio 720 / MT8189",
              "precision": "CPU FP32 + NPU FP16", "protocol": "nuScenes detection_cvpr_2019, v1.0-mini mini_val",
              "dataset": {"version": "v1.0-mini", "split": "mini_val", "scenes": 2,
                          "scope": "完整 mini_val,不代表 trainval/val 6019 帧基准"},
              "evaluated_samples": len(samples), "expected_samples": expected_count,
              "board_accuracy": accuracy, "reference_accuracy": reference["accuracy"] if reference else None,
              "accuracy_change_percentage_points": {name: (accuracy[name] - reference["accuracy"][name]) * 100
                                                     for name in accuracy} if reference else None,
              "npu_mean_ms": npu, "pipeline_mean_ms": elapsed, "peak_rss_mib": rss,
              "timing_scope": "pipeline 不含文件加载及图像预处理;NPU 为每帧所有 SDK inference 调用之和",
              "memory_scope": "Python 混合推理进程峰值,评分前采集,不含全部驱动内存",
              "model_size_bytes": sum(path.stat().st_size for path in files),
              "nuscenes_devkit": importlib.metadata.version("nuscenes-devkit"),
              "missing": [] if reference else ["同协议完整 FP32 参考尚未提供,差值留空"]}
    report["evaluation"] = accuracy_result("mit_bevfusion", "npu", accuracy, len(tokens), "frame")
    from summary import publish

    publish(ctx, report)



if __name__ == "__main__":
    main()
