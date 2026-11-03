"""评测 FastSAM ONNX 全量精度,在 Docker 内执行."""

import argparse
import json
import math
import os
from pathlib import Path

from tools.storage import REPO_ROOT, load_model_config
from tools.deploy import context

from tools.accuracy.evaluate_onnx import evaluate_model

from tools.evaluation import LIMITS, is_test, result_status


def onnx_eval(config: dict, paths: dict, config_path: Path) -> None:
    """在 Docker 评测全量 ONNX 精度,只输出同协议参考指标."""
    reference = paths["work"] / ("onnx_reference_test.json" if is_test() else "onnx_reference.json")
    # 先写入临时文件,校验完成后再发布本次报告.
    pending = paths["work"] / "tmp" / ("onnx_reference_test.pending.json" if is_test() else "onnx_reference.pending.json")
    pending.unlink(missing_ok=True)
    evaluate_model(model="fastsam", onnx=paths["output"] / "FastSAM-s.onnx",
                   dataset_root=paths["dataset"], output=pending,
                   temp_dir=paths["work"] / "tmp")
    if not pending.is_file() or pending.stat().st_size == 0:
        raise FileNotFoundError(f"ONNX 评测未生成报告: {pending}.")
    report = json.loads(pending.read_text(encoding="utf-8"))
    accuracy = report.get("accuracy")
    if (report.get("status") != result_status() or report.get("model") != "fastsam"
            or report.get("metric") != "segm AR@100" or report.get("samples") != (LIMITS["fastsam"] if is_test() else 5000)
            or type(accuracy) not in (int, float) or not math.isfinite(accuracy)
            or not 0 <= accuracy <= 1):
        raise ValueError(f"FastSAM ONNX 全量评测报告不完整: {pending}.")
    if is_test() and (report.get("evaluation_scope") != "subset" or
                      not report.get("sample_ids_sha256")):
        raise ValueError("FastSAM 快速参考缺少子集范围或样本指纹.")
    pending.replace(reference)
    print(f"[ONNX 精度] 报告路径: {reference.resolve()}", flush=True)


def main() -> None:
    """直接读取 YAML 配置并执行本文件对应的操作."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config.yaml")
    parser.add_argument("--work-dir", type=Path, help="覆盖默认的模型外部资源 build 目录.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_model_config(config_path)
    model = Path(__file__).resolve().parents[2]
    work = context(__file__, config_path, args.work_dir).work
    work = work.resolve()
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    paths = {"model": model, "repo": REPO_ROOT, "work": work,
             "dataset": Path(config["data"]["dataset_dir"]),
             "output": model / config["model"]["output_dir"]}
    print(f"[步骤] {Path(__file__).stem}; 工作目录: {work}", flush=True)
    onnx_eval(config, paths, config_path)


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
