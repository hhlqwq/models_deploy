"""按同协议直接计算 TensorFlow 或 ONNX 浮点精度."""


import os
import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from audio_utils import evaluation_ids, load_records, load_waveform, mapping, metrics, preprocess

from tools.deploy import configure_parser
from tools.output_metrics import record_reference, finish_reference

from tools.evaluation import (
    audio_records, is_test, result_status, scope_metadata,
)


from export_model import original_models
from tools.evaluation import accuracy_result, record_selection


def evaluate(args):
    """对全量 2000 条音频求窗口平均分数,保存结果和校准折外指标."""
    records = audio_records(load_records(args.dataset))
    indices = mapping(args.models_dir / "original/upstream/yamnet_class_map.csv")
    if args.mode == "tensorflow":
        frames, _ = original_models(args.models_dir)
    else:
        # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(args.models_dir / "yamnet.onnx"),
                                       options, providers=["CPUExecutionProvider"])
    all_scores = []
    for row in tqdm(records, desc=f"YAMNet {args.mode} 全量精度"):
        waveform = load_waveform(args.dataset / "audio" / row["filename"])
        if args.mode == "tensorflow":
            patch_scores = frames(waveform)[0].numpy()
        else:
            window_scores = []
            for patch_index, patch in enumerate(preprocess(waveform)):
                tensor = patch[None]
                output = session.run(None, {"audio": tensor})[0]
                record_reference(args.models_dir / "yamnet.onnx", [tensor], {"output": output},
                    sample_id=f"{row['filename']}:window:{patch_index}",
                    sample_files=[args.dataset / "audio" / row["filename"]])
                window_scores.append(output)
            patch_scores = np.concatenate(window_scores)
        all_scores.append(patch_scores.mean(axis=0))
    scores = np.stack(all_scores)
    if scores.shape != (len(records), 521) or not np.isfinite(scores).all():
        raise ValueError("浮点分数覆盖不完整或含非有限数.")
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "scores.npy", scores)
    summary = {"status": result_status(), **scope_metadata(), "samples": len(records), "backend": args.mode, "metric": "ESC-50 projected macro AP",
               "metrics": metrics(scores, records, indices)}
    if args.mode == "onnx" and args.reference:
        reference = np.load(args.reference / "scores.npy")
        np.testing.assert_allclose(scores, reference, atol=3e-5, rtol=3e-4)
        summary["maximum_reference_score_error"] = float(np.max(np.abs(scores-reference)))
    summary["evaluation"] = accuracy_result("yamnet", "onnx" if args.mode == "onnx" else "tensorflow",
        {"ESC-50 macro AP (folds 2-5)": summary["metrics"]["held_out"]["map"]},
        len(evaluation_ids(records, indices)), "audio_clip",
            selection_id=record_selection(evaluation_ids(records, indices)))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[OK] {summary}", flush=True)
    if args.mode == "onnx":
        finish_reference()


def main():
    """评测选定浮点后端,每个 Python 环境直接执行同一份评测实现."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", dest="mode", choices=["tensorflow", "onnx"], default="onnx")
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    ctx = configure_parser(parser, __file__, {
        "models_dir": lambda c: c.output, "dataset": lambda c: c.dataset,
    })
    args = parser.parse_args()
    mode = "test" if is_test() else "full"
    args.output = args.output or ctx.work / "evaluation" / mode / args.mode
    if args.mode == "onnx" and args.reference is None and not is_test():
        args.reference = ctx.work / "evaluation" / mode / "tensorflow"
    evaluate(args)
    if not is_test() and args.mode == "onnx":
        from upload import prepare_reference

        prepare_reference(Namespace(model_root=ctx.model, models_dir=ctx.output,
                                    dataset=ctx.dataset, work=ctx.work))


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
