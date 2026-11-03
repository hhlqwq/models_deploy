"""用常驻 Python Neuron Runtime 执行 ResNet101 示例或全量推理."""

import json
from pathlib import Path
import platform
import time

import numpy as np
import PIL

from resnet101 import (
    MODEL, label_signature, preprocess, read_labels,
    validate_contract, validation_images,
)
from tools.board import context
from tools.evaluation import is_test, select_samples
from tools.files import read_json, require_file, sha256
from tools.runtime import Runtime, peak_rss_kib, quantize
from tools.output_metrics import OutputComparison


def input_images(ctx) -> list[tuple[str, Path]]:
    """从验证集选择固定子集或完整范围,展示图片另由渲染步骤筛选."""
    paths = select_samples(validation_images(ctx.dataset), MODEL, key=lambda path: path.name)
    return [(path.name, path) for path in paths]



def main() -> None:
    """示例不采集指标,仅全量评价预热并记录纯 NPU、端到端和 RSS."""
    ctx = context(__file__, new_run=True)
    if ctx.result.exists():
        raise FileExistsError(ctx.result)
    metadata = read_json(ctx.root / "build/reference/quantization.json")
    validate_contract(metadata)
    dla = require_file(ctx.root / "model_int8.dla")
    print("[检查] 核对部署模型、原生 INT8 张量契约和输入范围.", flush=True)
    if sha256(dla) != metadata["dla_sha256"]:
        raise ValueError("DLA 与量化契约不属于同一次编译.")
    images = input_images(ctx)
    labels_digest = None
    if not is_test():
        labels_digest = label_signature(read_labels(ctx.dataset / "val_labels_0based.txt"))
        from evaluate import matched_reference

        # 在耗时全量推理前拒绝已知不匹配的参考,不自动补跑主机评价.
        matched_reference(ctx, metadata, labels_digest)
    ctx.work.mkdir(parents=True)
    warmup_runs = 0 if is_test() else 5
    comparison = OutputComparison(ctx)
    with Runtime(dla) as runtime:
        if tuple(map(len, runtime.sizes)) != (1, 1):
            raise ValueError("ResNet101 要求单输入单输出.")
        item = metadata["input"]
        storage, view = runtime.tensor(0, item["shape"], np.int8, fill=item["zero_point"])
        output, logits = runtime.tensor(0, metadata["output"]["shape"], np.int8, output=True)
        if warmup_runs:
            print(f"[预热] {warmup_runs} 次,不计入正式计时.", flush=True)
            np.copyto(view, quantize(preprocess(images[0][1]), item["scale"], item["zero_point"]))
            for _ in range(warmup_runs):
                runtime.infer([storage], [output])
        with (ctx.work / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for index, (name, path) in enumerate(images):
                started = time.perf_counter() if not is_test() else None
                values = preprocess(path)
                np.copyto(view, quantize(values, item["scale"], item["zero_point"]))
                milliseconds = runtime.infer([storage], [output])
                # 逐张量正 scale 保持类别排序,相同 INT8 分数按类别索引稳定排序.
                order = np.argsort(-logits[0].astype(np.int16), kind="stable")[:5]
                row = {"index": index, "image": name, "top5": order.tolist(),
                       "backend": "python_neuron_runtime_hw"}
                if started is not None:
                    row.update(npu_ms=milliseconds, end_to_end_ms=(time.perf_counter() - started) * 1000)
                stream.write(json.dumps(row) + "\n")
                item_out = metadata["output"]
                comparison.update([values], {"output": (
                    logits.astype(np.float32) - item_out["zero_point"]) * item_out["scale"]},
                    sample_id=path.name, sample_files=[path])
                if (index + 1) % 500 == 0 or index + 1 == len(images):
                    print(f"[推理] {MODEL} {index + 1}/{len(images)}.", flush=True)
        if not is_test():
            details = {
                "model": MODEL, "run_id": ctx.run_id, "samples": len(images),
                "backend": "python_neuron_runtime_hw", "evaluation_scope": "full",
                "onnx_sha256": metadata["onnx_sha256"], "dla_sha256": metadata["dla_sha256"],
                "labels_sha256": labels_digest, "warmup_runs": warmup_runs,
                "pillow": PIL.__version__,
                "python": platform.python_version(),
                "peak_rss_kib": peak_rss_kib(),
                "input_padded_bytes": runtime.sizes[0][0],
                "output_padded_bytes": runtime.sizes[1][0],
            }
            (ctx.work / "inference.json").write_text(
                json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[推理完成] 本次过程目录: {ctx.result}.", flush=True)
    comparison.finish()


if __name__ == "__main__":
    main()
