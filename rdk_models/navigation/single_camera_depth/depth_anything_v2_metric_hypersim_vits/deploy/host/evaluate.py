"""使用已有 ONNX 计算正式参考,不触发模型转换或编译."""

import sys

import numpy as np
import onnxruntime as ort

from context import context
from deploy import upload


def main():
    """评价完整 654 帧并上传同版本参考和必要的原始评价资源."""
    ctx = context()
    sys.path.insert(0, str(ctx.model / "deploy"))
    from model import NYUDataset, fingerprint, preprocess, restore_depth, save_json, score_depth

    onnx = ctx.output / "depth_anything_v2_metric_hypersim_vits.onnx"
    artifacts = list((ctx.output / "build").glob("*.hbm"))
    if len(artifacts) != 1:
        raise ValueError("需要唯一已有 HBM,评价入口不会重新编译.")
    dataset = NYUDataset(ctx.dataset, ctx.config["evaluation"])
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    metrics = []
    try:
        for step, index in enumerate(dataset.test, 1):
            image, truth = dataset.read(int(index))
            tensor = preprocess(image, ctx.config["export"]["input_shape"])
            prediction = session.run(["metric_depth"], {"image": tensor})[0][0]
            restored = restore_depth(prediction, *image.shape[:2])
            metrics.append(score_depth(restored, truth, ctx.config["evaluation"]))
            if step % 25 == 0 or step == len(dataset.test):
                print(f"[ONNX 正式参考] {step}/{len(dataset.test)}", flush=True)
    finally:
        dataset.close()
    labeled = ctx.dataset / ctx.config["evaluation"]["labeled_file"]
    splits = ctx.dataset / ctx.config["evaluation"]["split_file"]
    print("[评价资源] 绑定数据与模型身份,随后上传原始 MAT 和参考摘要.", flush=True)
    reference = {
        "backend": "onnx_cpu", "evaluation": ctx.config["evaluation"],
        "export": ctx.config["export"],
        "identity": {
            "hbm_sha256": fingerprint(artifacts[0]), "onnx_sha256": fingerprint(onnx),
        },
        "dataset_identity": {labeled.name: fingerprint(labeled), splits.name: fingerprint(splits)},
        "sample_ids": [int(index) + 1 for index in dataset.test],
        "metrics": {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]},
    }
    path = ctx.work / "reference.json"
    save_json(path, reference)
    upload(ctx, [
        (path, "build/reference/reference.json"),
        (labeled, "datasets/NYU_Depth_V2/" + labeled.name),
        (splits, "datasets/NYU_Depth_V2/" + splits.name),
    ])
    print("[正式参考完成] 请在板端模型目录执行 bash evaluate.sh.", flush=True)


if __name__ == "__main__":
    main()
