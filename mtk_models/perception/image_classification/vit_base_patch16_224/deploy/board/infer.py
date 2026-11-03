"""使用 Python 执行 ViT 预处理与 NPU 推理,评分由独立步骤完成."""

import json

import cv2
import numpy as np

from tools.board import context
from tools.evaluation import is_test, select_samples
from tools.runtime import Runtime, peak_rss_kib, quantize
from tools.output_metrics import OutputComparison


def floating_preprocess(path):
    """沿用短边 256、双三次缩放、中心裁剪,返回 RGB NCHW FP32 输入."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法读取图片: {path}.")
    height, width = image.shape[:2]
    ratio = 256 / min(height, width)
    width, height = round(width * ratio), round(height * ratio)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    left, top = (width - 224) // 2, (height - 224) // 2
    values = image[top:top + 224, left:left + 224, ::-1].transpose(2, 0, 1)
    return values.astype(np.float32)[None] / np.float32(255.0)


def preprocess(path, scale, zero_point):
    """保留独立量化预处理入口,与浮点参考共用同一输入."""
    return quantize(floating_preprocess(path), scale, zero_point)


def main():
    """常驻加载模型,保存 Top-5 预测并仅在评价阶段采集性能."""
    ctx = context(__file__, new_run=True)
    if ctx.result.exists():
        raise FileExistsError(ctx.result)
    images = ctx.dataset / "val"
    if not images.is_dir():
        images = ctx.dataset / "ILSVRC2012_img_val"
    paths = sorted(images.glob("*.JPEG"))
    if len(paths) != 50000 or any(
            path.name != f"ILSVRC2012_val_{index + 1:08d}.JPEG"
            for index, path in enumerate(paths)):
        raise ValueError("ImageNet val 必须包含连续编号的 50000 张图片.")
    paths = select_samples(paths, "vit_base_patch16_224", key=lambda path: path.name)
    metadata = json.loads((ctx.root / "build/reference/quantization.json").read_text())
    scale, zero = metadata["input_scale"], metadata["input_zero_point"]
    ctx.work.mkdir(parents=True)
    comparison = OutputComparison(ctx)
    with Runtime(ctx.root / "model_int8.dla") as model:
        if tuple(map(len, model.sizes)) != (1, 1):
            raise ValueError("ViT 要求单输入单输出.")
        inputs, view = model.tensor(0, (1, 3, 224, 224), np.int8, fill=zero)
        outputs, logits = model.tensor(0, (1, 1000), np.int8, output=True)
        with (ctx.work / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for index, path in enumerate(paths):
                floating_input = floating_preprocess(path)
                np.copyto(view, quantize(floating_input, scale, zero))
                milliseconds = model.infer([inputs], [outputs])
                order = np.argsort(-logits[0].astype(np.int16), kind="stable")[:5]
                record = {"backend": "python_neuron_runtime_hw", "index": index, "image": path.name, "top5": order.tolist()}
                if not is_test():
                    record["npu_ms"] = milliseconds
                stream.write(json.dumps(record) + "\n")
                comparison.update([floating_input], {"output": (
                    logits.astype(np.float32) - metadata["output_zero_point"]) * metadata["output_scale"]},
                    sample_id=path.name, sample_files=[path])
                if (index + 1) % 500 == 0 or index + 1 == len(paths):
                    print(f"[推理] ViT {index + 1}/{len(paths)}", flush=True)
    if not is_test():
        (ctx.work / "memory.json").write_text(json.dumps({
            "peak_rss_kib": peak_rss_kib(), "backend": "python_neuron_runtime_hw"}))
    comparison.finish()



if __name__ == "__main__":
    main()
