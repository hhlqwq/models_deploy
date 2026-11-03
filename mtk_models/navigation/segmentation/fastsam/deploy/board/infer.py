"""使用 Python 完成 FastSAM 推理与掩码编码,只保留评分所需结果."""

import json
import time

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from fastsam_utils import OUTPUT_NAMES, OUTPUT_SHAPES, postprocess, preprocess
from full_accuracy_board import save_checkpoint
from tools.board import context
from tools.evaluation import coco_ids, is_test
from tools.runtime import Runtime, peak_rss_kib, quantize, read_contract
from tools.output_metrics import OutputComparison


def main():
    """常驻加载模型,逐图保存 RLE 掩码,评价阶段同时采集性能."""
    ctx = context(__file__, new_run=True)
    if ctx.result.exists():
        raise FileExistsError(ctx.result)
    protocol = json.loads((ctx.root / "board/accuracy_protocol.json").read_text())
    images = sorted((ctx.dataset / "images").glob("*.jpg"))
    if len(images) != protocol["images"]:
        raise ValueError("COCO val2017 图片数量不符.")
    selected = set(coco_ids(int(path.stem) for path in images))
    images = [path for path in images if int(path.stem) in selected]
    input_info, outputs = read_contract(ctx.root / "build/reference/runtime_config.csv",
                                        OUTPUT_NAMES, OUTPUT_SHAPES)
    indices = [next(index for index, item in enumerate(outputs) if item["name"] == name)
               for name in OUTPUT_NAMES]
    checkpoint_dir = ctx.work / "predictions_by_image"
    checkpoint_dir.mkdir(parents=True)
    comparison = OutputComparison(ctx)
    with Runtime(ctx.root / "model_int8.dla") as runtime:
        if tuple(map(len, runtime.sizes)) != (1, 10):
            raise ValueError("FastSAM 要求单输入和十个输出.")
        inputs, view = runtime.tensor(0, input_info["shape"], np.int8,
                                      fill=input_info["zero_point"])
        buffers, views = [], []
        for index, item in enumerate(outputs):
            storage, tensor = runtime.tensor(index, item["shape"], np.int8, output=True)
            buffers.append(storage)
            views.append(tensor)
        for count, path in enumerate(images, 1):
            started = time.perf_counter()
            values, geometry = preprocess(cv2.imread(str(path)))
            floating_input = values
            np.copyto(view, quantize(values, input_info["scale"], input_info["zero_point"]))
            milliseconds = runtime.infer([inputs], buffers)
            values = [(tensor.astype(np.float32) - item["zero_point"]) * item["scale"]
                      for tensor, item in zip(views, outputs)]
            boxes, scores, masks = postprocess(
                [values[index] for index in indices], geometry,
                confidence=protocol["confidence"], iou=protocol["nms_iou"],
                max_det=protocol["max_detections"], show_progress=False)
            predictions = []
            for score, mask in zip(scores, masks):
                encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                encoded["counts"] = encoded["counts"].decode("ascii")
                predictions.append({"image_id": int(path.stem), "category_id": 1,
                                    "segmentation": encoded, "score": float(score)})
            record = {"image_id": int(path.stem), "protocol": protocol,
                      "backend": "python_neuron_runtime_hw", "predictions": predictions}
            if not is_test():
                record.update(npu_ms=milliseconds, peak_rss_kib=peak_rss_kib(),
                              end_to_end_ms=(time.perf_counter() - started) * 1000)
            save_checkpoint(checkpoint_dir / f"{path.stem}.json", record)
            comparison.update([floating_input], {item["name"]: value
                                                for item, value in zip(outputs, values)},
                              sample_id=path.name, sample_files=[path])
            if count % 50 == 0 or count == len(images):
                print(f"[推理] FastSAM {count}/{len(images)}", flush=True)
    comparison.finish()


if __name__ == "__main__":
    main()
