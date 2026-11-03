"""使用 Python 执行 RTMPose WholeBody 人体裁剪和 NPU 推理."""

import json
import time

import cv2
import numpy as np

from rtmpose_utils import decode_prediction, preprocess_image
from tools.board import context
from tools.evaluation import is_test, pose_image_ids
from tools.runtime import Runtime, peak_rss_kib, quantize
from tools.output_metrics import OutputComparison


def main():
    """保留原始 detection_id、检测框评分及 SimCC 量化协议."""
    ctx = context(__file__, new_run=True)
    if ctx.result.exists():
        raise FileExistsError(ctx.result)
    path = ctx.dataset / "person_detection_results/COCO_val2017_detections_AP_H_56_person.json"
    detections = json.loads(path.read_text())
    if len(detections) != 104125:
        raise ValueError("人体检测框必须覆盖完整 104125 条记录.")
    selected = set(pose_image_ids(detections))
    rows = []
    for index, item in enumerate(detections):
        if item["category_id"] != 1 or len(item["bbox"]) != 4 or min(item["bbox"][2:]) <= 0:
            raise ValueError(f"无效人体检测框: {index}.")
        if int(item["image_id"]) in selected:
            rows.append(dict(item, detection_id=index))
    rows.sort(key=lambda item: (item["image_id"], item["detection_id"]))
    ctx.work.mkdir(parents=True)
    timings = []
    comparison = OutputComparison(ctx)
    with Runtime(ctx.root / "model_int8.dla") as runtime:
        if tuple(map(len, runtime.sizes)) != (1, 2):
            raise ValueError("RTMPose 要求一输入两输出.")
        inputs, view = runtime.tensor(0, (1, 3, 256, 192), np.int8, fill=-128)
        output_x, x = runtime.tensor(0, (1, 133, 384), np.int8, output=True)
        output_y, y = runtime.tensor(1, (1, 133, 512), np.int8, output=True)
        last_id, image = None, None
        with (ctx.work / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for count, item in enumerate(rows, 1):
                started = time.perf_counter()
                if item["image_id"] != last_id:
                    image = cv2.imread(str(ctx.dataset / "images" / f"{item['image_id']:012d}.jpg"))
                    last_id = item["image_id"]
                    if image is None:
                        raise ValueError(f"无法读取 COCO 图片: {last_id}.")
                values, geometry = preprocess_image(image, tuple(item["bbox"]))
                np.copyto(view, quantize(values, 1.0, -128))
                if count == 1 and not is_test():
                    for _ in range(20):
                        runtime.infer([inputs], [output_x, output_y])
                    started = time.perf_counter()
                prepared = time.perf_counter()
                milliseconds = runtime.infer([inputs], [output_x, output_y])
                inferred = time.perf_counter()
                pred_x = (x.astype(np.float32) + 22) * 0.007186536211520433
                pred_y = (y.astype(np.float32) + 14) * 0.007497822400182486
                keypoints, area = decode_prediction(pred_x, pred_y, geometry)
                ended = time.perf_counter()
                comparison.update([values], {"simcc_x": pred_x, "simcc_y": pred_y},
                    sample_id=f"{item['image_id']}:detection:{item['detection_id']}",
                    sample_files=[ctx.dataset / "images" / f"{item['image_id']:012d}.jpg", path])
                stream.write(json.dumps({"backend": "python_neuron_runtime_hw", "detection_id": item["detection_id"],
                    "image_id": item["image_id"], "bbox_score": item["score"],
                    "area": area, "keypoints": keypoints}) + "\n")
                if not is_test():
                    timings.append([(prepared - started) * 1000, milliseconds,
                                    (ended - inferred) * 1000])
                if count % 500 == 0 or count == len(rows):
                    print(f"[推理] RTMPose {count}/{len(rows)}", flush=True)
    if not is_test():
        values = np.asarray(timings)
        report = {"processed_detections": len(rows), "peak_rss_kb": peak_rss_kib(),
                  "backend": "python_neuron_runtime_hw", "npu_p95_ms": float(np.percentile(values[:, 1], 95))}
        for index, name in enumerate(("preprocess_mean_ms", "npu_mean_ms", "postprocess_mean_ms")):
            report[name] = float(values[:, index].mean())
        (ctx.work / "timing_summary.json").write_text(json.dumps(report))
    comparison.finish()


if __name__ == "__main__":
    main()
