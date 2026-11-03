"""一帧真实输入检查与 BEV 展示,不发布精度或性能报告."""

from pathlib import Path
import json

from tools.board import context

import numpy as np
from PIL import Image, ImageDraw

from pipeline import Pipeline


def main():
    """核对原生张量与浮点子图并保存少量真实 3D 检测示例."""
    ctx = context(__file__)
    root = ctx.root
    with np.load(root / "examples/input/sample.npz") as archive:
        sample = {key: archive[key] for key in archive.files}
    pipeline = Pipeline(root / "models", compare=True)
    output_errors = {}
    try:
        boxes, scores, labels, _, _ = pipeline(sample)
        for stage in pipeline.stages.values():
            for name, metric in stage.output_metrics.items():
                values = metric.result()
                output_errors[f"{stage.name}/{name}"] = values
                print(f"[输出误差] 单帧 {stage.name}/{name}: {values}", flush=True)
    finally:
        pipeline.close()
    ctx.work.mkdir(parents=True, exist_ok=True)
    if not output_errors:
        raise ValueError("BEVFusion 未获得任何 NPU 子图输出误差.")
    (ctx.work / "output_metrics.json").write_text(json.dumps({
        "scope": "one_frame_npu_subgraphs_same_stage_inputs", "frames": 1,
        "reference_backend": "onnxruntime_cpu_on_board",
        "nrmse_definition": "100 * RMSE / RMS(reference)",
        "outputs": output_errors}, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    destination = root / "examples/output"
    destination.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (900, 940), "#111827")
    draw = ImageDraw.Draw(image)
    points = sample["points"]
    pixels = ((points[:, :2] + 54) / 108 * 900).astype(np.int32)
    valid = ((pixels >= 0) & (pixels < 900)).all(axis=1)
    for x, y in pixels[valid][::3]:
        draw.point((int(y), int(899 - x)), fill="#65758b")
    visible = 0
    for box, score, label in zip(boxes, scores, labels):
        if score < 0.25:
            continue
        x, y, _, width, length, _, yaw = box[:7]
        corners = np.array([[-width, -length], [-width, length], [width, length], [width, -length]]) / 2
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = (corners @ rotation.T + [x, y] + 54) / 108 * 900
        polygon = [(float(corner[1]), float(899 - corner[0])) for corner in corners]
        color = ("#22d3ee", "#fbbf24", "#fb7185", "#a78bfa")[int(label) % 4]
        draw.line([*polygon, polygon[0]], fill=color, width=2)
        visible += 1
    draw.text((16, 914), f"MIT BEVFusion | camera + lidar | score >= 0.25: {visible}", fill="white")
    image.save(destination / "bev.png")
    np.savez(destination / "detections.npz", boxes=boxes, scores=scores, labels=labels)
    print(f"[通路完成] 一帧完整 camera + lidar,输出 {len(boxes)} 个候选框,示例 {destination / 'bev.png'}.",
          flush=True)


if __name__ == "__main__":
    main()
