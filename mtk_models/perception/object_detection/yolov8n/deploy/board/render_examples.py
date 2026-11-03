"""从真实板端 COCO 预测生成三个固定公开示例的检测效果."""

import argparse
import json
from pathlib import Path

import cv2

from tools.board import configure_parser


def draw_detection(image, row, label, occupied):
    """绘制真实检测框,避开已有标签,使密集场景的类别与分数可读."""
    x, y, width, height = row["bbox"]
    x1, y1, x2, y2 = map(round, (x, y, x + width, y + height))
    color = (0, 145, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {row['score']:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    label_width, label_height = text_width + 6, text_height + baseline + 6
    left = max(0, min(x1, image.shape[1] - label_width))
    top = max(0, min(y1 - label_height, image.shape[0] - label_height))
    # 优先沿框上方错开标签,必要时扫描其他位置,不改变预测框或分数.
    positions = [top] + list(range(0, image.shape[0] - label_height + 1,
                                   label_height + 2))
    positions = sorted(set(positions), key=lambda value: abs(value - top))
    for candidate in positions:
        rectangle = (left, candidate, left + label_width, candidate + label_height)
        if all(rectangle[2] <= old[0] or rectangle[0] >= old[2]
               or rectangle[3] <= old[1] or rectangle[1] >= old[3]
               for old in occupied):
            top = candidate
            break
    occupied.append((left, top, left + label_width, top + label_height))
    cv2.line(image, (x1, y1), (left, top + label_height), color, 1)
    cv2.rectangle(image, (left, top), (left + label_width, top + label_height),
                  color, cv2.FILLED)
    cv2.putText(image, text, (left + 3, top + text_height + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    """只绘制指定样例,记录图片地址和许可证的清单随输入保留."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    configure_parser(parser, __file__, {
        "input_dir": lambda c: c.root / "examples/input",
        "output_dir": lambda c: c.root / "examples/output",
        "work_dir": lambda c: c.work,
    })
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "samples.json").read_text(encoding="utf-8"))
    selected = {sample["image_id"]: [] for sample in manifest["samples"]}
    with (args.work_dir / "predictions.json").open(encoding="utf-8") as source:
        for row in json.load(source):
            if row["image_id"] in selected and row["score"] >= 0.25:
                selected[row["image_id"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(manifest["samples"], 1):
        image = cv2.imread(str(args.input_dir / sample["file"]))
        if image is None:
            raise ValueError(f"示例图片无法读取: {sample['file']}.")
        occupied = []
        for row in selected[sample["image_id"]]:
            label = manifest["categories"][str(row["category_id"])]
            draw_detection(image, row, label, occupied)
        output = args.output_dir / f"sample_{index}_detections.jpg"
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"无法保存示例: {output}.")
        print(f"[示例 {index}/3] {output}", flush=True)


if __name__ == "__main__":
    main()
