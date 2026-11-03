"""输入单张 RGB 图片,输出米制深度图及指定位置的估计深度."""

import argparse
from datetime import date
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort

from model import preprocess, restore_depth


def main():
    """执行一次 ONNX 推理,保存原始深度和带米制标注的可视化."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="输入彩色图片")
    parser.add_argument("--model", type=Path, help="ONNX 路径,默认读取本模型本地配置")
    parser.add_argument("--output-dir", type=Path, help="输出目录,默认使用日期临时目录")
    parser.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"), help="原图像素坐标,默认中心")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="手工指定原图区域,统计其有效深度中位数,右下边界不包含")
    args = parser.parse_args()
    if args.model is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "host"))
        from context import context
        args.model = context().output / "depth_anything_v2_metric_hypersim_vits.onnx"
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"输入图片不可读: {args.image}")
    height, width = image.shape[:2]
    point = args.point or [width // 2, height // 2]
    x, y = point
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("取样点必须位于原图范围内.")
    if args.roi:
        x1, y1, x2, y2 = args.roi
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("ROI 必须为原图内有效的左上与右下坐标.")
    print("[1/3] 加载已有 ONNX,仅使用 CPUExecutionProvider.", flush=True)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != "image"
            or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 3, 518, 686]
            or len(outputs) != 1 or outputs[0].name != "metric_depth"
            or outputs[0].shape != [1, 518, 686]):
        raise ValueError("ONNX 不符合本模型固定 FP32 输入输出契约.")
    tensor = preprocess(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), inputs[0].shape)
    print("[2/3] 单张 RGB 推理,输出每像素米制深度.", flush=True)
    prediction = session.run(["metric_depth"], {"image": tensor})[0]
    depth = restore_depth(prediction[0], height, width)
    if not np.isfinite(depth).all():
        raise ValueError("预测深度包含 NaN 或 Inf.")
    output = args.output_dir or (
        Path("D:/tmp/hailongcodex" if sys.platform == "win32" and Path("D:/").exists()
             else "C:/tmp/hailongcodex" if sys.platform == "win32" else "/tmp/hailongcodex")
        / date.today().strftime("%Y%m%d") / "depth_anything_v2" / args.image.stem)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "depth_m.npy", depth)
    color = cv2.applyColorMap(
        np.round(np.clip(depth / 20.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    annotated = image.copy()
    cv2.drawMarker(annotated, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    text = f"Pixel ({x},{y}): {depth[y, x]:.3f} m"
    print(f"[取样点] ({x},{y}) 的估计深度: {depth[y, x]:.3f} 米", flush=True)
    if args.roi:
        region = depth[y1:y2, x1:x2]
        valid = region[np.isfinite(region) & (region > 0)]
        if not valid.size:
            raise ValueError("ROI 内没有有效正深度值.")
        median = float(np.median(valid))
        cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
        text += f" | ROI median: {median:.3f} m"
        print(f"[手工区域] 深度中位数: {median:.3f} 米,不是自动识别的物体距离.", flush=True)
    canvas = np.concatenate([annotated, color], axis=1)
    canvas = cv2.copyMakeBorder(canvas, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(canvas, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if not cv2.imwrite(str(output / "depth_visualization.jpg"), canvas):
        raise OSError("深度可视化保存失败.")
    print(f"[3/3] 已保存 {output / 'depth_m.npy'} 和 {output / 'depth_visualization.jpg'}", flush=True)


if __name__ == "__main__":
    main()
