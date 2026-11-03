"""将 YOLOv5s TorchScript 模型量化为 MTK INT8 TFLite."""

import argparse
from collections.abc import Iterator
from pathlib import Path

import cv2
import mtk_converter
import numpy as np
from tqdm import tqdm

from tools.deploy import configure_parser
from tools.storage import storage_roots


def preprocess_image(image_path: Path, image_size: int = 640) -> np.ndarray:
    """读取图片并生成 YOLOv5 NCHW FP32 输入."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取校准图片: {image_path}")
    height, width = image.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    nchw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(nchw, axis=0)


def calibration_data(
        calibration_dir: Path, sample_count: int) -> Iterator[list[np.ndarray]]:
    """按固定顺序生成校准数据,并显示准备进度."""
    image_paths = sorted(
        path for path in calibration_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    selected_paths = image_paths[:sample_count]
    if len(selected_paths) < sample_count:
        raise ValueError(
            f"校准图片不足: 需要 {sample_count},实际 {len(selected_paths)}")
    for image_path in tqdm(selected_paths, desc="准备 YOLOv5s 校准数据"):
        yield [preprocess_image(image_path)]


def convert_model(args: argparse.Namespace) -> None:
    """创建 MTK PyTorch Converter 并执行 INT8 PTQ."""
    converter = mtk_converter.PyTorchConverter.from_script_module_file(
        str(args.torchscript), input_shapes=[(1, 3, 640, 640)])
    converter.quantize = True
    # 可选: 输出端追加 DEQUANTIZE.MT8189 部署走 --suppress-output 直取
    # MDLA 原生 INT8 输出, 默认无需开启.
    converter.append_output_dequantize_ops = args.append_output_dequantize
    converter.calibration_data_gen = lambda: calibration_data(
        args.calibration_dir, args.samples)
    converter.convert_to_tflite(str(args.output))


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchscript", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--append-output-dequantize",
                        dest="append_output_dequantize",
                        action="store_true")
    parser.set_defaults(append_output_dequantize=False)
    configure_parser(parser, __file__, {
        "torchscript": lambda c: c.work / "yolov5s.torchscript",
        "calibration_dir": lambda c: c.calibration,
        "samples": lambda c: c.config["data"]["calibration_samples"],
        "output": lambda c: c.work / "model_int8.tflite",
    })
    return parser.parse_args()


if __name__ == "__main__":
    convert_model(parse_args())
