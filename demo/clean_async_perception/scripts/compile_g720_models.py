"""将本 Demo 的两份 ONNX 校准为 INT8, 编译 G720 DLA 并生成张量契约."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import cv2
import mtk_converter
import numpy as np
import onnx
from onnx import helper, numpy_helper
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.quantization import tensor_metadata


def prepare_yolox(image):
    """按 Demo 的 BGR、左上角 letterbox 和 0–255 输入约定准备数据."""
    height, width = image.shape[:2]
    scale = min(640 / width, 640 / height)
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
    canvas = np.full((640, 640, 3), 114, np.float32)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])


def prepare_mobilenetv2(image):
    """按 Demo 的中心裁剪和 ImageNet RGB 归一化准备分类输入."""
    height, width = image.shape[:2]
    side = min(height, width)
    y, x = (height - side) // 2, (width - side) // 2
    image = cv2.resize(image[y:y + side, x:x + side], (320, 320))
    image = cv2.resize(image, (256, 256))[16:240, 16:240, ::-1].astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    stddev = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image / 255 - mean) / stddev
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def fold_yolox_focus(source, output):
    """将 Focus 切片和首层 3×3 卷积等价合并为 stride=2 的 6×6 卷积."""
    model = onnx.load(source)
    graph = model.graph
    nodes = list(graph.node)
    image = graph.input[0].name
    index = next(i for i, node in enumerate(nodes) if node.op_type == "Conv")
    stem = nodes[index]
    producers = {name: node for node in nodes[:index] for name in node.output}
    concat = producers.get(stem.input[0])
    attributes = {attr.name: helper.get_attribute_value(attr) for attr in stem.attribute}
    if (concat is None or concat.op_type != "Concat" or len(concat.input) != 4
            or len(concat.attribute) != 1 or concat.attribute[0].i != 1
            or attributes != {"dilations": [1, 1], "group": 1, "kernel_shape": [3, 3],
                              "pads": [1, 1, 1, 1], "strides": [1, 1]}):
        raise ValueError("YOLOX 首层与预期的 Focus 卷积结构不符.")

    def scalar(name):
        """读取 Focus Slice 的单元素常量."""
        node = producers[name]
        if node.op_type != "Constant":
            raise ValueError("Focus Slice 参数必须为常量.")
        return numpy_helper.to_array(node.attribute[0].t).item()

    parities = []
    for value in concat.input:
        offsets = {}
        while value != image:
            node = producers[value]
            if node.op_type != "Slice" or len(node.input) != 5:
                raise ValueError("Focus 分支不是预期的二维切片.")
            value, start, end, axis, stride = node.input
            start, axis = scalar(start), scalar(axis)
            if (axis not in (2, 3) or axis in offsets or start not in (0, 1)
                    or scalar(stride) != 2 or scalar(end) < 640):
                raise ValueError("Focus 切片参数不符.")
            offsets[axis] = start
        parities.append((offsets[2], offsets[3]))
    if len(set(parities)) != 4 or any(
            name in producers for node in nodes[index + 1:] for name in node.input):
        raise ValueError("Focus 节点存在重复偏移或被后续层引用.")

    weights = next(tensor for tensor in graph.initializer if tensor.name == stem.input[1])
    values = numpy_helper.to_array(weights)
    if values.shape[1:] != (12, 3, 3):
        raise ValueError("Focus 卷积权重形状不符.")
    fused = np.zeros((values.shape[0], 3, 6, 6), dtype=values.dtype)
    for channel, (dy, dx) in enumerate(parities):
        fused[:, :, dy::2, dx::2] = values[:, channel * 3:(channel + 1) * 3]
    weights.CopyFrom(numpy_helper.from_array(fused, weights.name))
    stem.input[0] = image
    del stem.attribute[:]
    stem.attribute.extend([
        helper.make_attribute("kernel_shape", [6, 6]),
        helper.make_attribute("strides", [2, 2]),
        helper.make_attribute("pads", [2, 2, 2, 2]),
    ])
    del graph.node[:index]
    onnx.checker.check_model(model)
    onnx.save(model, output)


def convert(name, source, preprocess, images, input_shape, output_shapes, output,
            adapted_source=None):
    """按实际 TFLite 张量顺序记录量化参数, 然后编译至 MDLA 5.3."""
    def calibration_data():
        """逐张提供与板端预处理一致的校准输入."""
        for path in tqdm(images, desc=f"{name} 校准"):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"无法读取校准图像: {path}")
            yield [preprocess(image)]

    print(f"[CONVERT] {source.name}", flush=True)
    tflite = output / f"{name}.tflite"
    converter = mtk_converter.OnnxConverter.from_model_proto_file(str(adapted_source or source))
    converter.quantize = True
    converter.use_per_output_channel_quantization = True
    converter.append_output_dequantize_ops = False
    converter.calibration_data_gen = calibration_data
    converter.convert_to_tflite(str(tflite))
    reader = mtk_converter.TFLiteParser(str(tflite))
    inputs = [tensor_metadata(item) for item in reader.get_input_tensor_details()]
    outputs = [tensor_metadata(item) for item in reader.get_output_tensor_details()]
    if len(inputs) != 1 or inputs[0]["shape"] != input_shape:
        raise ValueError(f"{name} 输入张量不符合 Demo 预期: {inputs}")
    if sorted(item["shape"] for item in outputs) != sorted(output_shapes):
        raise ValueError(f"{name} 输出张量不符合 Demo 预期: {outputs}")

    print(f"[COMPILE] {name} → MDLA 5.3", flush=True)
    dla = output / f"{name}.dla"
    subprocess.run(["ncc-tflite", "--arch=mdla5.3", "--suppress-output", "--disallow-bridge",
                    str(tflite), "-o", str(dla)], check=True)
    if dla.stat().st_size == 0:
        raise ValueError(f"编译结果为空: {dla}")
    return {"input": inputs[0], "outputs": outputs,
            "onnx_sha256": sha256(source.read_bytes()).hexdigest(),
            "dla_sha256": sha256(dla.read_bytes()).hexdigest()}


def main():
    """读取用户提供的业务权重和校准目录, 生成可打包的模型目录."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    images = sorted(path for path in args.calibration_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.samples <= 0 or len(images) < args.samples:
        parser.error("校准图片不足或 --samples 非正数.")
    models = Path(__file__).resolve().parents[1] / "models"
    output = models / "g720"
    output.mkdir(parents=True, exist_ok=True)
    contract = output / "tensors.json"
    contract.unlink(missing_ok=True)
    shapes = [[1, channels, size, size] for size in (160, 80, 40, 20)
              for channels in (4, 6)]
    yolox = models / "yolox_s_p2_detect.onnx"
    adapted = output / "yolox_focus_folded.onnx"
    print("[ADAPT] 等价合并 YOLOX Focus 首层.", flush=True)
    fold_yolox_focus(yolox, adapted)
    metadata = {
        "yolox": convert("yolox", yolox, prepare_yolox, images[:args.samples],
                         [1, 3, 640, 640], shapes, output, adapted),
        "mobilenetv2": convert("mobilenetv2", models / "mobilenetv2.onnx", prepare_mobilenetv2,
                               images[:args.samples], [1, 3, 224, 224], [[1, 2]], output),
    }
    contract.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {contract}", flush=True)


if __name__ == "__main__":
    main()
