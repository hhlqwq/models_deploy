"""验证镜像工具链及实际 GPU 执行能力."""

from datetime import date
import importlib
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from importlib.metadata import version

import cv2
import mtk_converter
import mtk_llm_sdk
import pyarrow
import transformers
import numpy as np
import torch
import onnxruntime as ort
from onnx import TensorProto, helper


def verify_models():
    """检查全部已接入模型的必要导入,并调用轻量官方评分接口."""
    modules = {
        "resnet101": ("torchvision", "PIL"),
        "vit_base_patch16_224": ("torchvision", "cv2"),
        "yolov5s": ("pycocotools.cocoeval", "torchvision"),
        "yolov8n": ("ultralytics", "pycocotools.cocoeval"),
        "mobilefacenet": ("torchvision", "cv2"),
        "rtmpose_body2d": ("mmpose", "mmcv", "mmengine", "mmdet", "xtcocotools.cocoeval"),
        "fastsam": ("ultralytics", "pycocotools.cocoeval"),
        "depth_anything_v2_small": ("torchvision", "cv2"),
        "yamnet": ("tensorflow", "tf_keras", "tf2onnx"),
        "whisper_tiny": ("whisper", "tiktoken"),
        "mit_bevfusion": ("nuscenes.eval.common.loaders", "nuscenes.eval.detection.algo",
                          "pyquaternion", "sklearn"),
        "minicpm5_2b": ("mtk_llm_sdk", "transformers", "tokenizers", "pyarrow"),
        "deepseek_r1_distill_qwen_1_5b": ("mtk_llm_sdk", "transformers", "tokenizers", "pyarrow"),
    }
    for task, names in modules.items():
        for name in names:
            importlib.import_module(name)
        print(f"[模型依赖] {task}: 通过", flush=True)
    from nuscenes.eval.detection.algo import calc_ap
    from nuscenes.eval.detection.data_classes import DetectionMetricData

    if calc_ap(DetectionMetricData.no_predictions(), 0.1, 0.1) != 0:
        raise RuntimeError("nuScenes 空预测评分异常.")
    from pycocotools import mask

    encoded = mask.encode(np.asfortranarray(np.ones((2, 2), dtype=np.uint8)))
    if int(mask.area(encoded)) != 4:
        raise RuntimeError("COCO 掩码接口异常.")
    print("[评分接口] nuScenes AP 和 COCO 掩码: 通过", flush=True)


def verify_converter():
    """用合成小图验证 Converter 和 NCC,不读取或重新编译项目模型."""
    import onnx

    base = Path("/tmp/hailongcodex") / date.today().isoformat()
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mtk_sdk_check_", dir=base) as temporary:
        work = Path(temporary)
        graph = helper.make_graph(
            [helper.make_node("Add", ["input", "bias"], ["output"])], "sdk_check",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 1])],
            [helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, work / "check.onnx")
        converter = mtk_converter.OnnxConverter.from_model_proto_file(str(work / "check.onnx"))
        converter.quantize = False
        converter.convert_to_tflite(str(work / "check.tflite"))
        ncc = Path(os.environ["NCC_ROOT"])
        environment = dict(os.environ, LD_LIBRARY_PATH=str(ncc / "lib"))
        subprocess.run([str(ncc / "bin/ncc-tflite"), "--arch=mdla5.3", "--suppress-output",
                        "--disallow-bridge", "--relax-fp32", "--suppress-input",
                        str(work / "check.tflite"), "-o", str(work / "check.dla")],
                       env=environment, check=True)
        if not (work / "check.dla").is_file() or not (work / "check.dla").stat().st_size:
            raise RuntimeError("SDK 合成小图没有生成 DLA.")
    print("[SDK] 合成 ONNX 转换及 NCC 编译: 通过", flush=True)


def main():
    """构建时检查依赖, 启动时额外检查真实 CUDA 运算."""
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print("[VERIFY]", sys.version, mtk_converter.__version__, cv2.__version__)
    llm_version = version("mtk-llm-sdk")
    if llm_version != "2.7.5":
        raise RuntimeError(f"需要 mtk-llm-sdk 2.7.5,实际为 {llm_version}.")
    expected = {"numpy": "1.26.4", "onnx": "1.13.1", "onnxruntime-gpu": "1.18.0",
                "torch": "2.0.0+cu118", "torchvision": "0.15.1+cu118",
                "mtk-converter": "8.16.0", "mtk-quantization": "8.2.1",
                "transformers": "4.44.2", "nuscenes-devkit": "1.2.0"}
    for package, required in expected.items():
        actual = version(package)
        if actual != required:
            raise RuntimeError(f"{package} 版本不符: {actual},要求 {required}.")
    print("[LLM]", llm_version, transformers.__version__, pyarrow.__version__)
    verify_models()
    verify_converter()
    if "--build" in sys.argv:
        return
    assert torch.cuda.is_available(), "CUDA 不可用"
    with torch.no_grad():
        output = torch.nn.Conv2d(3, 8, 3).cuda()(torch.ones(1, 3, 64, 64).cuda())
        torch.cuda.synchronize()
    print("[GPU]", torch.cuda.get_device_name(0), tuple(output.shape))
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])], "gpu_check",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor("w", TensorProto.FLOAT, [2, 2], [1, 0, 0, 1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    session = ort.InferenceSession(model.SerializeToString(),
                                   providers=["CUDAExecutionProvider"])
    assert session.get_providers()[0] == "CUDAExecutionProvider", session.get_providers()
    result = session.run(None, {"x": np.ones((2, 2), np.float32)})[0]
    np.testing.assert_allclose(result, 1)
    print("[VERIFY] Torch 和 ONNX Runtime GPU 运算通过.")


if __name__ == "__main__":
    main()
