"""导出完整混合部署的固定子图,逐个尝试 G720 FP16 编译."""

import json
from pathlib import Path
import sys
import subprocess

import numpy as np
import torch

from tools.deploy import compile_dla, context

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network import Camera, CameraDownsample, Fusion, Head


def main():
    """复用已下载权重,明确记录每个子图的执行后端及失败原因."""
    import mtk_converter

    ctx = context(__file__)
    package = ctx.output / "models"
    package.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    state = torch.load(ctx.config["model"]["weights"], map_location="cpu", weights_only=True)["state_dict"]
    stages = (
        ("camera", Camera, [(1, 256, 704, 3), (1, 256, 704, 1)], ["image", "depth"],
         ["depth_probability", "image_features"]),
        ("camera_downsample", CameraDownsample, [(1, 360, 360, 80)], ["bev"], ["camera_bev"]),
        ("fusion", Fusion, [(1, 180, 180, 80), (1, 180, 180, 256)], ["camera", "lidar"],
         ["features", "heatmap"]),
        ("head", Head, [(1, 200, 128), (1, 200, 2), (1, 200, 10), (1, 32400, 128), (1, 32400, 2)],
         ["query", "query_position", "classes", "features", "key_position"], ["predictions"]),
    )
    manifest = {"format": "MIT-BEVFusion-hybrid", "revision": "db75150717a9462cb60241e36ba28d65f6908607",
                "stages": {}, "sparse_backend": "numpy_cpu"}
    consumed = set()
    for index, (name, implementation, shapes, input_names, output_names) in enumerate(stages, 1):
        print(f"[子图 {index}/{len(stages)}] {name}: 导出 FP32 ONNX.", flush=True)
        module = implementation(state).eval()
        tensors = tuple(torch.zeros(shape) for shape in shapes)
        with torch.no_grad():
            outputs = module(*tensors)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        unused = module.names - module.used
        unused = {key for key in unused if not key.endswith("num_batches_tracked")}
        if unused:
            raise ValueError(f"子图有未消费的官方参数: {sorted(unused)}")
        consumed.update(module.names)
        onnx_path = package / f"bevfusion-det_{name}.onnx"
        torch.onnx.export(module, tensors, onnx_path, opset_version=13,
                          input_names=input_names, output_names=output_names, do_constant_folding=True)
        item = {"backend": "onnxruntime_cpu", "file": onnx_path.name,
                "inputs": [{"name": key, "shape": list(shape)} for key, shape in zip(input_names, shapes)],
                "outputs": [{"name": key, "shape": list(tensor.shape)}
                            for key, tensor in zip(output_names, outputs)], "dtype": "float32"}
        tflite = ctx.work / f"{name}.tflite"
        try:
            converter = mtk_converter.OnnxConverter.from_model_proto_file(str(onnx_path))
            converter.quantize = False
            converter.convert_to_tflite(str(tflite))
            dla = package / f"{name}.dla"
            compile_dla(ctx, tflite, dla, extra=("--relax-fp32", "--suppress-input"))
        except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
            # 明确使用 CPU 后端,不声称整个模型都在 NPU 执行.
            item["npu_compile_error"] = (f"NCC exit={error.returncode}" if isinstance(
                error, subprocess.CalledProcessError) else str(error))
            print(f"[CPU 子图] {name}: {error}", flush=True)
        else:
            item.update(backend="mtk_npu", file=dla.name, dtype="float16")
        manifest["stages"][name] = item
        del module, tensors, outputs
    sparse = {key: tensor.numpy() for key, tensor in state.items() if key.startswith("encoders.lidar.")}
    geometry = {key.rsplit(".", 1)[-1]: tensor.numpy() for key, tensor in state.items()
                if key in {"encoders.camera.vtransform." + name for name in ("dx", "bx", "nx", "frustum")}}
    consumed.update(sparse)
    consumed.update("encoders.camera.vtransform." + key for key in geometry)
    if consumed != set(state):
        raise ValueError(f"完整模型权重覆盖不一致: {set(state) - consumed}")
    np.savez(package / "sparse_weights.npz", **sparse)
    np.savez(package / "geometry.npz", **geometry)
    (package / "contract.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not any(item["backend"] == "mtk_npu" for item in manifest["stages"].values()):
        raise RuntimeError("没有可用 NPU 子图,不能交付为 CPU + NPU 混合部署.")
    print("[编译完成] 子图编译记录已保存,原生 I/O 及完整通路仍需板端核实.", flush=True)


if __name__ == "__main__":
    main()
