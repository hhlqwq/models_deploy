"""完整 camera + lidar 混合推理,直接通过 SDK C ABI 调用 NPU."""

import json
import ctypes
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from geometry import decode, pool_bev, projected_depth, select_queries
from sparse import SparseEncoder
from tools.runtime import Runtime
from tools.output_metrics import Moments


def cpu_session(path):
    """限制 ORT 工作线程并仅使用 CPU 后端."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class Stage:
    """依据已编译契约执行固定子图,严格核对原生缓冲区大小."""

    def __init__(self, root, name, contract, reference=False, compare=False):
        """浮点参考也使用同一导出图,不替换权重或模型版本."""
        self.contract, self.name = contract, name
        self.cpu, self.runtime = None, None
        self.inputs, self.outputs = [], []
        self.elapsed_npu_ms = 0
        self.reference = cpu_session(root / f"bevfusion-det_{name}.onnx") if reference else None
        self.comparison = (cpu_session(root / f"bevfusion-det_{name}.onnx")
                           if compare and contract["backend"] == "mtk_npu" else None)
        self.output_metrics = {}
        if contract["backend"] == "mtk_npu":
            self.runtime = Runtime(root / contract["file"])
            for output, tensors, target in ((False, contract["inputs"], self.inputs),
                                             (True, contract["outputs"], self.outputs)):
                for index, tensor in enumerate(tensors):
                    target.append(self.native_tensor(index, tensor["shape"], output))
            print(f"[NPU 契约] {name}: bytes={self.runtime.sizes}", flush=True)
        else:
            self.cpu = self.reference or cpu_session(root / contract["file"])
            print(f"[CPU 子图] {name}", flush=True)

    def native_tensor(self, index, shape, output):
        """读取 SDK 原生 NHWC 对齐维度,允许检测头的 2->4 通道补齐."""
        kind = "Output" if output else "Input"
        # SDK Types.h 的 RuntimeAPIDimensions 固定包含四个 uint32_t.
        dimensions = (ctypes.c_uint32 * 4)()
        function = getattr(self.runtime.library, f"NeuronRuntime_get{kind}PaddedDimensions")
        function.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32 * 4)]
        function.restype = ctypes.c_int
        self.runtime.call(f"get{kind}PaddedDimensions", self.runtime.handle, index, ctypes.byref(dimensions))
        padded_shape = tuple(dimensions)
        logical_shape = (1,) * (4 - len(shape)) + tuple(shape)
        size = self.runtime.sizes[int(output)][index]
        if (padded_shape[:-1] != logical_shape[:-1] or padded_shape[-1] < logical_shape[-1]
                or int(np.prod(padded_shape)) * 2 != size):
            raise ValueError(f"{self.name}/{kind}[{index}] 原生 FP16 布局不符: "
                             f"logical={logical_shape}, native={padded_shape}, bytes={size}.")
        storage = np.zeros(padded_shape, dtype=np.float16)
        view = storage[..., :logical_shape[-1]].reshape(shape)
        return storage, view

    def __call__(self, *values):
        """输出转回 FP32 交给 CPU 几何与稀疏计算,保留 NPU 调用耗时."""
        feed = {item["name"]: np.ascontiguousarray(value, dtype=np.float32)
                for item, value in zip(self.contract["inputs"], values)}
        if len(values) != len(self.contract["inputs"]):
            raise ValueError(f"{self.name} 输入数量不一致.")
        for item, value in zip(self.contract["inputs"], values):
            if list(value.shape) != item["shape"] or not np.isfinite(value).all():
                raise ValueError(f"{self.name}/{item['name']} 输入形状或数值错误.")
        if self.runtime:
            for (_, view), value in zip(self.inputs, values):
                view[...] = value
            self.elapsed_npu_ms += self.runtime.infer([item[0] for item in self.inputs],
                                                      [item[0] for item in self.outputs])
            result = [item[1].astype(np.float32) for item in self.outputs]
        else:
            result = self.cpu.run([item["name"] for item in self.contract["outputs"]], feed)
        if any(not np.isfinite(item).all() for item in result):
            raise ValueError(f"{self.name} 输出有非有限数值.")
        if self.comparison is not None:
            names = [item["name"] for item in self.contract["outputs"]]
            expected = self.comparison.run(names, feed)
            for name, actual, floating in zip(names, result, expected):
                self.output_metrics.setdefault(name, Moments()).update(floating, actual)
        if self.reference and self.runtime:
            expected = self.reference.run([item["name"] for item in self.contract["outputs"]], feed)
            for actual, floating in zip(result, expected):
                error = np.linalg.norm((actual - floating).reshape(-1)) / max(
                    np.linalg.norm(floating.reshape(-1)), 1e-8)
                if not np.isfinite(error) or error > 0.1:
                    raise ValueError(f"{self.name} 原生 I/O 与浮点图不一致,relative_l2={error:.6f}.")
            # 首次调用足以核对该子图契约,六相机不重复运行参考.
            self.reference = None
            print(f"[契约通过] {self.name}: 原生布局及 FP16 数值检查通过.", flush=True)
        return result

    def close(self):
        """释放当前阶段的硬件句柄."""
        if self.runtime:
            self.runtime.close()


class Pipeline:
    """串接六相机深度特征,稀疏 LiDAR,融合解码与完整检测头."""

    def __init__(self, root, check=False, floating=False, verbose=True, compare=False):
        """只初始化一次,正式评价复用相同模型和稀疏权重."""
        root = Path(root)
        self.contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
        self.verbose = verbose
        self.stages = {}
        try:
            for name, item in self.contract["stages"].items():
                if floating:
                    item = {**item, "backend": "onnxruntime_cpu", "file": f"bevfusion-det_{name}.onnx"}
                self.stages[name] = Stage(root, name, item, reference=check, compare=compare)
        except Exception:
            self.close()
            raise
        with np.load(root / "sparse_weights.npz") as archive:
            self.sparse = SparseEncoder({key: archive[key] for key in archive.files}, verbose)
        with np.load(root / "geometry.npz") as archive:
            self.geometry = {key: archive[key] for key in archive.files}

    def __call__(self, sample, measure=False):
        """每次返回 3D 框,分数,类别及完整推理和纯 NPU 调用耗时."""
        for stage in self.stages.values():
            stage.elapsed_npu_ms = 0
        started = time.perf_counter() if measure else None
        camera_bev = np.zeros((1, 360, 360, 80), dtype=np.float32)
        for index in range(6):
            if self.verbose:
                print(f"[相机 {index + 1}/6] 深度特征与 CPU BEV pooling.", flush=True)
            augmentation = sample["augmentations"][index]
            depth = projected_depth(sample["points"], sample["lidar_to_image"][index], augmentation)
            probability, features = self.stages["camera"](sample["images"][index:index + 1], depth)
            camera_bev += pool_bev(probability, features, sample["camera_to_lidar"][index],
                                   sample["intrinsics"][index], augmentation, self.geometry)
        camera_bev = self.stages["camera_downsample"](camera_bev)[0]
        lidar_bev = self.sparse(sample["points"])
        if self.verbose:
            print("[融合与检测] 保留完整 32400 BEV 键及 200 查询.", flush=True)
        features, logits = self.stages["fusion"](camera_bev, lidar_bev)
        inputs, labels, scores = select_queries(features, logits)
        prediction = self.stages["head"](*inputs)[0]
        boxes, scores, labels = decode(prediction, labels, scores)
        elapsed = (time.perf_counter() - started) * 1000 if measure else None
        npu = sum(stage.elapsed_npu_ms for stage in self.stages.values())
        return boxes, scores, labels, elapsed, npu

    def close(self):
        """显式释放所有已创建的 NPU 实例."""
        for stage in self.stages.values():
            stage.close()
