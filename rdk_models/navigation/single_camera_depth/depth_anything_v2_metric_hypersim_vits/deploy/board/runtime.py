"""通过官方 hbm_runtime 执行 S100 米制深度推理."""

import numpy as np


class Runtime:
    """只接受显式 FP32 公共输入输出契约,不猜测量化张量格式."""

    def __init__(self, path, shape):
        """加载 HBM 并检查模型数、张量名称、形状和数据类型."""
        from hbm_runtime import HB_HBMRuntime, hbDNNDataType

        self.runtime = HB_HBMRuntime(str(path))
        if len(self.runtime.model_names) != 1:
            raise ValueError("当前运行器只支持单模型 HBM.")
        self.name = self.runtime.model_names[0]
        if (self.runtime.input_names[self.name] != ["image"]
                or self.runtime.output_names[self.name] != ["metric_depth"]
                or list(self.runtime.input_shapes[self.name]["image"]) != list(shape)
                or list(self.runtime.output_shapes[self.name]["metric_depth"]) != [1, *shape[-2:]]
                or self.runtime.input_dtypes[self.name]["image"] != hbDNNDataType.F32
                or self.runtime.output_dtypes[self.name]["metric_depth"] != hbDNNDataType.F32):
            raise ValueError("HBM 输入输出与本模型的 FP32 米制深度契约不符.")

    def run(self, tensor):
        """同步执行一次 BPU 推理,返回原始模型分辨率的米制深度."""
        result = self.runtime.run({"image": np.ascontiguousarray(tensor, dtype=np.float32)})
        depth = np.asarray(result[self.name]["metric_depth"])
        if not np.isfinite(depth).all():
            raise ValueError("BPU 输出包含非有限值.")
        return depth[0]
