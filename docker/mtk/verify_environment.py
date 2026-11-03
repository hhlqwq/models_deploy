"""验证镜像工具链及实际 GPU 执行能力."""

import subprocess
import sys

import cv2
import mtk_converter
import mtk_llm_sdk
import pyarrow
import transformers
import numpy as np
import torch
import onnxruntime as ort
from onnx import TensorProto, helper


def main():
    """构建时检查依赖, 启动时额外检查真实 CUDA 运算."""
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print("[VERIFY]", sys.version, mtk_converter.__version__, cv2.__version__)
    print("[LLM]", transformers.__version__, pyarrow.__version__)
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
