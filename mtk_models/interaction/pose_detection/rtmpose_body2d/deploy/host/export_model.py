"""从 OpenMMLab MMPose 官方 RTMPose-M 权重离线导出 ONNX."""

import argparse
import importlib.machinery
from pathlib import Path
import sys
import types

import mmpose
import numpy as np
import onnx
import torch
from torch import nn
from onnx import helper, numpy_helper

from tools.deploy import configure_parser


class MissingMmcvOps(types.ModuleType):
    """为未使用的 MMCV 扩展符号提供失败即停的导入占位."""

    def __getattr__(self, name: str):
        """返回一旦调用就明确失败的扩展函数.

        Args:
            name: MMCV 扩展函数名称.

        Returns:
            不可调用的占位函数.
        """
        def unavailable(*args, **kwargs):
            """阻止 RTMPose 导出静默调用未安装的 MMCV 扩展."""
            del args, kwargs
            raise RuntimeError(
                f"RTMPose 导出意外调用未安装的 MMCV 扩展: {name}")

        return unavailable


def prepare_mmcv_lite_import() -> None:
    """允许 MMPose 注册无关模型,但禁止实际调用缺失的 MMCV 扩展."""
    try:
        __import__("mmcv._ext")
    except ModuleNotFoundError:
        stub = MissingMmcvOps("mmcv._ext")
        stub.__file__ = "/virtual/mmcv/_ext.py"
        stub.__spec__ = importlib.machinery.ModuleSpec(
            "mmcv._ext", loader=None)
        sys.modules["mmcv._ext"] = stub


prepare_mmcv_lite_import()

from mmpose.apis import init_model


EXPECTED_MMPOSE_VERSION = "1.3.2"
INPUT_HEIGHT = 256
INPUT_WIDTH = 192
KEYPOINTS = 133
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


class RTMPoseExportWrapper(nn.Module):
    """导出 RTMPose 主干和 SimCC 头,并固定 RGB 输入归一化."""

    def __init__(self, model: nn.Module) -> None:
        """初始化导出包装器.

        Args:
            model: 由 MMPose 官方配置和权重构造的模型.
        """
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(
            self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对 NCHW RGB `[0,255]` 输入归一化并输出两个 SimCC 张量."""
        normalized = (image - self.mean) / self.std
        features = self.model.extract_feat(normalized)
        pred_x, pred_y = self.model.head.forward(features)
        return pred_x, pred_y


def export_model(config_path: Path, weights_path: Path, output_path: Path) -> None:
    """加载官方 MMPose 模型并导出固定输入 ONNX.

    Args:
        config_path: MMPose v1.3.2 官方配置路径.
        weights_path: 官方 RTMPose-M checkpoint 路径.
        output_path: ONNX 输出路径.
    """
    if mmpose.__version__ != EXPECTED_MMPOSE_VERSION:
        raise RuntimeError(
            "MMPose 版本不匹配: "
            f"expected={EXPECTED_MMPOSE_VERSION}, "
            f"actual={mmpose.__version__}")
    model = init_model(str(config_path), str(weights_path), device="cpu")
    model.eval()
    out_channels = int(model.head.out_channels)
    if out_channels != KEYPOINTS:
        raise ValueError(
            f"RTMPose 关键点数量错误: expected={KEYPOINTS}, "
            f"actual={out_channels}")
    wrapper = RTMPoseExportWrapper(model).eval()
    dummy_input = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["pred_x", "pred_y"],
    )
    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)
    print(f"[OK] RTMPose ONNX 已导出: {output_path}")


MAX_SUPPORTED_IR = 8
MAX_SUPPORTED_OPSET = 18


def validate_target_schemas(model: onnx.ModelProto) -> None:
    """确认标准 ONNX 节点在 opset 18 中存在且属性受支持."""
    for node in model.graph.node:
        domain = node.domain or ""
        if domain not in {"", "ai.onnx"}:
            continue
        schema = onnx.defs.get_schema(
            node.op_type, MAX_SUPPORTED_OPSET, domain)
        supported_attributes = set(schema.attributes)
        unknown_attributes = sorted(
            attribute.name for attribute in node.attribute
            if attribute.name not in supported_attributes)
        if unknown_attributes:
            raise ValueError(
                f"{node.name or node.op_type} 在 opset "
                f"{MAX_SUPPORTED_OPSET} 不支持属性: {unknown_attributes}")


def remove_safe_reshape_allowzero(model: onnx.ModelProto) -> int:
    """移除不改变当前模型语义但 MTK 导入器不支持的 allowzero 属性."""
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    removed = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        allowzero = next(
            (item for item in node.attribute if item.name == "allowzero"),
            None)
        if allowzero is None:
            continue
        if allowzero.i == 1:
            shape_tensor = initializers.get(node.input[1])
            if shape_tensor is None:
                raise ValueError(
                    f"无法安全移除动态 Reshape allowzero=1: {node.name}")
            shape = numpy_helper.to_array(shape_tensor)
            if np.any(shape == 0):
                raise ValueError(
                    f"无法安全移除含 0 shape 的 Reshape allowzero=1: "
                    f"{node.name}, shape={shape.tolist()}")
        node.attribute.remove(allowzero)
        removed += 1
    return removed


def expand_gau_mul_broadcast(model: onnx.ModelProto) -> int:
    """将 NCC 不支持的 GAU 双轴广播改写为同形状逐元素乘法.

    新导出图不依赖第三方节点名称,通过唯一常量形状 `(1,1,2,128)` 定位目标 Mul.
    如果图结构变化或出现多个候选,立即停止而不是猜测.
    """
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    matches = []
    for node in model.graph.node:
        if node.op_type != "Mul":
            continue
        constant_inputs = [initializers.get(name) for name in node.input]
        shapes = [tuple(numpy_helper.to_array(tensor).shape)
                  for tensor in constant_inputs if tensor is not None]
        if (1, 1, 2, 128) in shapes:
            matches.append(node)
    if len(matches) != 1:
        raise ValueError(f"GAU 广播 Mul 数量异常: {len(matches)}")
    node = matches[0]
    constant = next(
        (initializers.get(name) for name in node.input
         if initializers.get(name) is not None and
         tuple(numpy_helper.to_array(initializers[name]).shape) ==
         (1, 1, 2, 128)),
        None)
    if constant is None:
        raise ValueError("GAU 广播 Mul 缺少预期常量输入.")
    values = numpy_helper.to_array(constant)
    if values.shape != (1, 1, 2, 128):
        raise ValueError(f"GAU 广播常量 shape 异常: {values.shape}")
    expanded_values = np.repeat(values, 133, axis=1)
    replacement = numpy_helper.from_array(
        expanded_values.astype(values.dtype), constant.name)
    model.graph.initializer.remove(constant)
    model.graph.initializer.append(replacement)
    for value in list(model.graph.value_info):
        if value.name == constant.name:
            model.graph.value_info.remove(value)

    dynamic_input = next(name for name in node.input if name != constant.name)
    expanded_input = f"{dynamic_input}_repeat_axis2"
    concat = helper.make_node(
        "Concat",
        inputs=[dynamic_input, dynamic_input],
        outputs=[expanded_input],
        name="node_expand_gau_mul_input",
        axis=2)
    node_index = list(model.graph.node).index(node)
    model.graph.node.insert(node_index, concat)
    dynamic_index = list(node.input).index(dynamic_input)
    node.input[dynamic_index] = expanded_input
    return 1


def prepare_model(source: Path, output: Path) -> None:
    """合并外部权重并在语义不变时降低 ONNX IR 版本."""
    model = onnx.load(str(source), load_external_data=True)
    opsets = {(item.domain or "ai.onnx"): item.version
              for item in model.opset_import}
    standard_opset = opsets.get("ai.onnx", 0)
    print(f"[INFO] OpenMMLab RTMPose: IR={model.ir_version}, opset={opsets}")
    if standard_opset > MAX_SUPPORTED_OPSET:
        validate_target_schemas(model)
        for item in model.opset_import:
            if (item.domain or "ai.onnx") == "ai.onnx":
                item.version = MAX_SUPPORTED_OPSET
        standard_opset = MAX_SUPPORTED_OPSET
    if model.ir_version > MAX_SUPPORTED_IR:
        model.ir_version = MAX_SUPPORTED_IR
    reshape_count = remove_safe_reshape_allowzero(model)
    gau_mul_count = expand_gau_mul_broadcast(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(output), save_as_external_data=False)
    reloaded = onnx.load(str(output), load_external_data=False)
    external_count = sum(
        tensor.data_location == onnx.TensorProto.EXTERNAL
        for tensor in reloaded.graph.initializer)
    if external_count:
        raise RuntimeError(f"兼容模型仍包含 {external_count} 个外部权重引用.")
    print(f"[OK] MTK 兼容 ONNX: IR={model.ir_version}, "
          f"opset={standard_opset}, 移除 Reshape allowzero "
          f"x{reshape_count}, 展开 GAU Mul 广播 x{gau_mul_count}, "
          f"{output}")


def parse_args() -> argparse.Namespace:
    """从模型 YAML 读取默认参数,官方 MMPose 配置单独命名避免混淆."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmpose-config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compatible-output", type=Path)
    configure_parser(parser, __file__, {
        "mmpose_config": lambda c: Path(mmpose.__file__).resolve().parent / ".mim" /
            c.config["model"]["upstream_config"],
        "weights": lambda c: c.config["model"]["weights"],
        "output": lambda c: c.output / (
            "rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192.onnx"),
        "compatible_output": lambda c: c.output / (
            "rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192_mtk_compatible.onnx"),
    })
    args = parser.parse_args()
    if args.compatible_output is None:
        args.compatible_output = args.output.with_name(
            args.output.stem + "_mtk_compatible.onnx")
    return args


def main() -> None:
    """在同一导出步骤生成原始 ONNX 和语义等价的 MTK 兼容 ONNX."""
    args = parse_args()
    if not args.mmpose_config.is_file():
        raise FileNotFoundError(args.mmpose_config)
    print(f"[1/2] 原始模型: {args.weights}", flush=True)
    export_model(args.mmpose_config, args.weights, args.output)
    print("[2/2] 处理 MTK ONNX 兼容性.", flush=True)
    prepare_model(args.output, args.compatible_output)


if __name__ == "__main__":
    main()
