"""用 Python 完成 YOLOv8n 预处理、硬件推理和 DFL 解码."""

from tools.board import context
from tools.detection import infer
from tools.runtime import read_contract
from yolov8_utils import OUTPUT_NAMES, OUTPUT_SHAPES, postprocess, preprocess


def main():
    """按转换器输出索引绑定原生张量,解码时恢复模型语义顺序."""
    ctx = context(__file__, new_run=True)
    inputs, outputs = read_contract(ctx.root / "build/reference/runtime_config.csv",
                                    OUTPUT_NAMES, OUTPUT_SHAPES)
    indices = [next(index for index, item in enumerate(outputs) if item["name"] == name)
               for name in OUTPUT_NAMES]

    def decode(values, geometry):
        """将 Runtime 顺序转换为 box/score 语义顺序后执行后处理."""
        return postprocess([values[index] for index in indices], geometry)

    infer(ctx, inputs, outputs, preprocess, decode)


if __name__ == "__main__":
    main()
