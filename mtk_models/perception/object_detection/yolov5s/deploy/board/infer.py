"""用 Python 完成 YOLOv5s 预处理、硬件推理和检测解码."""

from tools.board import context
from tools.detection import infer
from yolov5_utils import INPUT, OUTPUTS, postprocess, preprocess


def main():
    """使用既有量化契约执行当前模型的小批量或全量推理."""
    infer(context(__file__, new_run=True), INPUT, OUTPUTS, preprocess, postprocess)


if __name__ == "__main__":
    main()
