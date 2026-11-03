"""使用公共 COCO 检测评测,保留 yolov5s 的结果状态规则."""

import os
from tools.coco import evaluate_coco, parse_args


def main():
    """读取本模型参数,交给公共评测主体执行."""
    evaluate_coco(parse_args(__doc__, entry=__file__), model="yolov5s",
                  status_without_reference="complete")


if __name__ == "__main__":
    os.environ.setdefault("MTK_EVAL_SCOPE", "full")
    main()
