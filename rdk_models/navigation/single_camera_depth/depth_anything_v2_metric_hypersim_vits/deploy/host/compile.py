"""生成 S100 独立编译配置并直接调用官方 hb_compile."""

import subprocess

import yaml

from context import context


def main():
    """使用显式 INT16 节点策略编译,不推断实际权重和激活位宽."""
    ctx = context()
    shape = ctx.config["export"]["input_shape"]
    working = ctx.output / "build"
    working.mkdir(parents=True, exist_ok=True)
    ctx.work.mkdir(parents=True, exist_ok=True)
    config = {
        "model_parameters": {
            "onnx_model": str(ctx.output / "depth_anything_v2_metric_hypersim_vits.onnx"),
            "march": ctx.platform["march"], "working_dir": str(working),
            "output_model_file_prefix": "depth_anything_v2_metric_hypersim_vits",
        },
        "input_parameters": {
            "input_name": "image", "input_shape": "x".join(map(str, shape)),
            "input_type_rt": "featuremap", "input_type_train": "featuremap",
            "input_layout_rt": "NCHW", "input_layout_train": "NCHW",
            "norm_type": "no_preprocess",
        },
        "calibration_parameters": {
            "cal_data_dir": str(ctx.work / "calibration"), "cal_data_type": "float32",
            "calibration_type": "default", "preprocess_on": False,
            "optimization": ctx.config["quantization"]["node_policy"],
        },
        "compiler_parameters": {
            "compile_mode": "latency", "core_num": 1, "optimize_level": "O2",
        },
    }
    path = ctx.work / "compile.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print("[编译] hb_compile,目标 S100 / nash-e,输出与错误直接显示.", flush=True)
    subprocess.run(["hb_compile", "--config", str(path)], cwd=ctx.work, check=True)


if __name__ == "__main__":
    main()
