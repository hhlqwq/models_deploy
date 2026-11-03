"""从 Google 原始权重直接导出固定窗口 ONNX."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from audio_utils import load_records, load_waveform, preprocess, stream_patches

from tools.deploy import configure_parser



def original_models(models_dir):
    """加载锁定的 Google Keras 实现,构造原始音频和固定窗口模型."""
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
    import tensorflow as tf
    import tf_keras
    sys.path.insert(0, str(models_dir / "original/upstream"))
    # 上游源码目录由参数指定并校验后加载,不能在解析参数前导入.
    import params
    # 上游源码目录由参数指定并校验后加载,不能在解析参数前导入.
    import yamnet
    parameters = params.Params()
    frames = yamnet.yamnet_frames_model(parameters)
    frames.load_weights(str(models_dir / "original/yamnet.h5"))
    inputs = tf_keras.layers.Input(shape=(1, 96, 64), batch_size=1, name="audio")
    features = tf_keras.layers.Permute((2, 3, 1))(inputs)
    scores, _ = yamnet.yamnet(features, parameters)
    patches = tf_keras.Model(inputs, scores)
    patches.load_weights(str(models_dir / "original/yamnet.h5"))
    return frames, patches


def export_model(args):
    """比较原始前向、NumPy 前处理和分块窗口,再导出固定 ONNX."""
    # 仅当前模型或后端需要此依赖,其他评测模式和板端环境不要求安装.
    import tensorflow as tf
    import tf2onnx
    frames, model = original_models(args.models_dir)
    records = load_records(args.dataset)
    maximum_feature_error = 0.0
    maximum_score_error = 0.0
    for row in records[:10]:
        waveform = load_waveform(args.dataset / "audio" / row["filename"])
        expected, _, mel = frames(waveform)
        patches = preprocess(waveform)
        chunks = [waveform[start:start + 1600] for start in range(0, len(waveform), 1600)]
        streamed = np.stack(list(stream_patches(chunks)))
        np.testing.assert_allclose(streamed, patches, atol=2e-5, rtol=2e-5)
        reference = np.stack([mel.numpy()[start:start + 96]
                              for start in range(0, len(mel) - 95, 48)])[:, None]
        np.testing.assert_allclose(patches, reference, atol=2e-4, rtol=2e-4)
        predicted = np.concatenate([model(patch[None]).numpy() for patch in patches])
        np.testing.assert_allclose(predicted, expected.numpy(), atol=2e-5, rtol=2e-4)
        maximum_feature_error = max(maximum_feature_error,
                                    float(np.max(np.abs(patches - reference))))
        maximum_score_error = max(maximum_score_error,
                                  float(np.max(np.abs(predicted - expected.numpy()))))
    args.models_dir.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(
        model, input_signature=[tf.TensorSpec((1, 1, 96, 64), tf.float32, name="audio")],
        opset=13, output_path=str(args.models_dir / "yamnet.onnx"))
    (args.work / "export_check.json").write_text(json.dumps({
        "samples": 10, "streaming_chunk_samples": 1600,
        "streaming_windows_match": True, "maximum_feature_error": maximum_feature_error,
        "maximum_original_score_error": maximum_score_error}, indent=2))
    print("[OK] 原始模型与固定窗口前向一致,ONNX 已导出.", flush=True)


def main():
    """直接导出模型,由 compile.sh 选择已配置的 TensorFlow Python 环境."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    ctx = configure_parser(parser, __file__, {
        "models_dir": lambda c: c.output, "dataset": lambda c: c.dataset,
    })
    args = parser.parse_args()
    args.work = ctx.work
    export_model(args)


if __name__ == "__main__":
    main()
