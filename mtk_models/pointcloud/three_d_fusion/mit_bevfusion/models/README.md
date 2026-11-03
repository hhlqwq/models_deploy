# 模型资源

权重和转换产物位于存储配置对应的 `pointcloud/three_d_fusion/mit_bevfusion` 外部资源目录,不提交 Git.

- `bevfusion-det.pth`: 用户下载的官方完整相机与 LiDAR 检测权重.
- `upstream/`: 固定 MIT commit 的离线源码,用于核对配置与算子语义.
- `build/`: `bevfusion-det_camera.onnx`、`bevfusion-det_camera_downsample.onnx`、`bevfusion-det_fusion.onnx`、`bevfusion-det_head.onnx`,三个 NPU FP16 DLA,张量契约,CPU 稀疏参数和几何参数.
- `build/sample.npz` 与 `sample.json`: 一帧 mini_val 编译输入及其来源.
- `build/samples.json`, `ground_truth.json`, `onnx_reference.json`: 只在独立评价入口运行后生成的全量评价输入.

具体来源见 [source_url.txt](source_url.txt).相机 ONNX 是 CPU 执行图;其余三个 ONNX 也用于板端原生契约检查和独立同协议浮点参考,不能删除后仍宣称这些功能可用.
