# RDK 模型

本目录独立维护地瓜平台模型的导出、量化、编译和 Python 板端实现,不调用 MTK 转换器或 Runtime.
平台配置见 `configs/platforms/s100.yaml`,编译环境见 [S100 Docker](../docker/rdk_s100/README.md).

| 模型 | 平台 | 状态 |
| --- | --- | --- |
| [Depth Anything V2 Metric Hypersim ViT-S](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/README.md) | S100 / nash-e | ONNX 单图验证、100 张校准与 HBM 编译上传已完成;板端三样本通路与对照已完成;正式精度待评价 |

每模型只提供 deploy/compile.sh 与 deploy/evaluate.sh 两个主要入口.文件存在不代表已通过编译或板端验证.
compile 仅进行转换、量化、编译、部署与少量通路检查并保存示例;evaluate 独立使用已有模型完成精度、性能和进程峰值 RSS.
实际评价只保存一个 results/summary.json,缺少实测时不填造指标.
