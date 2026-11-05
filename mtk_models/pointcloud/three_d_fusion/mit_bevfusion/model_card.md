# MIT BEVFusion 模型卡

模型来自 [MIT HAN Lab BEVFusion](https://github.com/mit-han-lab/bevfusion/tree/db75150717a9462cb60241e36ba28d65f6908607),固定 `db75150717a9462cb60241e36ba28d65f6908607` 的 `camera+lidar/swint_v0p075/convfuser.yaml`,使用用户下载的官方 `bevfusion-det.pth`.源码许可 Apache-2.0;数据集及权重使用条件仍需遵守各自原始条款,来源见 [资源记录](models/source_url.txt).

输入为六路 RGB 相机及 5 维 LiDAR 点云.图像原始 1600x900,验证时缩放 0.48 后底部居中裁剪至 256x704,使用 ImageNet RGB mean/std.点云范围 `[-54,-54,-5,54,54,3]`,体素大小 `[0.075,0.075,0.2]`,当前帧与最多 9 个历史 sweep,每体素最多 10 点,验证时最多 160000 个体素.历史不足时沿用官方加载规则: 完全无历史则补当前帧,有历史则使用实际可用帧.

SwinTiny 和 CPU 稀疏主干均保留完整计算.相机输出先由 CPU 完成真实标定几何变换与深度加权 BEV pooling,再融合 LiDAR BEV.检测头使用 32400 个 BEV 键和 200 个查询,输出 nuScenes 十类 3D 框,速度和分数;配置无额外 NMS.不采用相机单模态或伪造 LiDAR 特征替代完整模型.

G720 使用 CPU FP32 + NPU FP16 混合部署,不是纯 NPU 模型.三个 DLA 分别执行相机 BEV 降采样,ConvFuser+SECOND+SECONDFPN+稠密热图,以及 TransFusion 注意力和回归.相机 Swin+LSSFPN+DepthLSS 使用 ONNX Runtime CPU,LiDAR 稀疏主干使用 NumPy CPU;NPU 输入输出按 SDK 实际维度绑定,检测头包含通道补齐.

G720 正式评价覆盖 nuScenes v1.0-mini 的完整 mini_val，共 2 个场景、81 帧，按官方 `detection_cvpr_2019` 计算 mAP 与 NDS。浮点参考使用适配后的 FP32 图及相同 CPU 稀疏主干。[正式报告](results/summary.json)

[NVIDIA CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion)仅作为分图设计参考,未使用其 CUDA 引擎,预导出权重或平台成绩作为 MTK 实测.
