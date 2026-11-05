# 开发进度

这里记录已有代码和实验范围，细节以对应模型、应用与结果文件为准。更换设备、运行库、模型包或代码后，在该版本下单独记录新运行。

| 方向 | 当前记录 | 下一步 |
| --- | --- | --- |
| G720 模型适配 | 13 个模型已完成 G720 Python 板端子集与正式评价，报告和示例见 [MTK 索引](../mtk_models/README.md)。 | 各模型 README 记录样本范围、浮点对照、性能和峰值 RSS。 |
| S100 模型适配 | Depth Anything V2 Metric 和 Whisper tiny 有子集记录。Qwen3-0.6B 保留配置准备记录，适配已暂停。[深度子集](../rdk_models/navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/results/summary_test.json) · [Whisper 子集](../rdk_models/audio/stt/whisper_tiny/results/summary_test.json) · [Qwen3 记录](../rdk_models/gen_ai/llm/qwen3_0_6b/README.md) | 根据实验需要选择后端、数据和范围。 |
| G5100 | 有公开占位配置。[配置](../configs/platforms/genio_5100.yaml) | 核对实机、系统和工具链。 |
| MiniCPM5-2B 连续对话 | 有 C++ 应用和 CMake 入口。简化入口的构建与板端运行待验证。[Demo 记录](../demo/MiniCPM5_2B/results/summary.json) | 有运行需求时按当前构建说明复现。 |
| 异步感知 | 已有 Orbbec、ROS 2 输入和 Web/ROS 2 输出代码。BAG 回放路径有已知问题。[应用和流程](../demo/clean_async_perception/README.md) | 按场景核对输入路径和 SDK 组合。需要 BAG 时单独排查。 |
| 小车语音控制 | 上一版 S100 FILE 有小样本板端记录。重构后的 FILE 与 MIC 均已记录 AArch64 交叉编译结果，新版板端运行待核对。[运行与记录](../demo/smart_car/voice_control/README.md) | 按需验证新版 FILE 输出、连续 MIC 输入及繁忙和断开。 |
| 小车视觉与整车集成 | 视觉感知、绕障和整车接口处于设计阶段。[视觉设计](../demo/smart_car/visual_perception/README.md) · [总体设计](../demo/smart_car/README.md) | 有应用需求时先开发独立输入输出链路，再联调。 |

新增记录包含目标与后端、模型或应用版本、已完成步骤、实际运行范围、结果位置和后续问题。计划中的方向见 [路线图](roadmap.md)。
