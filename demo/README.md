# Demo 总览

每个 Demo 按应用场景独立维护。运行前从应用 README 核对目标设备、模型包、依赖、配置和输入。先阅读运行流程，按需要选择文件回放、真实外设或 ROS 2 联调。

| Demo | 运行目标和输入 | 输出 | 入口 |
| --- | --- | --- | --- |
| MiniCPM5-2B 连续对话 | G720，命令行文本 | 流式回复和多轮会话 | [构建与流程](MiniCPM5_2B/README.md) |
| 异步感知 | Linux + Intel 推理 SDK，Orbbec 或 ROS 2 图像，另有 BAG 回放入口 | YOLOX 检测、MobileNetV2 地面分类、可选 Web 预览、ROS 2 结果 | [应用说明与流程](clean_async_perception/README.md) |
| 小车语音控制 | RDK S100，本地音频或 USB 麦克风 | 转录和控制 JSON | [运行指南](smart_car/voice_control/README.md) · [FILE/MIC 流程](smart_car/voice_control/docs/flow.md) |
| 小车视觉感知 | 设计中，RGB 图像与相机内参 | 拟输出米制深度、局部栅格、路径和状态 | [设计与流程](smart_car/visual_perception/README.md) |

[小车总体设计](smart_car/README.md)说明语音、视觉和未来联调的职责划分。现有实验进度以各 Demo 记录为准。新增应用的文档要求见 [开发规范](../docs/development.md#demo-开发)。
