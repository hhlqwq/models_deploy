# 文档导航

本仓库用于个人的多平台模型适配、实验和 Demo 开发。先选要完成的任务，再到对应模型、平台或 Demo 查看实际入口。

| 任务 | 文档 |
| --- | --- |
| 理解现有目录和演进方案 | [项目架构](architecture.md) |
| 查新增模型、Demo 和公共代码的约定 | [开发规范](development.md) |
| 准备开发机、编译容器和测试设备 | [开发与测试环境](environment.md) |
| 接入新的板卡、系统或运行后端 | [新增运行目标](guides/add-platform.md) |
| 查看现有模型和运行记录 | [MTK 模型](../mtk_models/README.md) · [RDK 模型](../rdk_models/README.md) |
| 查现有平台的配置和运行入口 | [G720 板端指南](genio_720.md) · [S100 开发与运行](platforms/s100.md) |
| 查看实际进度、下一步和候选方向 | [开发进度](progress.md) · [路线图](roadmap.md) |
| 查现有精度对比口径 | [小样本输出数值对照](output_metrics.md) |
| 按已有命令复现模型或应用 | [G720 ResNet101 与 S100 语音示例](examples/README.md) |
| 查适配过程和排障案例 | [G720 YOLOv5s](study/Genio720_MT8189部署YOLOv5s总览.md) · [G720 MiniCPM5-2B](G720适配MiniCPM5-2B流程.md) |
| 找应用 Demo | [Demo 总览](../demo/README.md) |

平台专用命令以对应的 Docker、模型和 Demo README 为准。`study/` 里的材料保留写作时的方案和实验背景，当前状态以对应项目和实际结果为准。
