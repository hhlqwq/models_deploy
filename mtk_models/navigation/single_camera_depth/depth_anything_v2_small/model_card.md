# Depth Anything V2 Small 模型卡

| 项目 | 内容 |
| --- | --- |
| 任务 | 单目相对深度估计 |
| 官方源码 | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)，固定提交 `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` |
| 官方权重 | [Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)，版本 `03876f8651c73a60fe4c2c48294e09fcb6838fcf` |
| 许可 | Small 模型 Apache-2.0，见同目录 [LICENSE](LICENSE) |
| 本次输入 | RGB 518×518，NCHW，ImageNet 均值和标准差归一化；原生 INT8 输入行步长 528 字节 |
| 本次输出 | 518×518 INT8 相对深度图；量化参数随转换产物保存 |
| Genio 720 | DA-2K 点对准确率 85.78%,NPU 131.31 ms,峰值 RSS 32.66 MiB |
| Genio 5100 | `not_started` |

模型输出仅表达单张图内的相对深度结构，不能直接用于米制测距或安全避障。
