# FastSAM-s 模型卡

| 项目 | 配置 |
| --- | --- |
| 任务 | 类别无关实例分割 |
| 模型来源 | CASIA-LMC-Lab/FastSAM 官方 FastSAM-s.pt |
| 官方源码参考提交 | b4ed20c2fed75eadc5aa7d8b09fedd137b873b52 |
| 实际导出实现 | 容器预装 Ultralytics 8.0.111,已验证真实权重前向和 ONNX 导出 |
| 输入 | RGB, float32, NCHW, [1,3,640,640], /255 |
| 缩放 | 保持比例,居中 letterbox,填充值 114 |
| NPU 输出 | stride 8/16/32 的 64 通道框 logits、1 通道分数 logits、32 通道掩码系数,以及 [1,32,160,160] 原型 |
| 板端实现 | Python Neuron Runtime API + OpenCV |
| CPU 后处理 | Python DFL、Sigmoid、类别无关 NMS、原型线性组合、原图掩码还原 |
| Demo 默认阈值 | confidence=0.4, NMS IoU=0.9, max_det=100；正式 AR 评测 confidence=0.001 |
| 提示支持 | 全图、单前景点、单 xyxy 框; 提示语义见 Demo 文档 |
| 尚不支持 | 文本/CLIP、负点组合、多图批处理、G5100 实测 |
| 核心精度 | 类别无关 segm AR@100: ONNX FP32 39.02%,板端 NPU 37.57%,下降 1.45 个百分点 |
| 当前结果 | COCO val2017 全量 5000 张,NPU 平均 14.72 ms,峰值 RSS 64.55 MiB |

FastSAM 不输出 COCO 80 类语义标签.本次正式精度采用类别无关协议,
不能把所有预测随意设置为某个 COCO 类别后宣称完成标准实例分割 mAP.
640 输入是本项目端侧配置,不直接比较官方 1024 输入结果.
