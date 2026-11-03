# 模型契约

- 模型: Depth Anything V2 Metric Hypersim Small,ViT-S 编码器,室内米制深度.
- 作者配置: features=64,out_channels=[48,96,192,384],max_depth=20.
- 导出契约: ONNX IR9/opset17;image,FP32 RGB NCHW [1,3,518,686];metric_depth,FP32 [1,518,686],单位米.
- 预处理: 固定 518x686 的 OpenCV bicubic,除以 255 后按 ImageNet mean/std 归一化.该尺寸专用于 NYU 480x640 图像比例,其他比例图片采用拉伸,不得当作保持原比例的官方通用输入流程.
- 后处理: align_corners=True 双线性恢复原图尺寸;显示色阶固定 0-20 米.
- 校准: NYU 标准 train 中 seed=17 选 100 张,与 test 完全互斥.
- 编译: S100 / nash-e,请求 set_all_nodes_int16 策略.实际逐算子精度及权重/激活位宽以编译产物为准,未编译前不宣称已实现某种 W/A 位宽.
- 正式评价: NYU 标准 test 654 张,使用 depths 米制真值,有效范围 0.001-10 米,裁剪 [45:471,41:601],逐图指标均值,不做尺度对齐.
- 指标: AbsRel、RMSE(米)、log10、delta1/2/3;浮点 ONNX 与 BPU 使用相同预后处理和样本.
- 性能: Python Runtime 同步调用与端到端耗时分别记录;前者含绑定开销,不等于纯 BPU 设备耗时.
- 当前证据: 2026-11-03 已从 NAS 室内权重完成静态图改写检查与 ONNX 导出,重新读取文件确认 IR9/opset17、静态输入输出及 99,652,380 字节大小.ONNX checker 和独立单文件 CPU 单图推理已完成,输出 [1362,2048] FP32 米制深度且全部有限.样例没有真实米制标注,未验证测距误差;量化、编译和板端评价未执行.已有 NAS 权重的官方发布一致性仍待核实.
