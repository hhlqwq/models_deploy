# 模型契约

- 模型: Depth Anything V2 Metric Hypersim Small,ViT-S 编码器,室内米制深度.
- 作者配置: features=64,out_channels=[48,96,192,384],max_depth=20.
- 导出契约: ONNX IR9/opset17;image,FP32 RGB NCHW [1,3,518,686];metric_depth,FP32 [1,518,686],单位米.
- 预处理: 固定 518x686 的 OpenCV bicubic,除以 255 后按 ImageNet mean/std 归一化.该尺寸专用于 NYU 480x640 图像比例,其他比例图片采用拉伸,不得当作保持原比例的官方通用输入流程.
- 后处理: align_corners=True 双线性恢复原图尺寸;显示色阶固定 0-20 米.
- 校准: NYU 标准 train 中 seed=17 选 100 张,与 test 完全互斥.
- 日常数据格式: RGB PNG 与米制 FP32 NPY,indices.txt 保留官方划分顺序.原始 MAT 只归档在 original/,仅一次性整理工具需要 h5py/SciPy.
- 编译: S100 / nash-e,请求 set_all_nodes_int8 策略.实际逐算子精度及权重/激活位宽以编译产物为准,未编译前不宣称已实现某种 W/A 位宽.
- 正式评价: NYU 标准 test 654 张,使用 depths 米制真值,有效范围 0.001-10 米,裁剪 [45:471,41:601],逐图指标均值,不做尺度对齐.
- 指标: AbsRel、RMSE(米)、log10、delta1/2/3;浮点 ONNX 与 BPU 使用相同预后处理和样本.
- 性能: Python Runtime 同步调用与端到端耗时分别记录;前者含绑定开销,不等于纯 BPU 设备耗时.
- 子集评价: NYU test 中固定间隔选择 100 张,保留三个示例,同协议比较 ONNX/HBM,发布 results/summary_test.json.仅对三个示例进行原始输出数值比较.
- 当前证据: ONNX IR9/opset17 导出及 CPU 单图通路已验证;本次 INT8 编译与100张板端子集验证进行中,尚未发布新指标.不复用旧版本指标.已有 NAS 权重的官方发布一致性仍待核实.
