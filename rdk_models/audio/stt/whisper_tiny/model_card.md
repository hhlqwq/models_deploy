# Whisper-Tiny 模型卡

- 来源为 OpenAI Whisper v20250625 的多语言 tiny.pt,约 39M 参数,MIT 许可.
- 目标平台为 RDK S100 / nash-e,使用 OpenExplorer 3.7.0 和官方 Python hbm_runtime.
- 复用仓库已有权重与 FP32 ONNX,采用按行缩放的浮点等价 LayerNorm、one-hot embedding 与静态单 Token Decoder.
- 当前默认 W8A16 配方,常量仿射层实际为 INT16 激活 / INT8 权重;注意力动态矩阵乘两路输入实际为 INT16 / INT8,公开 I/O 为 FP32.执行位置与实际位宽审计写入子集报告.
- 编码器输入为 16 kHz 单声道音频得到的 FP32 [1,80,3000] Log-Mel,输出 FP32 [1,1500,384] audio_features.
- Decoder 的词表为 51865,每步输入 1 Token,8 路 Cache 均为 [6,1,64,256],公开输入输出 FP32,绝对位置上限 448.超过 256 Token 后使用滑动 Cache.
- 本次为英语贪心转写,文本使用官方 EnglishTextNormalizer 规范化,按整个语料的编辑距离除以参考词数计算 WER.
- 校准为 test-clean 排序前 8 条,每条最多 32 个真实自回归状态,共 187 组;子集为排除这 8 条后的固定间隔 50 条,仅对 3 条 Encoder 输出计算 CosSim/RMSE/NRMSE.
- 2026-11-04 实测 50 条,参考 1128 词,ONNX 与 BPU 均 83 个编辑错误,WER 均 7.3582%,BPU 50/50 正常遇到 EOT.
- 完整 test-clean 为 2620 条,包含校准音频;本次尚未运行全量评价、正式性能或峰值 RSS.
- 固定子集成绩只能描述该范围.Whisper 可能漏词、误识别或生成输入中没有的文本.
