# Model Card：Whisper-Tiny

## 模型

- 上游：`openai/whisper`.
- 版本：`v20250625`.
- 权重：多语言 `tiny.pt`,约 39M 参数.
- 许可证：MIT.
- 目标：MT8189 / Genio 720.

## 预处理与解码

输入音频转换为 16 kHz 单声道,并按 OpenAI Whisper 规则生成 `[1, 80, 3000]` Log-Mel.
Decoder 使用语言 Token、`transcribe` 和 `notimestamps` 提示,贪心解码的总长度上限为 448 Token,包括提示词.KV Cache 固定 256,满后采用保留最近 Token 的滑动窗口,浮点参考与板端协议一致.

当前配方为 W8A16，主计算权重为 INT8，激活及原生 I/O 为对称 INT16。LayerNorm 等价改写后由 SDK 分解为整数基础算子。默认校准使用 test-clean 前 8 条音频，和正式范围有重叠。G720 正式评价覆盖全部 2620 条音频。[正式报告](results/summary.json)

## 限制

`complete` 只表示指定 Run 无失败或缺失样例,不表示精度达到产品要求。Whisper 可能产生
遗漏、错误文本、繁简体差异或重复幻觉,不应用于未经人工复核的高风险决策.
