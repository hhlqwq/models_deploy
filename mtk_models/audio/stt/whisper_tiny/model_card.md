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

当前部署目标为 W8A16,主计算权重 INT8、激活及原生 I/O 为对称 INT16.LayerNorm 等价改写后由 SDK 分解为整数基础算子,主计算层和 I/O 类型由转换后审计确认,不允许浮点回退.默认校准使用 test-clean 前 8 条音频,每条最多 32 个 Decoder 步骤,与评价集存在重叠.2026-10-27 已编译并执行板端小样本,50 条非校准音频的 NPU WER 为 182.27%,同范围 ONNX 为 5.81%,精度未通过验收;尚未执行全量评价,历史浮点结果不能作为 W8A16 成绩.

## 限制

`complete` 只表示指定 Run 无失败或缺失样例,不表示精度达到产品要求。Whisper 可能产生
遗漏、错误文本、繁简体差异或重复幻觉,不应用于未经人工复核的高风险决策.
