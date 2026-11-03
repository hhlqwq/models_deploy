# 模型产物

原始权重 `tiny.pt` 与转换产物保存在 NAS 对应模型资源目录,本目录仅记录公开来源.

本目录预期生成以下产物:

- `tiny_encoder.onnx`：固定 `[1, 80, 3000]` Log-Mel 输入的 Encoder.
- `tiny_decoder_step.onnx`: 单 Token、256 Token KV Cache、448 个绝对位置的 Decoder.
- `build/tiny_encoder_w8a16.tflite`、`build/tiny_decoder_w8a16_1t256c.tflite`: MTK W8A16 量化产物.
- `tiny_encoder_w8a16.dla`、`tiny_decoder_w8a16_1t256c.dla`: Genio 720 DLA.
- `build/runtime_contract.json`: 转换后实际 INT16 I/O 量化契约,上传至板端 `models/`.

实际文件、Shape、dtype 和编译参数必须以当前运行清单为准.
