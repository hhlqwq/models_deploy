# 模型与运行资源

实际文件在本地 S100 配置指定的 NAS 模型根目录下的 audio/stt/whisper_tiny,代码仓库仅保存来源、程序和实测报告.

```text
original/
    tiny.pt                         # 复用原有官方权重
    source_url.txt
    runtime_resources/
        mel_filters_f32.bin         # 80x201 FP32 Whisper Mel 滤波器
        decode_config.txt           # 提示 Token、禁止 Token、Cache 及位置配置
        whisper/assets/multilingual.tiktoken
        vendor/                     # 英文规范化及 more-itertools 源码和许可
build/
    w8a16_precision.json             # 逐图实际精度及执行位置审计
tiny_encoder.onnx
tiny_decoder_step.onnx
tiny_encoder_w8a16.hbm
tiny_decoder_step_w8a16.hbm
```

ONNX 与已有 MTK 导出文件一致,已核对原始 tiny.pt 与原记录的官方发布一致性.配套滤波器、解码规则和官方文本资源从已有资源复用,不重新下载权重.完整编译过程、校准张量、编译 YAML 与 PTQ 审计图放主机 /tmp/hailongcodex/当天日期/s100/whisper_tiny/.

板端 HBM、滤波器、解码规则、词表与 runtime_contract.json 放 models/,Python及规范化组件放 board/,选定参考放 build/reference/;原始训练权重留在 NAS.Encoder HBM 为 37,848,392 字节,Decoder HBM 为 55,305,320 字节.两个 HBM 的公开接口均为 FP32,内部位宽以实际审计为准.

正式全量评价准备全部 2620 条音频和同协议浮点参考.日常子集与全量报告分别保存 results/summary_test.json 和 results/summary.json.
