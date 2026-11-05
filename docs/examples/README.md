# 示例

以下示例对应仓库已有脚本与历史运行记录。命令按需执行，先确认当前环境、制品、输入和资源路径。

## G720 ResNet101 模型适配

在已有 G720 编译环境，从仓库根目录执行单模型入口。提前按 [模型说明](../../mtk_models/perception/image_classification/resnet101/README.md) 准备官方权重、校准输入、子集参考所需数据和板端连接。

```bash
bash mtk_models/perception/image_classification/resnet101/deploy/compile.sh
```

当前入口依次导出 ONNX、量化、编译、计算浮点子集参考并上传。需要板端子集结果时，到终端给出的部署目录运行 `bash compile.sh`，结果在 `results/summary_test.json`，展示图在 `examples/output/`。已有 ImageNet 独立评价可先查 [历史 `summary.json`](../../mtk_models/perception/image_classification/resnet101/results/summary.json)，其中记录模型、样本覆盖、精度和计时范围。只排查某一步时，可先阅读该模型的 `deploy/host/` 实现并按当前任务选取对应入口。

## S100 小车语音 FILE 模式

在已经准备程序、配置及模型包的 S100 上执行。使用自备短音频，例如一段「请向前走」，中文输入选择 `whisper.language=zh`。

```bash
./voice_control_file config.json /path/to/forward.wav > run.jsonl
```

终端显示诊断，`run.jsonl` 按有效段保存控制结果。识别为前进时，结果包含 `command=forward`，并带原转录和状态。静音输入可以没有结果，多条指令文件可以输出多行。构建、模型路径、字段和实际板端记录见 [语音控制指南](../../demo/smart_car/voice_control/README.md)，流程见 [FILE/MIC 运行图](../../demo/smart_car/voice_control/docs/flow.md)。
