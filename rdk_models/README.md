# RDK 模型

本目录独立维护地瓜平台模型的导出、量化、编译和 Python 板端实现,不调用 MTK 转换器或 Runtime.
当前运行目标为 S100，配置和运行入口见 [S100 开发与运行](../docs/platforms/s100.md) 与 [S100 Docker](../docker/rdk_s100/README.md)。独立应用见 [Demo 总览](../demo/README.md)。

| 模型 / 版本 | 场景 / 类别 | 输入 / 参数量 | 原始权重 / FP32 ONNX | 数据集 / 指标 / 参考基线 | S100 量化方式 / 编译与性能 | S100 **CosSim**/RMSE/NRMSE/精度 |
| --- | --- | --- | --- | --- | --- | --- |
| [Depth Anything V2 Metric Hypersim ViT-S](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/README.md)<br>INT8 基线 | 导航 / 单目深度 | RGB `1×3×518×686`<br>— | 原始: —<br>ONNX: 99.65 MB | NYU test 固定100张<br>δ1,ONNX: 62.96% | **INT8 基线**<br>HBM: 57.96 MB<br>Runtime: 36.54 ms<br>RSS: 109.38 MiB | 0.927799/1.621510/47.5908%/38.18%<br>(-24.78%)<br>[报告](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/results/summary_test.json) |
| [Depth Anything V2 Metric Hypersim ViT-S](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/README.md)<br>INT8 + Softmax A16 | 导航 / 单目深度 | RGB `1×3×518×686`<br>— | 原始: —<br>ONNX: 99.65 MB | NYU test 固定100张<br>δ1,ONNX: 62.96% | **INT8 + Softmax A16**<br>HBM: 62.38 MB<br>Runtime: 40.48 ms<br>RSS: 105.06 MiB | 0.901099/1.659613/48.7091%/24.04%<br>(-38.92%)<br>[报告](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/results/summary_test.json) |

数值误差依次为 CosSim、输出 RMSE、NRMSE,NRMSE 单位百分比.深度模型覆盖三张固定图片,RMSE 单位米,任务精度展示100张子集的δ1.括号采用板端减浮点的有符号百分点差值,沿用 MTK 的 `%` 写法.缺少实测的数据以 `—` 表示.全局A16方案已由用户明确舍弃,从候选表移除.其历史实测记录保存在报告中.各量化实验对照见模型 README.

[Whisper tiny](audio/stt/whisper_tiny/README.md) 的历史子集包含 50 条非校准音频，任务指标为语料级 WER，Encoder 数值对照覆盖其中三条音频。当前配方、选样调整和实测范围见 [子集结果](audio/stt/whisper_tiny/results/summary_test.json) 与模型说明。[Qwen3-0.6B](gen_ai/llm/qwen3_0_6b/README.md) 保留已暂停的配置准备记录。

已有完整运行路径的模型使用 `deploy/compile.sh` 与 `deploy/evaluate.sh`。`compile` 包含转换、量化、编译、部署和子集相关步骤，`evaluate` 使用已有模型按需记录精度、性能和进程峰值 RSS。子集与独立评价分别记录在 `results/summary_test.json` 和 `results/summary.json`，实际范围以报告为准。

总入口使用与MTK相同的四脚本方式.编译服务器执行 deploy/compile.sh 或 deploy/evaluate.sh;板端执行 deploy/compile_board.sh 或 deploy/evaluate_board.sh.公共部署根目录由本地配置指定,模型使用末级名称作为子目录.板端总入口完成后回传结果,每端compile.log与evaluate.log覆盖保存.
