# RDK 模型

本目录独立维护地瓜平台模型的导出、量化、编译和 Python 板端实现,不调用 MTK 转换器或 Runtime.
平台配置见 `configs/platforms/s100.yaml`,编译环境见 [S100 Docker](../docker/rdk_s100/README.md).

| 模型 / 版本 | 场景 / 类别 | 输入 / 参数量 | 原始权重 / FP32 ONNX | 数据集 / 指标 / 参考基线 | S100 量化方式 / 编译与性能 | S100 **CosSim**/RMSE/NRMSE/精度 |
| --- | --- | --- | --- | --- | --- | --- |
| [Depth Anything V2 Metric Hypersim ViT-S](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/README.md)<br>INT8 基线 | 导航 / 单目深度 | RGB `1×3×518×686`<br>— | 原始: —<br>ONNX: 99.65 MB | NYU test 固定100张<br>δ1,ONNX: 62.96% | **INT8 基线**<br>HBM: 57.96 MB<br>Runtime: 36.54 ms<br>RSS: 109.38 MiB | 0.927799/1.621510/47.5908%/38.18%<br>(-24.78%)<br>[报告](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/results/summary_test.json) |
| [Depth Anything V2 Metric Hypersim ViT-S](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/README.md)<br>INT8 + Softmax A16 | 导航 / 单目深度 | RGB `1×3×518×686`<br>— | 原始: —<br>ONNX: 99.65 MB | NYU test 固定100张<br>δ1,ONNX: 62.96% | **INT8 + Softmax A16**<br>HBM: 62.38 MB<br>Runtime: 40.48 ms<br>RSS: 105.06 MiB | 0.901099/1.659613/48.7091%/24.04%<br>(-38.92%)<br>[报告](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/results/summary_test.json) |
| [Depth Anything V2 Metric Hypersim ViT-S](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/README.md)<br>全局 A16 策略 | 导航 / 单目深度 | RGB `1×3×518×686`<br>— | 原始: —<br>ONNX: 99.65 MB | NYU test 固定100张<br>δ1,ONNX: 62.96% | **全局 A16 策略**<br>HBM: 129.17 MB<br>Runtime: 463.82 ms<br>RSS: 110.62 MiB | 0.999789/0.071238/2.0908%/63.27%<br>(+0.31%)<br>[报告](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/results/summary_test.json) |

数值误差依次为 CosSim、输出 RMSE、NRMSE,NRMSE 单位百分比.深度模型覆盖三张固定图片,RMSE 单位米,任务精度展示100张子集的δ1.括号采用板端减浮点的有符号百分点差值,沿用 MTK 的 `%` 写法.缺少实测的数据以 `—` 表示.三组为同条件板端实测,最终保留方案等待用户选择.全局A16的LayerNorm权重含INT16.各量化实验对照见模型 README.

Whisper 数值列显示三条 Encoder 输出的范围,任务精度为50条音频的语料级WER,WER越低越好.当前同子集WER相同,保留W8A16.动态注意力两路输入实际为INT16/INT8,细节见模型说明.

每模型只提供 deploy/compile.sh 与 deploy/evaluate.sh 两个主要入口.文件存在不代表已通过编译或板端验证.
compile 进行转换、量化、编译、部署、固定子集任务精度与少量输出数值检查并保存示例;evaluate 独立使用已有模型完成精度、性能和进程峰值 RSS.
compile 子集保存 results/summary_test.json,evaluate 全量保存 results/summary.json,缺少实测时不填造指标.
