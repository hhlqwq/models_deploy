# RDK 模型

本目录独立维护地瓜平台模型的导出、量化、编译和 Python 板端实现,不调用 MTK 转换器或 Runtime.
平台配置见 `configs/platforms/s100.yaml`,编译环境见 [S100 Docker](../docker/rdk_s100/README.md).

| 模型 / 版本 | 场景 / 类别 | 输入 / 参数量 | 原始权重 / FP32 ONNX | 数据集 / 指标 / 参考基线 | S100 量化方式 / 编译与性能 | S100 **CosSim**/RMSE/NRMSE/精度 |
| --- | --- | --- | --- | --- | --- | --- |
| [Depth Anything V2 Metric Hypersim ViT-S](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/README.md)<br>室内米制深度 | 导航 / 单目深度 | RGB `1×3×518×686`<br>— | 原始: —<br>ONNX: 99.65 MB | NYU test 固定100张子集<br>δ1<br>ONNX: 62.96% | **INT8 PTQ**<br>HBM: 57.96 MB<br>BPU耗时: —<br>RSS: — | 0.927799/1.621510/47.5908%/38.18%<br>(-24.78%)<br>[报告](navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/results/summary_test.json) |

数值误差依次为 CosSim、输出 RMSE、NRMSE,覆盖三张固定图片,RMSE 单位米,NRMSE 单位百分比.精度仅展示 δ1,覆盖100张子集.括号采用板端减浮点的有符号百分点差值,沿用 MTK 的 `%` 写法.缺少实测的数据以 `—` 表示.各量化实验对照见模型 README.

每模型只提供 deploy/compile.sh 与 deploy/evaluate.sh 两个主要入口.文件存在不代表已通过编译或板端验证.
compile 进行转换、量化、编译、部署、固定子集任务精度与少量输出数值检查并保存示例;evaluate 独立使用已有模型完成精度、性能和进程峰值 RSS.
compile 子集保存 results/summary_test.json,evaluate 全量保存 results/summary.json,缺少实测时不填造指标.
