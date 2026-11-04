# Depth Anything V2 Metric Hypersim ViT-S

面向 RDK S100 的室内米制单目深度模型.使用作者提供的 Hypersim 微调 Small 权重,编码器为 ViT-S,官方推理配置 max_depth=20,输出单位为米.

- 官方源码: https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
- 官方使用说明: https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md
- 官方权重下载地址: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
- 权重文件: depth_anything_v2_metric_hypersim_vits.pth

已有 NAS 权重位于本地 storage.models_root 下的 vision/depth_anything_v2/original/,不复制到仓库,不重复下载.
已实际加载 NAS 权重并完成 ONNX 导出,尚未核实该文件与官方发布文件的一致性.官方 Metric Depth Python 源码已固定在 a561b849ebae10a6f5ef49e26c83cbbcd36c71bf,连同许可证归档至 NAS tools/Depth-Anything-V2-a561b849ebae/.

与 MTK 目录下 depth_anything_v2_small 的相对深度版本分开维护,不得复用其 DA-2K 历史结果作为米制深度评价.
校准使用 NYU Depth V2 标准训练划分中 seed=17 选取的 100 张图像,正式评价使用互斥的标准测试划分 654 张.使用米制真值、固定裁剪与有效深度范围,不进行逐图尺度对齐,契约见 [模型卡](model_card.md).

当前暂存 HBM 为 INT8 PTQ 基线,正式保留方案等待精度与性能对照后由用户选择.该基线已完成 100 张 NYU test 子集验证，文件大小为 57,961,064 字节。公开输入输出均为 FP32。

## 单张 RGB 图片推理

在具备 NumPy、OpenCV、ONNX Runtime 的 Python 环境中执行:

infer_onnx.py 是自包含文件,不导入仓库其他文件、不读取 YAML 或本地配置.已将该文件放到 NAS ONNX 旁边,交给测试者时只需提供这两个文件.示例依赖安装命令:

```bash
python -m pip install "numpy<2" "opencv-python<4.12" "onnxruntime==1.19.0"
```

复制到独立目录后执行:

```bash
python infer_onnx.py --model depth_anything_v2_metric_hypersim_vits.onnx --image rgb.jpg --output-dir output
```

在仓库中也可执行:

```bash
MODEL_DIR=rdk_models/navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits
python "$MODEL_DIR/deploy/infer_onnx.py" --image /path/to/rgb.jpg --model /path/to/depth_anything_v2_metric_hypersim_vits.onnx
```

默认在日期临时目录保存 depth_m.npy 和 depth_visualization.jpg,并标出原图中心像素的估计深度.可用 --output-dir 指定输出目录,用 --point X Y 指定原图像素,或用 --roi X1 Y1 X2 Y2 统计手工区域的有效深度中位数.右下边界不包含.
原始深度数组与输入图片同尺寸,depth_m[y,x] 的单位为米.可视化颜色使用固定 0-20 米色阶.

模型直接输出每像素的估计深度,不自动检测或命名物体.只有先确定目标区域,才能将相应深度统计解释为该区域的估计值;包含背景的矩形中位数不一定代表目标表面.沿光轴的深度与到相机光心的三维直线距离也不同,后者需要相机内参.该脚本是 ONNX CPU 推理,不代表 S100 BPU 推理.

单独导出无需 NYU 数据,在 S100 容器内的仓库根目录执行:

```bash
python "$MODEL_DIR/deploy/host/export.py"
```

IR 是模型容器格式版本,opset 是算子集版本,本模型分别设置为 9 与 17.仅对导出器生成的不高于目标 IR 的图提升版本,不通过直接改数字强行降级高版本图.

本次验证仅复制独立脚本与已有室内客厅图片到日期临时目录,在没有仓库 Python 路径和配置的环境中执行一次 CPU 推理.实际环境为 NumPy 1.23.0、OpenCV 4.11.0、ONNX Runtime 1.19.0.输出 FP32 深度为 [1362,2048],与原图同尺寸,全部数值有限,范围为 0.663-9.801 米.像素 (1300,920) 估计 2.802 米,手工电视屏幕区域 [560:760,450:570] 的深度中位数为 4.976 米.图片没有真实米制标注,这些数字只证明输出通路和区域取值已运行,不能据此判断测距误差.

## 数据准备

在本地 dataset_root 对应的 NAS 公共数据集目录下放入:

```text
NYU_depth_v2/
    original/                # 官方下载文件仅作来源归档
        nyu_depth_v2_labeled.mat
        splits.mat
    train/
        indices.txt          # 官方 795 张样本编号和顺序
        rgb/                 # 0001.png 等 RGB 图片
        depth/               # 同编号 FP32 NPY,单位米
    test/
        indices.txt          # 官方 654 张样本编号和顺序
        rgb/
        depth/
```

- 标注数据约 2.8 GB: https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
- 标准划分: http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
- 官方说明: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

首次下载文件放在数据集根目录,使用 `tools/prepare_nyu_depth_v2.py --dataset-dir /path/to/NYU_depth_v2` 一次性整理.该步骤需要 h5py 和 SciPy,逐张核对 RGB/深度数值无损后才将两个 MAT 移到 original/,不覆盖冲突文件.不需要下载 Raw 428 GB 数据.
日常校准、推理和评价仅读取 RGB PNG、米制 FP32 NPY 及样本索引,不再读取 MAT,不依赖 h5py 或 SciPy.数据整理需要对这个数据集目录的可写挂载,日常编译和评价继续只读挂载.

## 编译与部署

在已配置的 S100 容器内,仓库根目录执行:

```bash
bash rdk_models/navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/deploy/compile.sh
```

主机顺序执行导出、校准、量化编译和上传,错误直接显示,不自动运行板端推理.输入为 FP32 RGB [1,3,518,686],模型输出为 FP32 米制深度 [1,518,686].
量化采用 SDK `set_all_nodes_int8` 策略,不配置全局 INT16.实际权重、激活及 CPU/BPU 分配以本次编译报告为准,不以配置名称代替产物核对.校准集与评价集互斥.

主机在标准 test 划分上使用与 MTK 相同的固定间隔选样规则,保留 NYU 1、2、9 三张示例,总计 100 张.计算同协议 ONNX 指标,向公共数据集目录上传这 100 张 RGB PNG、米制 FP32 NPY 和选样索引.只有三张示例保留原始 ONNX 输出用于数值对比,不保存全子集浮点张量.

登录板端模型部署目录执行 `bash compile.sh`,核对模型、参考和子集数据身份,执行相同 100 张 HBM 推理并计算 AbsRel、RMSE(米)、log10、delta1/2/3 及相对浮点的损失.误差类指标越低越好,delta 指标越高越好,正的损失表示变差.三张示例额外计算 CosSim、输出 RMSE 和 NRMSE,其中 NRMSE=100*RMSE/RMS(ONNX),单位百分比.

子集任务精度、覆盖范围、量化配置及三张数值误差统一保存到 `results/summary_test.json`,不改写全量 `summary.json`,不统计性能或 RSS.示例保存到 `examples/output/`.

## 正式评价

主机执行同目录 `deploy/evaluate.sh`,只使用已有 ONNX/HBM 生成同协议参考并上传 test 的 RGB PNG、FP32 NPY、索引和参考摘要,不上传训练集或 MAT,不重新量化或编译.随后在板端模型部署目录执行 `bash evaluate.sh`.

板端环境需要 hbm_runtime、NumPy 和 OpenCV.新增依赖安装包统一从 NAS tools/wheels 准备,不在模型入口在线安装.板端编译小批量和正式评价均不需要 h5py 或 SciPy.
正式结果只写 results/summary.json,不生成逐样本深度文件、额外报告或日志副本.报告区分 Runtime 调用和端到端耗时,纯 BPU 设备计时缺失时状态保持 partial.

## 当前子集结果

固定 100 张 NYU test 子集实测见 [summary_test.json](results/summary_test.json)。

| 量化版本 | 数据集 / 指标 / 参考基线 | 量化方式 / 编译与性能 | **CosSim**/RMSE/NRMSE/精度 |
| --- | --- | --- | --- |
| INT8 基线 | NYU test 固定100张<br>δ1,ONNX: 62.96% | **INT8 基线**<br>HBM: 57.96 MB<br>Runtime: 36.54 ms<br>RSS: 109.38 MiB | 0.927799/1.621510/47.5908%/38.18%<br>(-24.78%)<br>[报告](results/summary_test.json) |
| INT8 + Softmax A16 | NYU test 固定100张<br>δ1,ONNX: 62.96% | **INT8 + Softmax A16**<br>HBM: 62.38 MB<br>Runtime: 40.48 ms<br>RSS: 105.06 MiB | 0.901099/1.659613/48.7091%/24.04%<br>(-38.92%)<br>[报告](results/summary_test.json) |
| 全局 A16 策略 | NYU test 固定100张<br>δ1,ONNX: 62.96% | **全局 A16 策略**<br>HBM: 129.17 MB<br>Runtime: 463.82 ms<br>RSS: 110.62 MiB | 0.999789/0.071238/2.0908%/63.27%<br>(+0.31%)<br>[报告](results/summary_test.json) |

三组使用相同100张训练图校准、相同100张测试图评分,输出误差覆盖同样三张输入.计时预热10次,重复同一输入100次.耗时为同步 Python Runtime 调用,包含绑定和输出转换,峰值RSS覆盖同一评分与计时进程.括号为板端减浮点的百分点差值,沿用MTK的百分号写法.

全局A16策略的57个主要权重校准节点为INT8,25个LayerNorm权重校准节点为INT16,Softmax校准节点为INT16.

当前INT8作为部署基线,最终保留方案等待用户选择.全局A16在该子集上接近浮点精度,平均调用耗时约为INT8的12.70倍.仅Softmax A16在本轮的精度和速度均未超过INT8基线.

![NYU 1 的 INT8 HBM 深度示例](examples/output/sample_0001_depth.jpg)

## NAS 交付文件与板端目录

NAS 本模型资源根目录保留最终 ONNX 和 HBM,`original/` 保存原始权重,`build/` 保存编译中间文件.编译成功后将 HBM 移至资源根目录,打包及评价均读取外层交付文件.

板端部署包统一存放于本地配置 `board.models_root/<模型名>/`,公共数据集存放于 `board.datasets_root/<数据集名>/`.实际路径只写入 `configs/platforms/s100.local.yaml`.三张通路图片保留在模型 `examples/input/`,正式评价数据不再放进模型目录.
