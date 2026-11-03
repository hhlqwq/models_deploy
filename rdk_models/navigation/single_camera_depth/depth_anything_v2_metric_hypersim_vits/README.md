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

当前状态: 2026-11-03 已导出 depth_anything_v2_metric_hypersim_vits.onnx,IR=9,opset=17,文件 99,652,380 字节.静态图改写检查和 ONNX checker 已通过,保存后重新读取确认输入 [1,3,518,686]、输出 [1,518,686].尚未运行 ONNX 单图脚本、量化、编译或板端推理,无真实精度、性能或 RSS 报告.

## 单张 RGB 图片推理

在具备 NumPy、OpenCV、ONNX Runtime 的 Python 环境中执行:

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

## 数据准备

在本地 dataset_root 对应的 NAS 公共数据集目录下放入:

```text
NYU_Depth_V2/
    nyu_depth_v2_labeled.mat
    splits.mat
```

- 标注数据约 2.8 GB: https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
- 标准划分: http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
- 官方说明: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

保留原始 MAT,代码按需读取 RGB 和 depths,无需重复生成完整图片与深度副本.不需要下载 Raw 428 GB 数据.

## 编译与部署

在已配置的 S100 容器内,仓库根目录执行:

```bash
bash rdk_models/navigation/single_camera_depth/depth_anything_v2_metric_hypersim_vits/deploy/compile.sh
```

主机顺序执行导出、校准、量化编译和上传,错误直接显示,不自动运行板端推理.输入为 FP32 RGB [1,3,518,686],模型输出为 FP32 米制深度 [1,518,686].
INT16 节点策略不等于已核实每层实际位宽;最终以真实编译结果和板端元数据为准.

登录 S100 的模型部署目录后执行 `bash compile.sh`,只做三个固定示例的通路检查并保存并排输入与深度图,不生成精度、性能或 RSS 报告.

## 正式评价

主机执行同目录 `deploy/evaluate.sh`,只使用已有 ONNX/HBM 生成同协议参考并上传原始数据与摘要,不重新量化或编译.首次上传包含约 2.8 GB MAT,提前确认板端磁盘空间.随后在板端模型部署目录执行 `bash evaluate.sh`.

板端需要 hbm_runtime、NumPy、OpenCV、SciPy、h5py.新增依赖安装包统一从 NAS tools/wheels 准备,不在模型入口在线安装.当前板端 h5py 尚未安装.
正式结果只写 results/summary.json,不生成逐样本深度文件、额外报告或日志副本.报告区分 Runtime 调用和端到端耗时,纯 BPU 设备计时缺失时状态保持 partial.
