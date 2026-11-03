# 外部模型资源

正式资源保存在本地 S100 平台配置 storage.models_root 下的 vision/depth_anything_v2/.
original/ 保存已有原始权重,目录最外层保存当前 ONNX,build/ 保存官方编译器输出.
官方 Python 源码和依赖安装资源归档在 toolchain.root 对应的 NAS S100/tools.
校准张量与打包过程文件放主机日期临时目录,不重复保存完整图片数据集.
