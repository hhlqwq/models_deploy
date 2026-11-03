# ResNet101 资源

正式资源位于本地配置的 `models_root/perception/image_classification/resnet101/`,仓库只保留代码与来源记录.

请下载 [resnet101-cd907fc2.pth](https://download.pytorch.org/models/resnet101-cd907fc2.pth)到该目录.它对应 TorchVision `ResNet101_Weights.IMAGENET1K_V2`,约 170.5 MiB;不使用名称不同的 V1 权重或厂商导出产物替代.

编译入口生成 resnet101.onnx 和 model_int8.dla,转换中间产物放 NAS build/.评价参考放主机本地日期目录,compile 及 evaluate 各自完成浮点评估后自动上传板端 build/reference/.板端最终 DLA 位于模型根目录,不再使用 models/onnx_reference.json 或单独上传入口.

完整地址、用途与固定版本见 [source_url.txt](source_url.txt),权重未在本仓库内下载.
