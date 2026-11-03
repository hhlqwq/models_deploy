# RTMPose Body2d 模型卡

## 正式开源上游

- 官方项目：<https://github.com/open-mmlab/mmpose>
- 目标变体：RTMPose-M、COCO-WholeBody 133 点、256×192 输入.
- 固定版本：`v1.3.2`.
- 官方配置：`configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py`.
- 官方权重：<https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth>
- 文件大小：72,010,049 bytes.
- 许可证：MMPose Apache-2.0；正式发布时仍需保留权重附带的第三方声明.

`deploy/host/export_model.py` 从本地官方 `.pth` 自行导出双 SimCC 输出 ONNX.原始框架、
ONNX 和 MTK NPU 必须使用同一个官方开源权重完成同协议精度对比.

## 正式模型规格

- 参数量：17.9M.
- 输入分辨率：256×192.
- 正式输入张量：`image [1,3,256,192]`,NCHW RGB float32 `[0,255]`.
- 导出图内使用 MMPose 官方 ImageNet mean/std 完成归一化,不包含 RGB/BGR 通道重排.
- 输出：`pred_x [1,133,384]`、`pred_y [1,133,512]`,SimCC split ratio 2.0.

模型输入是人体框裁剪结果,不包含人体检测器.本项目采用 1.25 倍人体框边距并调整到
192:256 比例后执行仿射变换.精度报告需要锁定原始数据集、人体框来源、仿射变换和
SimCC 解码配置.
