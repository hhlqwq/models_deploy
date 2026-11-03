"""固定 MIT 权重的纯 PyTorch 推理图,不加载训练框架或 CUDA 扩展."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Network(nn.Module):
    """直接使用官方参数名称,记录实际参与计算的权重."""

    def __init__(self, state, prefixes):
        """仅注册当前子图需要的参数,避免重复导出整个检查点."""
        super().__init__()
        self.names = {key for key in state if key.startswith(prefixes)}
        self.used = set()
        for key in self.names:
            self.register_buffer(key.replace(".", "__"), state[key])

    def get(self, key):
        """读取权重并记录消费情况,缺失参数直接报错."""
        self.used.add(key)
        return getattr(self, key.replace(".", "__"))

    def bias(self, prefix):
        """按检查点决定卷积或线性层是否有偏置."""
        return self.get(prefix + ".bias") if prefix + ".bias" in self.names else None

    def linear(self, x, prefix):
        """应用末维线性层,保持官方参数布局."""
        return F.linear(x, self.get(prefix + ".weight"), self.bias(prefix))

    def norm(self, x, prefix, eps=1e-5):
        """按最后一个通道维度执行 LayerNorm."""
        weight = self.get(prefix + ".weight")
        return F.layer_norm(x, weight.shape, weight, self.get(prefix + ".bias"), eps)

    def bn(self, x, prefix, eps=1e-5):
        """冻结运行统计量,不改变检查点的 BN 参数."""
        return F.batch_norm(x, self.get(prefix + ".running_mean"),
                            self.get(prefix + ".running_var"),
                            self.get(prefix + ".weight"), self.get(prefix + ".bias"),
                            training=False, eps=eps)

    def conv(self, x, prefix, stride=1, padding=0):
        """根据权重维数执行 Conv1d 或 Conv2d."""
        weight = self.get(prefix + ".weight")
        operation = F.conv1d if weight.ndim == 3 else F.conv2d
        return operation(x, weight, self.bias(prefix), stride=stride, padding=padding)

    def cbr(self, x, prefix, stride=1, padding=0, eps=1e-5):
        """执行 OpenMMLab ConvModule 的 Conv-BN-ReLU 推理路径."""
        return F.relu(self.bn(self.conv(x, prefix + ".conv", stride, padding),
                              prefix + ".bn", eps))

    def sequential(self, x, prefix, strides=(1, 1, 1), eps=1e-5):
        """执行 DepthLSS 和融合模块的三个顺序卷积块."""
        for index, stride in zip((0, 3, 6), strides):
            weight = self.get(f"{prefix}.{index}.weight")
            x = self.conv(x, f"{prefix}.{index}", stride, int(weight.shape[-1]) // 2)
            if f"{prefix}.{index + 1}.weight" in self.names:
                x = F.relu(self.bn(x, f"{prefix}.{index + 1}", eps))
        return x


class Camera(Network):
    """单路相机 SwinTiny + LSSFPN + DepthLSS,六路共用同一网络."""

    def __init__(self, state):
        """保持窗口大小 7 和官方 256x704 图像契约."""
        super().__init__(state, ("encoders.camera.backbone.", "encoders.camera.neck.",
                                "encoders.camera.vtransform.dtransform.",
                                "encoders.camera.vtransform.depthnet."))

    def window_attention(self, x, prefix, heads, shift):
        """按官方右下补零和循环移位执行窗口注意力."""
        height, width, channels = map(int, x.shape[1:])
        pad_height, pad_width = (-height) % 7, (-width) % 7
        padded_height, padded_width = height + pad_height, width + pad_width
        x = F.pad(x, (0, 0, 0, pad_width, 0, pad_height))
        mask = None
        if shift:
            x = torch.roll(x, shifts=(-3, -3), dims=(1, 2))
            labels = np.zeros((1, padded_height, padded_width, 1), dtype=np.float32)
            counter = 0
            for row in (slice(0, -7), slice(-7, -3), slice(-3, None)):
                for column in (slice(0, -7), slice(-7, -3), slice(-3, None)):
                    labels[:, row, column] = counter
                    counter += 1
            labels = labels.reshape(1, padded_height // 7, 7, padded_width // 7, 7, 1)
            labels = labels.transpose(0, 1, 3, 2, 4, 5).reshape(-1, 49)
            differences = labels[:, :, None] - labels[:, None, :]
            mask = torch.from_numpy(np.where(differences != 0, -100, 0).astype(np.float32))
        window_indices = np.arange(padded_height * padded_width).reshape(
            padded_height // 7, 7, padded_width // 7, 7)
        window_indices = window_indices.transpose(0, 2, 1, 3).reshape(-1)
        inverse_indices = np.argsort(window_indices)
        # 固定索引替代六维窗口转置,保持 G720 算子的 rank <= 4.
        flat = x.reshape(1, padded_height * padded_width, channels)
        windows = flat[:, torch.from_numpy(window_indices)].reshape(-1, 49, channels)
        qkv = self.linear(windows, prefix + ".qkv")
        q, k, v = [part.reshape(-1, 49, heads, channels // heads).permute(0, 2, 1, 3)
                   for part in qkv.split(channels, dim=-1)]
        scores = (q * (channels // heads) ** -0.5) @ k.transpose(-2, -1)
        indices = self.get(prefix + ".relative_position_index").long()
        bias = self.get(prefix + ".relative_position_bias_table")[indices.reshape(-1)]
        scores = scores + bias.reshape(49, 49, heads).permute(2, 0, 1)[None]
        if mask is not None:
            scores = scores + mask[:, None]
        windows = (scores.softmax(-1) @ v).transpose(1, 2).reshape(-1, 49, channels)
        windows = self.linear(windows, prefix + ".proj")
        x = windows.reshape(1, padded_height * padded_width, channels)
        x = x[:, torch.from_numpy(inverse_indices)].reshape(1, padded_height, padded_width, channels)
        if shift:
            x = torch.roll(x, shifts=(3, 3), dims=(1, 2))
        return x[:, :height, :width]

    def forward(self, image, depth):
        """NHWC 图像和稀疏深度输入,返回 NHWC 深度概率及相机特征."""
        root = "encoders.camera.backbone."
        x = self.conv(image.permute(0, 3, 1, 2), root + "patch_embed.projection", 4)
        x = self.norm(x.permute(0, 2, 3, 1), root + "patch_embed.norm")
        outputs = []
        for stage, (blocks, heads) in enumerate(zip((2, 2, 6, 2), (3, 6, 12, 24))):
            for block in range(blocks):
                prefix = root + f"stages.{stage}.blocks.{block}"
                x = x + self.window_attention(self.norm(x, prefix + ".norm1"),
                                               prefix + ".attn.w_msa", heads, block % 2)
                ffn = self.linear(self.norm(x, prefix + ".norm2"), prefix + ".ffn.layers.0.0")
                x = x + self.linear(F.gelu(ffn), prefix + ".ffn.layers.1")
            if stage > 0:
                outputs.append(self.norm(x, root + f"norm{stage}").permute(0, 3, 1, 2))
            if stage < 3:
                # 官方 nn.Unfold 顺序为 C,kh,kw,不是原始 Swin 的四角拼接顺序.
                channels = int(x.shape[-1])
                x = torch.cat((x[:, 0::2, 0::2], x[:, 0::2, 1::2],
                               x[:, 1::2, 0::2], x[:, 1::2, 1::2]), dim=-1)
                unfold_order = np.arange(4 * channels).reshape(4, channels).T.reshape(-1)
                x = x[..., torch.from_numpy(unfold_order)]
                prefix = root + f"stages.{stage}.downsample"
                x = self.linear(self.norm(x, prefix + ".norm"), prefix + ".reduction")
        root = "encoders.camera.neck."
        for level in (1, 0):
            up = F.interpolate(outputs[level + 1], size=outputs[level].shape[-2:],
                               mode="bilinear", align_corners=False)
            outputs[level] = self.cbr(torch.cat((outputs[level], up), 1),
                                      root + f"lateral_convs.{level}")
            outputs[level] = self.cbr(outputs[level], root + f"fpn_convs.{level}", padding=1)
        root = "encoders.camera.vtransform."
        depth = self.sequential(depth.permute(0, 3, 1, 2), root + "dtransform", (1, 4, 2))
        x = self.sequential(torch.cat((depth, outputs[0]), 1), root + "depthnet")
        return x[:, :118].softmax(1).permute(0, 2, 3, 1), x[:, 118:].permute(0, 2, 3, 1)


class CameraDownsample(Network):
    """CPU BEV pooling 后的相机 BEV 卷积."""

    def __init__(self, state):
        """加载相机 downsample 参数."""
        super().__init__(state, ("encoders.camera.vtransform.downsample.",))

    def forward(self, bev):
        """将 360x360 相机 BEV 降采样为 180x180."""
        x = self.sequential(bev.permute(0, 3, 1, 2),
                            "encoders.camera.vtransform.downsample", (1, 2, 1))
        return x.permute(0, 2, 3, 1)


class Fusion(Network):
    """ConvFuser + SECOND + SECONDFPN + TransFusion 稠密热图."""

    def __init__(self, state):
        """加载融合,BEV 解码器及候选生成前的卷积参数."""
        super().__init__(state, ("fuser.", "decoder.", "heads.object.shared_conv.",
                                "heads.object.heatmap_head."))

    def forward(self, camera, lidar):
        """输出候选查询使用的 128 通道特征与未激活的类别热图."""
        x = torch.cat((camera, lidar), -1).permute(0, 3, 1, 2)
        x = F.relu(self.bn(self.conv(x, "fuser.0", padding=1), "fuser.1"))
        outputs = []
        for stage in range(2):
            root = f"decoder.backbone.blocks.{stage}"
            for index in range(0, 18, 3):
                stride = 2 if stage == 1 and index == 0 else 1
                x = self.conv(x, root + f".{index}", stride, 1)
                x = F.relu(self.bn(x, root + f".{index + 1}", 1e-3))
            prefix = f"decoder.neck.deblocks.{stage}"
            if stage == 0:
                up = self.conv(x, prefix + ".0")
            else:
                up = F.conv_transpose2d(x, self.get(prefix + ".0.weight"), stride=2)
            outputs.append(F.relu(self.bn(up, prefix + ".1", 1e-3)))
        features = self.conv(torch.cat(outputs, 1), "heads.object.shared_conv", padding=1)
        heatmap = self.cbr(features, "heads.object.heatmap_head.0", padding=1)
        heatmap = self.conv(heatmap, "heads.object.heatmap_head.1", padding=1)
        return features.permute(0, 2, 3, 1), heatmap.permute(0, 2, 3, 1)


class Head(Network):
    """CPU 选取 200 个候选后的 TransFusion 注意力和回归图."""

    def __init__(self, state):
        """固定 200 查询,32400 个 BEV 键,不裁剪注意力范围."""
        super().__init__(state, ("heads.object.class_encoding.", "heads.object.decoder.",
                                "heads.object.prediction_heads."))

    def posembed(self, position, prefix):
        """执行官方可学习的二维位置编码."""
        prefix += ".position_embedding_head"
        x = self.conv(position.transpose(1, 2), prefix + ".0")
        x = F.relu(self.bn(x, prefix + ".1"))
        return self.conv(x, prefix + ".3").transpose(1, 2)

    def attention(self, query, key, prefix):
        """保持位置编码同时加入 K 和 V 的 MIT TransFusion 语义."""
        weight, bias = self.get(prefix + ".in_proj_weight"), self.get(prefix + ".in_proj_bias")
        q = F.linear(query, weight[:128], bias[:128]).reshape(1, -1, 8, 16).transpose(1, 2)
        k = F.linear(key, weight[128:256], bias[128:256]).reshape(1, -1, 8, 16).transpose(1, 2)
        v = F.linear(key, weight[256:], bias[256:]).reshape(1, -1, 8, 16).transpose(1, 2)
        output = ((q * 0.25) @ k.transpose(-2, -1)).softmax(-1) @ v
        return self.linear(output.transpose(1, 2).reshape(1, -1, 128), prefix + ".out_proj")

    def forward(self, query, query_position, classes, features, key_position):
        """B,P,C 契约输入,返回 center,height,dim,rot,vel,heatmap 拼接结果."""
        root = "heads.object."
        query = query + self.conv(classes.transpose(1, 2), root + "class_encoding").transpose(1, 2)
        prefix = root + "decoder.0"
        query_embedding = self.posembed(query_position, prefix + ".self_posembed")
        key_embedding = self.posembed(key_position, prefix + ".cross_posembed")
        positioned = query + query_embedding
        query = self.norm(query + self.attention(positioned, positioned, prefix + ".self_attn"),
                          prefix + ".norm1")
        query = self.norm(query + self.attention(query + query_embedding, features + key_embedding,
                                               prefix + ".multihead_attn"), prefix + ".norm2")
        ffn = self.linear(F.relu(self.linear(query, prefix + ".linear1")), prefix + ".linear2")
        query = self.norm(query + ffn, prefix + ".norm3")
        results = []
        for name in ("center", "height", "dim", "rot", "vel", "heatmap"):
            prefix = root + f"prediction_heads.0.{name}"
            result = self.cbr(query.transpose(1, 2), prefix + ".0")
            result = self.conv(result, prefix + ".1").transpose(1, 2)
            if name == "center":
                result = result + query_position
            results.append(result)
        return torch.cat(results, -1)
