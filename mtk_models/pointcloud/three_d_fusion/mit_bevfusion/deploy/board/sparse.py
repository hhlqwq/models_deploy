"""CPU 稀疏主干,保持官方 spconv 的 xyz 坐标与 active-site 语义."""

import itertools

import numpy as np


def voxelize(points, max_voxels=160000):
    """按输入顺序保留体素和每体素前 10 点,对应官方 hard voxelization."""
    lower = np.array([-54, -54, -5], dtype=np.float32)
    size = np.array([0.075, 0.075, 0.2], dtype=np.float32)
    coords = np.floor((points[:, :3] - lower) / size).astype(np.int32)
    valid = ((coords >= 0) & (coords < [1440, 1440, 40])).all(axis=1)
    coords, points = coords[valid], points[valid]
    if not len(coords):
        raise ValueError("点云范围内没有有效点.")
    keys = np.ravel_multi_index(coords.T, (1440, 1440, 41))
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")[:max_voxels]
    remap = np.full(len(first), -1, dtype=np.int64)
    remap[order] = np.arange(len(order))
    groups = remap[inverse]
    indices = np.flatnonzero(groups >= 0)
    indices = indices[np.argsort(groups[indices], kind="stable")]
    sorted_groups = groups[indices]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1]
    ranks = np.arange(len(indices)) - np.repeat(starts, np.diff(np.r_[starts, len(indices)]))
    indices = indices[ranks < 10]
    sums = np.zeros((len(order), 5), dtype=np.float32)
    np.add.at(sums, groups[indices], points[indices])
    counts = np.bincount(groups[indices], minlength=len(order))
    return sums / counts[:, None].astype(np.float32), coords[first[order]]


def indice_pairs(coords, shape, kernel, stride, padding, submanifold):
    """构造每个核偏移的输入输出配对,SubM 不生成新的 active sites."""
    shape = np.asarray(shape, dtype=np.int64)
    stride = np.broadcast_to(stride, (3,))
    padding = np.broadcast_to(padding, (3,))
    output_shape = shape if submanifold else (shape + 2 * padding - kernel) // stride + 1
    offsets = list(itertools.product(*(range(n) for n in kernel)))
    candidates = []
    source_indices = []
    for offset in offsets:
        numerator = coords + padding - offset
        output_coords = numerator // stride
        valid = ((numerator % stride == 0) & (output_coords >= 0)
                 & (output_coords < output_shape)).all(axis=1)
        source_indices.append(np.flatnonzero(valid))
        candidates.append(np.ravel_multi_index(output_coords[valid].T, output_shape))
    if submanifold:
        output_coords = coords
        output_keys = np.ravel_multi_index(coords.T, shape)
        sorted_indices = np.argsort(output_keys)
        sorted_keys = output_keys[sorted_indices]
    else:
        output_keys = np.unique(np.concatenate(candidates))
        output_coords = np.stack(np.unravel_index(output_keys, output_shape), axis=1)
        sorted_keys, sorted_indices = output_keys, np.arange(len(output_keys))
    pairs = []
    for source, keys in zip(source_indices, candidates):
        positions = np.searchsorted(sorted_keys, keys)
        valid = positions < len(sorted_keys)
        valid[valid] &= sorted_keys[positions[valid]] == keys[valid]
        pairs.append((source[valid], sorted_indices[positions[valid]]))
    return output_coords, tuple(output_shape), pairs


def sparse_convolution(features, weight, output_count, pairs):
    """对每个核偏移执行小矩阵乘法,不创建全分辨率稠密张量."""
    output = np.zeros((output_count, weight.shape[-1]), dtype=np.float32)
    for matrix, (source, target) in zip(weight.reshape(-1, *weight.shape[-2:]), pairs):
        if len(source):
            # 同一核偏移的映射为单射,不需要有重复索引的高级索引累加.
            output[target] += features[source] @ matrix
    return output


class SparseEncoder:
    """加载官方稀疏参数,执行 4 个残差阶段并输出 256x180x180 BEV."""

    def __init__(self, weights, verbose=True):
        """接受 NumPy 权重映射,板端不依赖 PyTorch 或自编扩展."""
        self.weights = weights
        self.verbose = verbose

    def norm(self, features, prefix):
        """按官方 BN1d 的 eps=0.001 应用推理归一化."""
        state = self.weights
        scale = state[prefix + ".weight"] / np.sqrt(state[prefix + ".running_var"] + 1e-3)
        return (features - state[prefix + ".running_mean"]) * scale + state[prefix + ".bias"]

    def layer(self, features, coords, shape, prefix, stride=1, padding=1, submanifold=True):
        """执行卷积,返回特征及更新后的稀疏坐标."""
        weight = self.weights[prefix + ".weight"]
        output_coords, output_shape, pairs = indice_pairs(
            coords, shape, weight.shape[:3], stride, padding, submanifold)
        output = sparse_convolution(features, weight, len(output_coords), pairs)
        return output, output_coords, output_shape

    def __call__(self, points):
        """逐阶段打印进度,最终只展开低分辨率稀疏输出."""
        features, coords = voxelize(points)
        shape = (1440, 1440, 41)
        root = "encoders.lidar.backbone."
        features, coords, shape = self.layer(features, coords, shape, root + "conv_input.0")
        features = np.maximum(self.norm(features, root + "conv_input.1"), 0)
        for stage in range(1, 5):
            prefix = root + f"encoder_layers.encoder_layer{stage}"
            if self.verbose:
                print(f"[CPU 稀疏主干 {stage}/4] active voxels={len(coords)}", flush=True)
            for block in range(2):
                block_prefix = prefix + f".{block}"
                identity = features
                features, _, _ = self.layer(features, coords, shape, block_prefix + ".conv1")
                features = np.maximum(self.norm(features, block_prefix + ".bn1"), 0)
                features, _, _ = self.layer(features, coords, shape, block_prefix + ".conv2")
                features = np.maximum(self.norm(features, block_prefix + ".bn2") + identity, 0)
            if stage < 4:
                padding = (1, 1, 0) if stage == 3 else 1
                features, coords, shape = self.layer(
                    features, coords, shape, prefix + ".2.0", 2, padding, False)
                features = np.maximum(self.norm(features, prefix + ".2.1"), 0)
        features, coords, shape = self.layer(
            features, coords, shape, root + "conv_out.0", (1, 1, 2), 0, False)
        features = np.maximum(self.norm(features, root + "conv_out.1"), 0)
        if shape != (180, 180, 2):
            raise ValueError(f"稀疏主干输出形状错误: {shape}.")
        dense = np.zeros((*shape, 128), dtype=np.float32)
        dense[tuple(coords.T)] = features
        # 官方 C,D,H,W 合并,输出统一使用 NHWC 边界.
        return dense.transpose(0, 1, 3, 2).reshape(1, 180, 180, 256)
