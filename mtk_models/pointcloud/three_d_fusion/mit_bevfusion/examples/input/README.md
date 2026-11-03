# 编译输入

一帧 nuScenes `v1.0-mini/mini_val`,sample token `3e8750f331d7499e9b5123e9eb70f2e2`.
通过模型资源目录 `build/sample.json` 记录原始六相机及 LiDAR 文件和标定,`sample.npz` 提供预处理后的真实输入.
数据来源为用户已有的官方 nuScenes 数据,不向 Git 复制数据集文件.正式评价不会使用该单帧替代 val 6019 帧.
