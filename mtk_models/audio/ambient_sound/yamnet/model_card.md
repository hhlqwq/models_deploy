# YAMNet 模型卡

- 任务: 机器人环境声音事件分类,521 类,多标签分数.
- 上游: Google TensorFlow Models / AudioSet YAMNet,版本见资源来源记录.
- 功能参考: https://aihub.qualcomm.com/models/yamnet
- 输入: `1x1x96x64` Log-Mel,默认原生 INT16,局部INT16方案使用INT8输入.
- 输出: `1x521` Sigmoid 类别分数,窗口均值用于音频级评测.
- 前处理: 16 kHz 单声道,25 ms 周期 Hann,10 ms 帧移,96 帧窗,48 帧步长.
- 部署: Genio 720 / MT8189 / MDLA 5.3,CPU 音频前处理,Python Neuron Runtime.
- 评测: ESC-50 固定 47 类投影,第二至第五折主评测,同步报告全量指标.
- 默认交付: `w8a16`,全网络INT8权重、INT16激活,原生INT16输入与INT16输出.
- 可选部署: `int8_tail16`,第12层pointwise输出至分类头使用INT16激活,
  前段INT8激活,权重全为INT8,原生INT8输入与INT16输出;所有PTQ方案仅用第一折校准.
- W8A16: INT8 权重、INT16 激活、INT32 偏置,主计算层精度在转换后严格核验.
- 范围: 环境声音分类,不是语音转文字、声源定位或设备故障诊断模型.
- 源码许可: Apache-2.0;ESC-50 整体数据许可 CC BY-NC 3.0.
- Genio 5100: 尚未部署测试.
- 实测结果: Genio 720 三后端全量完成,见 README 和 `results/summary.json`.
