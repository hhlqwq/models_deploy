# Genio 720 板端规范

## 平台配置

板端连接、资源目录及工具链参数以 [genio_720.yaml](../configs/platforms/genio_720.yaml) 为准.目标为 Genio 720 EVK,MT8189 / MT8391,MDLA 5.3.系统和 Runtime 版本随每次评价记录,不能由历史文档推断当前设备状态.

运行前确认 Neuron Runtime、APUSYS 设备节点和所需动态库可用.推理、前后处理和评价统一由 Python 直接调用 SDK Runtime,不依赖自编 C++ 程序或桥接库.组件存在或版本查询成功不代表模型已通过推理和精度验证.

## 编译与部署

编译在 Ubuntu Docker 容器执行,板端执行真实 NPU 推理.使用 `--arch=mdla5.3` 和 `--disallow-bridge`,整数原生 I/O 的转换参数须与实际张量契约一致.系统与模型所需 Runtime 不匹配时停止运行并核对版本,不覆盖系统库强行兼容.

主机 compile.sh 完成导出、量化、编译及固定子集浮点精度后,自动上传部署文件及必要参考.用户登录板端执行 compile.sh,完成同子集 NPU 精度、损失、固定少量输入的 CosSim/RMSE/NRMSE 和示例,保存 results/summary_test.json.全量 evaluate.sh 不重复采集数值参考.主机不会自动启动板端推理.普通入口无参数,主机总入口为 mtk_models/deploy/compile.sh 与 evaluate.sh,板端为 deploy/compile_board.sh 与 evaluate_board.sh;各入口依次运行全部模型,首个失败即停止.

入口直接顺序调用命令,输出和错误直接显示在终端,只补充必要阶段提示,不默认保存编译日志.Docker 公共环境由创建容器入口统一设置,板端路径复用公共工具.只保存运行、评分和排错必需的数据.

## 独立评价

在已有编译容器的模型目录执行 `bash deploy/evaluate.sh`,只读取已有浮点模型,生成同协议参考与必要输入.不导出、量化、编译或构建.主机参考及预测保存在本地日期过程目录,入口自动上传至板端 `build/reference/full.json`;LLM 的题目、Token 和质量评价资源在 `build/reference/full/`,BEVFusion 的样本清单及评分标注也在 `build/reference/`.这些是必要评价输入,不属于对外交付报告.

在现有板端部署目录执行:

```bash
bash evaluate.sh
```

评价入口使用已有 DLA 和 Python 脚本,执行完整精度评价、必要预热和性能统计,采集推理进程峰值 RSS,最终只给出 `results/summary.json`.不重新量化或编译;结束后调用 compile.sh 的仅回传分支,不会重新运行模型.重复评价成功后更新这一个报告,运行编号、数据集范围、版本和缺失项写在报告内部.

所有模型的中间预测、日志和内部指标统一位于各自部署目录的 `build/work/`.compile 和 evaluate 开始推理时覆盖该模型上次的 build/work,后续评分、展示和报告步骤复用同一目录;不按日期、阶段或运行编号累积目录,不写入板端 /tmp 内存盘.失败现场保留到下次推理开始,不拿上次指标冒充本次结果.覆盖范围只限本模型 build/work,不影响 models、examples、results 或其他模型.历史 /tmp/hailongcodex 不自动迁移或清理.

MiniCPM 正式评价为 MMLU 固定 1000 题、57 学科、seed=17 的分层子集与完整 WikiText2 test,不复用旧 `evaluation/full/` 资源.DeepSeek 仍使用 MMLU 完整 14042 题及完整 WikiText2 test.若已有上下文不足,明确报错,不截题或自动重编译.Python 生成使用官方 Token 输入,样例速度不能标为标准定长基准.新接入评价代码不代表已完成实测,成绩以对应版本的实际报告为准.

## 目录

```text
<board_root>/open_models/<model>/
├── compile.sh                 同子集精度、数值误差及示例
├── evaluate.sh                全量精度、性能和峰值 RSS
├── config.json                配置快照及模型目录声明
├── model_int8.dla              普通模型最终 DLA
├── models/                    多文件模型的必要运行包
├── board/                     Python 代码及公共模块
├── build/reference/           浮点参考和必要评价输入
├── build/work/                当前运行数据,覆盖复用
├── examples/input/            少量展示输入
├── examples/output/           展示结果
├── results/summary_test.json  子集精度、损失及数值误差
└── results/summary.json       全量精度、损失、性能和 RSS
```

原生 LLM 双 DLA、tokenizer 与脚本布局按模型说明维护,同样遵守两个入口、两阶段各自唯一报告,运行资源均位于 models/.正式模型、数据集和评价输入存放在配置指定的资源位置,临时目录不作为正式模型的唯一存储.

## 评价口径

- 精度使用相同样本、预处理、后处理和评分规则对比原框架、标准格式及 NPU.报告实际覆盖数量、基线和差值.
- 性能分别记录预热后的纯 NPU 调用和端到端耗时,注明输入形状、批量大小、重复次数及计时范围.LLM 的 Prefill 与 Decode 分别使用 token/s.
- 内存注明采集进程和范围,RSS 与系统可用内存变化分别记录.
- 自编样例、固定子集与全量基准必须区分.更换固件、Runtime 或模型后,旧指标不能直接标为新环境结果.

## 系统维护

刷机使用与板卡存储类型匹配的官方完整镜像,按 MediaTek Genio 工具说明操作.镜像、bootloader、kernel、firmware 和 Runtime 应来自匹配的发布组合.刷写和重启会中断设备任务,执行前确认维护授权.

Windows 刷写使用板卡对应的 USB 下载口和官方驱动,UART 用于串口诊断.解包镜像时应保留 Linux 符号链接语义,缺失文件别名时先修正解包结果,不要修改镜像内容.升级后重新确认网络、SSH、APUSYS 和 Runtime,再执行模型验证.
