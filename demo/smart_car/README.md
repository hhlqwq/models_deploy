# 小车 Demo 总体开发与工程计划

总体集成仍处于设计阶段,下述统一 Makefile、CMake 和 robot_demo 接口尚未实现.语音模块已恢复推进: 两种音频输入共用 Silero VAD、自行部署的 Whisper tiny 和板端原有 DeepSeek,输出控制 JSON;Whisper 来源已由用户明确确认.目标、计划和实际状态以 [voice_control](voice_control/README.md) 为准.本文其余总体构建命令仍为拟定接口,不代表已连接底盘.

## 组成与开发顺序

小车各子 Demo 按独立运行设计,后续可通过 ROS 2 集成.模型来源、转换和按需评价由对应的 [MTK](../../mtk_models/README.md) 或 [RDK](../../rdk_models/README.md) 模型目录维护.Demo 按所选目标引用模型包.

| 子 Demo | 独立运行 | 集成时职责 | 当前范围 |
| --- | --- | --- | --- |
| [voice_control](voice_control/README.md) | USB 麦克风或音频文件 → Silero VAD → 已部署 Whisper tiny → DeepSeek → 控制 JSON | 提供六类语音指令识别结果,供后续 ROS2 集成 | Silero 与 FILE 小样本链路已验证,真实 MIC 待验收;目标为控制台输出,不下发运动指令 |
| [visual_perception](visual_perception/README.md) | RGB → Metric 深度 → 局部栅格 → A-star 绕障路径 → ROS 2/RViz2 | 接收目标并提供地图、路径和有效状态 | 单障碍物绕行路径展示,不接底盘 |
| `robot_demo` | 后续整体联调入口 | 协调语音任务、规划、轨迹跟踪与任务反馈 | 后续开发,不预建空应用 |

先完成两个独立 Demo,再开发整体应用.整体应用启动并协调已有模块,复用原代码,不复制一份语音或视觉实现.首版采用语音、视觉独立进程,避免长时间文本生成直接阻塞摄像头循环;多进程不意味着 SDK 一定支持并发,NPU 与内存资源须在集成时核实.

## 计划目录

以下目录按实际实现需要建立,本轮仅更新文档:

```text
demo/
├── README.md
├── Makefile                   # 单个/多个 Demo 的统一构建入口.
├── CMakeLists.txt              # 唯一顶层依赖图和安装规则.
├── package.xml                # 唯一顶层 ROS 2 包 robot_demos.
├── mk/common.mk               # Make 公共变量与子入口转发约定.
├── cmake/dependencies.cmake   # 按所选 Demo 查找依赖、声明 imported targets.
├── common/                    # 确实共用的 C++ 类型与少量基础功能.
├── platforms/
│   ├── genio_720/              # 平台绑定、工具链描述与能力声明.
│   ├── mtk/                    # SDK Runtime 和模型引擎的 MTK 实现.
│   └── linux/                  # 摄像头、网络、蓝牙等操作系统实现.
├── voice_control/
│   ├── Makefile                # 选择语音应用,转发统一入口.
│   ├── CMakeLists.txt          # 语音 targets,由顶层 add_subdirectory 引入.
│   ├── core/                  # 会话逻辑、ASR/LLM 抽象接口.
│   ├── adapters/ros2/         # 语音结果发布与任务反馈对接.
│   ├── mobile/                # 手机 App,独立 Gradle 构建.
│   └── ...
├── visual_perception/
│   ├── Makefile                # 选择视觉应用,转发统一入口.
│   ├── CMakeLists.txt          # 视觉 targets,不另起构建体系.
│   ├── core/                  # 几何、栅格、A-star 与抽象接口.
│   ├── adapters/ros2/         # 深度、栅格、路径及状态发布.
│   └── ...
└── robot_demo/                 # 整体联调时再创建.
    ├── Makefile                # 选择整体应用及依赖模块.
    ├── CMakeLists.txt
    ├── src/                   # 任务协调,不复制子 Demo 算法.
    ├── launch/                # 整体启动.
    └── config/                # 公开任务与话题配置模板.
```

每个 Demo 保留自己的功能实现、入口、README、必要示例及部署入口.厂商和 OS 实现集中在 `platforms/`,不在两个 Demo 各复制一套 MTK Runtime 或 Linux 设备层.各模块模型特有的后端文件按名称区分,由对应应用按需链接,不做一个包含全部模型的巨型平台库.

`common/` 不包含 SDK、ROS 2 类型或某一 Demo 的业务规则.仅提取确实被复用的数据状态、错误等小范围功能,不建立日志框架、插件注册中心或通用任务管理系统.已有仓库公共工具可以复用时先复用,不平行维护同义实现.

## Makefile 与 CMake 分工

用户使用 Makefile 选择构建范围和平台;CMake 负责源码、targets、依赖、链接顺序和安装.不在各 Makefile 重复维护编译器参数、头文件列表和 SDK 库列表.

拟定的 Ubuntu Docker 内命令如下,首个支持平台为 `genio_720`:

```bash
# 单独编译某个 Demo 的板端 C++ 应用.
make -C demo/voice_control PLATFORM=genio_720
make -C demo/visual_perception PLATFORM=genio_720

# 同一次构建中选择多个 Demo,共享依赖只构建一次.
make -C demo PLATFORM=genio_720 DEMOS="voice_control visual_perception"

# 编译当时已实现的独立 Demo,不自动包含尚未实现的整体应用.
make -C demo all PLATFORM=genio_720

# robot_demo 完成后才提供,同时选择它需要的子应用.
make -C demo robot PLATFORM=genio_720
```

构建规则:

- 根目录与子目录 Makefile 默认目标均为 `build`;子目录通过 `$(MAKE)` 转发到同一根入口并固定自身 `DEMOS`,不再配置另一套 CMake 工程.公共变量和转发规则集中在 `mk/common.mk`.
- 多 Demo 构建一次配置、一次构建,不并行递归启动多个 CMake 去写同一缓存.并发度由统一入口控制.
- `all` 只包含已实现且明确支持的应用,不是成功空操作.未知 Demo、未实现的 `robot` 或不支持的平台应在构建前明确报错.
- 默认不编译手机 App、不安装依赖、不下载模型、不执行测试、不上传设备、不运行程序.手机 App 使用 Gradle,可在语音 Makefile 中提供显式 `mobile` 目标,不将 Android 工具链混入板端 C++ 构建.
- `install` 只将已选应用及允许打包的必要依赖安装到暂存目录,不写主机 `/usr` 或板端系统目录.如提供 `clean`,只允许清理已解析并确认属于本任务的构建子目录,不得触及模型资源及其他任务.
- Makefile 命令行按 Make 语法使用 Tab;C++ 按仓库规范使用 4 空格缩进,函数包含中文说明.构建只保留必要阶段提示,不默认创建日志副本.

根目录 `CMakeLists.txt` 是唯一配置入口,子目录 CMake 文件只声明所属 targets.ROS 2 使用顶层唯一 `package.xml` 和 `ament_cmake`;若需工作区集成,colcon 也构建同一个包和同一套 CMake targets,不再给每个嵌套子目录增加 `package.xml`.不同时在每个 Demo 维护独立的 Make、colcon 和 CMake 依赖清单.

构建树置于本地日期临时目录,例如 Linux 的 `/tmp/hailongcodex/YYYY-MM-DD/demo/<platform>/<arch>/<build-type>/<demo-set>/`.Windows 侧准备文件使用 `D:/tmp/hailongcodex/YYYY-MM-DD/`.所选 Demo 集合规范化后作为目录标识,工具链/SDK ABI 变化时使用新的构建配置,不在旧 CMake 缓存里切换编译器.暂存安装目录与构建目录分开,不复用历史目录中的未选应用进行打包.

## 公共依赖与安装策略

依赖集中声明,按所选应用查找.已配置 SDK 与系统包优先复用,不在配置或构建时执行隐式网络下载.

| 类别 | 使用方 | 管理方式 |
| --- | --- | --- |
| C++ 编译器、CMake、Make、交叉编译 sysroot | 全部 | 复用公共 Docker 与本地工具链配置,核对版本和目标 ABI |
| ROS 2、ament、rclcpp、标准消息、TF | 两个子 Demo 与整体应用 | 同一目标发行版与运行环境,通过对应包的 CMake targets 引入 |
| MTK SDK Runtime | 需要 NPU 的应用 | 一个 imported target 描述头文件与动态库;各模型适配器按需链接 |
| Qwen 官方生成运行库与 tokenizer | 语音 | 区别于通用 NPU Runtime,先核实官方 API、ABI 与许可 |
| 音频处理、Whisper tokenizer、WebSocket 库 | 语音 | 优先满足已有资源与最小依赖;选择后固定来源和版本 |
| 图像处理、摄像头与可选显示库 | 视觉 | 只查找需要的组件,不为语音构建强制引入整套视觉依赖 |
| Android SDK/Gradle | 手机 App | 与板端构建分离,仅显式构建手机 App 时需要 |

实现时在依赖声明及对应 README 记录实际最低/验证版本、许可证、来源 URL 和用途,本阶段不编造已验证版本.第三方源码、SDK 与大型模型放已有外部资源目录,不复制进各 Demo.缺少某个已选功能必需依赖时立即说明缺失项,不自动启用 CPU 后端、替换模型或跳过该功能.

CMake 通过 target 级 `PRIVATE`、`PUBLIC`、`INTERFACE` 表达真实依赖,避免全局 `include_directories`、`link_directories` 和无差别编译宏.厂商头文件不进入核心库的公共接口;应用安装时只携带许可允许再分发的库.系统 Runtime 只检查匹配性,不覆盖系统库,不将主机 x86 库打包给 AArch64.

安装产物保留可重定位的包内目录关系,通过相对 RPATH 或明确的运行环境定位随包动态库,不把编译主机路径固化到运行包.ROS 2 包安装规则与部署打包读取同一份安装结果,不再手工拼接第二套依赖文件列表.

相关机制参考 [CMake target 依赖说明](https://cmake.org/cmake/help/latest/command/target_link_libraries.html)与[交叉编译工具链说明](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html),实际版本以现有环境核实结果为准.

## 平台切换设计

平台选择的公开入口统一为 `PLATFORM`,平台描述负责组合 **芯片后端 + 操作系统实现 + 工具链 + 必需能力**.首版只声明并实现 Genio 720,不提前将 G5100 或其他芯片写成已支持.

依赖关系:

```text
voice_core / perception_core
    只定义业务与抽象接口
             ↑ 实现
platforms/mtk 或未来其他芯片后端
platforms/linux 或未来其他操作系统后端
             ↓ 由平台工厂组合
应用入口 → ROS 2 适配层 → 独立运行或整体启动
```

具体边界:

1. 核心层只使用标准 C++ 数据结构与显式单位,不包含 SDK、ROS、V4L2、Socket 或系统绝对路径.ROS 2 适配属于通信边界,与芯片平台分开,换芯片不应重写消息接口.
2. `platforms/mtk/` 保存通用 Runtime 和 MTK 版 `IAsrEngine`、`ILlmEngine`、`IDepthEstimator`.纯数学、与后端无关的预后处理可保留在对应模块;张量原生布局、量化与 SDK 绑定留在后端,不向业务暴露.
3. `platforms/linux/` 保存摄像头、Wi-Fi/蓝牙传输等 OS 实现.相同 Linux 接口可被多个芯片平台复用,不为每块板复制一份.
4. `platforms/genio_720/` 只负责平台描述、工具链规则和应用所需服务的组装;用编译期显式选择的工厂创建接口实现,不扫描插件、不在核心层散布平台 `#ifdef`.
5. 工具链文件在 CMake `project()` 前选择,目标库从匹配 sysroot/SDK 查找,构建用工具与目标库分开解析.公开文件不硬编码开发者 SDK 根目录或设备地址.
6. 真实 SDK、编译器、sysroot、模型包和板端路径沿用 `configs/platforms/*.local.yaml` 等现有本地配置.构建入口只读取并解析所需值,不把 YAML 当 Shell 脚本执行,不创建第二套人工维护的 SDK 路径清单.
7. 启动时核对模型契约、所需服务和实际 Runtime.换平台后必须换成匹配平台的模型产物;语音与视觉引擎对外接口保持一致,但不承诺 DLA、量化或官方生成包可以跨芯片复用.

后续移植到新平台的工作顺序为: 新增工具链与平台描述 → 实现或复用所需芯片/OS 接口 → 准备匹配模型 → 重新构建 → 经授权完成目标板验证.完成后用户侧以 `PLATFORM=<新平台>` 选择.仅更换参数不能替代适配与实测,缺少能力时直接报错.

## 子 Demo 与整体应用的接口

- 语音独立模式继续向手机和终端输出文本,同时通过 ROS 2 提供转录、回答和请求状态.首版不自动执行自然语言中的运动要求.整体集成阶段再定义明确的开始、停止、取消等任务语义,携带请求编号,返回接受、执行、完成或失败状态.
- 视觉的几何、栅格和 A-star 核心接收显式目标.独立演示由入口提供固定前方目标;整体应用通过 ROS 2 目标接口提供目标,不修改 A-star 业务实现.原有局部路径、栅格与状态输出继续复用.
- 顶层协调器负责将支持的任务映射为目标与取消动作,不直接解析 LLM 文本生成电机指令.轨迹跟踪与底盘驱动后续独立接入,运动命令只由指定控制模块输出.
- 目标、地图、路径统一约定坐标系、米制单位、来源时间、请求关联及失效处理.实际行驶阶段需要里程计/定位提供稳定坐标,不能将每帧车前目标当作固定世界目标.
- 整体启动统一配置话题、命名空间和 TF 所有者,避免两个模块重复发布同一坐标变换.平台服务只提供能力,任务状态由协调器管理.

集成时单独核实三个模型的常驻内存、SDK 多进程/并发能力、视觉刷新和生成延迟.若需要协调 NPU 访问,根据实际限制制定最小调度方案,不先做一个公共推理服务或默认通过杀进程释放资源.

## 后续落地顺序

1. 确认公共工程设计,先实现统一 Make/CMake、目标选择、平台描述与最少依赖查找.只建立当前会使用的目录.
2. 实现语音、视觉各自核心和平台接口,独立 Demo 使用同一构建入口.编译及板端验证须单独按授权范围执行.
3. 接通两个 Demo 的 ROS 2 输入输出并完成各自展示,保留模块独立运行能力.
4. 创建 `robot_demo`,通过启动配置与任务协调复用子模块;再按后续授权接入轨迹跟踪与底盘.

本次只审阅并更新计划文档,没有创建实际 Makefile、编译工程、安装依赖或验证新平台.
