# 异构机器人数据集比较与接入决策

调研日期：2026-11-01。阶段一已通过功能性验收，固定点见 [baseline_freeze.md](baseline_freeze.md)。本报告核对公开说明、元数据和官方控制代码；尚未下载轨迹、运行候选数据加载器或验证候选仿真。本阶段没有提交 GPU 作业。

## 选型结论

选取 **DROID、BridgeData V2、ALOHA Sim Transfer Cube Human**，分别覆盖同本体不同接口、异构单臂和异构双臂。ALOHA 选择有对应仿真任务的搬方块数据，便于后续验证接口。

| 候选 | 本阶段用途 | 数据接入判断 | 现有 `pi05_libero` 能否直接控制 |
| --- | --- | --- | --- |
| DROID | 优先核对 Panda 关节速度与末端控制差异 | OpenPI 有 DROID 适配器；需固定 RLDS 字段、核对单位与标定 | 尚无证据；其 8 维关节状态不同于 LIBERO 的 8 维末端状态 |
| BridgeData V2 | 单臂跨本体位姿、动作及相机缺失研究 | 需新适配器，处理姿态基准、动作重标注、尺度、频率及夹爪约定 | 暂不接入闭环；7 维动作不等于同一控制接口 |
| ALOHA Sim Transfer Cube Human | 有原生仿真的双臂接口对照 | OpenPI 有 ALOHA 配置及仿真客户端；当前数据格式仍需适配 | 双臂 14 维绝对关节目标与当前单臂 7 维末端增量不兼容 |

以上是源码层面的可行性判断，新增三者均未通过当前 checkpoint 的直接闭环验收。更换 `pi05_droid`、`pi05_base` 或 `pi0_aloha_sim` 是独立实验，不能沿用第一阶段分数。[官方默认策略与权重映射](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/serve_policy.py)

## 本体、任务与观测

图像尺寸按 H×W×C 记，表示源数据尺寸。

| 项目 | DROID | BridgeData V2 | ALOHA Sim Transfer Cube Human |
| --- | --- | --- | --- |
| 来源 | 真实机器人示范 | 真实遥操作及脚本示范 | 人类示范的仿真搬方块数据 |
| 本体 | Franka Panda：单臂 7 关节＋夹爪 | WidowX 250 6DOF：单臂＋夹爪 | 双 ViperX：每臂 6 关节＋夹爪 |
| 任务 | 多场景日常操作，按 episode 指令选取抓放子集 | 抓放、推、扫、开关门抽屉、堆叠等 | 双臂传递方块，单任务 |
| 相机 | RLDS 三路左目 RGB：两外部、一路腕部；180×320×3；立体/高清另见 raw 数据 | 主固定、两随机外部、腕部；640×480；部分轨迹有深度 | `observation.images.top`，480×640×3，单路 RGB 视频 |
| 视角缺失 | OpenPI 示例选一路外部＋腕部 | 多数只有主视角，不能假设均有腕部或四路图 | 没有腕部图；缺失视角使用显式 mask |
| 频率 | 官方机器人及示例 15 Hz | 5 Hz | 数据 50 Hz；仿真 DT=0.02 s |
| 语言 | RLDS 提供指令字段，额外标注另发 | 轨迹级自然语言；加载器需开启语言读取 | 单任务，可按任务表映射 `Transfer cube`；不是多任务泛化集 |

依据：[DROID 平台](https://droid-dataset.github.io/)、[RLDS schema](https://droid-dataset.github.io/droid/the-droid-dataset.html)、[BridgeData V2](https://rail-berkeley.github.io/bridgedata/)、[ALOHA 元数据](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/blob/main/meta/info.json)、[ACT 仿真](https://github.com/tonyzhaozh/act/blob/main/sim_env.py)、[仿真常量](https://github.com/tonyzhaozh/act/blob/main/constants.py)。

## 状态、动作与坐标系

| 接口 | 状态 | 动作 | 转换与待核对项 |
| --- | --- | --- | --- |
| LIBERO 冻结基线 | 8 维：末端 xyz、轴角、两夹爪关节位置 | 7 维：6 维 OSC_POSE 输入＋夹爪；20 Hz | 输入裁剪后经控制器缩放，不是直接的米/弧度位移；沿用已验收客户端 |
| DROID / OpenPI | 7 关节位置＋1 夹爪位置；RLDS 另有 Cartesian 6 维状态 | OpenPI 输出 8 维：7 关节速度命令＋夹爪位置 | 关节排序、零位、速度缩放及夹爪标定；转 LIBERO 需 FK/控制器转换及基座/世界系标定 |
| Bridge 常见表示 | xyz＋3 维 Euler＋夹爪；具体 release 需实查 | 原始命令与重标注标签须分开；默认重标注为前后状态前 6 维之差＋二值夹爪 | 姿态基准、Euler/轴角、源/目标参考系、5→20 Hz 时间语义及控制器缩放 |
| ALOHA Sim | 14 维：左 6 关节＋夹爪、右 6 关节＋夹爪 | 同顺序的 14 维**绝对关节位置目标** | 各关节零位与轴方向、左右基座变换、夹爪归一化与 OpenPI 内部校准 |

依据：[LIBERO 实测接口](action_space_notes.md)、[固定 DROID 适配器](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py)、[DROID 执行客户端](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/main.py)、[Bridge 加载器](https://github.com/rail-berkeley/bridge_data_v2/blob/main/jaxrl_m/data/bridge_dataset.py)、[ALOHA 关节仿真](https://github.com/tonyzhaozh/act/blob/main/sim_env.py)。

### DROID：同为 Panda，接口仍不同

维护者说明 Cartesian 姿态为 xyz Euler，Cartesian 速度在机器人基座系表达；旋转速度为 Euler 增量的回复带有不确定措辞，不能直接等同于角速度向量。机器人代码检查速度命令位于 [-1,1]，物理尺度仍需追到后端，本报告不把命令值标成已经确认的 m/s 或 rad/s。[坐标约定回复](https://github.com/droid-dataset/droid/issues/9)、[RobotEnv](https://github.com/droid-dataset/droid/blob/main/droid/robot_env.py)

已发现两处需在接入前修正的问题：官方 RLDS 文档把顶层 `action` 写成 7 维并注释为六关节速度，但同页 `action_dict.joint_velocity` 为 7 维；须检查实际 TFDS features，明确选择关节速度与夹爪位置字段。另外，固定 `pi05_droid` 配置 horizon=15，而示例客户端断言 `(10,8)`，须使客户端与实际输出一致，不能认定示例对 π₀.₅ 可原样运行。[RLDS 文档](https://droid-dataset.github.io/droid/the-droid-dataset.html)、[模型配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)、[示例断言](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/main.py)

官方另发部分轨迹的改进外参，需按 episode 和相机 ID 匹配；旧 RLDS 与补充标定不能直接混用。[补充标注与标定](https://huggingface.co/KarlP/droid)

### Bridge：区分原始命令与 reached-state 标签

加载器默认 `relabel_actions=True`，前六维标签来自到达状态之差，未必等于原始命令；提供统计量时还会再归一化。应分别保留原始动作、重标注动作及统计量版本。[官方加载器](https://github.com/rail-berkeley/bridge_data_v2/blob/main/jaxrl_m/data/bridge_dataset.py)

机器人源码的状态转换包含 `default_rotation`，姿态并非简单的世界系绝对 RPY；动作绕当前末端位置旋转，轴与代码所称 world 轴一致。实验辅助函数又存在直接取矩阵 RPY 的路径。因此需沿所选 release 的导出流程核对参考姿态、基座和工作空间标定，不能仅凭列名硬编码变换矩阵。[状态/动作变换](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/utils/transformation_utils.py)、[目标构造](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/base/robot_base_env.py)、[实验辅助转换](https://github.com/rail-berkeley/bridge_data_v2/blob/main/experiments/utils.py)

xyz 应先统一参考系，姿态应先还原成旋转矩阵，再构造目标控制器增量。Euler 差不能直接解释成轴角；5 Hz 一步位移不能在 20 Hz 下原样重复四次。WidowX 与 Panda 关节空间也不能靠重命名统一。

### ALOHA：原生仿真可对照，当前权重接口不兼容

关节仿真中夹爪 0 闭合、1 张开；关节量按各自关节轴表达。末端位姿需结合 XML/运动学及双臂基座变换。ACT 另有 16 维双臂末端位置＋四元数接口，与本数据的 14 维关节接口不同。[关节仿真](https://github.com/tonyzhaozh/act/blob/main/sim_env.py)、[末端仿真](https://github.com/tonyzhaozh/act/blob/main/ee_sim_env.py)

固定 OpenPI 的 ALOHA 适配器还有关节符号变换及夹爪非线性校准。top 图可映射为 `cam_high`，腕部缺失使用官方 mask；这只说明格式可适配，不保证策略表现。不能截取单臂或补零后将数据塞进 `pi05_libero`。[ALOHA 适配器](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/aloha_policy.py)

## 数据版本、许可与下载范围

| 候选 | 研究入口 | 版本/存储事项 | 许可记录 |
| --- | --- | --- | --- |
| DROID | 官方 GCS RLDS，先考虑 `droid_100` | 官方调试子集约 2 GB、100 episodes；不先下载 TB 级全量。TFDS 版本和实际 features 尚未取样固定 | 本次查阅原始发布页未明确给出数据许可，入库前补核；代码许可不能代替数据许可 |
| BridgeData V2 | 官方原始数据及 TFRecord 加载器 | 遥操作/脚本子集分开记录；归档及转换源码 SHA 尚未下载固定 | 官方数据 CC BY 4.0，保留归属和引用 |
| ALOHA Sim | `lerobot/aloha_sim_transfer_cube_human` | 50 episodes、20,000 frames；首页内嵌说明 v2，实际 `meta/info.json` 已为 **v3.0**。元数据修改提交 `f20d5ec6357efe737b6441017b219f3f7c46766a` 可作候选 revision，尚未下载校验 | 发布者数据卡标 MIT |

依据：[DROID 下载及 schema](https://droid-dataset.github.io/droid/the-droid-dataset.html)、[Bridge 官方许可](https://rail-berkeley.github.io/bridgedata/)、[ALOHA 数据卡](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)、[实际 metadata](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/blob/main/meta/info.json)、[元数据提交](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/commit/f20d5ec6357efe737b6441017b219f3f7c46766a)。

OpenPI 基线固定旧 LeRobot 提交，不能凭名称相同认定可以读取 v3 数据。后续在独立研究环境使用匹配读取器，或核验兼容的历史数据版本，保留来源映射；不升级基线。ALOHA metadata 只声明 train，不能把随机帧切分当作独立测试集。[固定依赖](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/pyproject.toml)、[ALOHA metadata](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/blob/main/meta/info.json)

## 仿真对照与下一步

| 数据 | 可用对照路径 | 当前缺口 |
| --- | --- | --- |
| DROID | LIBERO 的 Panda 可用于未来控制器接口测试 | RLDS 不会自动提供相同场景、物体初态、相机和接触参数，不能直接重放原轨迹 |
| Bridge | 可研究 WidowX 模型和任务重建；当前未确认可直接复用的原场景仿真 | 本体、标定、场景资产、控制器和成功判据待补；先做离线几何适配 |
| ALOHA Sim | OpenPI 的 `gym_aloha/AlohaTransferCube-v0` 客户端和 ACT MuJoCo 任务 | 尚未安装运行；精确回放还需物体初态、随机种子及环境版本，本次 metadata 不能证明这些齐全 |

ALOHA 客户端为 50 Hz，官方默认权重是 `pi0_aloha_sim`，并非已经验收的 π₀.₅ LIBERO 权重。[仿真客户端](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_sim/main.py)、[默认权重映射](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/serve_policy.py)

下一步先以 CPU 小样本核验：DROID 做同本体接口审计，Bridge 做跨本体几何审计，ALOHA Sim 做原生仿真接口验证。每个候选读取 1—3 个 episode，保存实际 schema、revision、文件哈希、时间戳统计、相机可用率及动作范围；这一步不做全量下载、训练或 GPU 闭环。

统一记录保留：数据 revision、本体和左右臂标识、关节名、状态/动作类型、单位、参考系、旋转表示、时间戳/控制周期、夹爪约定、相机标定和缺失 mask、原始/重标注标签及归一化来源。保留原生维度，由适配层转换，不强制压成 8 维状态和 7 维动作。

新 GPU 实验之前需确认：小样本可读取且语义已核对；目标仿真能执行对应接口并提供成功判据；checkpoint、训练需求和独立评测划分明确。当前已完成资料调研与候选选型，这些实测门槛尚未完成。
