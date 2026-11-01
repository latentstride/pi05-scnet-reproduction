# RoboTwin 2.0：固定权重条件下的实验可行性

更新：2026-11-01。第二阶段优先使用 RoboTwin；第一阶段的 [LIBERO 基线](baseline_freeze.md) 保持冻结。此次完成官方资料和固定版本源码审计，没有安装 RoboTwin、下载示范或权重、启动仿真或提交 GPU 作业。

## 研究问题与当前决定

当前可推进的研究问题是：**同一双臂任务在不同本体上，经世界坐标末端目标和各本体控制接口适配后，轨迹误差、规划可行性及任务完成情况如何变化？** 第一组实验使用专家轨迹，π₀.₅ 暂不参与动作生成。

| 实验 | 动作来源 | 能回答的问题 | 当前判断 |
| --- | --- | --- | --- |
| 坐标及接口审计 | 元数据、运动学和已记录位姿 | 坐标、工具点、四元数、关节及夹爪语义能否正确转换 | 可先做 CPU 离线核验，尚未执行轨迹取样 |
| 同本体 `qpos` / `ee` 回放 | 同一专家轨迹的两种表示 | 重建和控制路径各引入多少误差 | 有官方接口，须先验证环境、初态及时间对齐 |
| 跨本体末端轨迹迁移 | 源本体专家轨迹，经目标本体 IK/规划执行 | 共享空间内的适配、可达性及执行差异 | 第二阶段主实验候选；尚未证明可成功回放 |
| 固定 `pi05_libero` 双臂闭环 | 现有单臂 7 维策略 | 当前配置不能给出完整双臂策略所需动作 | 不作为已可运行实验，不补零或复制动作 |
| 另选匹配接口的预训练 π₀.₅，参数固定 | 另一个明确来源的 checkpoint | 通过兼容性验收后，才可测其跨本体迁移 | 条件性候选，需独立检查训练本体、动作语义和归一化 |

“项目不进行微调”与“只能使用现有 `pi05_libero`”是两个约束。若坚持后者，本阶段先做数据/控制适配实验；若以后选择已有双臂预训练权重并固定参数，应单独命名实验，记录来源及已有训练经历。固定参数不代表该权重从未在目标任务或本体上训练。

## 固定源码证据

本次审计 RoboTwin `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`、XPolicyLab `0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3`。后者是单独审计的版本，**未验证其与前者子模块固定点组成可运行组合**。逐文件 Git blob SHA 见 [审计清单](research/robotwin_source_audit.json)。

### RoboTwin 的本体、任务及控制语义

固定本体注册表包含 `aloha-agilex`、`piper`、`franka-panda`、`ARX-X5`、`ur5-wsg`。官方任务目录有 50 类双臂任务；配置允许两个单臂组成双臂系统。这说明平台具备对照基础，不意味着每种本体都已在每个任务上验证可行。[本体注册表](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/env_cfg/task_config/_embodiment_config.yml)、[任务目录](https://robotwin-platform.github.io/doc/tasks/index.html)、[本体配置方式](https://robotwin-platform.github.io/doc/usage/configurations.html)

| RoboTwin 接口 | 动作布局 | 物理语义与处理 |
| --- | --- | --- |
| `qpos` | 左关节＋左夹爪＋右关节＋右夹爪；每臂 6 关节时共 14 维，7 关节时共 16 维 | 目标关节位置，经过 TOPP 时间参数化；不是关节速度或 LIBERO OSC 增量 |
| `ee` | 每臂 xyz＋四元数＋夹爪；两臂各 8，共 16 维 | 世界系绝对目标；经过工具点/方向转换及各臂规划器，不是逐物理步直接设置末端位姿 |

四元数使用 `[qw,qx,qy,qz]`；夹爪归一化 0 对应闭合、1 对应张开，最终映射取决于本体 `gripper_scale`。双 Panda 的 `qpos` 和 `ee` 恰好均为 16 维，仍是完全不同的接口。[控制文档](https://robotwin-platform.github.io/doc/usage/control-robot.html)、[动作执行源码](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/_base_task.py)、[机器人转换源码](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/robot/robot.py)

重点检查工具点：`get_obs()` 的 `endpose` 取自 `get_arm_pose()`，继而调用 `get_left/right_ee_pose()`。`_trans_endpose(is_endpose=False)` 在链节世界姿态上乘 `global_trans_matrix`、`delta_matrix`，位置沿转换后的局部 x 轴偏移 `gripper_bias - 0.12`；`get_left/right_tcp_pose()` 则使用另一偏移。因此不能直接把这些字段当成原始链节位姿或 TCP。回放必须沿匹配读取/目标转换路径验证，不能擅自再施加一次相同校准。[观测与位姿读取](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/_base_task.py)、[工具点和目标转换](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/robot/robot.py)

## XPolicyLab π₀.₅ 审计结果

`Pi_05` 已提供安装、转换、训练、模型服务和评测入口；旧版仅有 π₀ 教程的描述不再适用于当前代码。入口存在与 checkpoint 的跨本体能力分别核验。[适配器说明](https://github.com/XPolicyLab/XPolicyLab/blob/0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3/policy/Pi_05/README.md)

| 检查项 | 已读源码的事实 | 实验影响 |
| --- | --- | --- |
| 默认配置 | `action_type: joint`；`pi05_base_aloha_full_sim_arx-x5_seed_0`；归一化资产 `arx_x5_sim` | 不能只换 checkpoint 路径便认定兼容 LIBERO/Panda |
| 训练入口 | 对应配置从 `pi05_base` 初始化，训练数据为 RoboDojo ARX-X5 | 默认示例描述训练流程，不是现有 LIBERO 权重的零样本路线 |
| 状态和图像 | 打包双臂关节/末端字段，三路图映射为 high、左腕、右腕 | 需核验字段、排序和缺失视角；接口打包不产生控制能力 |
| 输出维度 | 所选 ALOHA 输出适配器固定截取前 14 维；joint delta mask 也是 6＋夹爪、6＋夹爪 | 不能原样支持双 Panda 的 16 维 joint；也不能据此认定完整支持 16 维 `ee` |
| 转换器 | 所读 `process_data.py` 读取 joint state，将下一帧状态作为 action；`action_type` 参数用于命名；fps 硬编码 50 | 传 `ee` 不会自动生成正确的末端动作；时间和标签均需重新审计 |

证据：[deploy.yml](https://github.com/XPolicyLab/XPolicyLab/blob/0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3/policy/Pi_05/deploy.yml)、[训练配置](https://github.com/XPolicyLab/XPolicyLab/blob/0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3/policy/Pi_05/openpi/src/openpi/training/config.py)、[模型封装](https://github.com/XPolicyLab/XPolicyLab/blob/0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3/policy/Pi_05/model.py)、[ALOHA 输出适配器](https://github.com/XPolicyLab/XPolicyLab/blob/0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3/policy/Pi_05/openpi/src/openpi/policies/aloha_policy.py)、[数据转换器](https://github.com/XPolicyLab/XPolicyLab/blob/0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3/policy/Pi_05/openpi/scripts/process_data.py)。

结论：可复用基础设施，不能原样宣布五本体 π₀.₅ 已打通。框架 `pack/unpack` 支持不同维度，不代表某个策略的训练变换、输出维度和归一化也已匹配。[通用状态打包](https://github.com/XPolicyLab/XPolicyLab/blob/0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3/utils/process_data.py)

## 数据入口与版本问题

优先检查 [官方 RoboTwin2.0 数据](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset) 中**相同任务下实际存在的不同本体归档**。将本体资产目录与示范目录分开核验：有某本体 URDF 不代表有该任务示范。本次尚未逐一确认所选任务下两种本体的归档和 episode，因此不列出未经核验的下载文件名。

`lerobot/robotwin_unified` 当前 metadata 标为 aloha，状态与动作均 14 维、30 fps、三路相机；不能作为覆盖五本体的数据证据，也不能用其 fps 替代原始 RoboTwin 控制时间。[实际元数据](https://huggingface.co/datasets/lerobot/robotwin_unified/blob/main/meta/info.json)

官方近期示范和 LeRobot 导出有 RGB、关节合法性与 seed 信息更新。HDF5 图像须使用对应版本的 `decode_image_bit`，避免旧/新 JPEG 编码混用造成颜色交换。固定原始数据 revision、资产 revision、机器人配置、图像解码器、转换源码及标签时间对齐；不只记录数据集名字。[官方数据入口与解码约定](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/README.md)

## 第一组可验收实验

任务候选为 `handover_block`：源码明确包含抓取、双臂交接和放置，并有环境成功判据。配置候选为 Aloha-AgileX 与双 Panda，用于暴露 14/16 维差异；这只是可行性核验候选，尚未证明两者都可加载、可达或有配套示范。不为凑本体数量绕过共同任务验证。[任务与成功判据](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/handover_block.py)

| 顺序 | 工作 | 验收证据 | 本次状态 |
| --- | --- | --- | --- |
| 1 | 固定源码、检查两个配置和同任务数据入口 | 版本、实际关节名/维度、数据文件清单；分别成功加载任务的日志 | 源码接口已审计；数据文件与加载待验证 |
| 2 | 每本体读取 1—3 条示范 | schema、相机、时间戳、关节/夹爪范围、末端位姿与工具点说明 | 待执行 |
| 3 | 静态 FK/位姿/参考系往返转换 | 同一物理工具点的数值对照，单位和四元数顺序明确 | 待执行 |
| 4 | 同本体 qpos 与 ee 回放 | 同初态、同目标；记录实际时间、规划失败及轨迹误差 | 待执行 |
| 5 | 源专家 EE 轨迹在目标本体回放 | 目标 IK/碰撞可行率、误差、专家回放成功率及视频 | 待执行 |
| 6 | 模型侧兼容性 | 权重及 norm stats 来源，原生动作含义，输出维度和三路观测，最小真实闭环 | 现有 `pi05_libero` 尚不满足双臂接口 |

不要直接跨本体复制 qpos。先在源本体 FK 得到明确工具点的世界轨迹，必要时以物体/任务参考系表达，再转换到目标基座及工具点，经目标 IK/规划执行。左右臂、基座间距、工具偏移、抓取宽度和共享可达空间均需显式记录。

同一 seed 在不同本体的初始化过程中未必产生相同物体初态，应保存并核对实际物体位姿。先关闭域随机化、固定相机及任务几何，成功判据沿用同一任务；记录规划器、速度/加速度约束，避免把这些差异全部归因于坐标适配。

## 指标与报告口径

位置误差使用同一世界系、同一工具点：`e_p(t)=||p_exec(t)-p_ref(t)||`，报告每臂 RMSE、P95、最大值，单位 mm。姿态使用归一化四元数：`e_R(t)=2*acos(clamp(abs(dot(q_exec,q_ref)),0,1))`，转为度；绝对内积消除 q 与 -q 的二义性。另报夹爪误差、双臂同步偏差及规划失败率。

RoboTwin 的动作调用可能包含多个物理步及重新定时，不能按数据行号直接配对误差。保存实际仿真时间，以共同时间轴或明确的任务相位对齐；若使用相位对齐，另报耗时/进度差，不能隐藏执行变慢或停滞。规划失败及不可达 episode 进入总分母，不仅统计成功回放。

成功率分开命名：`expert_replay_success_rate` 用于专家轨迹；`policy_success_rate` 仅用于模型根据新观测自主生成动作的闭环。当前二者均无实测值，不能把第一阶段的 15/15 填入 RoboTwin 结果。

数值往返误差和动力学执行误差分别设阈值。先取得原生专家回放误差作为参考，再预先固定正式对照的容差和评测 episode；不凭空宣布 1 mm 或某个成功率已达到。

## 资源与下一步

现在先做源码、metadata、少量轨迹及离线数学核验。官方推荐 Linux / Python 3.10，并涉及 SAPIEN、渲染、Vulkan 和 CuRobo；已通过 LIBERO 的 CUDA 推理环境不等于已通过 RoboTwin 图形/规划栈。不能承诺完整仿真可在本机 Mac 或无 GPU 条件下直接运行。[官方安装与硬件要求](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)

后续环境放在个人项目的独立研究目录，不改基线或课题组共享环境。当前不租 GPU、不下载全量、不训练。完成小样本审计后，再根据最小仿真所需资源、实际剩余预算和模型兼容性证据安排验证。
