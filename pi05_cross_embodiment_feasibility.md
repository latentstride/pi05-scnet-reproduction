# Gate 2.5：固定权重 π₀.₅ 跨本体实验可行性审计

审计日期：2026-11-02。目标任务：`handover_block`。源本体：Aloha-AgileX；目标本体：ARX-X5。约束：本项目不更新模型参数，不改动已冻结的 LIBERO 基线。本轮仅做公开资源、源码、配置及 CPU 数据核验。

## 1. 验收结论

**Gate 2.5 审计完成，结论为“有条件继续”。** 已找到具有明确发布来源、公开参数分片及配套归一化的候选：`niejunnan25/pi05-robotwin` 中的 `handover_block` 专用 π₀.₅。可以推进独立环境中的源本体推理准备；跨本体闭环仍需控制适配和运行验收。当前不能宣布 ARX-X5 已可直接执行该策略。

本轮确认了真实参数分片的 HTTP 206 可读性；没有完整下载约 12.44 GB 参数，没有加载模型、测动作质量或启动仿真。因此“可获取”指公开资源入口已实测，不等于完整 checkpoint 校验或运行通过。

| 验收条件 | 本轮结果 |
| --- | --- |
| 明确的 π₀.₅ 来源和已有训练经历 | 通过：发布方从 `pi05_base` 对每任务 50 条 `demo_clean` 示范进行全参数微调，发布 step 30000 EMA 权重 |
| 权重可获取 | 通过入口检查：固定 HF revision、27 文件清单、参数分片前 4096 bytes 实际读取成功 |
| 本体与观测接口 | 源码发布评测配置为 Aloha-AgileX；三相机＋14 维关节/夹爪状态＋语言 |
| 动作与控制语义 | 已核对：服务输出 50×14 绝对关节/夹爪目标，模型训练内部使用关节 delta |
| 归一化 | 通过：任务专用统计下载成功，SHA-256 与发布 manifest 一致，14 维数值有限、分位数跨度均正 |
| ARX-X5 直接兼容 | 未通过：加载方式、运动学、相机及控制语义仍需适配验证 |

本项目采用此权重后不再训练，符合“使用已有权重并固定参数”。但它已由发布方在目标任务上训练，不能称为基础模型对该任务的零样本评测。Aloha→ARX 实验可称“固定源策略的跨本体迁移”；目标本体是否完全未出现在基础模型预训练中无法确认，也不能声称严格的目标本体零样本。

## 2. 候选对照与选择

| 候选 | 来源、本体及输入输出 | 配套归一化 / 控制语义 | 本轮判断 |
| --- | --- | --- | --- |
| 官方 `pi05_libero` | 已验收的单 Panda；双图像＋8 维状态；10×7 动作 | `assets/physical-intelligence/libero`；LIBERO 末端增量控制与夹爪 | 保留第一阶段基线；不能补零或复制成双臂策略 |
| 官方 `pi05_droid` | 单 Panda；外部与腕部图像＋7 关节及夹爪状态；15×8 输出 | `assets/droid`；官方客户端使用关节速度和夹爪位置模式 | 不作为当前双臂交接实验首选，不能把速度当绝对 qpos |
| 官方 `pi05_base`＋`pi05_aloha` 配置 | 官方基础权重；Trossen ALOHA 接口；14 维状态，输出截取 14 维 | `assets/trossen`；包含关节翻转、夹爪角度/线性转换及 delta→absolute | 真实候选，但 Trossen ALOHA 与 Aloha-AgileX/ARX-X5 不同；不因维度相同就直接使用 |
| **PACE 的 `handover_block` π₀.₅** | 发布方任务级微调；发布评测 Aloha-AgileX；三图像＋14 状态→50×14 | 任务自己的 `assets/robotwin`；无 Trossen 标定转换；最终绝对 qpos | **首选：最贴近当前任务、原有 JAX/OpenPI 路线和六关节双臂布局** |
| FlashVLA 的 RoboTwin π₀.₅ | 第三方 RoboTwin 多任务微调，ALOHA 数据配置；三图像＋14 状态→14 动作 | 保存的 QUANTILES pre/postprocessor；流式模型，配置 chunk=20、执行步数=10 | 备选；需独立 FlashVLA 栈，增加执行调度变量；本轮只核验资源与接口，不运行 |
| XPolicyLab `Pi_05` 默认示例 | RoboDojo ARX-X5 训练/服务配置 | `arx_x5_sim`，14 维 joint 适配器 | 是接口代码；已读说明未提供本任务可直接下载的匹配权重，不据此选定策略 |

官方对照依据：[模型发布清单](https://github.com/Physical-Intelligence/openpi#model-checkpoints)、[固定基线训练/推理配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)、[DROID 实际客户端](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/main.py)。本次三组官方统计均可公开读取：base/trossen 为 14 维，LIBERO state/actions 为 8/7 维，DROID 文件保存 32 维填充统计；统计长度不等于执行器动作维度。

备选依据：[FlashVLA 模型发布](https://huggingface.co/z-lab/flashvla-pi05-robotwin/tree/47c90d4fd00f36f68bee30c5f8e68c35f4a65e0b)、[RoboTwin 数据和评测说明](https://github.com/z-lab/flashvla/blob/8539f1cdfcf878e3a93c39f3d6c885c19ede6af8/sim_eval/robotwin/README.md)、[实际模型封装](https://github.com/z-lab/flashvla/blob/8539f1cdfcf878e3a93c39f3d6c885c19ede6af8/sim_eval/robotwin/policy/pi05_flashvla/pi05_flashvla_model.py)、[XPolicyLab 适配器](https://github.com/XPolicyLab/XPolicyLab/blob/0075cc96a1bf24c5d66a4a3a5e29eae6d739fef3/policy/Pi_05/README.md)。FlashVLA 参数文件元数据为 16,573,735,272 bytes；其两份小型归一化 safetensors 已下载并解析字段形状，未下载模型参数。

## 3. 首选权重：固定来源与获取证据

| 项目 | 固定值 |
| --- | --- |
| 发布项目 | PACE / OpenPI-PACE，发布方 `niejunnan25`；第三方权重，非 Physical Intelligence 官方 RoboTwin 权重 |
| 模型仓库 revision | `17ef545270cf26ed81169893b60f845f80ce6557` |
| checkpoint 子目录 | `handover_block/`，JAX/Orbax OCDBT |
| 参数＋任务资产 | 27 个文件，合计 12,441,552,737 bytes，约 12.44 GB；共享 tokenizer 另计 |
| 配置 | `pi05_robotwin_handover_block` |
| 当前审计模型源码 | OpenPI-PACE `f4e52e6de4cc487a11585a3a7414d610b2883dae` |
| 当前审计评测源码 | PACE `fd2e624cc19325cc9b20a620c82d51c0d8e25cc1` |
| 评测接口基础版本 | RoboTwin stable_2.0，`13c3c47ff4312dd62484bcd51be034af55c062d1` |
| 训练与评测本体证据 | 发布方 `task_config/demo_clean.yml` 为 `embodiment: [aloha-agilex]`；模型卡不单列逐 checkpoint 本体，训练本体归属仍以该项目配置证据为依据 |

模型仓库公开、未 gated。已读取发布 manifest、任务 norm stats、Orbax `_METADATA`，并从一个约 70 MB 的实际参数分片仅读取前 4 KB，响应为 `206`、`Content-Range: bytes 0-4095/70041649`。27 个文件的路径和大小与固定 revision 的 HF tree 对应；完整参数 SHA-256 尚未实测。

注意版本差异：模型卡与 manifest 记录的原始验证源码为 `ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12`；该版本已读 `config.py` 没有当前新增加的任务注册模块。当前发布说明把配置放在 `pace/src/openpi_pace/configs.py`，并称补丁在官方 OpenPI `2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea` 验证。**不能只 checkout 原始版本就认定新配置可用，也不能直接修改本项目冻结的 OpenPI 环境。** 应固定上述当前源码、权重 revision 和资产，另建研究环境；二者在 SCNet 的组合尚未运行。

来源：[固定模型卡](https://huggingface.co/niejunnan25/pi05-robotwin/blob/17ef545270cf26ed81169893b60f845f80ce6557/README.md)、[发布清单](https://huggingface.co/niejunnan25/pi05-robotwin/blob/17ef545270cf26ed81169893b60f845f80ce6557/manifest.json)、[当前模型配置](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/pace/src/openpi_pace/configs.py)、[源本体配置](https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/task_config/demo_clean.yml)。

## 4. 输入、动作及控制语义

| 层级 | 必须保持的约定 |
| --- | --- |
| 外部观测 | `state=[左6关节,左夹爪,右6关节,右夹爪]`；关节按对应 URDF 顺序，转动量按 rad，夹爪按归一化开合值 |
| 相机 | head→`cam_high`，left→`cam_left_wrist`，right→`cam_right_wrist`；客户端将 HWC uint8 转 CHW，服务内部 resize 到 224×224；语言通过 `prompt` |
| 本体专用变换 | `adapt_to_pi=False`：不施加 Trossen 关节符号翻转或其夹爪机构标定 |
| 训练动作变换 | `use_delta_joint_actions=True`，mask 为 `6,-1,6,-1`；12 个关节相对查询时状态，两个夹爪通道保持绝对值 |
| 模型内部 | π₀.₅ action horizon=50；内部填充维度与对外14维分开，不能将填充通道当作可用机器人控制通道 |
| 服务返回 | 去归一化→恢复绝对关节目标→截取14维；50×14，夹爪不再做 Trossen 反标定 |
| 环境执行 | 发布客户端逐行调用 `env.take_action(action)`，是关节目标路径；不是世界系末端 xyz/四元数，也不是关节速度 |

证据：[观测与执行循环](https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/pace/src/pace/rollout.py)、[ALOHA 适配器](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/src/openpi/policies/aloha_policy.py)、[变换链和 mask](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/src/openpi/training/config.py)、[统计加载及输出变换顺序](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/src/openpi/policies/policy_config.py)。

发布方按 50 Hz 序列设置 PACE；这只描述其执行协议，不能恢复我们旧版 HDF5 缺失的原始时间戳。首轮应使用固定执行窗口、记录实际仿真时间；暂不加入 PACE 自适应窗口、异步流水线等额外变量。发布入口硬编码 100 episodes，不能原样作为预算内最小验收，应另设单次短时 smoke test。

## 5. 归一化核验与现有样本检查

首选必须使用同目录的 `assets/robotwin/norm_stats.json`。文件 SHA-256 为 `8a1f114a8873dfa0788c7f512851ee3063224ef2b8af270421076a2e4e544960`，与发布 manifest 一致。state/actions 均有 14 维 mean、std、q01、q99，全部有限，q99−q01 均正。

π₀.₅ 配置使用 quantile normalization。关节动作统计描述经过 delta 变换后的量；不能用它直接归一化原始绝对目标。state 是绝对状态，夹爪为绝对开合。运行时应让发布的变换链处理，不额外手动重复 delta、归一化或反归一化。

CPU 诊断将已有六条轨迹的**状态**与源统计分位区间比较，没有构造伪动作标签：

| 样本 | 状态帧数 | 超出源 q01/q99 的关节标量比例（不计夹爪） |
| --- | --- | --- |
| Aloha-AgileX，各3条 | 854 | 2.54% |
| ARX-X5，各3条 | 581 | 4.82% |

这是数值覆盖诊断，不是策略能力、物理安全或失败率。夹爪 q99=0.9998，而样本张开值可为1，会放大“任一通道超界”比例，不能把该比例当作 OOD 成功/失败判据。两本体通道数相同也不代表关节物理语义相同。没有替换为 ARX 统计、裁剪数据或用目标数据重新拟合归一化。

证据：[固定任务统计](https://huggingface.co/niejunnan25/pi05-robotwin/blob/17ef545270cf26ed81169893b60f845f80ce6557/handover_block/assets/robotwin/norm_stats.json)。本轮机器可读结果见 [gate25_audit.json](research/gate25/gate25_audit.json)，复验脚本见 [audit_gate25_stats.py](scripts/audit_gate25_stats.py)；脚本需已有六条样本及该任务统计。

## 6. 可以继续哪一种 RoboTwin 实验

**可以继续准备：Aloha-AgileX 源策略原生接口验证，然后评估 Aloha→ARX 控制适配。暂不直接启动双本体成功率对照。**

当前 PACE loader 读取一个本体配置后供左右臂使用，并固定 `dual_arm_embodied=True`。我们的 Aloha 资产是一个双臂 URDF，ARX 资产是单臂 URDF 的两个实例；换一个本体名字不足以完成加载适配。两者基座、关节链和观测相机也需要分别验证。[实际 loader](https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/pace/src/pace/robotwin.py)、[已完成的本体审计](robotwin_data_probe.md#7-本体配置补读已完成)

| 路线 | 可回答的问题 | 当前决定 |
| --- | --- | --- |
| Aloha 原生 checkpoint＋原生观测/动作 | 候选权重是否能在源本体闭环执行 | 下一次模型运行的首要验收，尚未执行 |
| 直接把同一14维策略接 ARX qpos | 同通道布局的朴素直接迁移表现 | 可作为后续对照，但要先证明状态/零位/关节顺序/夹爪兼容；目前不放行直接闭环 |
| 源策略 qpos→源 FK→共同工具点→目标 IK/规划 | 固定源策略经几何与控制适配后能否迁移 | 研究候选；还需解决目标观测如何转成策略所需源状态，以及图像与工具点差异 |
| 专家轨迹的同本体/跨本体回放 | 纯数据和控制适配是否正确 | 独立验收，不可把它记为 π₀.₅ 自主控制 |

仅做输出侧 FK/IK 仍不完整：策略每次查询还依赖源本体语义的 state 和相机。ARX 原生14维状态不能因为形状相同就冒充源状态；可行的状态映射、源构型分支连续性、双臂协同、相机视角和夹爪标定都必须记录。IK/规划不可达必须记为失败，不得只保留可执行片段。

下一次正式运行前需完成：完整单任务权重及共享 tokenizer 获取并校验；独立模型环境加载与真实观测单次推理；GPU Vulkan/SAPIEN/CuRobo 通过；源本体短闭环；再验收 ARX 加载与控制/观测映射。所有运行资产继续放在 SCNet 的个人项目根目录内。本轮不自动提交 GPU 作业，也不下载50任务合集。

实验命名应区分 `source_policy_success_rate`、`target_policy_success_rate` 和 `expert_replay_success_rate`。第一阶段 LIBERO 15/15 不进入上述 RoboTwin 指标。若源本体自身无法稳定完成任务，应先定位版本、观测、归一化和控制错误，再讨论跨本体效果。
