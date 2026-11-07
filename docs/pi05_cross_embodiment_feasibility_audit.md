# Aloha-AgileX → ARX-X5 跨本体可行性审计

## 专家数据语义

**CONFIRMED。** 官方 RoboTwin2.0 固定版本 `3dc3b798668feb99ac61cc9086d84cbcc3d79186` 提供两个本体的 `handover_block/*_clean_50.zip`。ZIP 目录各列 50 个 HDF5、50 个规划 PKL。CPU 复读的 Aloha/ARX 各三条 HDF5 只有 `joint_action/vector` 与分字段，没有独立根 `action`，14 列拼接逐值相同。关节字段取驱动位置目标，夹爪字段取归一化请求，`endpose` 另记录物理场景的末端。公开转换器生成的训练标签是下一记录帧，不能用来回填实际关节或实际控制时钟。 [官方任务](https://robotwin-platform.github.io/doc/tasks/handover_block.html) · [样本清单](../research/robotwin/sample_manifest.json) · [采集函数与转换路径](pi05_robotwin_protocol_audit.md)

**CONFIRMED。** 已有 Aloha 缓存和本轮读取的 ARX 三条规划 PKL 均对应每条 7 段左臂、4 段右臂，含六维位置及速度。按公开任务顺序重建同步进度、夹爪插值和记录点后，两本体各三条的 14 列 HDF5 全部逐值匹配。ARX 仅用约 214 kB 的 HTTP Range 读取目录和三个 PKL，成员 ZIP CRC 和 SHA-256 留证。ARX 是命令时序复算，无物理推进。

| 本体 | episode 0/1/2 保存帧 | 重建物理 tick | 以 4 ms 推算的持续时间 |
| --- | --- | --- | --- |
| Aloha | 295 / 277 / 282 | 3945 / 3655 / 3764 | 15.780 / 14.620 / 15.056 s |
| ARX | 195 / 192 / 194 | 2437 / 2401 / 2422 | 9.748 / 9.604 / 9.688 s |

多数帧间隔 15 tick，阶段边界存在 0、1 及其他间隔。这个时钟来自缓存和固定源码的独立重建。两份原始 HDF5 生成 commit、逐样本时间戳和作者训练 LeRobot 采样时钟仍为 **UNKNOWN**。公开的两本体轨迹并不因相同 episode 编号而自动成为成对初态。 [Aloha 缓存复算](../research/g2/expert-metadata-20261106/audit.json) · [ARX 缓存复算](../research/g2/protocol-audit/arx-cache-replay.json)

## 本体、坐标和夹爪

| 项目 | Aloha-AgileX | ARX-X5 |
| --- | --- | --- |
| 臂关节顺序 | 左 `fl_joint1…6`，右 `fr_joint1…6`，一个双臂 URDF | 左右是两个独立实例，各 `joint1…6`，终端 `joint6`/move group `link6` |
| 方向与限位 | 名义局部轴依次 z,y,y,y,z,x，六关节 URDF 原始范围均 ±10 rad | 名义局部轴同序，第 6 关节原始范围 ±3.14 rad，其余 ±10 rad。运行还需审计每臂有效标定范围 |
| 基座 | 名义双臂根 `(0,−0.65,0)` m | 单臂名义根 `(0,−0.35,0.784)` m，实际双臂基座由场景放置并逐次读取。正式几何使用实例实测根矩阵 |
| 工具点 | 末端关节、child link、RoboTwin EE、TCP 有不同定义 | 使用自己的 URDF 固定变换与工具偏置，不能复用 Aloha qpos。两者配置中的旋转矩阵相同也不能省略根坐标和链路变换 |
| 夹爪 | 归一化 0 闭、1 开，原生 `set_gripper` 映射到预载闭合至全开目标，实测位置可能受接触阻挡 | 两个实际手指分别读数。项目 `Grip.measure` 使用已标定开度通道，`Grip.target` 生成本体指驱动目标。接触时实际不对称与标量驱动分别保留 |

以上 URDF 是资产字段值，不能用作厂家实际安全额定值。关节原点/安装朝向、世界基座、末端偏置以及 `global_trans_matrix`、`delta_matrix`、`gripper_bias` 共同决定方向和物理工具点。项目 `g2_assets.load_arms` 核验资产哈希，`ArmChain.__init__` 取 URDF 与标定界限的交集，`Arm.offset` 应用 child/joint/TCP/tool 变换。两本体 episode0 的零臂关节首帧就有不同的世界末端位置。 [本体资产](../research/robotwin/embodiment_config_audit.json) · [本体实测初帧](../research/robotwin/probe_results.json) · [固定 Robot 源码](https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/envs/robot/robot.py) · [本体绑定](../scripts/g2_assets.py)

## 从源 FK 到目标路径

`g2_adapter.Adapter.actions` 在观测帧校验通过后，用源命名关节的 FK、源基座与工具偏置得到任务系末端目标。场景要求源、目标使用同一任务系。随后 `solve_pair/tool_ik` 在目标根坐标和工具点下逆解，验证工具 FK 回代、夹爪投影、关节限位及双臂碰撞。端点通过后 `g2_motion.QuinticMotion.plan` 才做限速/限加速度和路径检查。`Adapter.observe` 的方向相反，从 ARX 实测末端重建虚拟 Aloha 状态，源代理碰撞或连续性拒绝可发生在模型收到请求之前。 [适配器](../scripts/g2_adapter.py) · [时间参数化](../scripts/g2_motion.py) · [场景绑定](../scripts/g2_task_runtime.py)

| 检查阶段及分母 | 结果 | 证据范围 |
| --- | --- | --- |
| 原生专家驱动/FK 与实测回读，3 条共 854 帧 | Aloha 原始驱动命令逐值复原。物理回放实测 EE 对归档每条最大 0.167/0.014/0.314 mm，角误差 0.0561/0.0079/0.0325° | 源实测一致性。驱动目标 FK 与实际 EE 在接触时允许不同，不能把上述误差误称驱动跟踪差 |
| Aloha 驱动 FK 到 ARX 名义 TCP 的离线 IK，854 帧 | 双臂 IK 854/854。逐条最大回代位置 0.20114/0.19864/0.19861 mm，角度 0.00935/0.00906/0.00999° | 检查已枚举名义目标的几何可达性，未包含生产碰撞和真实接触 |
| 选定 IK 分支的帧间静态插值，854 帧段 / 3533 点 | ARX 托盘碰撞阻断 24/24/16 段，共 **64/854（7.49%）** | 采样步不超过 0.02 rad、1 mm。只对该插值及已知几何成立，持物位姿和连续扫掠未完整验证 |
| 后续 ARX 参考的实测帧边界，854 帧 | 799 clear、55 右指—托盘拒绝，**55/854（6.44%）** | 历史动态参考取样后静态 FCL，不能与上行合并。放置后缀 3/3 静态通过，未推进物理 |
| 动态专家参考，3 条/本体 | Aloha 原时序 3/3，ARX Aloha 映射后重定时的中心限力参考 3/3，均满足各记录的冻结任务判据及保持 | 后者使用直接物理驱动与影子接触监测，生产碰撞门禁 0/3 实际执行。ARX TCP 动态最大偏差 5.90/4.60/5.38 mm，与 IK 残差不同 |

因此静态端点结果支持「存在几何可达目标」。专家物理参考进一步支持「特定重定时与接触控制能完成任务」。新版生产链的完整动态安全路径仍缺验收。原生专家末段也曾有静态驱动几何指—托盘重叠，实测物理最大穿透 <0.013 mm；这只能说明须区分目标与实测，不能用来豁免策略候选或 ARX 碰撞。 [全帧 IK 与静态路径](pace_g2_workspace_stage6.md) · [Aloha 物理回放](pace_g2_native_expert_replay.md) · [ARX 中心限力验证](pace_g2_centered_grip.md) · [生产路径边界与后缀](g2_remaining_cpu_acceptance.md)

## 专家与预测的可比证据及 ARX 失败

- **CONFIRMED。** `127054052` 的四个 ARX 拒绝均发生于端点筛选，在该次拒绝发生前没有进入该目标的 QuinticMotion。seed0 两条为目标世界左指候选和物块重叠，seed1 K1 为实测工具转回源代理时虚拟 Aloha 左腕自碰撞，seed1 K4 为目标 ARX 右臂自碰撞。扩展 70/65/104/225 对仍拒绝。三条预测动作的源 FK 可从已存请求复核，源代理拒绝的那条没有新模型动作。 [碰撞归因](g2_arx_collision_analysis.md) · [四事件 CPU 几何复查](../research/g2/offline-repair-20261106-r1/static-replay-127054052-r3/replay.json)
- **CONFIRMED。** 在四个捕获的接近场景中，各测试官方 Aloha 专家 tick 0/25/100 的源 FK 目标和路径，12/12 通过。这个对照回答同场景存在可规划的专家接近目标。它没有声称专家 tick 与失败策略时刻对齐。与 seed0 原生接近对照一致，专家能先接近再闭爪，策略候选的指—块冲突发生在该记录的闭爪之前。 [专家目标对照](g2_offline_counterfactual.md) · [原生阶段对照](../research/g2/final-cpu-review/expert-approach-archived.json)
- **CONFIRMED。** `127139405` seed0 ARX 完成 25 个动作后无安全有界进展端点/路径。seed1 ARX 完成 156 个策略动作、4 个恢复动作，下一次源观测恢复请求触及额度。该次终止证明恢复预算有限，未证明当前或未来状态全局无 IK 解。旧监督器的误分类仅解释后十项没有运行。 [实际作业证据](../research/g2/final-cpu-review/result-127139405.json)
- **SUPPORTED_HYPOTHESIS。** 目标碰撞、源代理筛选和恢复限制均已在特定样本出现。某个源模型目标是否在原生状态也不合理、所有目标 IK 分支是否均无解、以及改变映射后是否仍处于模型训练分布，是不同问题。现有有限枚举和非同步专家相位对照无法给出全轨迹模型误差或全局不可达结论。禁止将不同本体、不同时刻或不同夹具状态的 qpos 逐帧相减作为模型错误。

后续匹配必须先按实测初态和接触相位确认可比性，并在共同任务系比较同一工具目标的原生、映射及真实执行误差。当前仍为 **UNKNOWN** 的项目包括目标资产在全部场景的完整动态约束、在线模型生成的接管目标可达性、真实持物后的恢复/放置路径，以及由原生评测结果向 ARX 迁移的成功率。
