# RoboTwin 跨本体控制适配规范

2026-11-03 · 阶段 4 · `handover_block` · Aloha-AgileX → ARX-X5。

本阶段交付坐标链、双向适配设计、失败协议及 CPU 回归。依据 [同状态 SAPIEN FK 验收](docs/robotwin_sapien_fk_smoke.md)、[阶段三数据契约](research/robotwin/data_contract.md)和 [PACE 变换验证](research/robotwin/pace_transform_validation.json)。历史 HDF5 的 `recorded_joint_target` 与 `recorded_endpose` 分别保留，`actual_qpos` 缺失，历史残差沿用[原记录](robotwin_fk_validation.md)。本阶段的数值验收仅适用于静态同状态几何。IK 可达率、控制跟踪、接触与任务成功率留待后续实测。

## B1 · 坐标链和工具点

列向量约定。矩阵 `T_A_B` 把 B 中的点变换到 A，平移在最后一列，最后一行为 `[0,0,0,1]`。组合 `T_A_C = T_A_B @ T_B_C`，求逆用 `[Rᵀ, −Rᵀp; 0, 1]`。长度用 m，旋转关节用 rad。对外位姿为 `[x,y,z,qw,qx,qy,qz]`。WXYZ 四元数先检查有限性和非零范数，再单位化。q 与 −q 表示相同旋转，连续插值先对齐符号并走最短弧，姿态误差按 SO(3) 角距离计算。

| 本体 / 侧 | 静态夹具 `T_W_B` 的平移 | 根姿态 WXYZ（归一化前） | 六关节顺序 | 末端 child / joint |
| --- | --- | --- | --- | --- |
| Aloha 左 | (0, −0.65, 0) | (0.707, 0, 0, 0.707) | `fl_joint1…6` | `fl_link6 / fl_joint6` |
| Aloha 右 | (0, −0.65, 0) | (0.707, 0, 0, 0.707) | `fr_joint1…6` | `fr_link6 / fr_joint6` |
| ARX 左实例 0 | (−0.30, −0.35, 0.784) | (0.707, 0, 0, 0.707) | `joint1…6` | `link6 / joint6` |
| ARX 右实例 1 | (+0.30, −0.35, 0.784) | (0.707, 0, 0, 0.707) | `joint1…6` | `link6 / joint6` |

Aloha 双臂共享一个 articulation 根，URDF 保留两侧各自的安装偏移。ARX 配置根平移是 `(0, −0.35, 0.784)`，场景沿世界 x 加 `±embodiment_dis/2`。表中 0.60 m 是 FK 夹具的显式间距，历史采集基座仍未知。运行时保存每侧实际根矩阵、关节名及 articulation ID，ARX 同名关节以实例区分。由 active joint 名称查索引，排除轮、其他臂和夹爪。侧别由本体配置确定。

对每个本体和侧，令 W 为世界、B 为 articulation 根、C 为末关节的 URDF child link、J 为 SAPIEN 运动后的末关节坐标系、E 为 RoboTwin EE、P 为 RoboTwin TCP。

```text
T_W_C(q) = T_W_B @ FK_B_C(q)
T_W_J(q) = T_W_C(q) @ T_C_J         # T_C_J = pose_in_child
T_W_E(q) = T_W_J(q) @ T_J_E
T_W_P(q) = T_W_J(q) @ T_J_P
G = global_trans_matrix; D = delta_matrix; R = G @ D
T_J_E = rigid(R, R @ [gripper_bias - 0.12, 0, 0])
T_J_P = rigid(R, R @ [gripper_bias,        0, 0])
T_E_P = Trans(+0.12, 0, 0)          # 沿 E 的 x
```

URDF FK 累乘 `T_origin(xyz,RzRyRx) @ T_motion(axis,q)`。SAPIEN 的相同状态另满足 `T_W_parent @ pose_in_parent @ motion_x(q) = T_W_C @ pose_in_child`。当前两份配置均为 `G = diag(1,−1,−1)`、`D = I`、`gripper_bias = 0.12 m`，所以 E 与 J 原点重合，轴向仍有旋转。按固定源码，`get_*_ee_pose` 返回 E，`get_*_tcp_pose` 返回 P。几何链以 SAPIEN 3.0.0b1 导出及源码 `13c3c47ff4312dd62484bcd51be034af55c062d1` 为准。 [源码](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/envs/robot/robot.py) · [配置审计](research/robotwin/embodiment_config_audit.json)

**共同任务坐标与工具点。** 每个场景定义任务系 K（例如桌面标定坐标系），每个本体、每侧提供 `T_W_K` 和从 RoboTwin TCP 到共同操作工具系 U 的 `T_P_U`。用 `T_K_U = inv(T_W_K) @ T_W_P @ T_P_U` 传递双臂目标。共用同一世界与任务锚点时可明确登记 `T_W_K = I`。CPU 几何夹具可登记 `T_P_U = I` 来比较 RoboTwin 的名义工具点。接触回放前须确认夹指几何、抓取轴和实际操作点，记录每侧外参、标定方法、版本及容差。目标端通过 `T_W_P = T_W_K @ T_K_U @ inv(T_P_U)` 恢复自己的 TCP，再通过 `T_W_C = T_W_P @ inv(T_J_P) @ inv(T_C_J)` 恢复 child-link 目标。若后端 IK 接受 base 下的 child-link 目标，传 `inv(T_W_B) @ T_W_C`。若使用 RoboTwin 的 `*_plan_path` 封装，传入世界 E 位姿（由目标 P 右乘沿工具 x 的 −0.12 m 平移），由该封装处理其内部变换，先用已知 q 做目标回代，避免对同一工具偏移重复补偿。测试夹具通过仅确认名义点和轴的数学定义。

**夹爪。** PACE 两侧的槽位分别为 6、13，`g ∈ [0,1]`，0 闭合、1 张开。配置指令坐标 `s = −0.01 + 0.055g` m。Aloha 各用 `fl_/fr_joint7`，ARX 各用所在实例的 `joint7`。两者 `joint8` 均按 `1·s + 0`，指轴方向由 URDF 决定。该坐标不是双指间距。URDF 每指允许 Aloha `[0,0.04765]` m、ARX `[0,0.044]` m。按配置直算时，两者的闭合端都低于下限，ARX 的张开端也超上限。严格预检将越界原始 drive 值返回 `position_limit`。运行前须核对后端饱和、夹指实际开口及接触行为，版本化任何端点重标定。输入侧优先用同帧实测指关节和标定逆映射构造 Aloha 归一化开度，同时检查 mimic 一致性。未标定时可离线记录配置逆映射 `(s+0.01)/0.055`，标记为开度代理。源码的 `get_normal_real_gripper_val()` 读取 drive target，实际夹爪状态须另取。

## B2 · 输入侧状态适配

1. 在同一物理步采集 ARX 两侧 actual qpos、指关节、基座和相机。快照含 `episode_id, step_id, sim_time_s`。异步采集须声明最大时差，超时、跨 reset 或缺任一侧即拒绝整帧。
2. 按名称取 ARX 各侧六关节，做 FK 并转换为 K 下的两个 U 位姿。用 Aloha 每侧的根和工具外参转换为对应 IK 目标。保持左右身份及双臂相对位姿。
3. 求 Aloha IK，以上一成功的双臂代理状态作种子。首次使用已记录的源本体初始参考状态，允许固定顺序的有限多种子搜索。先筛残差、位置限位、奇异性、碰撞及双臂可行性，再按与上次解的加权距离及离限位余量排序，等分按种子次序选。允许关节周期表示时只在配置许可且满足限位的等价角中选择。任一侧跳支超出时序边界则整帧失效。
4. 生成 `float32[14]` 的 `[q_AL(6),g_AL,q_AR(6),g_AR]`，语义标记为 `aloha_proxy_from_arx_actual`。三路 RGB 和 prompt 沿用阶段三接口。PACE 的已审计输入配置 `adapt_to_pi=False`，模型变换负责归一化。代理是从实际 ARX 状态构造的 Aloha 表示，另记录它与 PACE 发布评测所用驱动目标状态的分布差异。
5. 同时检查 Aloha URDF 限位及版本化的源本体控制边界。若本轮时间间隔已知，还需核对代理状态连续性及速度。标注超源数据支持域的状态和动作，示范范围及统计分位数用于记录分布偏离，执行边界由明确配置给出。标定、有效关节/速度边界或双臂解缺失时阻断执行。不可达点默认返回失败，近似投影仅可另作离线实验，需记录位姿残差与任务约束变化。

## B3 · 输出侧动作适配

1. 每个 PACE 输出行按 Aloha 绝对 14 维目标解释。对外输出已完成反归一化和关节 delta 还原，保持原有夹爪语义。逐行检查形状、有限值、夹爪范围、Aloha 位置与时序边界。动作块按顺序处理，行间及重规划时的首段均需检查。
2. 以 Aloha 左右各自的关节顺序和基座做源 FK，传递 `T_K_U` 与归一化开度意图。夹爪由目标侧标定生成指关节控制目标。禁止将 Aloha 六关节逐索引复制为 ARX 物理等价控制。若另做直接索引基线，只能标记为 `offline_negative_control_index_copy` 并做离线统计。
3. 以当前 ARX actual qpos 作目标 IK/规划起点，按 B1 生成目标 link 位姿并复查目标 FK 残差。两臂规划必须共享时间网格，联合检查自碰、双臂互碰、场景碰撞与持物约束，且同一 handover 事件中两臂和夹爪的顺序明确。对全路径核对关节位置、速度、加速度、夹爪可执行范围及插值后的中间状态。
4. 全部通过后，控制后端才接收带步号和时间的 ARX 左右位置、速度与夹爪轨迹。双臂指令按同一控制 tick 提交。新计划替换旧计划时从新鲜实际状态重建起点。IK/规划器、碰撞资产及后端执行验收属于后续工作。

## B4 · 失败协议与回归

输入适配与动作适配统一返回 `status, detail, diagnostic, payload`。仅 `ok` 可带可用 payload，输入为整帧观测，输出为完整同步计划。其他状态的 payload 为 null，控制提交门只接受 `ok`。诊断包含 episode/步号、阶段、侧别、资产与标定版本、出错关节或候选、残差和耗时。CPU 数学函数对非法几何输入抛 `ValueError` 等输入异常，适配边界转为 `invalid_input`。现有 CPU 代码只返回局部段检查结果 `Check(status, detail)`，它的 `ok` 只说明本项检查通过。

| 状态 | 触发与处置 |
| --- | --- |
| `invalid_input`, `unsynced`, `uncalibrated` | 缺字段、重复或缺少关节名、形状或数值错误、双臂不同步、缺标定。拒绝整帧或动作块。 |
| `position_limit`, `velocity_limit`, `timing_unknown` | 超执行边界、超速度或无法判断时序。记录失败并由明确的重定时流程生成新候选。位置越界需重新求解。 |
| `unreachable`, `ik_failed`, `timeout` | 仅在可达域预检或后端有明确证据时标记不可达。有限搜索无合格解用 `ik_failed`，达到时间预算用 `timeout`。 |
| `collision`, `plan_failed`, `backend_error` | 碰撞约束失败、规划无有效路径或后端异常。舍弃候选，后端已部分下发时进入双臂同步停止流程并重新读取实际状态。 |

失败时不推进代理种子或策略块索引，不补零、不提交单臂残留解、不沿用过期目标。执行中的失败由控制器进入预先验收的双臂 hold/stop，再按新快照重算，重复失败超过预算则结束试次。离线失败只记日志。CPU 段检查计算 `|Δq|/Δt` 并检查两个端点，它是必要条件，不能覆盖插值峰值、加速度和碰撞。

从仓库根目录用 Python 和 NumPy运行：

```bash
python -m unittest discover -s tests -v
```

9 项测试覆盖 SE(3)、WXYZ、非交换工具旋转、任务系工具点传递、动作布局与关节名、夹爪配置及 URDF 边界、时序和限位失效返回。读取归档 [SAPIEN FK JSON](research/robotwin/env-build/evidence-20261102/sapien_fk_smoke.json) 的 9 个静态快照和 12 个末端，用已导出的同 actual qpos 的 URDF child FK 作为输入，重算基座、joint、EE、TCP 并与 SAPIEN 记录对比，同时检查由 TCP 反推 child 的方向。回归沿用 1 mm / 0.1° 同状态容差。测试无需启动 SAPIEN 或读取历史 HDF5。

## B5 · 时间语义与专家路径重定时

历史样本的 `original_sample_period_s = "unknown"`，时间轴保留 `frame_index`。当前仿真步长、PACE 输出块长度及历史记录帧数各自有独立语义。归档中 `recorded_joint_target` 和 `recorded_endpose` 可分别作为候选几何路径的来源，回放方案须固定其中一个并保留帧与来源。关节目标路径经源 FK 得到工具路径，直接用末端记录时须先验证其坐标与工具定义。由相邻帧构造的量标为派生，原始数据另存。 [离线预检查](research/robotwin/offline_trajectory_precheck.json)

- **几何路径**以单调参数 u 组织共同任务系中的双臂 waypoint 和夹爪事件。位置分段插值，旋转用符号连续的 SLERP。保留抓取、交接、释放点和相对位姿要求。对较大步进细分并检查中间 IK 与碰撞。初始实际状态接入路径首点也要规划。
- **重定时**针对已生成的 ARX 联合关节路径和夹爪轨迹。由每段 `max_i(|Δq_i|/v_i,max)` 给出时间下界，再用明确版本的轨迹参数化器施加实际速度、加速度与事件停留约束。位置可行性及碰撞仍沿全路径复查。两臂共用一个时间标尺，离散化到运行时控制 tick 后重新核对限制。配置应固定速度/加速度来源，审计资产中的 `1000 rad/s` 不能当作实际运行速度上限。配置缺失、无法重定时或插值越界时拒绝执行。
- **执行账本**逐试次记录 `source_frame_count`、`source_frame_index`、`original_sample_period_s`、`retime_method/version`、新 waypoint 时间、`control_period_s`、`physics_dt_s`、`control_call_count`（后端调用次数，含失败尝试）、`physics_step_count`（实际 step 次数）、`sim_elapsed_s`（仿真钟差或步长求和）、`planning_wall_s`（单调墙钟，逐次及累计，含失败尝试）。可另记双臂控制 tick 数和策略查询数。一条指令可跨多个物理步，一次 tick 也可涉及多次调用。CPU 离线检查的控制调用和物理步为 0，规划未运行时耗时用 null。
