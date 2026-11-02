# RoboTwin FK 运动学对应验证

日期：2026-11-02。范围：`handover_block`，Aloha-AgileX / ARX-X5 各三条既有示范。本轮仅 CPU；未下载 PACE 大权重，未租 GPU。

**本轮 FK 对照完成，但运动学对应验收未通过，PACE 下载门槛保持关闭。** 关节链可计算，不等于历史数据中的关节值和末端位姿已被证明是同一物理状态。

## 方法和预设标准

沿已校验 URDF 的 parent→child 链累乘 `T_origin(xyz,RzRyRx) × T_joint(axis,q)`，按本体配置的关节名顺序读六关节，不把夹爪放入末端祖先链。两份 URDF 的 SHA-256 与既有审计一致。程序通过一个独立的解析案例：z 轴转90度后，1 m 的 x 轴固定杆端点为 `[0,1,0]`，旋转与 WXYZ 四元数计算一致。未使用外部求解器拟合关节参数。

在首次对照前固定最大位置残差≤1 mm、最大角度残差≤0.1°，用于判断“同状态、同工具点”的数值对应；本轮没有根据结果放宽阈值。这不是动态控制器跟踪容差。

每臂共检查 Aloha 854 帧、ARX 581 帧。程序分别记录：

1. 配置默认基座＋URDF 子链节＋当前源码的姿态/工具变换；另检查末关节转动之前的 joint-origin 假设。
2. 仅由 episode0/frame0 求得一个完整基座刚体变换，其余帧留出验证。该假设会错误地把工具姿态差吸收到基座中，结果也未通过，保留作排错记录。
3. 保持配置基座方向，用同一首帧只标定一个平移和一个**右乘**工具旋转；不逐帧、不逐 episode 拟合。以下表格仅为第3类诊断中的 child-link 假设，不能当作独立坐标系验收。

## 留出帧残差

| 本体 / 臂 | 留出帧数 | 位置 RMSE / 最大值（mm） | 姿态 RMSE / 最大值（°） | 结果 |
| --- | --- | --- | --- | --- |
| Aloha 左臂 | 853 | 1.882 / 5.964 | 0.268 / 0.731 | 未通过 |
| Aloha 右臂 | 853 | 7.051 / 37.912 | 0.531 / 2.727 | 未通过 |
| ARX 左臂 | 580 | 1.826 / 8.433 | 0.334 / 1.439 | 未通过 |
| ARX 右臂 | 580 | 10.275 / 60.807 | 1.018 / 5.740 | 未通过 |

默认配置检查还出现姿态接近180°的差异；ARX 默认基座未加双臂间距时位置差约300 mm。源码对两个单臂实例的 x 平移为 `±embodiment_dis/2`，而归档没有记录该任务采集时实际基座。首帧平移诊断可解释间距量级，不能据此宣布已独立恢复历史基座。

完整结果保留全部假设、基座矩阵、工具旋转、各 episode 的残差，见 [fk_results.json](research/robotwin/fk/fk_results.json)；复验脚本见 [verify_robotwin_fk.py](scripts/verify_robotwin_fk.py)。

## 发现的字段语义问题

在已固定的 RoboTwin 源码中，`get_obs()` 的 `joint_action` 调用 `get_left/right_arm_jointState()`，后者读取 **`joint.get_drive_target()[0]`**。另有 `get_left/right_arm_real_jointState()` 才读取 articulation 的 `get_qpos()`，但不在上述观测路径中。

因此当前源码对应的该字段是驱动目标，不是实际关节测量。历史采集源码版本仍未证明与当前源码完全一致，不能仅凭本次源码阅读重新断言全部旧文件的标签来源。但已有证据足以否定“字段名 qpos/joint_action 必然表示实际关节位置”的假设。

末端读取则使用 SAPIEN joint `global_pose` 再施加方向矩阵和工具偏移。SAPIEN 关节坐标系与 URDF child-link 坐标系需要通过 `pose_in_parent` / `pose_in_child` 对照，不能直接等同。本轮测试的 child/pre-terminal 两种假设不代替实际 SAPIEN joint-frame 导出。

来源：[固定机器人源码](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/robot/robot.py)、[固定 get_obs](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/_base_task.py)、[SAPIEN 关节坐标系说明](https://github.com/haosulab/SAPIEN/discussions/71)。

上述残差包含目标/实际关节状态不同、工具坐标及采集版本差异的潜在影响，不能当作纯 URDF 精度误差，也不能拿它证明控制器有多少跟踪误差。本轮不做任意时间平移、额外姿态补丁或逐帧拟合来制造通过结果。

## 下一项最小验证

在独立 SAPIEN 研究环境中加载同一 URDF（数学检查可移除 mesh visual/collision 依赖），对明确给定的 qpos 同时导出实际 qpos、drive target、root pose、末端 child-link pose、joint global pose、pose_in_parent 和 pose_in_child。先比较独立 URDF FK 与 SAPIEN child-link FK，再用导出的 joint-frame 关系验证 RoboTwin 的端点转换。固定关节顺序和基座后，记录实际 qpos 与 endpose 的同一物理步快照。

只有上述独立几何对应通过，才能把控制器跟踪误差与坐标适配误差分开。旧轨迹缺实际 qpos 的情况应保留为数据限制，不强求用 drive target 重现每帧实际末端位置。

## 后续独立检查结果（2026-11-02）

已在新建的独立 SAPIEN CPU 环境完成同状态静态几何对应：两种本体 9 个快照、12 次末端 FK 对照与 219 次关节坐标系对照全部通过，最大 child-link FK 位置残差约 0.000347 mm。确认 `joint.global_pose` 需通过 child-link 的 `pose_in_child` 转换后对应，详见 [SAPIEN FK 报告](docs/robotwin_sapien_fk_smoke.md) 与 [导出脚本](scripts/robotwin_sapien_fk_smoke.py)。

这使独立运动学对应的下一项验证完成；上文历史 HDF5 的控制目标/实际状态缺失问题及其旧残差仍保留，不改写为历史轨迹验收通过。未申请 GPU，未下载 PACE 权重。
