# SAPIEN 静态 FK 与关节坐标系验收

日期：2026-11-02。**第二步脚本准备及 SCNet CPU 实测已完成。** 使用独立研究镜像中的 SAPIEN 3.0.0b1，仅创建 `PhysxCpuSystem`，不创建 renderer、不加载 CuRobo planner、不申请 GPU。第三步 GPU 图形验收按用户要求暂不执行。

## 方法与输出

脚本：[robotwin_sapien_fk_smoke.py](../scripts/robotwin_sapien_fk_smoke.py)。先校验已抽取的 Aloha-AgileX、ARX-X5 URDF 和配置 SHA-256，然后在独立输出副本中移除 visual、collision、material，保留所有 joint 定义、连杆关系、惯性、关节轴与限位。逐个 joint 的 XML 内容与原文件一致，原 URDF 未修改，无需下载 mesh 资产。

Aloha 加载一个双臂 articulation，ARX 加载两个单臂 articulation。每个实例分别设置 neutral、正向偏移、负向偏移三组限位内关节位置，并实读：

- active joint 名称与实际 DOF 顺序、`get_qpos()`、`get_drive_target()`；
- root pose、parent/child link world pose、joint global pose；
- `pose_in_parent`、`pose_in_child`；
- 按固定 RoboTwin 源码公式计算的 EE 与 TCP 世界位姿。

所有快照均在设置 qpos 后直接读取，**没有推进物理时间，也没有回放 HDF5**。驱动目标人为偏移约 0.025 rad，用于确认“实际状态”和“控制目标”是两组独立的 API 读数；该差值不代表控制器跟踪误差。ARX 基座间距 0.6 m 为显式诊断夹具参数，不宣称恢复了历史采集基座。

## 预设验收与实测

沿用前轮固定阈值：最大位置误差 ≤1 mm、最大姿态误差 ≤0.1°，不根据结果调整。独立 URDF FK 与 SAPIEN 在**相同实际 qpos、相同基座、相同 child-link** 下比较；另检验 PhysX 的 x 轴关节运动关系：

`T_world_parent × pose_in_parent × joint_motion(q) = T_world_child × pose_in_child`。

| 本体 | 静态快照 | 末端对照 | 关节坐标系对照 | child-link FK 最大位置残差 | 关节坐标系最大姿态残差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aloha-AgileX | 3 | 6 | 159 | 0.000347 mm | 0.009757° |
| ARX-X5 | 6 | 6 | 60 | 0.000148 mm | 0.0° |
| 合计 | 9 | 12 | 219 | 0.000347 mm | 0.009757° |

**全部通过。** child-link FK 的姿态残差输出为 0.0°（当前计算精度）。两种本体的 EE 对照一致支持 `joint.global_pose = T_world_child × pose_in_child`。因此 URDF child-link 不能直接当作 SAPIEN joint frame；应先使用实际 `pose_in_child` 再施加 RoboTwin 的 `global_trans_matrix`、`delta_matrix` 与工具偏移。这些转换来自仿真器与源码定义，没有通过拟合轨迹误差增加补偿量。

完整原始导出：[sapien_fk_smoke.json](../research/robotwin/env-build/evidence-20261102/sapien_fk_smoke.json)；URDF 准备校验：[prepared_manifest.json](../research/robotwin/env-build/evidence-20261102/prepared_manifest.json)；最终日志：[fk-smoke.log](../research/robotwin/env-build/evidence-20261102/fk-smoke.log)。

## 运行方式

从 `$HOME/code_cpf/pi05` 执行：

```bash
bash scripts/run_robotwin_env.sh python "$PWD/scripts/robotwin_sapien_fk_smoke.py" --prepare-only
bash scripts/run_robotwin_env.sh python "$PWD/scripts/robotwin_sapien_fk_smoke.py"
```

默认输出为个人项目 `research/robotwin/sapien-fk-smoke/`。仅准备 URDF 的模式不导入 SAPIEN；实际检查模式仍为 CPU，不需要 `ROBOTWIN_GPU=1`。

## 本轮结论的范围

独立 URDF FK 与同一仿真状态下的 SAPIEN 运动学对应已通过，第二步所需的状态和坐标系导出接口已跑通。此前 [历史数据 FK 报告](../robotwin_fk_validation.md) 的残差保留：旧 HDF5 未提供同一时刻的 actual qpos，不能因此改写为历史轨迹对应通过。

NVIDIA Vulkan 枚举、SAPIEN RGB 渲染、CuRobo CUDA 规划、带完整碰撞/视觉资产的任务与 π₀.₅ 闭环均不在本轮验收内。PACE 权重未下载。

来源：[SAPIEN 3.0 PhysX API](https://sapien-sim.github.io/docs/api/sapien.pysapien.physx.html)、[关节坐标系与运动轴定义](https://sapien-sim.github.io/docs/user_guide/getting_started/create_articulations.html)、[固定 RoboTwin robot.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/envs/robot/robot.py)、[本体归档与 URDF 哈希审计](../research/robotwin/embodiment_config_audit.json)。
