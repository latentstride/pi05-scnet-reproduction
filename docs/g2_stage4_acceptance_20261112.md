# 第四步「动作空间与坐标系统一」最终验收

**结论：COMPLETED。** 验收日期为 2026 年 11 月 12 日。原始完成标准是建立一套可验证的统一动作表示与本体适配接口，并用真实仿真状态和控制接口验证 FK、IK 与坐标转换。本次复用已有实测证据、代码和归档统计。

| 验收项 | 核对结果 |
|---|---|
| B1. 坐标链与工具点 | 固定世界、基座、末端连杆、关节坐标系和 TCP 的变换方向、左右侧身份及 WXYZ 约定。已有同状态 SAPIEN 对照覆盖两本体的 9 个快照、12 个末端和 219 项关节坐标系检查。最大 child-link FK 位置残差 0.000347 mm，最大关节坐标系姿态残差 0.009757°，满足原 1 mm / 0.1° 阈值。[坐标规范](../robotwin_control_adapter_spec.md#b1--坐标链和工具点) · [实测对照](robotwin_sapien_fk_smoke.md) |
| B2. 输入侧状态适配 | ARX 实测关节经目标 FK 进入共同任务系的双 TCP，再由 Aloha IK 构造同步 14 维代理观测。夹爪按实测双指开度映射为归一化标量，保留左右侧、原生坐标和不对称读回。[观测接口](../scripts/g2_adapter.py#L307) |
| B3. 输出侧动作适配 | PACE 接口使用左六关节、左夹爪、右六关节、右夹爪的 14 维绝对目标。模型内部使用关节增量和源归一化统计，输出时还原绝对值，夹爪始终以 0 闭、1 开表达。源 FK 经共同任务系传递双 TCP，目标 IK、同步重定时和夹爪标定生成 ARX 双臂控制目标。[动作语义](../research/robotwin/data_contract.md) · [动作适配](../scripts/g2_adapter.py#L358) · [ARX 执行](../scripts/g2_robotwin_backend.py#L240) |
| B4. 解耦测试与异常处理 | 已有几何回归覆盖 SE(3)、四元数、工具偏置、关节顺序、夹爪、速度和限位。实际控制验证覆盖同步、规划拒绝、重观测失败和安全停止。不可达、越限、碰撞、时序异常和预算耗尽均有结构化诊断。[几何回归](../tests/test_robotwin_control_geometry.py) · [适配回归](../tests/test_g2_adapter.py) · [CPU 接口回执](../research/g2/arx-projection-online/cpu-summary.json) |
| B5. 时间语义与路径重定时 | 历史采样周期缺失时保留帧索引并标为 `unknown`。重定时使用明确的双臂共同时间标尺，区分规划耗时、控制调用和实际 PhysX 步数。[时间语义](../robotwin_control_adapter_spec.md#b5--时间语义与专家路径重定时) · [轨迹实现](../scripts/g2_motion.py) |

全量几何筛查覆盖 **47,628** 个单臂目标。Aloha → ARX 为 **28,268/28,268**，ARX → Aloha 为 **19,273/19,360**。有解结果均满足 0.5 mm / 0.05° 阈值。反向 87 个目标的有限搜索未解有逐项记录。筛查采用名义夹具，碰撞与动态执行另由控制回放验证。[全量报告](../research/robotwin100_course/REPORT.md) · [逐目标结果](../research/robotwin100_course/data/ik_results_full.csv)

已有两个独立 ARX 快照的 CPU PhysX 回放各执行 4 个真实控制动作，分别推进 495 和 610 步，随后从实测状态完成源代理重映射并发出下一请求。模型响应和 RGB 在该 CPU 接口验证中使用测试替身。两项已归档 GPU 试次另外验证了真实图像、冻结模型新响应和控制接口的衔接，任务均在接近阶段安全停止。完整抓取、交接和放置属于第六步的策略评测。[CPU 验证](g2_arx_projection_online.md) · [真实试次回执](../research/g2/arx-projection-online/gpu-results-20261112.md)

本次验收只读取仓库代码、归档统计和已有回执，没有重复测试或新增 GPU 作业。
