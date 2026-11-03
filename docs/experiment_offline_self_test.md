# 阶段 6–7 离线自测记录

2026-11-03 · Python 3.13.9 · jsonschema 4.25.0

| 检查 | 结果 |
| --- | --- |
| Draft 2020-12 schema | 校验通过。 |
| 混合 JSON/CSV 输入 | 三类动作来源分别出表，完整记录可通过 episode ID 从失败索引回查。 |
| 统计计算 | 核对位置 RMSE、姿态 P95、规划失败率、越限率、夹爪状态、推理及控制耗时、执行窗口统计。合成配对的一方成功、一方不可达，差值为 −1。 |
| 初态与覆盖 | 同 seed 但物体位置偏差 5 mm 的配对被拒绝。不同 seed、物体姿态相同的配对通过。夹爪初态不同的同本体消融配对被拒绝。缺失条件和 reset 前故障分别进入配对覆盖统计。 |
| 输入隔离 | 拒绝重复试次、专家条件携带 checkpoint、错误数据类别、无效计数、权重不一致和条件漂移。 |
| 冻结清单 | 用合成记录演练正式入口。缺少预定试次或 seed 与清单不符时拒绝出表，完整且匹配时通过。 |

从仓库根目录复跑。

```bash
python -m unittest discover -s tests -p test_summarize_experiments.py -v
python -m unittest discover -s tests -p test_robotwin_control_geometry.py -q
```

新增评测自测 **8/8 通过**，现有控制几何回归 **9/9 通过**。评测自测在临时目录生成合成 JSON/CSV。它验证记录与分析接口，正式 RoboTwin 任务成功率待仿真采集。 [评测协议](../experiment_protocol.md)
