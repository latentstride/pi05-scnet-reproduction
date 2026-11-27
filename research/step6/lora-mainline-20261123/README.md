# 微调 π₀.₅ → ARX：2026-11-23 操作与证据快照

本目录保存本轮 Notebook 代码副本与已落盘的 JSON 证据。主线是微调模型使用真实 ARX 观测自主抓取；冻结 Aloha 适配和辅助抓取不作为微调成功。状态以本快照为准，后续运行不会自动更新此目录。

## 当前结论

- step-0600 的离线抓取几何优于混合 step-0400，但闭爪仍不稳定。
- 两组各 100 步的监督对照完成；夹爪损失进入 ARX 总损失，梯度到达全部 20 个 LoRA 参数叶子，仍未改善对应验证锚点的首次闭爪行号。
- 重复观测存在冲突的未来动作标签，已做保留原始文件的标签视图对照；改善有限，不能称为充分根因。
- 原始初态 seed 1、seed 0 的标签修复模型均未自主抓住。初始化补偿修复后的 seed 1 仍只有单指接触，没有双指夹持或持物抬升。
- 十步采样输出上的辅助监督仍是进行中的实验。保存失败状态存在，但恢复保真与局部修复尚未在本快照中得到验证。

## 代码与证据

| 路径 | 内容 |
|---|---|
| `lora_mainline_status_20261123.md` | 本轮完整归因、训练身份、接口区分和执行结果 |
| `mixed-0400-*-metrics.json`、`mixed-0600-*-metrics.json` | 已有模型 Train/Test 锚点评估 |
| `k4-learning/completed-comparison.json` | 两组 100 次更新及验证记录 |
| `k4-learning/grip-gradient-audit.json` | 夹爪梯度与 Adam 更新贡献 |
| `k4-learning/exact-input-label-aliasing.json` | 完全相同观测对应不同闭爪标签 |
| `k4-learning/unique_observations.py`、`run_unique_observations.py` | 标签视图修复与匹配对照，不改原 HDF5 |
| `k4-learning/matched-label-repair-result.json` | 标签修复的有限改善 |
| `k4-learning/physical-results.json` | 初始化修复前两条纯模型物理结果 |
| `k4-learning/initial-gravity-counterfactual.json` | 初始化重力补偿因果对照 |
| `k4-learning/initial-gravity-physical-result.json` | 初始化修复后仍未抓住的物理结果 |
| `k4-learning/initial-expert-path-distance.json` | 初始预测与整段专家接近路径的空间偏差 |
| `k4-learning/sampled_actions.py`、`run_sampled_*.py` | 进行中的实际采样监督代码，未宣称验证成功 |
| `initialization-gravity.patch` | 本次唯一应用到仓库运行代码的最小修复及回归测试 |
| `snapshot-manifest.json` | 快照文件 SHA-256、范围与缺失项 |

`step5_arx/`、`step6_arx/` 与 `k4-learning/` 是 Notebook 审计副本，保留当时的导入及路径依赖。它们不是克隆仓库后即可直接运行的独立软件包。运行需要已有固定版本 OpenPI/PACE、模型权重、ARX/Aloha 数据、机器人资产、SAPIEN 与对应 Notebook 环境；不上传这些大文件，也不在本次同步时发起作业。精确身份和结果目录见完整报告及 JSON 元数据。

## 已验证的最小代码修复

`joint_space_setup.calibrated_gripper` 补回原生 `set_gripper()` 对两臂的 `_entity_qf()` 调用。原实现只替换指关节标定，遗漏初始化期间的被动力补偿，导致机械臂下垂。

- 同 seed 初始 TCP target/actual 位置差：23.69199 → 0.009364 mm。
- 角度差：3.75389 → 0.001192°。
- 此修复不指定抓取目标、不覆盖模型动作、不修改碰撞阈值。
- 此分支独立运行 5 项相关测试通过：`test_g2_runtime.RuntimeTests.test_bootstrap_gripper_path_uses_declared_native_endpoints` 和 `test_g2_robotwin_backend`。

修复后的物理结果仍为 `SAFETY_STOP`：动作索引 26 单指接触；索引 35、请求索引 9 被拒绝，link6 与方块预测穿透约 0.255 mm。方块最高高度变化 1.159 mm，不是双指持物抬升；持物抬升为 0 mm。

## 下一步边界

优先恢复 Notebook 已保存的 `failure-state.pkl` 和 `latest-request-state.pkl`，验证恢复保真并定位腕部向方块运动的原因；恢复不完整时报告缺失项。局部模型自主动作修复有效后，再从原始初态验收双指夹持与抬升。局部恢复结果与原始初态结果分开记录，不靠人工对准、夹爪覆盖、挑有利种子或放宽碰撞阈值通过。
