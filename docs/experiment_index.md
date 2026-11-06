# 实验索引

审计基准：2026-11-07的main `d64b0e4feba82c4a97dbbfc8c6b518c67743a0f1`。下表只汇总仓库中已有报告，不新运行实验，不改写结果。源码Commit、运行发布标识、摘要发布Commit分别列明；无法证明的作业ID或源码版本标为unknown。

|阶段 / 目的|可核验版本与作业ID|结果 / 失败原因|证据入口与后续事项|
|---|---|---|---|
|Stage 1 · 官方模型功能性复现|冻结Commit `4f87a9085d9f8a2e1bcc14840443aeb5c14db63a`；公开摘要未提供具体作业ID，unknown|真实LIBERO观测、闭环及3任务×5初态15/15通过；非完整benchmark|[冻结记录](../baseline_freeze.md)、[清单](../baseline_manifest.json)、[逐轮摘要](../evaluation_results.csv)；保持基线不重跑|
|Stage 2 · RoboTwin示范/接口读取|官方数据revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`；无GPU作业|双本体各3条示范静态分析；时间与动作语义单独追查|[数据审计](../robotwin_data_probe.md)、[适配合同](../robotwin_control_adapter_spec.md)|
|环境 · 图形/规划失败复测|报告修订 `0425156`；作业126711883；运行Commit在main历史未独立核验，保留为报告标识|宿主CUDA/Vulkan通过，容器Vulkan/EGL及CuRobo预热失败|[复测报告](robotwin_gpu_retest_result.md)、[完整在线审计](robotwin_scnet_online_audit.md)；不泛化为所有节点故障|
|环境 · 最小图形/规划验收|报告修订 `2ffb4fc`；作业126712735|Vulkan/SAPIEN渲染与UR5e关节规划通过；非RoboTwin完整任务|[验收结果](robotwin_gpu_acceptance_result.md)、[在线审计](robotwin_scnet_online_audit.md)|
|G1 · 秒退失败|作业126779874；运行发布 `1f66d619...`见报告，GitHub可达执行Commit未独立核验|运行5秒，在图形驱动发现退出，三个阶段跳过|[失败复盘](pace_g1_failure_126779874.md)；后续修复证据独立保留|
|G1 · 统一接口验收|运行发布标识 `02ff7dd0fa8108c2e70fe990dfe3576982139807`（来源报告，不在本次main历史166项中，不能称已核验的GitHub main执行Commit）；摘要发布Commit `0192277`；作业126905534|单模型、真实观测推理及8步原生闭环通过；完整任务成功率未测|[实机报告](pace_g1_result_126905534.md)、[摘要](g1_acceptance_126905534.json)|
|G2 CPU · 原生专家回放|源码与数据哈希按对应证据；无GPU作业|3条真实CPU物理专家回放满足原任务谓词，非π₀.₅自主策略|[原生专家报告](pace_g2_native_expert_replay.md)；完整原始轨迹不在公开库|
|G2 CPU · ARX接触/持物|多轮历史探针；无GPU作业；不将vN文件名当Commit|低力/负载条件存在滑移；原10g中心反馈3条CPU专家参考通过，范围不含原生策略成功|[ARX专家失败](pace_g2_arx_expert_replay.md)、[中心反馈](pace_g2_centered_grip.md)|
|G2 CPU · 生产门禁/放置|r3源码哈希和清单见报告；无GPU作业|16项门禁功能通过，27524步完成交接但CPU截止前未放置；托盘段有真实/候选拒绝|[门禁报告](pace_g2_production_gate.md)、[局部放置](pace_g2_placement_suffix.md)；不等于完整专家生产路径通过|
|G2 GPU · 首次完整诊断|r2清单SHA `dbcb8d40...`，run lock `49281639...`（均是发布/配置哈希，不是Git commit）；源码/证据公开Commit `f0bc9a1`/`1922d07`/`7201306`；作业127054052|COMPLETED/0:0，真实策略0/8；Aloha四次达到动作上限、ARX四次被映射/碰撞检查拒绝|[作业结果](pace_g2_run_127054052.md)、[失败归因](g2_failure_root_cause_report.md)；历史统计不得改写|
|G2-E CPU · 窗口选择/跟踪|G2-E资料公开于 `469ba58`、`b9fc153`、`a39b8a9`；既有840预测块探索|据旧数据选26/31，长窗口已做CPU合同及静态检查；不是新闭环成功|[窗口分析](g2e_window_analysis.md)、[执行协议](g2e_execution_protocol.md)、[准备验收](g2e_acceptance.md)|
|G2-E GPU · 原生执行窗口|发布清单SHA `fb24a4c7...`，run lock `f054d149...`；结果公开Commit `d64b0e4`；作业127081074|COMPLETED/0:0、01:15:50；0/8任务成功、0/8完整交接序列；6次候选拒绝，K26滑移，K31闭合端微增触发张爪检查|[最终报告](g2e_run_127081074.md)、[结果目录](../research/g2/run-127081074/README.md)、[摘要](../research/g2/run-127081074/audit-summary.json)|
|剩余G2 CPU · 恢复/RRT/放置|公开Commit `d1d2002`及前序静态证据；无新GPU作业|188项完整工作区回归+13合成；原4精确目标仍拒绝，4恢复/绕障及3放置后缀为静态证据|[修复](g2_remaining_cpu_repair.md)、[CPU边界](g2_remaining_cpu_acceptance.md)；真实闭环未验证|
|剩余统一GPU计划|最多14试次、1次分配、34.20元上限；未提交，无新作业ID|`submission_allowed=false`；旧K1/K4原生成功门槛可能跳过全部ARX，须研究修订|[当前方案](g2_remaining_unified_acceptance_plan.md)、`scripts/g2_remaining_plan.py`；本维护PR不修改名单、门槛或预算|

## 版本追溯说明

本次通过GitHub API核对main的166项Commit元数据，并从最新main建立独立维护分支。完整历史摘要保存在[审计数据](repository_audit_data.json)的`version_control.history`；本地源checkout是浅克隆，其14项本地可见日志不能代表远端全部历史。

Stage 1冻结点在远端main历史中存在，四份公开摘要当前SHA-256与原清单逐项一致。G1的`02ff7dd...`是报告记录的执行发布标识，本次没有据此伪造GitHub可达性。G2/G2-E的manifest与run lock哈希完整值应从原报告/JSON读取，表中缩写仅供导航。

历史报告中的“等待审批”“当前运行中”“尚未完成”描述当时状态；最新结果以相应作业结果报告为准。CPU回放、静态几何、合成episode与真实策略试次分开计数，专家不进入策略成功率分母。没有日志或可达版本证据时保留unknown，不从文件名推断任务成功。
