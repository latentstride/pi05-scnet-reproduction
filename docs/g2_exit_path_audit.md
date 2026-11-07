# G2 试次退出与监督器审计

本轮审计覆盖 `g2_remaining_coordinator.py`、`g2_remaining_episode.py`、`g2_episode.py`、`g2_final_trial.py`、`g2_final_campaign.py`、`g2_final_expert.py`，以及它们调用的场景、控制和进程清理入口。诊断试次的任务谓词和整场监督器的退出状态分别记录。已结束的试次仍可能任务失败。

## 继续条件

子进程必须正常退出，回执必须能解析且 ID 匹配，监督器必须确认专用进程组已经退出。每个已运行试次还须提交独立的实测初态和几何验证结果、成功的 `close()` 与场景上下文退出凭证。失败试次还须证明 `hold()` 成功，两个清理调用均不得推进仿真。可控滑移须附带同一物理步的终止检查和清理凭证。危险状态和未知错误始终停止整场。

下一试次使用对应本体和 seed 的启动准入，并在独立子进程中重新构造场景，重复实测初态和几何检查。专家入口在此前提下还需实测双手共享持物和右手接管才能进入放置。启动探针不执行动作，因此只需场景关闭、上下文退出和进程组凭证。资产阶段明确拒绝且场景尚未打开时，仅须确认进程组退出。

## 退出路径覆盖矩阵

| 触发来源 | 结果分类 | 清理或准入证据 | 整场决定 | CPU 回归 |
| --- | --- | --- | --- | --- |
| Native、ARX 或专家的冻结任务谓词成立 | `success` / `diagnostic_complete` | 关闭、上下文、进程组、独立准入 | 继续 | A、C |
| 任务谓词为假，动作/查询预算耗尽，或专家序列正常结束 | `task_unsuccessful` / `diagnostic_complete` | 另需成功 hold | 继续 | A、C、E |
| 预测/候选碰撞、目标限位、源代理几何拒绝、发送前禁止提前开爪 | `diagnostic_complete` | 候选来源或已知规划门禁，另需成功 hold | 继续 | A、C、D |
| IK 无解、目标不可达、无安全候选路径、专家无实测交接或放置阶段条件不成立 | `diagnostic_complete` | 受控规划失败，另需成功 hold | 继续 | A、C、E |
| ARX 恢复额度用尽 | `recovery_budget_exhausted` / `diagnostic_complete` | 结构化计数等于本试次冻结上限，另需成功 hold | 继续 | A、B、F |
| 通用恢复申请时已持物、释放或指尖仍有接触 | `actual_safety_stop` | 不执行通用恢复，尽力清理 | 停止 | B、F |
| 物体滑移或失去双指接触，且该步剩余安全检查均通过 | `controlled_contact_stop` | 同一物理步的终止检查、hold、close、上下文和进程组 | 继续 | G |
| 滑移与危险故障同时发生，或滑移终止凭证缺失 | `actual_safety_stop`，清理失败另记基础设施故障 | 不采用不完整凭证 | 停止 | F、G |
| 初态或推进中的真实碰撞、实测关节越界、跟踪超限 | `actual_safety_stop` | 记录实测诊断，尽力清理 | 停止 | C、D、E、F、G |
| 观测、轨迹起点、几何状态或物理时钟失同步 | `actual_safety_stop` | 不接受候选或额度耗尽结果覆盖实测故障 | 停止 | C、F、G |
| 策略请求异常、共享模型或监控进程退出 | `infrastructure_failure` | 保全回执和已有日志，清理活动进程组 | 停止 | C、F、H |
| 试次或探针软/硬超时、外部中断、剩余时间不足以容纳试次硬上限 | `infrastructure_failure` 或整场预算停止 | 尽力清理，硬超时由主机终止进程组 | 停止 | C、F、H |
| hold、close、上下文退出失败或清理推进物理时间 | `infrastructure_failure` | 尝试其余清理，保留原故障，拒绝放行 | 停止 | E、F、G、I |
| 进程组无法确认退出，或试次非零退出、回执缺失、损坏、错 ID | `infrastructure_failure` | 检查进程组并保全可读证据 | 停止 | F、H |
| 可解释的启动资产/初态/模型输入拒绝 | `not_admitted` | 未开场景或已核验清理，按本体及 seed 隔离 | 跳过未准入条件，其他已准入条件继续 | E、J |
| 试次独立初态验证缺失、专家轨迹无效、未知异常、显式整场停止 | `infrastructure_failure` 或显式停止 | 缺少充分安全证据，保全已有日志 | 停止 | C、E、F |

回归索引

- A · `CampaignExitTests.test_all_14_continue_after_fourth_recovery_cap`
- B · `ExitPathTests.test_archived_recovery_cap_requires_structured_result_and_new_cleanup`、`test_historical_four_receipts_cannot_authorize_continuation`、`test_recovery_cap_and_contact_precedence`
- C · `ExitPathTests.test_classifier_exit_paths`
- D · `ExitPathTests.test_actual_collision_is_not_retried_as_a_planning_candidate`、`DeployTests.test_candidate_and_measured_rejection_are_separate`
- E · `TrialReceiptTests.test_expert_and_probe_receipts_require_real_context_exit`、`ExitPathTests.test_expert_preserves_actual_provenance_and_requires_measured_handoff`
- F · `CampaignExitTests.test_global_failures_never_launch_a_later_trial`
- G · `ContactStopTests.test_slip_stops_on_first_tick_and_requires_completed_cleanup`、`test_concurrent_measured_collision_or_failed_close_cannot_continue`、`test_slip_with_simultaneous_tracking_failure_stops_globally`、`test_campaign_continues_aloha_slip_to_arx_and_expert_only_with_proof`
- H · [启动与进程组回归](../tests/test_g2_final_startup.py)、`ExitPathTests.test_deployment_stage_timeout_reaps_process_and_preserves_log`、[主机超时回归](../tests/test_g2e_host.py)
- I · `ExitPathTests.test_finish_attempts_close_even_when_hold_fails`、`test_session_exits_once_even_when_close_fails`、`test_cleanup_must_not_advance_physics`
- J · [独立动态准入回归](../tests/test_g2_final_deploy.py)

## 历史证据与验证边界

[作业 127139405 的终态记录](../research/g2/final-cpu-review/result-127139405.json)载明四个启动探针通过，四个诊断试次实际运行，任务谓词成功 0 次，十个试次未运行。第四个试次 ARX seed1 在 156 个策略动作和 4 个恢复动作后以旧格式退出。CPU 重放按这些计数进入真实协调器的额度检查，模拟场景提供本次测试的 hold、关闭和退出证据。历史记录自身缺少当前协议要求的凭证，因此不会触发放行，也不用于补写现场清理结论。

生产分支的 14 项 CPU 调度模拟验证了第四项退出后，K26、K4、专家、恢复消融和基线依次启动。故障注入验证首项、中段和末项停机分支。真实 GPU 模型服务、渲染和逐试次准入，新版本在实际 SAPIEN/PhysX 中的第四项停机清理，以及集群容器内子进程退出行为仍待现场核验。CPU 测试结果只支持审阅候选发布。
