# 127192740 候选碰撞异常传递与安全退出修复（CPU）

## 真实失败与原因

作业 `127192740` 于北京时间 2026-11-09 06:49:32–07:26:27 运行，Slurm `FAILED / 1:0`，耗时 36 分 55 秒。共享模型服务启动一次，六个动态准入均通过。仅第一个 `unified-seed0-native-w50` 实际运行：2 次模型请求、27 个已完成动作，0 次完整任务成功；其余九项未运行。不能将未运行项记作策略失败，也不能把本轮 CPU 模拟计入真实成功。

[历史结果](../research/g2/candidate-stop-127192740/result-127192740.json)与 [Slurm 摘录](../research/g2/candidate-stop-127192740/sacct-excerpt.txt)单独追加，不修改原提交时的 PENDING 快照、历史 127147943 或其他冻结结果。

**CONFIRMED**：新的接近屏障在物理 tick 3575 后丢弃旧块余下 23 个动作，并实际发出第二次模型请求。新块再次触发手指候选碰撞；第一组记录的候选穿透为 4.735735 mm，原允许值仍为 2 mm。三档精细 IK 的内层诊断均保留候选碰撞证据，但 `refined_pair()` 的汇总异常只有 `attempts`，顶层 `geometry` 丢失，随后经原生执行器包装为控制失败。旧分类器无法证明该异常仅来自候选检查，因此归为 `actual_safety_stop`，监督器终止余下九项。

这证明异常来源在传递中丢失，**不证明模型产生了合法目标，也不证明旧现场可以安全继续**。旧日志有 hold、close、场景退出和进程组回收记录，但没有本次新增的终止状态复核及 hold 读回凭证，不能事后补造。

公开 [回归样本](../research/g2/candidate-stop-127192740/historical-fixture.json)包含原异常树、关键事件、末个物理状态和清理回执；仅移除私有视频路径，明确记录脱敏字段。原始未脱敏 receipt SHA256：`3c92400ebe610eab5677bdb4f0290086e03776d5c18d93490a463c79c6d34be3`。

## 最小修复与继续条件

- `refined_pair()`、`bounded_progress()` 的汇总异常携带结构化 `geometry=candidate`、原因与完整内层异常。任意深度出现 `geometry=actual` 均立即传播，不继续搜索或覆盖成候选。
- 候选拒绝后禁止在当前场景再次执行动作。规划入口通过 `finish_episode → RuntimeSession.note_failure → RuntimeReset` 注册失败；原生执行入口及专家异常入口同样注册，并保留结构化结果。
- 在同一物理 tick 读回实际关节、双指、基座、物块与任务物体状态，验证原限位、实际几何、接触和跟踪约束。接触检查使用 tracker 副本重查当前 tick，保留原抓持锚点、阈值和实时 tracker；不通过重置锚点隐藏滑移，也不额外推进仿真。
- 仍调用原 hold；读回机械臂目标等于实测关节、驱动速度为零。Aloha 保持原实测手指 hold 语义；ARX 正在持物时保留原闭合驱动目标。hold 前后实际状态哈希与 tick 必须一致。物理速度可以非零，因为仿真已暂停；这里证明的是暂停与驱动保持，不宣称经过了额外动力学稳定过程。
- hold、close、场景上下文退出必须全部成功，清理不能推进物理时间。凭证绑定 episode、tick、状态哈希及完整失败诊断哈希；上一试次或另一异常的凭证不能复用。
- 监督器还须确认子进程组全部退出、共享服务存活、独立 reset 与准入有效。然后才把本次记为 `controlled_candidate_stop / success=false`，进入原名单下一项；下一项仍新建场景并执行原实测初态、几何准入。不会重试当前项或申请新分配。

| 结果/凭证 | 试次结果 | 整场决策 | CPU 覆盖 |
| --- | --- | --- | --- |
| 候选碰撞，来源、现场复核、hold 读回和清理全部通过 | 单试次失败 | 继续下一预定项 | `test_live_check_hold_readback_and_cleanup_create_bound_proof_without_steps`；10 项模拟 |
| 候选标记存在，但凭证缺失、episode/异常哈希不匹配 | 安全状态无法证明 | 停止 | `test_real_failure_refinement_preserves_candidate_provenance`；missing/wrong proof 注入 |
| 内层或顶层实际碰撞、接触丢失/滑移、跟踪越限、失同步 | 原安全停止 | 停止 | nested actual、geometry、tracking、stale、same-tick slip |
| hold 异常、速度/手指读回错误、基座移动、close/上下文失败 | 清理/停止未证明 | 停止 | `test_terminal_hold_close_or_context_failure_never_has_complete_proof` |
| ARX 持物目标被 hold 意外改成实测开度 | 保持失败 | 停止 | `test_arx_retains_grip_drive_and_checks_stop_without_advancing` 正反例 |
| 服务退出、未知异常、超时、独立 reset 失败、进程组仍存活 | 基础设施/安全故障 | 停止 | 19 类批量注入，分布于第 1、4、7、10 项 |
| 旧恢复额度耗尽、正常任务失败/成功、原可控接触停止 | 原分类与证据要求 | 原决策不变 | 既有退出路径全量回归，含 127139405 重放 |

## CPU 验收与不变量

完整回归 **278/278**，合成 episode **13/13**；这些不是模型闭环成功。新增覆盖包含真实异常的三档汇总重现、再次包装、原生停止读回、ARX 保持、同 tick 接触复核及危险优先。

[调度分析](../research/g2/candidate-stop-127192740/schedule-analysis.json)使用真实异常数据与明确标注的模拟现场/清理凭证，调用生产 `classify()`、`supervise()`：首项安全候选停止后，原十项按原顺序完成模拟调度，共一个模型服务、六项启动准入。19 类故障注入均在预定位置停止，后续项保持未运行。**模拟凭证从未写回历史 receipt**。

[回归日志](../research/g2/candidate-stop-127192740/regression-tests.log)、[合成回执](../research/g2/candidate-stop-127192740/synthetic-acceptance.json)、[源码与原输出哈希](../research/g2/candidate-stop-127192740/regression-provenance.json)和 [发布不变量](../research/g2/candidate-stop-127192740/release-validation.json)分别保存。既有 848 个受保护研究/安全文件逐一与父提交比对未变。十份配置及所有非运行源码成员与实际运行旧包逐字节相同；保持原十项顺序、一次分配、一个模型服务、7 小时、不自动重试。原 2 mm 等安全阈值、最大恢复次数、checkpoint、norm stats、任务判据、历史结果和 LIBERO 基线均未改变。

候选包新增 `g2_candidate_stop.py`，共 76 个清单文件。独立 `python -S -B … --verify`、清单、Python AST、Shell 语法及配置绑定检查通过；提交开关保持 `false`。[新候选包](../research/g2/candidate-stop-127192740/g2-candidate-stop-candidate.tar.gz) SHA256 为 `be7af44b4cb70ad49362b0475829c12e90de09075e545d45f0b39d93c75e8d71`；清单自身 SHA256 为 `96834d54c8293a3c4ac6f2b5467660eecfd5adfc287ceb442aafbc047c05007e`。旧包 `24e337e1…` 保持原样。本轮没有 GPU 推理、仿真、作业操作或提交授权。

## 复现

在已具备冻结 OpenPI client 与轻量 CPU 依赖的环境中，将 `OPENPI_CLIENT_SRC` 指向既有 `packages/openpi-client/src`：

```bash
export PYTHONPATH="scripts:tests:$OPENPI_CLIENT_SRC"
python scripts/g2_candidate_schedule_cpu.py --output /tmp/g2-candidate-audit-new
G2_PYTHON=python bash scripts/g2_cpu_preflight.sh /tmp/g2-candidate-regression-new
```

候选构建与只读验证（必须使用新目录，不替换旧包）：

```bash
python scripts/g2_final_prepare_review.py . /tmp/g2-candidate-release-new \
  --unified-transfer --native-preload-hold --native-contact-preload \
  --bounded-approach-repair --archive /tmp/g2-candidate-release-new.tar.gz
python -S -B /tmp/g2-candidate-release-new/scripts/g2_final_deploy.py \
  /tmp/g2-candidate-release-new --verify
```

## 尚未验证

新停止凭证的真实 SAPIEN/FCL 现场生成、GPU 进程退出时序仍未经过新作业验证。CPU 正例使用实际停止逻辑与模拟几何/服务，不能替代现场测量。修复解决的是异常传递和整场退出条件，不解决第二块模型候选仍不合法，也不能保证 Aloha 抓持、交接或 ARX 策略成功。现场任何复核失败仍须全场停止。

本轮完成 CPU 修复与证据交付后等待评审，不自动进入下一次 GPU 实验，不合并 main。
