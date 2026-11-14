# G2 最后一轮集中 CPU 修复与统一候选验收

本轮基于 PR #2–#5、`db1ceb95ea652fc8459a274f7ae2b956da38dd8b` 和真实作业 `127147943`。修复已知拒绝点、抓持控制和源代理前向准入，保留原 10 项、原顺序、一次分配和一个模型服务。历史模型策略 **0/8**、ARX 专家 **2/2** 不变。CPU 重放、静态路径和合成测试均不计任务成功。

机器可读结果见 [analysis.json](../research/g2/final-concentrated-cpu/analysis.json)、[原始证据哈希](../research/g2/final-concentrated-cpu/raw-evidence-hashes.json)、[发布验收](../research/g2/final-concentrated-cpu/validation.json)。本报告替代旧预载整合评估中的当前状态说明，旧报告及旧发布包继续作为历史证据保留。

## 1. Aloha seed0：端点本身阻塞，存在合法的受限改目标接近

**CONFIRMED**：CPU 动力学按原作者 `Base_Task.take_action(qpos)` 重放，旧控制在第 3,528 步、第 27 个动作中重现同一候选拒绝，已完成 26 个动作。手指候选穿透约 2.081 mm，原阈值仍为 2 mm；当前实际几何检查通过，候选拒绝不能记成实际危险碰撞。

在动作开始的第 3,455 步，尚无手指接触；完整原模型端点的手指穿透约 **9.884 mm**。因此绕路后仍到同一端点不能解决此拒绝。到第 3,528 步已出现单指接触，也不能再用自由空间恢复。

有限 CPU 平移网格（4、8、11.5 mm，±x/y/z）仅找到左工具 **+z 11.5 mm** 的合法候选：实算位姿变化 11.500143 mm、0.0000228°，小于既有 12 mm/1°修正包络，完整静态密集路径和原限位检查通过。原生动力学验证只改这一动作的左臂目标，右臂与两侧夹爪请求保留，原每步候选、碰撞、接触、跟踪、时限保护继续执行。第 27 个动作在第 3,568 步完成，未触发保护。修正前两臂关节和双指测量与历史一致；物块位置的最大数值差约 1.22 μm，未把全部 JSON 声称为字节一致。

修正严格限于**无持有、无释放、无测得手指接触且左夹爪没有正在闭合**的接近阶段；合法端点不改，其他碰撞不套用此修正。记录原始和批准后的 14 维请求、位姿改变量及 `original_goal_reached=false`。这是受限的目标修正，不能称原模型目标已到达，也不能称官方原样 Fixed H50。H50/K31/W4 使用同一候选控制协议。

整块反例还暴露了一个衔接缺口：在修正第 27 个动作后继续执行旧块第 28 个动作，会在任何新增物理步之前被原门禁再次拒绝。不能据此声称整个 H50 块已经通过。最终控制器增加结构化 `NativeApproachBarrier`：完成修正动作后立即丢弃旧尾部，保留真实完成计数，阻止未经新捕获的再次执行。主执行链只在显式新协议、原原生保护、正确 episode/物理时刻/动作数凭证齐全时接受提前返回；下一次观测必须与屏障完成时刻一致，随后才重新查询模型。 最终发布代码的真实 CPU 重放在第 3,568 步完成 27 个动作并返回此屏障，剩余 23 个旧动作未执行、无新增物理步或伪造模型请求；重新观测再查询由独立闭环服务替身正反例验收。任意字典、缺少协议、过期观测都停止。它与 ARX 受限修正后的重新观测规则一致，不是跳过碰撞或复用旧块冒充新推理。

**剩余风险**：越过首个拒绝点不保证以后抓取、接管或完成任务；修正后的新观测与新预测必须现场形成。原始 H50/K31 现在比较的是同一候选控制协议下的窗口，遇到已验证接近屏障时都会提前截断，不再能视为始终无中断执行 50/31 步。

## 2. Aloha seed1：预载时序与幅度已隔离，安全余量增加但仍有限

已确认首次双指冲量在第 **5,086** 步，原三步稳定确认在 **5,088** 步。旧方案在确认持有后锁存较深驱动，3 cm 抬升的最小平移余量约 0.141793 mm。本轮仅改变接触后的驱动目标选择，不改变确认次数、持有锚点、接触材料、刚度、阻尼、力限或 5 mm/5°保护。

候选在首次真实双指接触后建立等驱动预载：`max(min(实际两指开度) − gap, max(当前两指驱动))`，裁入原物理限位，速度前馈为零；随后继续原确认与锁存规则。没有额外等待动作，没有延迟/重置锚点。

| gap | CPU 结果 | 最大平移滑移 | 最小平移余量 |
| --- | --- | --- | --- |
| 1 mm | 第 5,090 步失去双指接触，原保护拒绝 | 失败，保留负例 | 不作为通过方案 |
| 2 mm | 6,138 步，3 cm 抬升及保持 | 4.737720 mm | 0.262280 mm |
| 4 mm | 6,138 步，3 cm 抬升及保持 | 4.409619 mm | 0.590381 mm |
| 正式 4 mm 控制器 | 6,888 步，10 cm/4 s 抬升及 2 s 保持 | 4.412072 mm | **0.587928 mm** |

选择 4 mm 是这个有限网格内的结果，不代表全局最优。正式 10 cm 分支实测物块上升 **99.956453 mm**，峰值角滑移 4.158476°，末秒平移滑移变化约 −0.004325 mm。正式控制器与先前已验收保持分支在第 5,086 步之前的物理状态/驱动数值一致，第 5,088 步持有锚点完全一致；相关断言可重复运行。

进一步按已保存 H50 的两个 50 动作块重放 **100 个动作、7,045 个物理步**，原滑移/接触保护均未触发，最终仍为左手持有，物块比首次持有时高约 134.352 mm；峰值平移滑移 **4.676206 mm**、余量 **0.323794 mm**，角滑移 4.469779°。这比理想化竖直抬升的余量更小，必须同时报告。原始 replay 回执中 `unchanged_prefix_errors` 的 seed1 数值覆盖与旧 GPU 日志重叠的整个时间段，包含干预后差异；本报告的“前缀一致”来自独立的 5,086 步前逐字段重算。

**剩余风险**：余量仍不足 1 mm，不能保证新预测、快速腕部运动、碰撞扰动或长时持有。保存的第二块来自历史观测，重放不等于对干预后观测重新推理。完整真实双手交接和物理释放未被此 CPU 前缀证明。

## 3. ARX seed0：目标 IK 端点阻塞，受限修正通过实际规划器

两个原拒绝状态（第 4,675 / 4,840 步）实际 ARX 几何均通过；目标左指候选穿透约 **11.811 / 12.844 mm**。原 IK、三档更精细 IK 和额外有界分支搜索均拒绝同一端点；不能声称只是 IK 精度或“换一条路即可”，也没有证明全局无解。

同样的 +z 11.5 mm 候选经过原 FK→IK、原限位、原密集碰撞检查和实际 `RecoveryCoordinator.action_plan` + `JointRouter`：旧路径拒绝，新路径通过；第 4,840 步得到 26 控制采样的单动作路径，实算左工具变化 11.499900 mm，源 Aloha 代理的前向 IK/几何/连续性也通过，源观测种子没有被改写。修正计入原恢复额度（上限仍为 4），执行后必须实际重新观测，不假装原目标已到达。

第 4,675 步另做正式协调器对照：旧版与新版都只返回第一个合法动作，保留原有 `safe_prefix_before_rejection`；未把较晚 4,840 步的正例伪称为整段历史闭环必然重现。

静态重建器的桌腿比生产版本保守偏大，桌面、物块和机器人碰撞几何一致；上述手指/物块端点结论不依赖桌腿。该重建不替代现场准入；不能把静态路径通过写成 ARX 的动态抓取成功。

## 4. ARX seed1：源代理自碰撞在执行前可发现，重复恢复被阻断

**CONFIRMED**：历史五次源代理拒绝均涉及 `fl_link4 / fl_link6`，穿透约 0.217、0.224、0.263、0.253、0.260 mm。精细同目标 IK 和额外分支搜索没有找到合法同位姿代理。这不能归因于单纯的 IK 残差。

历史首次陷入发生在第 35 次请求后；物理恢复实际能返回，再查询、再执行四步后又进入相同障碍，直到第 39 次请求及原恢复额度耗尽。没有证据支持代码字节码不一致之类解释。

在第 35、39 次请求上，用原计划每个端点和相应历史实测端点分别检查：**前两个端点合法，第 3、4 个端点都源代理自碰撞**。新 `checked_prefix` 与正式协调器均只保留原有前两个动作，不改其数值，不制造观测或提前替换 seed，不加一次模型查询；执行后由原闭环取得真实新状态再决定后续动作。该提前截断计入原恢复额度，上限仍为 4。持有/释放/接触阶段保留原专用规则，不使用通用自由空间修正。

另以结构化碰撞对和数值穿透记录恢复历史；对同一障碍做过真实恢复后再次出现且没有改善，直接结束该试次，不再花额度返回同一个陷阱。未知诊断不猜测障碍类型，实际危险故障优先于额度耗尽或去重判断。预期候选失败仍须完整 hold/close/场景退出/进程组清理及下一试次独立准入，才能继续整场。

**剩余风险**：前向截断避免了历史中的已知坏端点，不保证在截断后的真实状态上策略会给出有效新目标。重复障碍终止是避免无效恢复，不是完成 ARX 任务。新 ARX 路径的真实动力学和新观测闭环仍未验证。

## 5. CPU 回归、发布包与不变量

全量 **269/269** 测试和 **13/13** 合成 episode 通过。[回归日志](../research/g2/final-concentrated-cpu/regression-tests.log) · [合成回执](../research/g2/final-concentrated-cpu/synthetic-acceptance.json)。新增覆盖原生修正后真实新观测/再查询、旧块尾部禁止复用、无效/过期凭证拒绝，以及接触预载建立/丢失、原三步确认和锚点、接管后释放、提前开爪拒绝、原真实危险优先、源代理前向截断、重复障碍/改善反例，以及 6 个原生/8 个非专家开关与专家隔离。

原每步保护、退出分类和整场监督器不放宽。现有整场故障注入、清理失败、服务故障、超时、恢复额度及后续试次继续/停止回归包含在全量执行中。模块依赖、独立 stdlib `--verify`、AST、Shell 语法、10 个 plan/config 哈希绑定均通过；在集群 CPU 环境完整解包，**75 个候选文件**哈希、冻结源码/配置及原模型 CPU 凭证哈希验证通过。没有模型加载、真实渲染或 GPU 推理。

候选：`research/g2/final-concentrated-cpu/g2-unified-concentrated-candidate.tar.gz`。

- Archive SHA256：`24e337e1f820d38d028e625c0eda7b9cf6b9f942763226d912efa20951422b8d`
- SHA256SUMS 自身 SHA256：`40d5ca77dd7d3d202aa9528e0dcfb9158e6740b5518cef7eb63c281c37b64a67`
- `submission_allowed=false`，不生成提交授权，不改旧包，不自动重试/重新排队。

[配置不变量复核](../research/g2/final-concentrated-cpu/local-release-acceptance.json)逐字段证明：原试次及顺序、窗口、初态、提示、RNG、动作/物理步/时间预算、checkpoint、norm stats、安全阈值、成功判据不变。6 个 Aloha 启用相同接触预载/保持；8 个策略试次启用同一自由空间修正契约；2 个专家配置、轨迹和执行逻辑不变。原生两臂驱动不再能统称“完全不改”：仅在已记录的合法接近修正中改变左臂目标，接触闭合/持有/交接绕过此修正，右臂原始关节目标保留。 原生屏障可能同时截断 H50/K31 的同一个前缀，也可能丢弃块后段尚未执行的闭爪预测；必须按真实请求、已执行动作、丢弃尾部和重新查询次数解释结果，不能把离线保留的右闭爪当作现场接管。完整双手持有后的释放仍走原规则，不能用重新观测屏障绕过。

| 顺序 | 原条件 | 窗口 |
| --- | --- | --- |
| 1 | seed0 Aloha | H50 |
| 2 | seed0 Aloha | K31 |
| 3 | seed1 Aloha | K31 |
| 4 | seed1 Aloha | H50 |
| 5 | seed0 Aloha | W4 |
| 6 | seed0 ARX 策略 | W4 |
| 7 | seed1 ARX 策略 | W4 |
| 8 | seed1 Aloha | W4 |
| 9 | seed1 ARX 修复专家 | 1 |
| 10 | seed0 ARX 修复专家 | 1 |

原资源封套仍为 1×RTX4090、4 CPU、32 GiB、**7 h**，一次模型服务，`--no-requeue`。试次硬时限合计 16,800 s，模型/现场准入等预留 4,200 s，总计 21,000 s（5 h 50 min），距分配上限还有 4,200 s。按历史 ¥3.60/卡时参考，7 h 封顶 **¥25.20**，低于原 ¥34.20；本轮没有取得新的实时价格，也没有提交授权。修复使策略可能活得更久，不能根据旧早停耗时保证完整跑满所有动作；剩余分配时间不足仍按原规则停机并记录未运行项，不能补开分配。

## 6. 证据、复现与人工审查

固定 PACE `fd2e624cc19325cc9b20a620c82d51c0d8e25cc1`，Openpi-PACE `f4e52e6de4cc487a11585a3a7414d610b2883dae`，CPU SAPIEN 3.0.0b1 / mplib 0.2.1。原作者控制函数源码哈希继续记录在重放回执。原始 CPU 证据在本地/集群私有归档，Git 仅提交派生报告、关键片段、哈希与小型可部署候选；不上传原始日志整包或权重。

- 完整 r2 原始 CPU 归档：`d40beefeb9fcbee82acc07580bddd56fcb6f5426819a0f274f3ebe2fbb99cf83`。
- 正式 ARX 协调器 r3 归档：`558ae1aaad4a2f5a405ecd3136406e53a0f5834b16f9b16255386549c996d6ca`。
- 原生旧尾部反例及最终观测屏障 r4 归档：`82f05cedd71913798732082dc6c2746d40e76e8dcf5d304d7a7795c493c8e42c`。
- 旧 0.142 mm 原生保持归档：`af16302a540a726ee3ca3df35c6aae84678219269ff8c0c290769f9a401d378b`。
- [逐文件证据哈希](../research/g2/final-concentrated-cpu/raw-evidence-hashes.json)、[接触关键帧](../research/g2/final-concentrated-cpu/contact-evidence-excerpts.json)及当前候选文件清单保留完整追溯。

在已有 CPU 仿真环境中，几何/正式协调器重放均接受参数 `--raw "$job127147943" --release "$old_frozen_release" --assets "$fixed_assets" --output "$new_output"`：

```bash
# 均为 CPU-only，不启动模型服务；输出目录必须全新。
CUDA_VISIBLE_DEVICES= "$cpu_python" scripts/g2_final_geometry_sweep_cpu.py \
  --raw "$job127147943" --release "$old_frozen_release" --assets "$fixed_assets" --output "$geometry_out"
CUDA_VISIBLE_DEVICES= "$cpu_python" scripts/g2_source_admission_probe_cpu.py \
  --raw "$job127147943" --release "$old_frozen_release" --assets "$fixed_assets" --output "$source_out"
CUDA_VISIBLE_DEVICES= "$cpu_python" scripts/g2_coordinator_repair_probe_cpu.py \
  --raw "$job127147943" --release "$old_frozen_release" --assets "$fixed_assets" --output "$coordinator_out"

# seed1 正式控制器的真实 CPU 动力学抬升，不是策略续推理。
CUDA_VISIBLE_DEVICES= "$cpu_python" scripts/g2_seed1_dynamics_replay_cpu.py \
  --raw "$job127147943" --release "$old_frozen_release" --assets "$fixed_assets" \
  --pace "$fixed_pace" --window 50 --mode candidate-preload --production-contact-preload \
  --settle-ticks 300 --lift-m 0.1 --lift-ticks 1000 --post-lift-ticks 500 --output "$lift_out"

# 从保留归档重算全部派生指标和物理前缀/锚点断言。
PYTHONPATH=scripts "$cpu_python" scripts/g2_concentrated_evidence_cpu.py \
  --evidence "$complete_r2" --arx "$arx_r3" --baseline "$old_preload_lift" \
  --additional "$native_barrier_r4" --project . --output "$analysis_out"
G2_PYTHON="$cpu_python" OPENPI_CLIENT_SRC="$client_src" \
  bash scripts/g2_cpu_preflight.sh "$new_regression_out"
python3 -S -B "$candidate/scripts/g2_final_deploy.py" "$candidate" --verify
```

生产候选具备 CPU 发布审查条件；`READY_FOR_GPU_REVIEW` 仅表示可人工评审上述受限修复和动态风险。它不等于模型闭环成功、完整安全认证或 GPU 提交许可。本轮不提交 GPU、不合并 main、不自动进入下一阶段。
