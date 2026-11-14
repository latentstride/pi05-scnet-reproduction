# G2 接触后预载保持候选：执行接入与 CPU 验证

候选已接入现有 Aloha NativeBackend，默认关闭。它在既有监测器确认稳定双指持有后，保持当时的两指位置驱动目标、将两指速度前馈置零；不锁定实际关节，不修改夹持力参数，不移动抓持锚点。原模型输出、原生 qpos 接口和全部逐物理步保护仍保留。

本轮 CPU 动力学完成了同一 H50 seed1 前缀后的保持、3 cm 抬升诊断和 1 s 后保持。物块实测上升 29.576 mm，全部 6,138 步通过原保护。释放通过真实接触状态机和 NativeBackend 的合成接触正反例；完整物理交接后的释放、后续模型闭环和完整任务仍未验证。

## 接入与控制语义

- 实现：[g2_native_preload.py](../scripts/g2_native_preload.py)、[g2_native_backend.py](../scripts/g2_native_backend.py)、[g2_task_runtime.py](../scripts/g2_task_runtime.py)。
- 唯一新增配置：`native_preload_control: measured_contact_preload_hold_v1`，必须与 Aloha 原生逐步门禁及 `physical_aperture_v1` 同时启用。缺省行为不启用。ARX 携带该配置会被拒绝。
- 持有仍由原 `NativeReleaseTracker`、原接触冲量条件及三步稳定条件确认。每步实测碰撞、接触、跟踪和期限检查全部通过后才锁存驱动；从下一物理步开始生效。
- `NativeBackend._before_step()` 读取原控制器实际产生的位置/速度驱动，先检查**原始开爪请求**，再选择保持或释放。未被允许的左手开爪不能被保持逻辑隐藏。
- 保持中的更深闭合目标不再增加预载，双指驱动保持各自锁存值，速度前馈为零。手臂驱动保持原值，候选碰撞检查针对最终实际应用的驱动进行，接受后才能推进物理。
- 允许的向外开爪请求走原门禁和原释放授权。左手必须先有实测右手持有；右手最终释放沿用原规则。进入 releasing、失去持有或重新抓取时按实测状态撤销/重新建立锁存。
- 每个候选物理步新增 `native_preload_control` 记录，分开保存 requested/applied 驱动、速度、保持/释放状态和首次锁存 tick。原模型动作及 `native_action_request` 不改写。原生逻辑夹爪命令缓存和观测定义不替换为保持值，新增日志明确记录其与实际应用驱动的区别。
- `hold()`、`close()`、场景上下文及监督器清理仍使用原实现；安全停止后没有额外推进、等待、重试或恢复动作。

该控制开关只作用于源本体。ARX 力控、源代理、FK→IK、规划、专家和统一监督器未修改。跨本体目标继续保持 Aloha→ARX；没有新建试次名单、GPU 阶段、提交授权或 allocation。

[候选配置](../research/g2/native-preload-candidate/candidate-config.json) 除上述新增键外，与冻结 H50 seed1 配置逐字段一致。[保护源码完整性](../research/g2/native-preload-candidate/protected-integrity.json) 验证原接触/碰撞保护、ARX 后端、适配器、专家及监督器相对 `abcebf17a9856fefdc82ca171d8f54c3f6cbfa5b` 字节未变。checkpoint、norm stats、阈值、成功判据、历史结果和 LIBERO 基线未修改。

## CPU 动力学：保持后抬升

复用作业 `127147943` 的 H50 seed1 实际前缀、已验收的 CPU 场景、生产资产以及固定 PACE 源码。两种驱动位置逐步重建误差均为零。CPU SAPIEN/mplib 无 renderer、无模型推理、无 GPU。

本轮通过实际 `NativeBackend` 的 SceneClock、before/after-step 钩子执行控制与原安全监督，不是仅在离线数组上替换夹爪数值。首次持有后继续原臂/右爪序列至历史日志末尾，之后是明示的诊断保持与抬升，不代表未知的模型后续动作。

| 阶段 | tick / 时长 | 结果 |
| --- | --- | --- |
| 历史真实前缀 | 1–5088 | 实测关节、双指、TCP、物块位姿和速度与 CPU 原始分支完全一致；接触几何/冲量多重集合一致 |
| 开始预载保持 | 5089 | 锁存首次持有的 21.925857 / 21.925857 mm 位置驱动，速度前馈置零 |
| 原轨迹后缀与诊断保持 | 至 5388，持有后 1.2 s | 漂移 4.820406 mm / 4.255141°，通过原保护 |
| 有限抬升诊断 | 5389–5888，2 s | 原臂 FK/有界 IK 生成 30 mm 目标，再用五次关节插值；每步候选与实测保护继续运行 |
| 抬升后保持 | 5889–6138，1 s | 持续实测双指持有，没有安全拒绝 |

- 从抬升开始到最终：实际 TCP 上升 **29.580034 mm**，物块上升 **29.576004 mm**；物块横向位移约 0.759 / −0.506 mm。
- 抬升及后保持的最大累计漂移 **4.858207 mm / 4.255370°**，原锚点未更新。相对原 5 mm 平移阈值的最小余量仅 **0.141793 mm**。
- 最终漂移 **4.807021 mm / 4.253245°**，实际双指位置 **30.245911 / 29.679352 mm**，位置驱动仍为 21.925857 mm。没有把实际双指位置强行设为驱动目标。
- 首次持有至末尾共 **4.2 s**，整段 6,138 步 CPU 墙钟约 **254.074 s**；这不是 GPU 预算或模型吞吐测量。

**CONFIRMED：** 该候选能在复现的这一状态下保留预载，并在原保护内完成有限抬升与后保持。证据不再局限于原地保持 1.2 s。

**记录表示差异：** 新旧日志整体 JSON 并非逐字节相同。NativeBackend 按生产逻辑将 quaternion 转成 float64 旋转，旧诊断 helper 使用 float32 矩阵；tracker 物块矩阵最大元素差为 6.74e−8，接触列表顺序也可不同。物理状态、速度、驱动与排序后的接触几何/冲量完全相同，另有物理前缀指纹验证。保持分支内部的活跃锚点始终不变；没有利用表示差异重设锚点。

[分析](../research/g2/native-preload-candidate/analysis.json) · [关键日志](../research/g2/native-preload-candidate/key-excerpts.zip) · [公开片段独立复算](../research/g2/native-preload-candidate/public-verification.json)

## 释放与安全兼容性

新增 6 项有意义的 CPU 测试，使用真实 NativeReleaseTracker，并在 NativeBackend 钩子中注入明确标记的合成接触回执。

| 情形 | 预期与实测 CPU 行为 | 对应测试 |
| --- | --- | --- |
| 未完成三步接触确认 | 不建立保持；原驱动透传 | `test_latch_only_after_existing_three_sample_acquisition_and_preserve_anchor` |
| 持有后继续闭合 | 保持首次双指驱动，速度前馈零，原锚点不变 | 同上；`test_native_hooks_apply_after_candidate_acceptance_and_hold_on_failure` |
| 仅左手持有时开爪 | 原始请求触发 premature release，禁止推进 | `test_donor_opening_is_not_hidden_by_hold_and_release_waits_for_receiver` |
| 双手确认后左手释放 | 透传开爪、原 tracker 授权 releasing，随后撤销左手锁存，右手保持 | 同上；NativeBackend 钩子测试 |
| 右手最终释放 | 透传开爪，失去接触后清空锁存，避免旧预载重新关爪 | 同上；NativeBackend 钩子测试 |
| 重新抓取 / 非对称双指驱动 | 按新的实测持有建立各指新参考，不复用旧锁存 | `test_both_fingers_and_reacquisition_get_new_reference` |
| 原 5 mm 滑移、失去双指接触、过期状态 | 停止，不能靠保持继续 | `test_slip_contact_loss_and_stale_state_still_stop` |
| 候选碰撞拒绝 | 原判定停止，不新增物理步；随后 hold 零速度、close 仍执行 | NativeBackend 钩子测试 |
| 配置误用于 ARX / 无物理释放语义 / 未知版本 | 拒绝配置；默认不启用 | `test_configuration_is_explicit_native_only_and_historical_default_unchanged` |

既有全量回归同时覆盖实际接触停止、伴随跟踪故障、碰撞/清理失败不能继续、进程组清理及跨本体调度。最终 **244/244 CPU 测试、13/13 合成 episode** 通过。合成 episode 的“success”只检验软件控制流，不是机器人任务成功。

**UNKNOWN：** 未做这一候选的完整真实物理右手接管及托盘释放；接口允许释放不保证接触动力学可行。原模型下一次请求、其他初态、较大抬升/腕旋转、长时蠕滑以及 ARX 策略表现仍未验证。当前平移安全余量很小，不能据本轮宣布生产就绪、原门禁误报或模型能力已修复。

## 复现和证据保全

固定 PACE revision 为 `fd2e624cc19325cc9b20a620c82d51c0d8e25cc1`，客户端 Openpi-PACE 为 `f4e52e6de4cc487a11585a3a7414d610b2883dae`。冻结执行包 SHA256 为 `fc9ba06855e81cc90c9d49f83936df714302c82febb72df326bf494394e388f9`，未修改。候选实际执行的四份代码与本次发布代码字节一致，哈希见 source-manifest/raw-manifest。

```bash
# 既有集群 CPU 环境；变量分别指固定 release、PACE 与匹配客户端源码。
ROBOTWIN_GPU=0 bash scripts/run_robotwin_env.sh env CUDA_VISIBLE_DEVICES= \
  PYTHONPATH="$repo/scripts:$frozen_release/scripts:$client_src" \
  python "$repo/scripts/g2_seed1_dynamics_replay_cpu.py" \
  --raw "$job127147943" --release "$frozen_release" \
  --assets "$pace/assets" --pace "$pace" --window 50 \
  --mode candidate-preload --settle-ticks 300 --lift-m 0.03 \
  --output "$new_cpu_run"

PYTHONPATH=scripts python scripts/g2_native_preload_analysis.py \
  "$new_cpu_run" "$accepted_original_cpu_run" "$new_analysis_directory"
PYTHONPATH=scripts python scripts/g2_native_preload_analysis.py \
  --verify-published research/g2/native-preload-candidate

G2_PYTHON="$cpu_python" OPENPI_CLIENT_SRC="$client_src" \
  bash scripts/g2_cpu_preflight.sh "$new_cpu_preflight_directory"
```

完整 CPU 归档 **10,410,248 字节**，SHA256：`af16302a540a726ee3ca3df35c6aae84678219269ff8c0c290769f9a401d378b`。完整轨迹、实际执行源码、控制日志和命令保存在集群及本地私有证据目录；公开包保存 1,056 条关键物理采样与对应控制记录，没有权重或原始数据归档。

工程排错记录也已保留：首次启动的客户端路径错误在物理运行前退出；首轮物理重放已完成，但新增控制数组 JSON 序列化失败，未作为完整验收记录。修复仅涉及诊断 JSON 写出，第二轮保存所有日志并独立复验通过。原始失败输出未覆盖。生产 episode logger 本来支持 NumPy，未因此修改。

最终新脚本/NativeBackend Ruff、全部修改源码 AST、配置逐字段对照、保护源码字节审计和 Git diff 检查通过。`g2_task_runtime.py` 原有两处单行 if 的 E701 未在本轮整理。完整测试与 13 个合成 episode 结果见研究目录。

本次没有 GPU 操作，没有新建 GPU 提交授权，没有合并 main。候选保留在 PR #5，等待人工审阅。
