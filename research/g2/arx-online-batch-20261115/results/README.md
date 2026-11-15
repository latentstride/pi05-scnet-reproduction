# 2026-11-15 ARX-X5 在线闭环结果

四项均完成冻结 π₀.₅ 模型加载、真实场景准入和在线推理。Slurm 均为 `COMPLETED / 0:0`。原始 `handover_block.check_success` 为 **0/4**。四项均由 ARX 碰撞门禁受控停止，停止回执确认保持当前位置并完成场景清理。没有稳定持物、50 mm 抬升、交接或放置。

| 作业 | 场景 / 策略 RNG | 模型查询 | 执行动作 | PhysX 步 | 最远阶段 | 终止 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [G1 · 128047752](G1/trial-receipt.json) | 1 / 20261107 | 6 | 237 | 29,167 | 左单指接触，999 步 | 第 6 查第 2 行的 16 对候选 IK 全部碰撞，均含 ARX `link6` 与物块碰撞 |
| [G2 · 128047778](G2/trial-receipt.json) | 1 / 20261108 | 2 | 45 | 6,729 | 左单指接触，432 步 | 第 2 查第 9 行左指起点预测穿透 2.080 mm 超过 2.000 mm |
| [G3 · 128047805](G3/trial-receipt.json) | 0 / 20261107 | 2 | 26 | 7,755 | 左单指接触，2,901 步 | 第 2 查第 1 行执行期间，左指起点预测穿透 2.000122 mm 超过 2.000 mm |
| [G4 · 128047813](G4/trial-receipt.json) | 2 / 20261107 | 1 | 0 | 0 | 完成首查，无物理推进 | 第一行的 16 对候选 IK 全部碰撞，均含 ARX 同臂 `link2/link5` 自碰撞 |

G1、G2、G3 在首次接触实际停止的 tick 5,767、5,784、4,855 采集了新 RGB、状态并完成第二次真实推理。G1 又执行了四个完整的 50 行前缀。这个结果验证了接触后可恢复在线推理与执行，四项都未达双指接触。G2、G3 的数值出自候选安全检查，现有记录不足以断言阈值附近的预测等于实测伤害。G4 可确认首行在当前场景和映射产生受拒候选，单次试次不足以确定场景 2 的一般可行性。

同 seed 的历史作业与本批 G1/G2 的首块动作均非逐值相同。50×14 矩阵分别有 510、494 项不同，最大分量绝对差分别为 0.004809、0.002886（关节值用 rad，夹爪为归一化开度）。历史同 seed 对照可用于描述行为变化，无法单独归因于控制器更新。[接触与碰撞明细](contact-and-collision.json)

## 原始证据与复现

| 作业 | 在线证据包 | 头部相机视频 | 成员 SHA-256 |
| --- | --- | --- | --- |
| G1 | [归档](G1/online-evidence.tar.gz) | [视频](G1/head.mp4) | [清单](G1/members.json) |
| G2 | [归档](G2/online-evidence.tar.gz) | [视频](G2/head.mp4) | [清单](G2/members.json) |
| G3 | [归档](G3/online-evidence.tar.gz) | [视频](G3/head.mp4) | [清单](G3/members.json) |
| G4 | [归档](G4/online-evidence.tar.gz) | [视频](G4/head.mp4) | [清单](G4/members.json) |

在线证据包包含原始任务回执、RNG 回执、模型启动与阶段日志、三路 RGB NPZ、50×14 动作 NPY、计划及逐步物理事件。G4 视频只有初态。Slurm 输出文件未见非空内容，调度退出码来自 [原始 sacct 记录](slurm.psv)。集群完整归档在项目的 `tools/arx-online-results-20261115.5laR4lAf`，传输时验证 [源归档 SHA-256](source-SHA256SUMS)。GitHub 证据包保留任务分析所需文件并省略缓存和主机诊断目录。[结果 JSON](analysis.json)

从仓库根目录执行：

```bash
D=research/g2/arx-online-batch-20261115
OUT=$(mktemp -d)
for g in G1 G2 G3 G4; do
  tar -xzf "$D/results/$g/online-evidence.tar.gz" -C "$OUT"
done
python "$D/analyze.py" "$OUT" "$D/submission_receipts.json"
```
