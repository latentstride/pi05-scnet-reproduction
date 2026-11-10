# G2 三个 GPU 作业收尾

冻结基线为 Draft PR #5 的 `73e39e6dc385097044459957dd8df68d735a9103`。三次作业均使用 `handover_block` 的冻结 π₀.₅ checkpoint `17ef545270cf26ed81169893b60f845f80ce6557`、固定 norm stats、PACE `fd2e624cc19325cc9b20a620c82d51c0d8e25cc1` 及 Openpi-PACE runtime `f4e52e6de4cc487a11585a3a7414d610b2883dae`。每次部署提交、候选包和原事件哈希见[统一机器记录](../research/g2/final-system-review/three-jobs.json)。

| 作业 | 部署提交 | Slurm | 实际策略试次 | 口径 |
| --- | --- | --- | --- | --- |
| `127408633` | `93b6c069` | FAILED，1:15:45，1:0 | 6 项实际运行，0 次完成，5 项未运行 | 项目受保护统一诊断 |
| `127510335` | `fae63f80` | COMPLETED，0:14:42，0:0 | 1 项受控候选停止 | 作者场景、原生命令、额外门禁的单例诊断 |
| `127510363` | `fae63f80` | COMPLETED，0:13:21，0:0 | 1 项规划失败 | ARX seed1 受保护跨本体诊断 |

| 作业和实际场景 | 请求 | 完成动作 | 策略物理步 | 结果和阻断点 |
| --- | ---: | ---: | ---: | --- |
| A，Aloha seed0 W50 | 2 | 27 | 3,568 | 第二块候选碰撞，安全继续 |
| A，Aloha seed0 W31 | 2 | 27 | 3,568 | 同一阶段候选碰撞，安全继续 |
| A，Aloha seed1 W31 | 5 | 144 | 10,988 | 已确认左手持物后滑移，接触停止凭证通过 |
| A，Aloha seed1 W50 | 4 | 177 | 12,422 | 右指候选穿透 2.042 mm，安全继续 |
| A，Aloha seed0 W4 | 7 | 26 | 3,542 | 左指候选碰撞，安全继续 |
| A，ARX seed0 W4 | 9 | 26 | 4,965 | 目标侧规划拒绝，接触期恢复禁令，监督器停止整场 |
| B，Aloha 场景 100000 H50 | 1 | 42 | 4,746 | 第 43 行中途左指候选穿透 2.052 mm，受控停止 |
| C，ARX seed1 W4 | 36 | 138 | 16,160 | 第 36 次响应后，下一状态的源 Aloha 代理自碰撞，规划失败 |

物理步按原事件流提取，不计启动探针或作者复位稳定步。实际运行行中无任务成功、无资源截断。A 后五项包括 ARX seed1、Aloha seed1 W4、两项专家和作者 H50，均未进入正式试次，不能记为失败。A、B、C 的记录均未产生严格作者原生完整任务评测。Slurm 正常收尾仅说明进程完成；B 的 `SAFETY_STOP` 是删失，C 的 `planning_failed` 证据定位在源代理候选。A 的整场停止分类也只证明恢复禁令触发，底层记录未证明实测危险碰撞。

本轮仅追加汇总与 CPU 审计，不覆盖任何历史产物。直接参见 [A 原结果](g2_result_127408633.md)、[B 原结果](g2_author_h50_result_127510335.md)、[B 第 43 行复算](g2_author_h50_root_cause_127510335.md)、[C 原结果](g2_arx_result_127510363.md)。

```sh
audit=research/g2/final-system-review
python3 -B "$audit/summarize_jobs.py" . "$audit/unified-event-counts.json" > /tmp/g2-three-jobs.json
cmp "$audit/three-jobs.json" /tmp/g2-three-jobs.json
```
