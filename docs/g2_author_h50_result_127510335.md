# 独立作者 Aloha Fixed H50 诊断结果 · 127510335

2026-11-10 19:59:42–20:14:24（北京时间），作业运行 **14 分 42 秒**，Slurm 状态 `COMPLETED`，退出码 `0:0`。唯一策略试次为 **`SAFETY_STOP / controlled_candidate_stop`，任务未成功**。监督器将安全退出的诊断标为 `completed`，原生任务成功标志为 `false`。[Slurm 摘录](../research/g2/job-127510335/sacct-excerpt.txt) · [原始试次回执](../research/g2/job-127510335/trial-receipt.json)

## 实际执行

| 检查项 | 实测 |
| --- | --- |
| 部署 | Vulkan 身份与 SAPIEN 渲染均通过，一个模型服务，GPU 峰值 14,378 MiB |
| 场景 | 作者缓存场景 `100000` 的原生复位通过，2,500 个稳定步后实际采集 tick 0 接触 |
| 首次请求 | 冻结 checkpoint `17ef545270cf26ed81169893b60f845f80ce6557` 在 CUDA 上运行，三路真实非恒定图像为 3×240×320，14 维状态，模型 RNG 由 seed 0 初始化 |
| 实际指令 | `Grab the red block using the left arm.` |
| 执行动作 | 发出一次策略请求，返回 50×14。原始动作顺序和逐行哈希一致，完成前 42 行 |
| 停止位置 | 第 43 个动作从 tick 4699 开始，推进 47 个物理步后，于 tick 4746 的候选检查被拒绝。第 44–50 个动作未尝试，未发出第二次请求 |
| 停止凭证 | 同 tick 真实接触采样，hold 读回、close、场景退出与进程组回收通过 |

## 失败归因

- **确认触发原因。** 第 43 个动作进行中，左指 0 的候选位置与方块预测穿透 **2.052 mm**，超过冻结的 **2.000 mm** 限值约 **0.052 mm**。失败诊断标为 `geometry=candidate`，实际状态复核和停止凭证均通过。
- **停在抓取接近阶段。** 4,746 个策略物理步的接触跟踪状态始终为 `approach`，两手均无跟踪器确认的有效指—物接触或持有。原始 50 行中，左夹爪指令范围为 **0.996770–1.001125**，右夹爪为 **0.998284–1.001373**，均接近全开。首块尾部也没有明显闭爪请求，因此本次没有证据表明延长窗口能在该块内补回闭爪。候选碰撞涉及接近几何，日志无法单独区分模型末端目标、原生 TOPP 路径和几何表示各自的贡献。
- **协议和预算。** 本次用作者场景和原生 qpos 执行，首块未执行到第 50 行，因此真实的第 50 行后重查和后续预测未发生。实际耗时和物理步远低于软墙钟及 130,000 步限额，提前结束属于安全停止。该单场景诊断不能估计作者报告的成功率，也没有产生 ARX 迁移结果。

## 证据与复核

部署归档 SHA256 为 `1d60de6eeaffacab0962119186d8bc7aac4755537cfba1aa900c325664585a6e`。原始动作、观测、接触日志和视频产物保留在 SCNet 的项目结果目录 `results/pace-g2-standalone-author-127510335`。仓库发布试次回执、Slurm 摘录及从原始日志计算的指标与哈希。[机器可读结果](../research/g2/job-127510335/result.json) · [提交绑定](../research/g2/standalone-review/submission-author.json)

用现有 NumPy 环境复核，输入为原结果目录：

```sh
evidence=research/g2/job-127510335
python3 -B "$evidence/extract_result.py" /path/to/pace-g2-standalone-author-127510335 \
  research/g2/standalone-review/submission-author.json > /tmp/127510335-result.json
cmp "$evidence/result.json" /tmp/127510335-result.json
(cd "$evidence" && shasum -a 256 -c SHA256SUMS)
```
