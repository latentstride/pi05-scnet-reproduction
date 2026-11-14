# G2 独立作业部署与并行排队

基线为 Draft PR #5 的 `93b6c0691e236d0b871745852f358a4cea85416b`。A 保留原 11 项和第 11 项作者基线。B、C 从 A 实际绑定的归档构建，保留全部冻结策略、执行与安全代码，独立调度层每次只选中一条试次。

| 作业 | 试次 | 时限 | ¥3.60/卡时上限 |
| --- | --- | --- | --- |
| A | 冻结 11 项，作者场景在第 11 项 | 7:00 | ¥25.20 |
| B | `author-native-h50-seed100000` | 1:20 | ¥4.80 |
| C | `unified-seed1-arx-full-w4` | 1:30 | ¥5.40 |

B 使用作者场景 100000、原生 H50、800 动作、130,000 策略物理步、2,400 秒软限和 2,700 秒硬限。C 仅选择冻结的 ARX seed1 完整映射与 W4 策略，保留该试次的准入探针。A 内已有同 seed 的 Aloha W4 对照。历史作业 127147943 的 ARX seed1 推进 156 个动作后卡在源代理候选碰撞的重复恢复上。当前包保留此前已通过 CPU 定向复查的前缀拒绝与重新查询控制。真实策略能否继续推进需要现场结果支持。[历史归因](g2_root_cause_127147943.md)

B 的 4,800 秒包括 1,080 秒模型启动预留、300 秒图形、2,700 秒试次、120 秒统计、120 秒清理、360 秒宿主与信号预留，余 120 秒。C 的 5,400 秒包括 1,680 秒模型启动及独立动态准入、300 秒图形、2,600 秒试次、120 秒统计、120 秒清理、360 秒宿主与信号预留，余 220 秒。场景复位、真实三相机及首轮推理在各自诊断或试次阶段。三项按最长分配合计 ¥35.40，B/C 增量 ¥10.20。[当前报价](../research/g2/standalone-review/scnet-quote.json)

固定 PACE `fd2e624cc19325cc9b20a620c82d51c0d8e25cc1`、Openpi-PACE `f4e52e6de4cc487a11585a3a7414d610b2883dae`。部署预检将模型文件大小、时间戳和冻结 CPU 验证记录逐一比较，并重新核对 norm stats 与 tokenizer 的实际 SHA256。启动时仍由冻结模型加载器独立核验。

独立作业各自启动一个服务并使用专用端口段、可写缓存、Slurm 日志和结果目录。服务、场景和安全停止凭证沿用现有监督器。作业包的 `submission_allowed=false` 保留，部署用另行审批的包哈希、模式、墙钟和单一 Job ID。Slurm 以 hold 方式接收后校验资源及无依赖，再绑定并释放该 ID。提交意向文件为单次写入，保留核对凭证；程序不会重试提交。

CPU 定向检查覆盖两包完整验签、实际调度命令各一项、进程与端口隔离、单服务和异常安全收尾。真实 Vulkan/EGL、三相机、模型推理、策略行为及占用量由各自正式分配留证。[CPU 日志](../research/g2/standalone-review/cpu-deployment.log)

复现离线验收命令

```sh
PYTHONPATH=scripts python3 -B -m unittest discover -s tests -p 'test_g2_standalone_deploy.py' -v
```

复现打包时以已部署的统一归档作为输入，`mode` 分别取 `author` 或 `arx`。

```sh
mode=author
python3 -B scripts/g2_standalone_package.py "$mode" . \
  research/g2/author-h50-unified-review/g2-author-h50-review.tar.gz \
  "/tmp/g2-standalone-$mode.tar.gz"
```

## 正式提交记录

2026-11-10 集群时钟（北京时间）。各作业均使用 `gnode39`、`ksagnormal01`、1 张 24 GiB RTX 4090、4 CPU 与 32 GiB 主存。

| 作业 | Job ID | 提交时间 | 状态（提交确认时） | 项目结果目录 |
| --- | --- | --- | --- | --- |
| A | `127408633` | 2026-11-09 17:10:19 | PENDING | `results/pace-g2-unified-127408633` |
| B | `127510335` | 2026-11-10 13:00:57 | PENDING | `results/pace-g2-standalone-author-127510335` |
| C | `127510363` | 2026-11-10 13:01:06 | PENDING | `results/pace-g2-standalone-arx-127510363` |

A 保留原 Job ID、提交时间、冻结包和顺序。B、C 分别一次提交，均已绑定并释放，Slurm 的 Dependency 为 `(null)`。实时分区 QoS 每用户提交上限 20 项，同时运行上限 12 项。调度采用多因素优先级，计入公平共享；两项可以同时排队，启动取决于实际 GPU 供应和优先级。提交时结果盘可用约 61.5 GiB，覆盖 32 GiB 的合计预留。

最初的集群预检在旧版 Git 的 `-C` 参数处停止。当时两项都没有调用 Slurm 提交。兼容修改只涉及提交端的源码版本核验，候选归档、模型和执行代码均与首次预检相同。最终两个包各核验 89 个成员。

[作业 B 回执](../research/g2/standalone-review/submission-author.json) · [作业 C 回执](../research/g2/standalone-review/submission-arx.json) · [证据 SHA256](../research/g2/standalone-review/SHA256SUMS)
