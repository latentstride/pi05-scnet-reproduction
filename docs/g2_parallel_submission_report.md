# G2 批量准入与提交记录

**七个独立 Slurm 作业均已提交并放行。** 真实 Job ID、队列状态及授权验证见本页最新记录。

| 作业 | 场景 | 方法 | CPU 物理准入 | 预估上限 | 提交 |
| --- | ---: | --- | --- | ---: | --- |
| G1 | 100000 | 原生 H50 | 通过 | ¥7.20 | PENDING |
| G2 | 200000 | 原生 H50 | 通过 | ¥7.20 | PENDING |
| G3 | 300000 | 原生 H50 | 通过 | ¥7.20 | PENDING |
| G4 | 100000 | 官方 PACE | 通过 | ¥7.20 | PENDING |
| G5 | 200000 | 官方 PACE | 通过 | ¥7.20 | PENDING |
| G6 | 300000 | 官方 PACE | 通过 | ¥7.20 | PENDING |
| G7 | 200000 | 受保护 H50 | 通过 | ¥7.20 | PENDING |

每项研究问题见[实验矩阵](g2_parallel_experiment_matrix.md)。预估按旧 ¥3.60/卡时，合计 14 卡时、¥50.40。本轮从 SCNet 商品页与同团队费用总览核实了当前报价和余额，并绑定本批次用户授权。旧三项授权不适用。

七份候选包在 SCNet 登录节点实际解包复验，`sbatch --test-only` 七项通过。该调用返回调度估算编号，没有占用 GPU、没有产生真实 Job ID。核查时该账户活动作业数为 0。完整候选清单 SHA256 为 `9ae1fb32d441b23d80311c71c9594ec946829f1a8d8ab8dca9adb193999fcc7d`。[调度器回执](../research/g2/parallel-batch/cpu/scnet-scheduler-test.json)

## 部署与失败隔离

每项有独立 Slurm 作业、授权、GPU 分配、WebSocket 端口段、缓存、日志及 `results/pace-g2-parallel/<episode_id>/<job_id>/`。模型权重、资产和锁定源文件只作输入。没有任务数组、跨作业依赖或自动重排队。G1 先进入 hold、先放行；实际优先级和启动次序由 Slurm 决定。一个作业运行时出错只终止自己的物理试次和进程组。

执行时先核验全部压缩包、哈希、实际资源、CPU 收据、报价和费用授权，再用 `sbatch --test-only` 核验调度参数。真实提交逐项采用 hold，拿到实际 Job ID 后核对节点、资源、计时、无依赖和持有原因，再绑定独立授权。七份授权全部通过才开始 release。Slurm 的多个 release 无事务保证。若中途失败，保留已放行作业，只在精确批次标记匹配且调度器确认仍处于 `JobHeldUser` 时撤回本批次的未启动项。提交回执不明确时只读查询并记录待核对项；不重复提交。持久意图按配置清单建立，换目录或重新打同一配置也不能自动重试。

## 命令

在 SCNet 登录节点使用本 PR 的源码。把七份压缩包及 `batch.json` 复制到仅自己可写的 `$PRIVATE_STAGE`，`$PROJECT_ROOT` 指向已部署项目，填写真实 Slurm 关联账户和 QoS。无调度器的本地检查用 `--dry-run`。登录节点用下列命令做不分配 GPU 的调度器检查。

```bash
bash scripts/submit_g2_parallel_batch.sh \
  --root "$PROJECT_ROOT" --stage "$PRIVATE_STAGE" \
  --account "$SLURM_ACCOUNT" --qos "$SLURM_QOS" --scheduler-test
```

正式提交前在私有 staging 提供 `platform-quote-evidence`、`platform-quote.json` 和 `batch-approval.json`。报价 JSON 绑定 SCNet 商品 ID、区域、观察时间、实际单价、余额覆盖与原证据 SHA256。批准 JSON 绑定本批次清单 SHA256、报价 JSON SHA256、批准的总金额、7 次分配以及 `automatic_retry=false`。具体字段由[提交脚本](../scripts/g2_parallel_submit.py)校验。仓库中的发布包始终保留 `submission_allowed=false`。待这些条件全部核实后，一次执行：

```bash
bash scripts/submit_g2_parallel_batch.sh \
  --root "$PROJECT_ROOT" --stage "$PRIVATE_STAGE" \
  --account "$SLURM_ACCOUNT" --qos "$SLURM_QOS" --execute
```

## 现场产物

每项独立记录真实初帧、查询关键帧、完整模型块、实际窗口、RNG 回执和原生状态。带 TCP、夹爪与物块的物理记录用于核对接触、推动、抓持与交接。视频由真实查询帧生成，帧率反映取样，不是逐物理步录像。超过 99 次查询时，原始文件编号及事件流可用于重排采样视频。GPU 监控、模型阶段计时、候选与实测接触、停止分类和清理分别留存；未取得证据的阶段记 `NOT_REACHED`。匹配首请求和首块后，才报告严格的窗口或门禁对照。ARX 和专家额外候选均未准入。

## 11 月 10 日正式提交复核

22:49（北京时间）查询的实际 Slurm 状态如下。

| 实验 | Slurm Job ID | 实际状态 | 等待原因 |
| --- | --- | --- | --- |
| G1 | 127556961 | PENDING | Priority |
| G2 | 127556962 | PENDING | Priority |
| G3 | 127556963 | PENDING | Priority |
| G4 | 127556966 | PENDING | Priority |
| G5 | 127556967 | PENDING | Priority |
| G6 | 127556975 | PENDING | Priority |
| G7 | 127556977 | PENDING | Priority |

SCNet 显示 ¥3.60/卡时，同团队可用余额覆盖 ¥50.40。每项 1 张 RTX 4090、4 CPU、32 GiB、最长 2 小时。项目实测约 96G，提交门禁检查时共享文件系统可用 73.92 GiB，高于 60 GiB 门槛。账户限制为最多提交 20 项、运行 12 项。七个冻结包和 CPU 收据复用原哈希，七项 Slurm test-only 均成功。

首次后台执行器缺少 sacctmgr，在 Slurm 提交之前退出。核对空队列、无批次意图和授权文件以及无同批调度记录后，在登录会话完成唯一一次真实批次。七项均先 hold，七份授权经验证后才 release。各项真实状态、独立性、授权哈希、结果和日志路径见[提交回执](../research/g2/parallel-batch-submission-20261110.json)。

报价凭证于 11 月 17 日 22:37（北京时间）到期。开跑前如已过期，部署入口会拒绝启动。试算给出的未来日期只代表调度预测。
