# G2 六项独立四小时发布审查

基线为 Draft PR #5 的 `4f624f58eabf335a696203da001424f9b09b34a6`。本次正式提交数为 **0**，发布包保持 `submission_allowed=false`。

| 场景 | 作者原生 Fixed H50 | 作者官方 PACE |
| --- | --- | --- |
| 100000 | G1 | G4 |
| 200000 | G2 | G5 |
| 300000 | G3 | G6 |

全部使用 Aloha-AgileX 的 `handover_block`。六份配置与历史 G1–G6 逐字段比较，差异限定为 `wall_s`、`trial_soft_s`、`trial_hard_s`。checkpoint、norm stats、RNG、三路相机、缓存场景、动作语义、PACE 校准和成功判据沿用原配置。G7 受保护 H50 及其证据保存在历史发布中。[历史七项报告](g2_standalone_seven_submission.md)

## 单作业时间账本

| 项目 | 秒 | 执行位置 |
| --- | ---: | --- |
| Slurm 分配 | 14,400 | `--time` 和 `--time-min` |
| 图形启动 | 300 | 启动命令最多 295 秒，保留 5 秒过渡 |
| 模型服务启动 | 900 | 原监督器的启动截止，独立于试次 |
| 试次软截止 | 11,400 | 主机传入同一时钟，客户端在物理步及模型调用检查 |
| 试次硬截止 | 12,000 | 已包含软限后的 600 秒，最后 10 秒用于主机回收试次进程组 |
| 统计和清理 | 480 | 120 秒统计／视频和 360 秒清理 |
| 宿主额外保留 | 360 | 包含 Slurm 在结束前 120 秒发送 TERM 的信号缓冲 |
| 完整硬限账本 | **14,040** | 图形 + 模型启动 + 试次硬限 + 统计清理 + 宿主保留 |
| 分配剩余 | **360** | 启动开销及调度计时余量 |

每项 1×RTX 4090（24 GiB）、4 CPU、32 GiB RAM，固定 `ksagnormal01/gnode39`。单请求 390 秒且仍受试次软截止约束。场景复位、三相机首帧、首请求推理和可能的 JAX 编译占试次预算。策略上限仍为 800 动作和 130,000 策略物理步。时间或步数提前耗尽为 `RESOURCE_CENSORED`，由试次回执或主机硬截止记录。只有按原任务谓词取得成功或正常到达动作上限才得到相应的完整任务结果。日志保留基础设施异常和进程回收状态。

## 启动与验收

继续使用历史作业 `127510335` 和 `127510363` 已运行的独立宿主、图形及模型链。包内模型环境、作者模型服务、驱动发现和渲染检查与历史母包字节相同。沿用此前修好的显式 `G2_PROJECT_ROOT`、Slurm `WorkDir` 核对、独立 Job ID 授权、端口、缓存和输出目录。每项继续使用原监督器与独立作者执行子进程。六份新包的真实 GPU 首次启动、驱动、三相机、模型编译和 PACE 长运行仍需正式计算节点留证。[历史启动身份](../research/g2/standalone-seven/launch-identity.json)

新 CPU 回执绑定各自的新配置哈希，并保留原场景复位及替身回执的旧配置和证据哈希。重算时间账本、运行 CPU 定向回归及生产监督器子进程测试。SCNet 登录节点试算使用六份新包的实际 SBATCH 文件。[CPU 回执及包清单](../research/g2/standalone-six-4h/) · [SCNet 试算与资源回执](../research/g2/standalone-six-4h/scnet-test-only.json)

正式运行仍保存原生任务结果、每个请求的相机／状态／动作块、选中窗口、实发夹爪命令、逐物理步物体位姿与接触及真实终止信息。已有摘要可定位首次闭爪、接触、双指接触和放置支撑。稳定持有、抬升和交接使用保存的逐步物块位姿、指块相对位姿及夹爪开度复核。单帧双指接触只作为代理信号，完整成功仍以原生谓词为准。[单试次摘要工具](../research/g2/parallel-batch/summarize_trial.py)

## 候选包与提交

六份候选包在 SCNet 项目下的 `tools/g2-standalone-six-4h-20261111/stage`。从本提交可复建同一批包。

```bash
python3 -B scripts/g2_parallel_package.py . \
  research/g2/standalone-review/g2-standalone-author.tar.gz \
  research/g2/standalone-six-4h "$NEW_STAGE"
```

最高 24 卡时。按历史 ¥3.60/卡时，单项 ¥14.40，整批 ¥86.40。正式提交前重新读取 SCNet 当前报价、账户余额、磁盘及配额，另取绑定新批次 SHA256 的六项费用授权。磁盘门槛维持 60 GiB，由每项 8 GiB 加批次 12 GiB 预留组成。历史 ¥50.40 的七项授权无法通过新批次检查。

[六条独立 `sbatch` 命令](../research/g2/standalone-six-4h/sbatch-preview.txt) · [候选包及批次 SHA256](../research/g2/standalone-six-4h/SHA256SUMS)

获批后通过已暂存的提交脚本运行。脚本连续进行六次独立的 `sbatch --hold`，核对每个 Job ID 的分配、包和授权后放行，无跨作业依赖。部分提交失败时，保留已有 Job ID 并按现有回执规则核对本批次的未放行作业。

```bash
bash "$STAGE/scripts/submit_g2_parallel_batch.sh" --root "$ROOT" --stage "$STAGE" \
  --account "$ACCOUNT" --qos "$QOS" --execute
```
