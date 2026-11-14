# G1–G7 独立作业发布

## 历史启动参照

| 历史作业 | 真实启动链 | 实际证据 |
| --- | --- | --- |
| 127510335 | 独立 SBATCH → `g2_standalone_deploy.py` → `g2_graphics.sh` → `g2_final_campaign.supervise` → 作者模型服务与受保护 H50 子进程 | CUDA 模型服务就绪，宿主/容器 Vulkan UUID 一致，SAPIEN 渲染通过，真实模型和动作已执行，最后受控停止。 |
| 127510363 | 同一独立启动链与渲染脚本 → 原监督器 → ARX 子进程 | CUDA 模型服务就绪，真实渲染通过，执行了 ARX 诊断，最后规划失败。 |

两项历史提交都在项目根目录调用 sbatch，并使用相同根目录的 `--chdir`。模型通过 `pace_g1_model_env.sh` 使用 `containers/openpi-ubuntu` 与模型虚拟环境，图形与仿真使用 `research/robotwin/robotwin-cu122`。两个容器使用项目根目录作为 `--pwd`，各作业有独立的 `G2_STANDALONE_CACHE` 和模型端口。[历史 Author 回执](../research/g2/standalone-review/submission-author.json) · [历史 ARX 回执](../research/g2/standalone-review/submission-arx.json) · [历史 Author 运行结果](../research/g2/job-127510335/result.json) · [历史 ARX 运行结果](../research/g2/job-127510363/result.json)

失败的七项从 staging 目录调用 sbatch，虽然 Slurm `WorkDir` 是项目根目录，旧批量入口仍把 `SLURM_SUBMIT_DIR` 作为运行根。实际结果写进 staging，图形脚本退出在设备发现前。[故障报告](g2_parallel_startup_failure_20261111.md)

## 本轮运行路径

七项均以历史独立 Author 发布包为母包。Slurm shell 核对明确传入的 `G2_PROJECT_ROOT` 和 `pwd -P`，从包内执行 `g2_standalone_deploy.py`，调用历史容器、驱动发现、Vulkan、SAPIEN、模型服务与监督器。`pace_g1_model_env.sh`、作者模型服务、驱动发现和 SAPIEN 渲染检查与实际运行过的历史 Author 包字节一致。图形脚本只改一行目录绑定。计算节点还会核对 Slurm 的 `WorkDir` 和当前作业的授权标签。

唯一运行适配是根据冻结配置选择同一个监督器中的一项试次，并绑定其场景、方法、模型端口、资源时钟、现有子进程和结果校验。没有独立 GPU 探针、跨作业依赖、自动重排队或批量 GPU 运行入口。保留故障日志、模型请求、完整动作块、物理接触和终止产物。G1–G7 的配置与上一轮逐值相同。正式 GPU 运行仍需验证新参数绑定在计算节点上的实际表现。

## 审查与一次提交

新候选已暂存在 SCNet 项目相对路径 `tools/g2-standalone-seven-20261111/stage`。登录节点针对七份包运行的 `sbatch --test-only` 全部通过，读回可用磁盘约 73.91 GiB。试算未创建作业。[SCNet 回执](../research/g2/standalone-seven/scnet-test-only.json)

从本次提交复建同一份候选时，指定全新的输出目录：

```bash
python3 -B scripts/g2_parallel_package.py . \
  research/g2/standalone-review/g2-standalone-author.tar.gz \
  research/g2/parallel-batch "$NEW_STAGE"
```


[七包冻结清单](../research/g2/standalone-seven/batch.json) · [七条实际 hold 命令模板](../research/g2/standalone-seven/sbatch-preview.txt) · [CPU 回归](../research/g2/standalone-seven/targeted-tests.txt)

```bash
# 从本次 PR 的源码在 SCNet 登录节点运行，STAGE 指向独立候选目录。
bash scripts/submit_g2_parallel_batch.sh --root "$ROOT" --stage "$STAGE" \
  --account "$ACCOUNT" --qos "$QOS" --dry-run

# 仅在取得此次七项费用授权、实时报价和余额凭证之后执行。
bash scripts/submit_g2_parallel_batch.sh --root "$ROOT" --stage "$STAGE" \
  --account "$ACCOUNT" --qos "$QOS" --execute
```

提交助手在登录节点连续调用七次 sbatch。每项先 hold，得到真实 Job ID 后校验相同项目目录、启动脚本和无依赖，逐项绑定新授权，再放行。全程不等待任何作业执行完。持久意图防止重复；若提交或授权失败，记录已创建 ID，仅撤回同批、明确仍由用户持有的作业。已经放行或状态不明的作业保留给核对。

每项 1 张 RTX 4090、4 CPU、32 GiB、最长 2 小时，七项 14 卡时。按上次核验费率 ¥3.60/卡时，整批最高 ¥50.40。正式执行需重新核验 30 分钟内的 SCNet 报价、真实余额、账户队列限制和至少 60 GiB 可用空间。新批准绑定批次哈希，包含 `submission_scheme=standalone-seven-v1`、`allocations_authorized=7` 与最高金额；旧授权不适用。

本轮没有创建 GPU 作业。已通过的 CPU 验证涵盖七包入口、历史脚本身份、不同场景与方法、共享监督器子进程、独立端口与目录、提交幂等和部分失败。真实 GPU 启动、GPU 设备可用性和排队时间仍待现场。
