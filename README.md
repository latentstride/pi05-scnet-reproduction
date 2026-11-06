# π₀.₅ 官方复现与跨本体机器人操作研究

项目目标是复现官方 π₀.₅ 的机器人操作接口，并研究 RoboTwin / PACE 中从 Aloha-AgileX 到 ARX-X5 的同任务迁移。官方权重、原始数据及部署环境保存在独立研究工作区；本仓库记录代码、协议、精简证据和失败归因。

## 结果与当前阶段

| 范围 | 已有结果 | 结论边界 |
|---|---|---|
| Stage 1 · 官方 `pi05_libero` | libero_spatial 任务0/1/2、各5个初态，15/15功能性验收通过 | 受限基线，不是完整LIBERO benchmark |
| Stage 2 · RoboTwin数据与环境 | 双本体示范、坐标/动作接口及CPU几何检查已有记录 | 数据可读、CPU检查通过不代表策略成功 |
| G1 · PACE接口 | 作业126905534完成真实观测推理及8步原生闭环 | 短段接口验收，完整任务成功率未测 |
| G2 · 原生/跨本体 | 作业127054052八次真实策略试验0/8成功 | 完整任务未通过 |
| G2-E · 执行窗口 | 作业127081074八次原生试验0/8成功，完整物理交接序列0/8 | K26/K31在seed1进入左手双指接触，尚未完成提起、交接或落盘 |
| 后续CPU研究 | 完整研究工作区188项回归、13个合成episode通过；恢复/RRT/放置路径有静态证据 | 不是新GPU策略成功；四个原ARX精确端点仍拒绝，动态修复未验收 |

最新G2原始结果见[127081074验收报告](docs/g2e_run_127081074.md)及[机器可读结果](research/g2/run-127081074/audit-summary.json)。剩余工作是原生碰撞/抓持/释放归因及跨本体闭环验收。现有[统一剩余方案](docs/g2_remaining_unified_acceptance_plan.md)仍需研究修订与人工审批；34.20元费用上限不构成GPU提交授权。

## Stage 1 冻结基线

冻结提交为 [4f87a908](https://github.com/latentstride/pi05-scnet-reproduction/tree/4f87a9085d9f8a2e1bcc14840443aeb5c14db63a)。[冻结记录](baseline_freeze.md)与[校验清单](baseline_manifest.json)固定官方来源、接口和摘要证据，维护工作不重装、不重跑、不改写该基线。

- 官方OpenPI：`215abfb217dbac7d5f1273282331b9b1866c0479`；LIBERO：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。
- 模型：官方 `gs://openpi-assets/checkpoints/pi05_libero`，权重不在Git中。
- [功能验收](acceptance_report.md)、[逐轮结果](evaluation_results.csv)、[评测报告](evaluation_report.md)、[动作接口](action_space_notes.md)。

## 代码与实验入口

| 阅读入口 | 用途与使用条件 |
|---|---|
| [scripts说明](scripts/README.md) | 早期部署记录；安装/下载示例属于历史流程 |
| [脚本索引](docs/script_inventory.md) | 全部重要脚本的用途、阶段、调用/测试关联与建议状态 |
| [实验索引](docs/experiment_index.md) | 真实实验、CPU检查、发布标识和失败记录的对应关系 |
| [g2_cpu_preflight.sh](scripts/g2_cpu_preflight.sh) | CPU总预检；完整回归依赖外部原始研究轨迹及固定OpenPI客户端 |
| [g2_run_task.py](scripts/g2_run_task.py) | G2任务执行库/入口；真实场景要求已授权分配，不能把它当提交命令 |
| [g2e_native_guard.py](scripts/g2e_native_guard.py) | G2-E实测状态与候选运动检查，当前失败证据待后续修复 |
| [g2_remaining_plan.py](scripts/g2_remaining_plan.py) | 剩余研究规划器，`submission_allowed=false`；不是可直接执行的最终发布 |
| [发布与准入说明](docs/pace_g2_single_job_release.md) | 历史G2 r2发布流程；旧授权、清单与运行结果分别保存 |

GPU/Slurm脚本只作研究与版本追溯入口。每次资源操作须在完整CPU准备、固定发布清单及人工授权后进行，不复用历史授权，不自动重排。

## 文档导航

- 研究设计：[数据集比较](dataset_comparison.md)、[RoboTwin可行性](robotwin_feasibility.md)、[控制适配合同](robotwin_control_adapter_spec.md)、[PACE运行设计](pace_runtime_plan.md)。
- G1结果：[126905534短段验收](docs/pace_g1_result_126905534.md)。
- G2失败：[127054052结果](docs/pace_g2_run_127054052.md)、[CPU根因报告](docs/g2_failure_root_cause_report.md)、[G2-E窗口分析](docs/g2e_window_analysis.md)、[127081074结果](docs/g2e_run_127081074.md)。
- 后续CPU工作：[剩余修复](docs/g2_remaining_cpu_repair.md)、[CPU证据验收](docs/g2_remaining_cpu_acceptance.md)、[执行协议](docs/g2e_execution_protocol.md)。
- 仓库维护：[审计报告](docs/repository_audit.md)、[维护与分支约定](docs/repository_maintenance.md)。

## 环境、数据与复查

Stage 1环境记录为Python 3.11.18、JAX/JAXLIB 0.5.3、PyTorch 2.7.1、NumPy 1.26.4，另有独立LIBERO环境；见[environment.md](environment.md)。RoboTwin/PACE使用独立环境及固定源码，见[环境记录](docs/robotwin_environment.md)。Git仓库本身不提供完整仿真依赖、模型或所有原始轨迹。

只读静态审计只需Python标准库：

```sh
python3 scripts/audit_repository.py
python3 scripts/audit_repository.py --check-docs
```

G2-E精简分析包的离线复查方式见[结果目录说明](research/g2/run-127081074/README.md)。它可重算保存预测的窗口截断统计，不能替代真实物理或策略闭环。公开数据范围及历史校验清单的解释见[同步范围](docs/g2_chat_sync_20261106.md)。

新增模型、环境、日志、临时结果、原始数据及认证材料保持在Git之外。历史发布快照和已经提交的精简证据保留原路径；不因重复哈希或命名后缀删除。后续改动使用独立分支和Pull Request，人工审查后再进入`main`。
