# 实验矩阵、资源预算与恢复

## 初始实验矩阵

| 条件 | 权重 / 统计 | 数据与指标 | 阶段 |
| --- | --- | --- | --- |
| F 原生冻结 | 原始 PACE / 同一套 ARX Train 统计 | Validation 与 Test；相同 ARX 输入、50×14 原生目标、固定噪声 | 已实现离线，对照待 GPU |
| L 原生 LoRA | PACE + 第五步 LoRA / 同一套 ARX Train 统计 | 第 100/200/300/400 步跑 Validation，锁定后跑 Test | 已实现离线，对照待 GPU |
| P 历史投影 | 归档冻结 PACE 与旧 Aloha→ARX 投影及安全配置 | 11 月 14 日诊断组 0/4，含三组投影开启和一组关闭 | 已有历史结果 |
| 历史冻结 ARX | 旧作业 128103655 的原生直接路径与原提示对照 | baseline 237 个策略动作、success_prompt 0 个，两条件均未成功 | 已有历史结果 |

P 的投影开启三试次在该批次均未获得可执行的投影动作。与运行设置有差异的历史条件均单独呈现，不合并到 F/L 成对计数。 [历史投影与消融回执](https://github.com/latentstride/pi05-scnet-reproduction/blob/3f95695d3c8bd9aabf9a626373032547c205bdb7/research/g2/arx-stage6-20261114/README.md)

验证和测试都使用每条轨迹固定窗口 `t=0,25,50,…` 加末个完整窗口，先按 episode 汇总，再等权汇总。F/L 使用同一 tokenization、相机、ARX 归一化、绝对关节恢复、10 步采样和按 episode/frame 派生的固定噪声。报告 flow-matching loss、一步与 50 步窗口的手臂 MAE（rad）、夹爪 MAE 和越界率。首先据验证 flow loss 选 checkpoint，按手臂和夹爪 MAE 打破平局。审阅三项指标后，如需调整学习率或步数，沿用 Validation 决策，在锁定方案前不运行 `test`。现配置为 400 步、3e-5，首轮范围 300–500 由验证曲线和原生动作误差决定。当前 Test 框架可比较一次选定方案。

在线拟采用环境初态 seed 0–4 配对，各 F/L 同一个场景快照、同一固定提示、每试次同一策略 seed/查询索引、H50 预测和固定 K=4 执行，最多 800 个策略动作。只走同一既有 ARX 原生护栏、碰撞/跟踪门禁和 RoboTwin 原成功判定。先做短窗一致性门禁，再按种子交替 F/L 执行，记录双指保持、共同持物、右手接管、放置、干预或终止与成功率。现有历史 P 投影仅作独立基线。新 websocket 可提供 F/L 原生预测；接入已有仿真运行器、在线场景锁和动态对照待实施。控制器签收请求需从同一观测快照读取原生 drive target 与三相机 RGB；独立记录真实 qpos 供跟踪审计。服务请求有 `state`、`images`、`prompt` 和显式单调 `query_index`，响应 `actions[50,14]`，每试次查询索引重置。

## 资源

| 项目 | 第一轮预算 / 硬条件 |
| --- | --- |
| GPU | 优先验证单张 RTX 4090（24 GiB），batch 1。以完整权重加载、真实单步和 20 步过拟合的实测显存与结果决定是否进入正式训练 |
| CPU / 内存 | 8 CPU、64 GiB RAM，批量 1 |
| 时间 | 6 h 硬上限。包括初始化、单步、20 步、400 步和四轮验证。预计步骤吞吐、峰显存和总历时待实测 |
| 磁盘 | 运行卷开工前至少 16 GiB 空闲。既有底座约 11.6 GiB，转换数据约 363 MiB。每个训练 checkpoint 写 adapter 与 Adam 矩，每份约 572 MiB 张量载荷，400 步常规 16 份约 8.9 GiB，再预留中断保存点、临时文件和日志 |
| 费用 | 本轮新增 GPU 费用为 0。现有 `ksagnormal01` RTX 4090 是首选候选。未来费用上限按获批分区的实时单价与硬时限计算 |

OpenPI 上游单卡 LoRA 的参考值为 >22.5 GB，并列出 RTX 4090。这是上游估算。仓库既有 PACE 推理峰值为 14,382 MiB。ARX H50 的训练峰值还受反向激活、XLA 工作区和编译影响，需实测。作业记录 JAX allocator 上限及单步、20 步与正式训练结束时的 allocator 统计。如果加载或更新显存不足，作业在相应门禁退出。 [OpenPI 硬件参考](https://github.com/Physical-Intelligence/openpi#requirements) [既有推理回执](../g2e_run_127081074.md)

## SCNet 路径与恢复

项目根取 `cd ~/code_cpf/pi05 && pwd -P`。已有底座在 `$root/cache/pace-handover`，tokenizer 在 `$root/cache/openpi-pace`，OpenPI 源码在 `$root/source`，模型 Python 为 `$root/tools/pace-model-venv/bin/python`，容器为 `$root/containers/openpi-ubuntu`。新的源码发布、缓存、报告、数据和训练输出均在该根下。当前转换文件位于 `$root/data/step5-arx/arx-drive-target-v1.h5`。不得用 git checkout、staging 或 Slurm 提交目录代替项目根。

在 CPU 和未来 GPU 阶段使用同一份发布代码。示例只执行 CPU，路径应先定义为根下已存在的绝对路径。

```bash
cd "$root"
bash "$release/scripts/step5_arx/env.sh" cpu "$root" -m scripts.step5_arx.train audit-cpu \
  --project-root "$root" --dataset "$dataset" --data-audit "$data_audit" --cpu-receipt "$cpu_receipt"
```

单作业脚本接收 `<root> <release> <dataset> <data-audit> <cpu-receipt> <new-run-dir> [--resume]`。未来提交时由获批的提交端设置选定的单卡分区和 GPU 类型、显式 Slurm 日志路径及硬时限。脚本不调用调度提交或自动重排。GPU 用当前分配内的显式 CUDA 设备，先校验运行录、安装与数据，再串行完成门禁。执行入口查磁盘和项目根，日志记录 JAX 设备信息和运行卷余量。

训练运行目录有排他文件锁。Orbax 保存 adapter、优化器状态，不重复写 PACE 底座。稳定的步数化 RNG 和窗采样保证从已完成步继续时使用下一步同一批次。Slurm 在硬时限前 300 秒请求停止，Python 在更新或验证窗口的边界处理；退出码 75 表示协作停止。每 25 步先提交一个完整 checkpoint。硬中止后忽略未完成的临时目录，以最大已完成步恢复。续跑要求相同源码、配置、数据和权重身份，指向原运行目录并传入 `--resume`；已通过且已写出的 GPU 门禁在同一运行目录内复用。在门禁未通过、文件缺失、损失不有限或梯度无更新时诊断相应错误。GPU 节点的 Slurm 信号转发、峰显存、续跑和 6 小时时限均待实测。
