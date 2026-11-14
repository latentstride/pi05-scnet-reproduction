# 冻结 π₀.₅ → ARX-X5 在线评测

研究任务为 RoboTwin `handover_block`。G1、G2 在同一场景初态复用历史作业 127999407、127999421 作配对对照。G3、G4 使用独立测得的场景 0、2 初态。模型不微调，原成功判据用于完整接近、抓取、交接、放置和撤离。实验单位是一次完整试次。

| 候选 | 场景 / 策略 RNG | 包与单作业入口 | CPU 初态证据 | 调度器 | 实际提交 |
| --- | --- | --- | --- | --- | --- |
| G1 | 1 / 20261107 | 原发布包，91 项通过 | 既有同场景锁定初态 | test-only 通过 | 0 |
| G2 | 1 / 20261108 | 原发布包，91 项通过 | 既有同场景锁定初态 | test-only 通过 | 0 |
| G3 | 0 / 20261107 | 独立新包，91 项通过 | [CPU 重建、FCL、初态读回](evidence/reset-G3.json) | test-only 通过 | 0 |
| G4 | 2 / 20261107 | 独立新包，91 项通过 | [CPU 重建、FCL、初态读回](evidence/reset-G4.json) | test-only 通过 | 0 |

`NOT_AUTHORIZED`。2026-11-15 的 SCNet 商品报价为 ¥3.60/卡时，余额覆盖单项 ¥5.40、四项 ¥21.60。现有提交入口要求提交时报价观察小于 30 分钟，作业启动时绑定报价仍在七天有效期内。四次正式包的 `sbatch --test-only` 均返回 0，并给出 2026-12-10 17:29:14（集群 UTC+08:00）的调度估计。按当前估计，启动时授权验证会在加载模型前拒绝。预估可能随队列改变。原门禁和节点约束保留。最终检查的 `squeue` 无同批活跃作业。现场没有真实 `sbatch`、held job、释放、计费或正式 Job ID。参见 [test-only 原始回显](evidence/slurm-test-only.log)、[提交回执](submission_receipts.json) 和 [SCNet 昆山 RTX 4090 商品](https://www.scnet.cn/ui/mall/detail/goods?type=base&goodsId=821418588551249920)。

## 发布与 CPU 验收

- 保留 G1/G2 原包及原 SHA-256。G3/G4 使用同一个单作业入口和独立场景绑定。G3 继承已锁定的场景 0 初态。G4 使用 [场景 2 实测初态](../pair-candidates-20261106/pair-probe.json)，其夹具哈希随包绑定。两个新场景的冻结任务、臂、夹爪、物体初态均已通过 CPU 检查。重建采用实机网格和 CPU PhysX 的任务几何替代体。真实 PACE 场景和渲染保留同分配启动验收。
- 原模型容器、SAPIEN/PhysX 资源、Vulkan 主机工具、包内脚本语法、checkpoint CPU 回执及其模型缓存、规范化统计、tokenizer、资产和源码版本已核对。真实项目根传给 `--chdir`，提交程序的子进程从同一目录调用 Slurm。新提交入口会在释放前核对 Slurm 记录中的实际 WorkDir 与脚本路径。用户配额为同时运行 12、活动提交 20。结果分区剩余约 67.7 GiB，结果父目录实测可写。历史两个同资源作业的 batch 峰值 RSS 约 11.2、16.8 GiB。
- 四包的控制器、映射、物理执行、模型服务与 Slurm 脚本逐文件哈希一致。只扩展单作业部署的场景选择与绑定校验。ARX 配置兼容历史命名 `arx-full-w4`，实际启用 `arx_direct_v1` 的 50 行候选和实时安全边界，允许首次接触、持物切换、安全拒绝或预算边界结束当前动作块。没有旧 TCP 投影、物理恢复或专家动作注入。
- 每项 1 GPU、4 CPU、32 GiB、90 分钟，试次硬限 2600 秒，动作/查询上限 800/300，关闭自动重排。独立目录和缓存按真实 Job ID 命名。服务绑定 `39000 + Job ID % 4000`，节点启动时检查端口可用。OpenPI 导入阶段 1200 秒，模型就绪阶段总限 1500 秒，模型及动态入口共 1680 秒。
- CPU 接触后闭环使用真实 PhysX 状态和本地 WebSocket 契约。实测 tick 5784 发生左单指接触并丢弃 49 行。第二次请求使用该 tick 新采集的双臂、夹爪及物块状态和三路独立 RGB 替代图，并经 WebSocket 收到模拟保持命令。该命令从 tick 5784 经过原检查执行至 tick 5799。见 [CPU 回执](evidence/contact-receipt.json) 与 [逐步物理证据](evidence/contact-physics.jsonl)。碰撞、越限、持物滑移、超时及失败退出回归见 [定向回归](evidence/targeted-regression.log)。历史 GPU 证据见 [原始结果](../arx-direct-online-20261114/README.md)。
- `GPU_REQUIRED` 包括实际 GPU 分配、CUDA、Vulkan/EGL NVIDIA ICD、真实 RGB、真实模型权重 GPU 加载与推理，以及实际抓持、抬升、交接、放置和任务判据。

## 复现

从仓库根执行以下命令。新包由已有 `write_archive` 和完整发布验证器生成。

```bash
D=research/g2/arx-online-batch-20261115
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts python "$D/build_scenes.py" /path/to/empty-build
sha256sum "$D"/packages/G{3,4}/g2-standalone-arx.tar.gz
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts python -m pytest -q \
  tests/test_g2_standalone_deploy.py tests/test_g2_unified_review.py \
  tests/test_g2_final_deploy.py tests/test_g2_arx_direct.py tests/test_g2_contact.py \
  tests/test_g2_contact_abort.py tests/test_g2_endpoint_cpu.py \
  tests/test_g2_physical_contact_records.py tests/test_g2_final_cpu.py \
  tests/test_g2_final_exit_audit.py
```

`test_g2_unified_review.py` 使用既有的 seed 0/1 `runtime-trace.npz`。它们也在原发布包的 `expert/seed{0,1}/` 中，按对应的 `expert-dense-episode*-20261106/checksums.json` 检查后可用于本地准备回归。

接触接口复用已验证 CPU 容器。以 `ROBOTWIN_GPU=0`、空 `CUDA_VISIBLE_DEVICES` 调用 `scripts/run_robotwin_env.sh`，将候选发布包的 `scripts/`、仓库的 `scripts/`（CPU 重放辅助脚本）和已安装 `openpi_client` 置于 `PYTHONPATH`。传入 `../arx-frozen-input-audit/evidence/inputs/q2.json`（相对本目录）、相同发布包的 `asset-audit.json`、冻结 PACE `assets/` 与独立输出目录。

```bash
python contact_exchange.py /path/to/q2.json /path/to/asset-audit.json \
  /path/to/PACE/assets /path/to/new-cpu-output --first-row 1
```

`evidence/slurm-test-only.log` 中的临时标识仅用于调度器模拟。正式提交应继续通过单作业入口 `scripts/g2_standalone_submit.py arx`，G3/G4 显式传 `--scene-seed 0` 或 `--scene-seed 2`。以绑定包、清单及配置的实际 SHA 为参数，从真实项目根发出。各调用独立，没有依赖或自动重试。真实作业提交仍需在有效报价和可启动的调度条件下进行。

## 结果与排障

见 [结果目录](results/README.md)。通过 `sacct` 记录调度终态，保留启动 stdout/stderr 和应用日志。若失败发生在授权或分配前，先查看 Slurm stdout。容器和图形错误查看 `graphics.log` 与 `deployment-failure.json`。模型初始化阶段查看 `model/phases.jsonl` 和 `model/startup.json`。任务失败查看 `trials/<episode>/receipt.json`、原始动作块及 `episode/events.jsonl` 的规划和物理回执。
