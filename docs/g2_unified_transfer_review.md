# G2 单次分配统一验收候选

基线为 Draft PR #2 的 `c6b6b54460c566c75b640f7a8f5ce76195ded94d` 和 Draft PR #3 的 `5838304e76f33b924c89997ac24d830968906f4e`。保留 RoboTwin `handover_block`、固定 π₀.₅ 权重 revision `17ef545270cf26ed81169893b60f845f80ce6557`、原任务谓词及 Aloha-AgileX → ARX-X5。使用一次分配、一个共享模型服务、四组本体/seed 准入及六个无动作探针。所有结果为诊断，单轮不估计总体成功率。

## 统一试次及对照

| 顺序 | seed | 本体 | 条件 | 单试次硬上限 |
| ---: | ---: | --- | --- | ---: |
| 1 | 0 | Aloha | H50 | 1200 s |
| 2 | 0 | Aloha | K31 | 1200 s |
| 3 | 1 | Aloha | K31 | 1200 s |
| 4 | 1 | Aloha | H50 | 1200 s |
| 5 | 0 | Aloha | K4 | 1200 s |
| 6 | 0 | ARX | K4，既有映射、有限恢复和路由 | 2600 s |
| 7 | 1 | ARX | K4，既有映射、有限恢复和路由 | 2600 s |
| 8 | 1 | Aloha | K4 | 1200 s |
| 9 | 1 | ARX | 官方 Aloha 专家映射加放置后缀 | 2200 s |
| 10 | 0 | ARX | 官方 Aloha 专家映射加放置后缀 | 2200 s |

- 1–4 项隔离原生窗口，顺序交叉。两条件的配置除窗口及标识字段外逐值一致，使用同一冻结初态、提示、RNG、原生 `qpos`、300 动作、25,000 物理步和 1150 s 协作时限。候选检查逐物理步执行，实际跟踪、接触、滑移、限位与清理仍走同一链路。
- 5–8 项隔离窗口长度差异，提供配对本体诊断。ARX 必须使用虚拟源观测、逆解及重定时，结果包含映射和目标控制的综合影响，不能归为单一本体因果效应。9–10 项验证相同 ARX 场景的专家可行性，独立于模型策略成功。
- 从旧方案删去 K26、ARX 无恢复和无路由消融，共 6 项，并加入 2 项 H50。保留 K4 是为了本体比较，不用于扩大原生窗口筛选。
- 本候选沿用历史已锁定的两个 10 g 夹具及两种本体的实测初态读回。作者缓存 100000/200000 只作为来源对照，公开缓存不含完整对象实测位姿，不能绑定到当前 ARX 专家及跨本体初态。本次选择原有配对夹具以维持同一研究问题。两夹具可能与公开训练分布相近，训练逐条样本无法确认。不能用本候选估计论文场景上的 52.6%。 [源协议审计](pi05_robotwin_protocol_audit.md)

## 准入、记录与止损

先启动共享模型并确认 GPU 内存、checkpoint 和环境。四个 Aloha 无动作探针分别为每个 seed 的 H50/K31，从独立重置、同种子 RNG 与同一提示得到完整的初态读回和首块。比较状态及三张 RGB 的 dtype、shape、字节、文本，以及输出 50×14 的 dtype、shape 和字节哈希。两条探针和清理均通过才准入该 Aloha seed。每个实际 Aloha 试次再次对照同一回执，在接受第一个动作前验证哈希。后续观测时间因窗口不同而改变。两个 ARX 探针独立验证实际初态、几何、代理输入和模型服务。

若安全清理成功，但一个 Aloha seed 的无动作探针初态或首块不匹配，保留该 seed 的 Aloha 为 `not_run`，独立准入的 ARX 可继续。实际试次首块漂移、真实碰撞、失同步、服务错误、硬超时或清理无法证明时，立即停止整场。候选阻断、动作预算耗尽、四次恢复额度耗尽、以及有原同 tick 凭证的可控接触停止，只有在完整清理后才允许下一试次独立重置。任何剩余时间不足一项硬上限时，保存其余 `not_run`。故障不会触发自动重试或另一个分配。

保留原始 50×14 输出、选择前缀、实际指令、左右手闭爪索引、实测 qpos 与工具误差、持有变化、指驱动、物块/接触、候选拒绝、任务谓词及清理回执。释放和托盘支撑须从持有变化及物理接触复核。任务谓词为真时也分别标注物理阶段证据。正常任务失败与基础设施故障分别记账。

作者 Fixed H50 按行调用原生 `take_action`，动作中每物理步检查成功，可提前返回。TOPP 仍由冻结 RoboTwin 执行，沿用其回退和夹爪插值。项目 H50 保持项目附加门禁，因此只与同门禁 K31 形成因果窗口对照。整个被选择前缀先走现有目标包络，H50 第 32–50 行越界可以让前缀在任何动作前被拒。这是需要记录的真实执行差异。 [冻结上游 rollout](https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/pace/src/pace/rollout.py)

## 资源预算

候选为 **1×RTX 4090 24 GB、4 CPU、32 GiB**。历史峰值 14,394 MiB，现场仍需保留至少 1024 MiB 显存余量。价格按 SCNet 昆山历史已核验的 **¥3.60/卡时**，本轮预算以此作为单价上限，未来需重新核验有效订单价。 [历史报价凭证](../research/g2/final-cpu-review/scnet-rate-contact-stop-20261107.json)

| 阶段 | 估时 | 预算根据与硬界限 |
| --- | ---: | --- |
| 图形/运行时准入 | 180 s | 历史入口耗时与 300 s 阶段预算，硬 300 s |
| 共享模型导入、校验、恢复 | 350 s | 127081074 分段实测约 285 s，模型就绪最多 960 s |
| 首次 JAX 编译及推理 | 90 s | 127081074 约 86 s，含在首个探针和整体启动上限内 |
| 六个无动作探针 | 1800 s | 127139405 的 Aloha 探针约 309–364 s、ARX 约 151–153 s。模型、编译与探针合计硬 3600 s |
| Aloha K31 两项 | 860 s | 历史 K31 完整外层记录约 390–427 s/项，硬 2400 s |
| Aloha K4 两项 | 1240 s | 历史约 473–621 s/项，硬 2400 s |
| Aloha H50 两项 | 1800 s | 每项 900 s 是高于历史 K31 的规划留量，硬 2400 s。H50 的实际速度和到达阶段未知 |
| ARX K4 两项 | 1800 s | 历史实际早停约 163–276 s，按有更长进展和恢复留量，硬 5200 s |
| ARX 专家两项 | 2500 s | 历史 CPU 参考每条约 110–119 s 仿真时间，真实策略速度和场景路径存在偏差，规划留量 1250 s/项，硬 4400 s |
| 统计及完整清理 | 180 s | 各估 90 s，统计硬 180 s、清理保留 120 s |

估时合计约 **3 小时、¥10.80**，为预算规划值。阶段及十项硬预算合计 **21000 s（5 小时 50 分）、¥21.00**。分配硬限 **25200 s（7 小时）、¥25.20**，内含 4200 s 机动余量。低于 9.5 小时及 ¥34.20 要求。按固定节点和资源核验，当前候选提交开关关闭。 [历史模型阶段](../research/g2/run-127081074/model/phases.jsonl) · [历史 Aloha 墙钟](../research/g2/run-127081074/batch-status.json) · [历史 ARX 与探针](../research/g2/final-cpu-review/result-127139405.json)

## CPU 复现及动态边界

```bash
export G2_PYTHON=/path/to/cpu/python
export OPENPI_CLIENT_SRC=/path/to/pinned/openpi/packages/openpi-client/src
export PYTHONPATH="$OPENPI_CLIENT_SRC" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
check_dir=$(mktemp -d)
bash scripts/g2_cpu_preflight.sh "$check_dir/cpu"
"$G2_PYTHON" scripts/g2_final_prepare_review.py "$PWD" "$check_dir/release" \
  --unified-transfer --archive "$check_dir/release.tar.gz"
python3 -B "$check_dir/release/scripts/g2_final_deploy.py" "$check_dir/release" --verify
```

打包读取私有研究工作区中已有的两条专家命令轨迹，只收录冻结校验通过的文件。GitHub 交付代码、摘要和哈希，包单独留存。真实 GPU 首次推理、RGB 位级重现、初态碰撞及接触、生产 ARX 专家后缀和集群进程清理仍为 `GPU_DEFERRED`。同一分配内的无动作探针是现场验收的一部分。 [CPU 验收与包指纹](../research/g2/unified-transfer-review/verification.json)
