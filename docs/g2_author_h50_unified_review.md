# 作者原生 Aloha Fixed H50 统一准入

**结论为 `READY_FOR_REVIEW`。** 建议在未来单次 7 小时分配中追加场景 `100000` 的作者基线，保留原十项和两个 H50/K31 配对。候选包保持 `submission_allowed: false`，部署仍须另行绑定批准的作业 ID、包清单哈希和当期报价。没有改变原 Slurm 脚本、冻结配置、检查点、统计、控制器或历史结果。 [候选包、账本和哈希](../research/g2/author-h50-unified-review/)

## 作者试次预算

| 边界 | 候选值 | 计入内容及依据 |
| --- | ---: | --- |
| 作者动作 | 800 | PACE 原生动作计数、原任务谓词和 50 行执行窗口。完整运行最多 16 次策略请求。 |
| 策略阶段物理步 | 130,000 | 复位后的诊断 tick 0 起计数。历史 243 个完整原生动作实测 19–178 步。按试次均值外推 800 动作为 66,026–106,308 步。上限对外推高端留约 22% 余量，折合 162.5 步/动作。若每步都达到已观测峰值 178，则需 142,400 步。场景 100000 的真实 TOPP 上界仍为 `UNKNOWN`。 |
| 软墙钟 | 2,400 秒 | 从父进程启动子试次前计时，包含容器启动、原生复位、真实 RGB 和首请求、暖查询、执行及试次证据 I/O。历史条件外推为 1,105–1,781 秒，不是上界。另加一次历史 73 秒级冷编译仍低于候选软限，现场是否额外编译待测。 |
| 硬墙钟 | 2,700 秒 | 父子共用同一绝对单调时钟。最后 300 秒中，前 290 秒留给安全暂停、hold/close、连接关闭及原子回执，最后 10 秒保留给宿主终止和回收。 |
| 连接及每次请求 | 最多 390 秒 | 连接握手与服务问候共用一次上限。每次已发送请求另用一次 390 秒上限，并受同一个 2,400 秒软截止约束。 |

作者复位的 2,500 个稳定步已在真实 CPU 中读回，计入作者试次墙钟。复位中的相机初始化和开爪也计入该墙钟，均在策略物理步预算之前。全场图形驱动初始化与原六项动态准入归启动账本。共享模型的首次编译随原动态准入计一次，只有作者输入实际触发额外编译时才占作者试次墙钟。作者每次请求保存三相机帧和原始动作块，另有逐步事件日志。原试次视频保存在原试次墙钟内，作者基线使用图像与动作回执。历史事件文件按 130,000 步线性外推约 3.5 GB，只用于规划；候选在部署和作者试次启动时分别要求 12 GiB、8 GiB 可用空间。 [历史计时与 TOPP 证据](../research/g2/author-native-h50/budget.json) · [前轮物理验收](g2_author_native_h50_admission.md)

## 7 小时统一账本

| 秒 | 原十项 | 追加作者基线 | 替换 seed0 H50 |
| --- | ---: | ---: | ---: |
| 原生 Aloha，每项 1,200 | 7,200 | 7,200 | 6,000 |
| ARX 策略，每项 2,600 | 5,200 | 5,200 | 5,200 |
| ARX 专家，每项 2,200 | 4,400 | 4,400 | 4,400 |
| 作者 H50 硬限 | 0 | 2,700 | 2,700 |
| 图形/Vulkan/EGL | 300 | 300 | 300 |
| 共享服务、首次编译和原六项准入 | 3,600 | 3,600 | 3,600 |
| 统计及证据汇总 | 180 | 180 | 180 |
| 整场清理 | 120 | 120 | 120 |
| 现行阶段与试次封套合计 | **21,000** | **23,700** | **22,500** |
| Slurm 提前通知预留 | 180 | 180 | 180 |
| 跨阶段进程切换、回收和落盘预留 | 180 | 180 | 180 |
| 25,200 秒中的剩余 | **3,840** | **1,140** | **2,340** |

这是各阶段按硬限占满后的封套比较。原方案列额外列出两笔评审预留，原部署逻辑保持原样。Slurm 脚本设 `TERM@120`，按文档再计 60 秒信号提前量。候选追加试次后用全局截止保护两笔预留和 180 秒统计、120 秒全场清理。跨阶段预留覆盖子进程启动和原试次计时外的回收等开销。作者自身 10 秒回收窗已在 2,700 秒内。替换能多留 1,200 秒，但会删去两个配对中的一个，令 seed0 K31 失去同初态 H50 对照。追加保留原十项的相对顺序和两组配对。原试次若发生真实危险、服务退出或无法验证清理，作者项也会按安全规则留为未运行。 [Slurm 信号文档](https://slurm.schedmd.com/sbatch.html#OPT_signal) · [机器可读账本](../research/g2/author-h50-unified-review/ledger.json)

## 调度、分类与 CPU 验收

原 `unified-plan.json` 和十个原命令按原字节和参数保留。候选包增加已校验的 sidecar，宿主仅在已验证且另获授权的包中构造第十一项。原六项准入和原客户端种子重置保持原样。服务扩展在同一个端口处理原协议和作者协议。作者连接独立初始化 key(0) 并校验请求序号、输入哈希和连续的 RNG 回执。作者试次在自己预算内完成真实三相机、复位读回和首次原始响应检查，然后才允许动作。 [统一适配源码](../scripts/g2_author_h50_campaign.py) · [作者入口](../scripts/g2_author_h50_trial.py)

- `SUCCESS` 与 `TASK_INCOMPLETE` 分别由作者原谓词和完成 800 动作结算。两者单列在作者诊断统计中。
- `controlled_candidate_stop` 或可证明的 `RESOURCE_CENSORED` 要求同 tick 状态与接触读回、几何和跟踪复核、hold 前后哈希一致、close、场景退出、匹配的时钟及进程组回收。服务须存活，且没有未收回的请求。资源截止不计任务失败。
- 危险、凭证缺失、未知异常、服务死亡、清理失败均停整场。硬截止记录 `RESOURCE_CENSORED` 并按基础设施停止处理，宿主发终止信号并验证进程组；即使进程已回收，也不会将其视为已证明的物理安全停止。请求发出后超时也记录资源截止，但因无法证明共享服务已释放请求，整场停止。实测危险优先于同 tick 的软截止。旧异常路径仍要求真实候选来源和原有停止凭证。
- 55 项针对性 CPU 测试通过。测试使用真实统一监督器、子进程和固定的上游 rollout，并注入模型、时钟和部分仿真替身，覆盖越过旧 1,150 秒、完整 800 动作、早成功、候选拒绝及失败关闭。原客户端和作者子进程共用真实 CPU WebSocket 服务，用键与模型替身核对重置和序号。宿主在无第三方包环境中验证了真实 CPU 候选停止和资源截止回执。另在真实 CPU PhysX、原生 TOPP 和原安全门禁下，以 1 步测试预算触发截断。实测恰好执行 1 个物理步、0 个完整动作，获得同 tick 停止凭证并成功回收进程组。 [针对性测试](../research/g2/author-h50-unified-review/targeted-tests.log) · [资源停止回执](../research/g2/author-h50-unified-review/physx-cutoff.json)

**GPU 现场检查**包括真实 Vulkan/EGL、三路相机、冻结模型和真实 JAX RNG 回执、首块编译/延迟、文件系统余量和真实 GPU 进程退出。作者策略在 130,000 步或 2,400 秒内能否走满 800 动作仍未知；保护和资源截止均单独记录。以上现场验证放在同一次已批准分配内。

## 复现

```bash
export PYTHONPATH="scripts:tests:$OPENPI_CLIENT_SRC"
export PACE_H50_ROLLOUT="$pinned_pace/pace/src/pace/rollout.py"
python -m unittest -v test_g2_author_h50_campaign test_g2_author_native_h50 \
  test_g2_final_deploy test_g2_final_startup test_g2_final_exit_audit \
  test_g2_candidate_stop test_g2_final_contact_stop test_g2_unified_review
CUDA_VISIBLE_DEVICES= python scripts/g2_author_native_h50_cpu.py \
  --pace "$pinned_pace" --release "$frozen_release" \
  --reference research/g2/author-h50-preflight/cpu-reset-readback.json \
  --output "$new_cpu_output" --case physics-cutoff
python scripts/g2_author_h50_campaign.py . "$frozen_release" "$new_release" \
  --archive "$new_archive"
python -S -B "$new_release/scripts/g2_final_deploy.py" "$new_release" --verify
```
