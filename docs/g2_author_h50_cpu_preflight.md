# 作者场景 100000 的原生 Aloha Fixed H50 CPU 预检

**GPU 准入结论为 `NOT_READY`。** 冻结候选仍以项目 seed0 夹具、预载和接近修正运行，不能执行作者原生 H50 基线。CPU 已复现首步前候选拒绝的一个退出边界。没有当前 tick 接触样本时，hold 和 close 仍执行，但无法签发安全继续凭证。真实首帧和模型加载是未来同一次分配的现场验收项，当前结论由 CPU 可验证的运行契约和退出缺口决定。

## 场景身份和物理读回

作者 PACE `fd2e624`，Openpi-PACE `ba98766/f4e52e6`，checkpoint `17ef545`，RoboTwin 数据 `3dc3b79`。`expert_scenes.json` 分片 0 首条为 `scene_seed=100000, episode_info={}`。缓存哈希和完整版本见[场景配置](../research/g2/author-h50-preflight/scene-config.json)。CPU 检查沿作者 `load_environment → setup_demo → restore_cached_scene_state → sample_instruction(unseen)`，本场景 restore 没有任务特定的额外赋值。

`Base_Task` 按 seed 重置 NumPy/Torch，建立 0.74 m 桌面和墙，加载共享双臂 Aloha URDF、相机和原生驱动，回 home 并开爪，再生成红块和蓝垫并静置检查。头、左腕、右腕用 D435 320×240，任务配置关闭背景、杂物、相机位置、桌高和光照随机化。相机创建依然消耗 NumPy 随机数，省略整个调用会改变后续生成的物体初态。块长方体半尺寸为 30/30/100 mm，原 Actor 包装将质量设为 10 g。保留原始 TCP 标定、绝对 qpos 及夹爪请求语义。作者从生成的 unseen 池抽样。本预检固定 Python RNG 起点，历史论文选中的字符串未知。 [协议源码及行号](../research/g2/author-protocol-alignment/source-evidence.json)

**CPU 实测复位通过。** 在现有 Linux 容器中，原环境先因 CuRobo 导入时触发 CUDA 初始化而失败。离线复现脚本改用 CPU PhysX、相机元数据句柄和保留碰撞网格的 URDF，使用作者原 NumPy 夹爪插值，跳过 CuRobo 启动和视觉组件。作者 `setup_demo` 确实推进了物理场景并完成静置。方块质量读回为 10 g，方块、目标垫、桌面和双臂基座的位置与 G1 档案差为 0，姿态误差低于 0.1°，14 维驱动状态最大差为 0。实测臂关节限位、夹爪标定范围及初态几何通过，计入未持物的静止几何也通过。原生谓词为假，未执行策略动作。复现提示为 `Grab the red block using the left arm.`。作者 `close_env` 返回，CPU 进程组回收。 [物理状态](../research/g2/author-h50-preflight/cpu-reset-readback.json) · [环境和版本](../research/g2/author-h50-preflight/cpu-environment.json) · [进程日志](../research/g2/author-h50-preflight/reset-process.log)

历史 G1 `126905534` 同一缓存版本与 seed 已实测三路非恒定 RGB 首帧。已从原归档重新验证 `3×240×320 uint8`、像素哈希、14 维驱动状态和物体/基座位姿。该次块位置约 `[-0.152160, 0.227931, 0.840000] m`，与 `127192740` 锁定 seed0 的 `[-0.173313, 0.197930, 0.840001] m` 明显不同。G1 只执行了 8 步、两次查询，提示也不同，作为历史现场初态和成像参考。 [G1 归档说明](pace_g1_result_126905534.md) · [归档复核](../research/g2/author-h50-preflight/offline-check.json)

## 作者条件、项目保护和偏离

| 条件 | 作者原生路径 | 现有项目路径与验收含义 |
|---|---|---|
| 输入和模型 | 三路 320×240 CHW uint8 RGB，按头、左腕、右腕绑定键名。服务端转 HWC，等比缩放并填充至 224×224，再转至 [−1, 1]。状态为左六关节驱动目标 rad、左爪请求、右六关节驱动目标 rad、右爪请求，夹爪 0 闭、1 开。任务 quantile stats 与 unseen 提示。公开服务 JAX key(0)，每次 query split。 | 模型资产和变换契约可复用。项目首项显式 seed 20261106、项目提示和 float32 cast。未来基线须从作者原生状态编码构造请求。若共享服务先做启动探针，基线开始前需恢复一次 key(0) 语义并留证，块间继续使用同一 RNG 流。 |
| 窗口和动作 | 预测 50×14 绝对关节/夹爪，原生 `take_action(qpos)` 逐行执行，按 TOPP 插值。若某臂 TOPP 失败或零长度，该臂跳过关节驱动更新，仍按 50 个子步插值夹爪。正常跑满块后重查。 | 现有包启用预载修改指驱动，接近修正可修改目标并强制重查。作业 `127192740` 在 27 动作后丢弃 23 行，不能标为作者 H50。项目额外前缀裁剪也须与作者原生逐步限幅区分并记录实际差异。 |
| 保护 | 每物理步检查任务成功。作者判据要求功能点 XY 误差各小于 3 cm、Z 误差小于 1 cm，且右爪开度请求大于 0.8。 | 原项目限位、候选与实测几何、跟踪和接触保护须保留。安全检查可接受或拒绝；拒绝后结束本试次并留证。任何额外动作修改或提前重查构成协议偏离。 |
| 结束和统计 | 原任务谓词，或 800 个动作耗尽。 | 原候选为 300 动作，25,000 物理步，并有试次及分配墙钟上限。保留保护触发的终止类别，不把早于 800 的中断计作原生任务未完成。 |
| 清理与隔离 | 作者调用 `close_env`，每场 reset。 | 继续需要当前场景的停止凭证、hold/close/上下文退出、进程组回收和下一场独立准入。首次候选若在 tick 0 拒绝，当前实现缺少已建立的同 tick 接触样本，现有严格门禁会整场停止。 |

没有证据表明必须改动控制阈值才能做这种对照。可以审查一条**只拒绝、不改写命令的受保护 Fixed H50 路径**，允许时保持作者动作和查询节奏，一旦需要安全干预就删失该试次。当前发布包的锁定入口不具备这一协议，CPU 预检没有实现新的执行器。

## 安全回归与 GPU 准入

CPU 实跑既有退出路径测试 **22/22**，覆盖有凭证的候选拒绝、失同步、hold/close 错误、模拟 10 项调度和后续试次隔离。额外 tick 0 注入通过生产包装器验证，候选来源保留，物理步为零，hold/close 均执行，`measure_stop` 返回 `candidate stop lacks current physical contact sample`，整场继续条件不成立。该注入只验证退出边界。真实碰撞、未验证停止、清理失败和进程组不退出仍为整场停止条件。 [测试日志](../research/g2/author-h50-preflight/stop-tests.log) · [注入回执](../research/g2/author-h50-preflight/offline-check.json)

CPU 阶段的阻断条件是**缺少可执行、保持作者语义且不削弱现有保护的冻结入口，以及首个物理步前拒绝的同 tick 安全凭证验证**。25,000 物理步上限对应 800 个动作平均最多 31.25 个子步，而 TOPP 的子步数随动作变化，单臂回退就会走 50 个子步。能否在现行物理步、试次墙钟及共享服务余量内覆盖作者 800 动作，仍待验证。此前 `127147943` 的 8 个策略失败、2 个专家成功是项目协议的结果。 `127192740` 只运行首项，旧候选分类触发整场停止。两者均无法补齐作者条件。 [历史实验](g2_root_cause_127147943.md) · [分类修复](g2_candidate_stop_127192740.md)

未来若获准，**同一次分配开头**检测 Vulkan/EGL，复位并保存真实三路首帧、相机参数、状态和场景读回，核对其与 CPU/历史证据的身份和差异，再加载冻结模型并留存输入、RNG、首块及服务回执。实际 GPU 进程退出时序一并验收。论文对该缓存场景实际选取的指令及当时的服务 RNG 状态仍未知，因此这一条固定场景只能验证单例闭环。任一关键现场条件失败即按原因退出，无需单独冒烟作业。首块和后续记录应区分原始模型请求、被执行的 qpos、保护拒绝与实际接触。

- **成功**。遵循受保护的 H50 语义，触发原生谓词，证明冻结策略在这条重建场景有一个成功实例。
- **任务未完成**。无协议偏离，正常耗尽 800 动作且原生谓词未触发，只回答该场景与固定随机流。
- **安全停止或协议偏离**。单列删失及触发原因。有完整安全停止和隔离凭证的单项失败可继续。凭证缺失或实际危险须整场停止。基础设施超时单列，不计入原生策略结果。

**统一方案唯一调整建议**。只有通过准入、并确认目标是优先回答作者源本体基线问题时，才考虑用这条受保护基线替代 `unified-seed0-native-w50`。它占相同的源本体 H50 位置并保留后续 ARX 项，但会失去原 seed0 的 H50/K31 同夹具配对，不能继续报告那组窗口对照。本轮原名单和历史结果照旧。

## 复现

```bash
export PYTHONPATH="scripts:tests:$OPENPI_CLIENT_SRC"
python scripts/g2_author_h50_preflight_cpu.py --output "$new_cpu_report"
python -m unittest -v test_g2_candidate_stop test_g2_final_exit_audit
# 在已有 Linux CPU SAPIEN 3.0.0b1/MPLib 0.2.1 环境中
CUDA_VISIBLE_DEVICES= python scripts/g2_author_h50_reset_cpu.py \
  --pace "$pinned_pace_with_assets" --release "$frozen_24e337e1_release" \
  --output "$new_reset_directory"
python scripts/g2_author_h50_preflight_cpu.py \
  --readback "$new_reset_directory/readback.json" --output "$new_comparison_report"
```
