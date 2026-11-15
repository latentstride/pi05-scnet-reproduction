# 第六步：π₀.₅ 原生 ARX-X5 配对闭环

研究问题：第五步在 ARX-X5 专家数据上训练的 LoRA，能否改善 `handover_block` 的抓取、交接、放置和原生任务成功率？每个场景比较 F（原始 PACE 底座）与 L（同一底座加第五步选定 LoRA）。协议与历史冻结 ARX 和 Aloha→ARX 投影不同，历史结果不进入此分母。

| 已有接口 | 第六步接线 |
| --- | --- |
| PACE `load_environment`、RoboTwin `setup_demo`、`get_obs`、`check_success` | 两个 ARX-X5 实例，seed 0–4 独立重建。使用原生演员、稳定化和任务判定，保存初始状态、相机标定、状态差异。 |
| PR #5 `arx_settings`、`joint_space_setup`、`capture_arms`、`capture_initial` | 复用资产哈希、双臂初始化和物理反馈；仅新增原生观测格式和配对核验。 |
| PR #5 `ARXBackend`、`QuinticMotion`、`JointCollisionWorld`、`ContactWorld`、`ContactTracker` | 单行关节目标按已有夹爪标定送入原门禁：同场 FCL、PhysX 自碰撞规则、保持/释放保护、实测 FK、运动限幅及跟踪。原生目标不需要 IK，无源机械臂、代理或 TCP 投影。 |
| PR #6 `native_observation`、`NativeAdapter`、`NativePolicy`、`Session`、`Dataset` | 使用其 HWC RGB、命令状态、归一化、采样和逆变换。评测端按原 WebSocket 契约收发；服务包装层只补充 checkpoint 身份。CPU 替身使用同一 wire codec 和真实 ARX norm stats。 |
| PR #5 视频和 JSON 写入 | 沿用头相机 MP4（按查询采样的回放视图）。新增逐物理步轨迹、二进制 WebSocket 请求与响应和各试次回执。 |

PR 以 #6 的 `3ff31cb` 为基线；安全运行模块从 #5 的 `3f95695` 单独部署，通过 [文件哈希](pr5_runtime_sha256.json) 限定运行代码。两份依赖无需合并。ARX 标定、控制和接触参数逐项取自既有 [原生 ARX 验收配置](https://github.com/latentstride/pi05-scnet-reproduction/blob/64e92b46d5e9b6eebdf4e6de7deb79ac3eeb2756/research/g2/arx-direct-online-20261114/evidence/127999421/release/configs/unified-seed1-arx-full-w4.json)，摘录见 [本轮固定安全配置](arx_safety_profile.json)。块质量必须与该配置已经验证的 10 g 一致；本入口不改质量或门禁。碰撞阈值和跟踪容差不因 F/L 或 CPU 改变。

### 数据流与配对

1. RoboTwin 稳定化结束后冻结物理时钟。采集三张 `240×320×3 uint8 HWC RGB`、相机内外参、命令与物理反馈。固定源码的相机从 SAPIEN `Color` 读取 RGBA 并去掉 alpha，在线路径不做 BGR 翻转。旧示范数据 JPEG 的通道修正已在第五步完成。
2. `joint_action/vector` 的顺序为左 `joint1…6`、左夹爪、右 `joint1…6`、右夹爪；手臂源自当前 drive targets，夹爪为 RoboTwin 的最新归一化命令，0 闭合、1 张开。检查采样命令与驱动读取一致。独立记录实际 arm/finger qpos、夹爪实测开度、四指驱动目标和机器人/物体位姿。命令、物理状态和服务转换后的模型 state 各有含义。
3. 每试次 `query_index` 从 0 起单调递增。两组使用相同策略 seed 42，继承 Step 5 的 `fold_in(seed, query_index)`。服务用 `NativePolicy` 将命令与图像转换成模型输入，采样得到内部 `50×32`，由同一 adapter 截取 14 列、逆归一化、仅为手臂加回当前命令 anchor、裁剪夹爪并返回 `50×14`。全部响应须有限、形状正确；每条要执行的行另通过 ARX 关节、夹爪、碰撞和跟踪门禁。未执行的尾部没有驱动意义。
4. H50、每次最多执行 K4、每试次最多 800 个模型行。每行从当前实测姿态走原 20 ms quintic 控制周期和 4 ms PhysX 步，运动可能有多个控制周期；不把模型行当固定物理时间。源示范缺逐帧时钟。原接触保护或持物转换会先丢弃余行再采样。逐步记下控制周期、实际 tick、物理和墙钟时间。
5. 配对条件见 [pairs.json](pairs.json)。同一 seed 在独立环境重置，不共享活跃 PhysX 状态。L 发首个请求前比对实际关节、位置、姿态、夹爪、驱动目标、速度、块质量和初始相机几何；保存 F/L 初始回执。物理位置公差 10 µm，姿态 0.002°，臂角 10 µrad；相机矩阵和其余数值使用绝对 `1e-5`。渲染噪声不要求像素哈希相同，原始帧均保存。姿态不配对则停止 L，排除该对。

### 阶段、停止与统计

| 事件 | 冻结的辅助判据 |
| --- | --- |
| 接近、左/右接触 | TCP 至红块中心 ≤ 120 mm。至少一只校准手指有 PhysX 块接触才记接触；两指分别有接触才记双指。阈值来自已有接触跟踪器。原生关节控制的 IK 字段为 `unavailable`，FCL 拒绝单列。 |
| 左手抓持和抬升 | PR #5 跟踪器认定 holder（双指正冲量、至少连续 3 个物理步、约束内的物体相对 TCP 变换）；同时较初始块中心抬高 ≥ 30 mm。接触过且未抬升/持有时，<= 10 mm 高度偏移、>=20 mm 水平位移标记侧推。非预期碰撞保存原安全原因、场景接触对象和冲量；分析标签不参与控制。 |
| 共同持物与接管 | 两侧都是实测 holder 且未在释放。其后左手已抬升过块、无左指接触、左夹爪实际开度 >0.8 且只剩右手持物，持续 ≥40 ms 才记右手接管。曾抬升后在目标外失持且观测到下降 ≥30 mm 标记掉落；安全停止时只有失持证据则保留原原因。 |
| 目标、释放与得分 | 原生块—目标功能点在世界 XYZ 各差 < (30,30,10) mm 记进入目标；完成接管后、仍在区域、开右夹爪且无保持/手指接触才记实测释放。最终任务成功只调用 RoboTwin `check_success()`。 |

所有指标均给出 `observed`（含首次 tick/时间）、`not_observed`、`unavailable`。其中 `not_observed` 只描述已记录的时间范围。安全退出、动作耗尽、资源截断、初态不一致、服务或仿真异常各自保留。墙钟耗尽与 SIGTERM 记 `RESOURCE_CENSORED`，意外硬杀导致缺回执时离线表记 `MISSING`。任务成功、动作到限或策略触发的执行门禁停止可计入正式分母，但须有真实模型、真实 RGB/视频及另一组的已匹配完整结果。CPU 替身、资源截断或基础设施异常均不会变成模型失败。

每条目录包括 `initial.json`、`trajectory.jsonl`、`collision-checks.jsonl`、`wire/`、`receipt.json`；真实渲染另写 `head.mp4`。二进制 wire 文件保留原始消息及对应哈希与时间。运行下列入口可从轨迹重算指标，与回执互检，产生 `pairs.csv` 和 `summary.json`：

```bash
PYTHONPATH="$step6" python -m scripts.step6_arx.analyze "$results" --output "$new_summary"
```

### CPU 验收与正式 checkpoint 接入

CPU 在已有 SCNet RoboTwin 容器启动，`ROBOTWIN_GPU=0` 与空 `CUDA_VISIBLE_DEVICES`。下例需要原 PACE checkout、OpenPI client、第五步数据和 PR #5 文件快照。建议把上述变量设置成项目根内的绝对路径：

```bash
export ROBOTWIN_GPU=0
bash "$root/scripts/run_robotwin_env.sh" env CUDA_VISIBLE_DEVICES= \
  PYTHONPATH="$step6:$root/tools/pace-smoke-deps:$openpi/packages/openpi-client/src" \
  python -B -u -m scripts.step6_arx.evaluate \
  --pr5-root "$pr5" --pace-root "$pace" \
  --asset-audit "$pr5/research/g2/g2e-20261106-r1/release/asset-audit.json" \
  --output "$results" --cpu-fixture --headless --dataset "$dataset" --invalid-variant lora
```

在第五步训练完成后，使用有效的第五步发布 `$release`、该发布对应的 `$cpu_receipt`、已锁定数据/审计和完整训练运行 `$run`。同一已获授权的分配内，推荐先做真实三相机与模型短窗核验，再按 F/L 配对跑正式矩阵。单台单卡无法同时容纳两服务时，先运行 F 的全部 seed 并结束 F 服务，然后从同一结果目录启动 L；仿真仍按 seed 比较独立重置。若显存已验证充足，可用两个 URL 执行默认交替顺序。以下只是已有分配内的进程入口：

```bash
# 每轮取 variant=frozen 或 lora；单个服务独占 GPU。
bash "$release/scripts/step5_arx/env.sh" gpu "$root" "$step6/scripts/step6_arx/serve.py" "$variant" \
  --project-root "$root" --dataset "$dataset" --data-audit "$data_audit" \
  --cpu-receipt "$cpu_receipt" --run "$run" --seed 42 --port 8000

# 在同一节点的 RoboTwin 仿真容器中执行；先 F 后 L，URL 选相应参数。
PYTHONPATH="$step6:$root/tools/pace-smoke-deps:$openpi/packages/openpi-client/src" \
  python -B -u -m scripts.step6_arx.evaluate \
  --pr5-root "$pr5" --pace-root "$pace" \
  --asset-audit "$pr5/research/g2/g2e-20261106-r1/release/asset-audit.json" \
  --output "$results" --variant "$variant" "--${variant}-url" ws://127.0.0.1:8000
```

服务启动仍走第五步 GPU 与发布哈希准入；包装层核对 `identity.json`、`selected.json` 与已提交的 Orbax checkpoint，回显底座、norm stats、训练身份及选中 checkpoint 内容哈希。它不写训练目录。冻结策略仍加载原始 PACE 权重；两个服务都读取同一第五步统计。没有有效 LoRA 时只可使用标明的 CPU fixture，或等待正式模型。

预计正式资源：一张第五步现有入口支持的 RTX 4090（24 GiB）串行服务、4–8 CPU、32–64 GiB RAM。现有 PACE 参考推理峰值约 14 GiB，仅供预估；L 的真实 GPU 恢复、编译与显存须先短窗验证。十试次每场 2,550 秒，整矩阵试次墙钟上限约 7 小时 5 分钟，另留容器、模型加载和清理时间。最大 2,000 次三相机请求约 1.3 GiB 原始 RGB，加上轨迹、FCL 审计和 MP4，建议独立结果卷预留至少 10 GiB，并满足第五步服务原有 16 GiB 余量门槛。未启动或提交任何 GPU 作业。
