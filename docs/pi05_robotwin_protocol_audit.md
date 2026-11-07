# π₀.₅ RoboTwin 原生执行协议审计

## 结论

固定 checkpoint 和归一化资产的身份核对通过。项目的原生 Aloha 诊断与作者评测在窗口、重置/初态、语言、随机流、安全门禁和终止预算上有明确差异。因此本地结果只能评价项目实际执行协议。现有证据未发现权重或统计串用、左右臂互换、弧度换算错误，或客户端再次累加关节增量。

作者论文附录 B.3 给出 `handover_block` 单项结果。Fixed H=5、25、50 和 PACE 分别为 **3.6%、33.2%、52.6%、53.1%**，每种方案报告 900 个 episode。模型卡声明该发布覆盖论文的单任务模型。这是作者协议下完整任务成功的外部证据。公开材料未给出表格每次运行的参数文件哈希、种子轨迹和完整运行回执。本项目尚未复现这些成功。 [PACE 论文 §4 与附录 B.3][paper] · [固定模型卡][card]

## 版本和证据

| 标识 | 固定 revision | 用途 |
| --- | --- | --- |
| HF | `17ef545270cf26ed81169893b60f845f80ce6557` | `niejunnan25/pi05-robotwin`，模型卡、manifest、任务统计 |
| TRAIN | `ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12` | OpenPI-PACE 发布所标原始验证配置 |
| SERVE | `f4e52e6de4cc487a11585a3a7414d610b2883dae` | 本项目固定 OpenPI-PACE 服务源码 |
| SIM | `fd2e624cc19325cc9b20a620c82d51c0d8e25cc1` | 作者公开 PACE 评测及 RoboTwin 仿真源码 |
| BASE | `13c3c47ff4312dd62484bcd51be034af55c062d1` | 作者声明的 RoboTwin stable_2.0 基线 |
| DATA | `3dc3b798668feb99ac61cc9086d84cbcc3d79186` | 官方 RoboTwin2.0 数据和本体资产 |
| LOCAL | `c6b6b54460c566c75b640f7a8f5ce76195ded94d` | 本轮审计前的 Draft PR #2 |

源码逐文件 Git blob ID 和 SHA-256、HF URL 与内容哈希在[源码锁](../research/g2/protocol-audit/source-lock.json)。TRAIN/SERVE 核对的 10 个配置、加载器、服务、策略和变换源码逐字节一致，README 有变化。SIM 的基础任务、任务文件、机器人、规划器和两份配置共 6 份与 `127054052` 冻结源哈希一致。历史运行 release 另见 [127054052 冻结上下文](../research/g2/failure-audit-127054052-r1/release-context/)、[127081074 发布哈希](../research/g2/g2e-20261106-r1/release-hashes.json)、[127139405 发布记录](../research/g2/final-cpu-review/submission-127139405.json)。

## 逐项核对

| 项目 | 作者协议与具体实现 | 本项目对照 |
| --- | --- | --- |
| 训练任务、本体和来源 | HF manifest 声明单任务、50 条 `demo_clean`、训练 seed 42、π₀.₅ base 初始化、EMA 0.99、step 30000。TRAIN `pace/src/openpi_pace/configs.py:make_configs` 绑定 `robotwin/handover_block`，AdamW、batch 32。SIM `task_config/demo_clean.yml` 明确为 Aloha-AgileX。DATA 对应 `demo_clean.zip` 中 50 个 HDF5 的路径都在 `handover_block/aloha_agilex/data/`。 | Aloha 是有直接公开材料支持的训练本体。训练时实际本地 LeRobot 数据快照、逐条成员清单、生成 commit 仍为 **UNKNOWN**。官方 `demo_clean.zip` 与本项目抽查的 `aloha-agilex_clean_50.zip` 有不同 LFS ID，50 个 HDF5 的大小与 CRC32 组合没有重合。不能将它们认作字节相同的训练样本。 [模型卡][card] · [训练配置][train-config] · [clean 配置][clean] · [ZIP 目录凭证](../research/g2/protocol-audit/official-zip-directory.json) |
| 训练状态、动作标签和时间 | TRAIN `LeRobotAlohaDataConfig.create` 映射 `observation.state` 和 `action`。`training/data_loader.py:create_torch_dataset` 从数据集 metadata 的 fps 取 `t/fps` 未来序列。SIM 随仓 `policy/pi05/scripts/process_data.py:data_transform` 把当前帧驱动记录作状态、下一记录帧作动作标签。随仓 LeRobot 转换器的 `create_empty_dataset` 填 50 fps。 | 确认参考转换语义和代码中的 50 fps；该 checkpoint 实际预处理代码、实际 LeRobot fps、尾部边界填充记录为 **UNKNOWN**。官方原始 HDF5 不是训练格式，内部帧间距也不是固定 20 ms。 [数据配置][openpi-config] · [训练读取][loader] · [参考预处理][preprocess] · [LeRobot 转换][convert] |
| 14 维排列、单位、夹爪 | SIM `Base_Task.get_obs` 和 `Robot.get_left/right_arm_jointState` 输出左 6 关节目标、左夹爪、右 6 关节目标、右夹爪。臂为绝对旋转关节位置，单位 rad。臂状态来自 drive target，夹爪来自归一化请求。0 闭、1 开。`Robot.set_gripper` 才将标量换算为本体驱动，Aloha 名义区间 −0.010 至 0.045 m，并使用实测开度限制单次开度变化。 | `pace_client_smoke.encode`、`NativeBackend.capture/execute`、`g2_capture.capture_snapshot` 保留模型状态字段和左右顺序。实测关节另用于控制监测，模型状态经 float32 转换。原生 `g2_episode.native_prefix` 额外做整前缀关节限位和夹爪 0.01 容差内截取，超包络则拒绝。未发现额外累加增量。 [上游机器人函数][robot] · [上游动作接口][base-task] · [项目原生执行](../scripts/g2_native_backend.py) |
| 相机和语言 | SIM `rollout.encode_observation` 把 head/left/right RGB 映射为 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 的 CHW。TRAIN `AlohaInputs.__call__` 将三路送入 base/wrist 输入，`ModelTransformFactory.__call__` 将图像缩放至 224×224，使用 Paligemma prompt 及离散状态路径。训练 prompt 由 `PromptFromLeRobotTask` 从数据任务元数据取；随仓转换器抽 `seen`。评测 `sample_instruction` 抽 `unseen`。 | 三路映射和 live RGB 入口对应。固定整任务提示、分布及渲染画面与作者逐 episode 抽样条件不同。参考预处理含 JPEG 解码和 640×480 缩放，真实训练 RGB/BGR、JPEG 与实际训练字节未核，不能宣布像素等价。提示差异是否因果仍为 **UNKNOWN**。 [原观察编码][rollout] · [指令抽样][sim-robotwin] · [Aloha 输入变换][aloha-policy] · [项目编码](../scripts/pace_client_smoke.py) |
| 归一化、反归一化与增量 | TRAIN `LeRobotAlohaDataConfig.create` 使用 `adapt_to_pi=False`、`DeltaActions/AbsoluteActions` 掩码 6,-1,6,-1，夹爪保留绝对值。PI05 使用 quantile norm。 `policy_config.create_trained_policy` 从任务 checkpoint 的 `assets/robotwin` 载入统计。输出顺序是反归一化、恢复绝对臂动作、截取 14 维。 | `pace_model_smoke.load_model` 使用同一 loader。HF manifest 的 28 个条目（含 tokenizer）与冻结清单的路径、大小、SHA-256 相同。独立读取的 3300 B 统计 SHA-256 为 `8a1f114a8873dfa0788c7f512851ee3063224ef2b8af270421076a2e4e544960`，state/actions 各含 14 维 mean、std、q01、q99。此前启动回执验证模型文件；本轮仅重新比对公开小文件与清单。 [固定 HF manifest][hf-manifest] · [模型加载][policy-config] · [变换][transforms] · [项目加载器](../scripts/pace_model_smoke.py) |
| 模型随机流、动作块和重查 | TRAIN `Policy.infer` 的 JAX RNG 默认由 key(0) 开始并逐 query split。SIM `evaluate.evaluate` 要求 50×14。论文基线 Fixed H=5/25/50，源码允许 1–50。PACE `core.select_horizon` 按两臂关节速度选最早合格谷点，排除夹爪，无谷时回退至最多 50。`run_episode` 消费前缀后重新观察，未用后缀被丢弃。 | `127054052` 为 K1/K4，`127081074` 为 K1/4/26/31，`127139405` 仅实际运行两条 Aloha K31。没有执行论文基线或 PACE。G2-E 及后续按试次主动重置 RNG，每 seed 分别为 20261106/20261107。首块同 seed 跨 K 完全相同，后续观察/输出会随执行改变。 [作者入口][evaluate] · [选择器][core] · [策略随机流][policy] · [项目窗口规则](../scripts/g2e_protocol.py) · [项目 RNG](../scripts/g2e_model_entry.py) |
| 原生 qpos 控制、物理步与完成 | SIM `Base_Task.take_action` 对每一行目标分别调用双臂 MPLib TOPP，按左右进度比例交错推进；起点使用关节 drive target。TOPP 采样与默认物理积分均为 1/250 s。单行动作的物理步数随规划长度变化。TOPP 异常/空轨迹一侧跳过臂驱动更新，但仍可在 50 步回退中发送夹爪。每物理步检查成功并允许提前返回。 | 本项目沿用同一 `env.take_action(..., action_type="qpos")`，G2-E 注入逐步候选、实测几何、接触和跟踪检查，可在动作中间停机。环境计数先增加，本项目另记录已完整动作。历史 TOPP 回退的逐调用触发数为 **UNKNOWN**。PACE 50 Hz 是速度特征标尺，不等于策略 50 次/秒或每动作固定 5 步。 [动作循环][base-task] · [原生驱动][robot] · [项目安全门禁](../scripts/g2e_native_guard.py) |
| 成功、预算与初态 | `handover_block.check_success` 比较物块功能点 0 到目标点 1，x/y 各 <30 mm，z <10 mm，右夹爪归一化请求 >0.8。SIM 评测上限 800 个 `take_action`。公开 `collect_episodes` 从 100000/200000/300000 开始筛选，每分片 100，筛选专家通过后重新 setup，或使用已发布缓存。 | 项目沿用任务判据，另记左持→双持→右持→释放/支撑过程。两条配置使用 `scene_seed=0/1` 和实测锁定夹具，不是上游评测分片；项目 native 限 300 个动作，G2-E 另有限时和 25000 物理步保护。初始化独立几何、物块质量与渲染验证、停止规则、日志与持有审计均是项目附加条件。 [成功判据][task] · [步数配置][steps] · [场景筛选][evaluate] · [项目场景实现](../scripts/g2_task_runtime.py) |
| 仿真版本 | 作者 README 指向 BASE，本文读取其 SIM 派生仓，`script/_install.sh` 另调整 SAPIEN/MPLib 安装源码。论文每批实际 SAPIEN、PhysX、MPLib、驱动及资产的完整版本清单未见，记为 **UNKNOWN**。 | 本项目已有 SAPIEN 3.0.0b1/MPLib 0.2.1 的 CPU 环境凭证和固定场景源。源码一致不能证明两个运行镜像和接触结果逐步一致。 [作者环境][sim] · [作者安装脚本][install] · [本项目环境凭证](../research/robotwin/env-build/evidence-20261102/cpu_validation.json) |

本轮没有启动模型或仿真。对真实闭环影响最大的已知协议差异需要成对控制。可复算流程、数据范围和下一项实验见[决策报告](pi05_next_experiment_decision.md)。

[paper]: https://arxiv.org/html/2606.00537v1
[card]: https://huggingface.co/niejunnan25/pi05-robotwin/resolve/17ef545270cf26ed81169893b60f845f80ce6557/README.md
[hf-manifest]: https://huggingface.co/niejunnan25/pi05-robotwin/resolve/17ef545270cf26ed81169893b60f845f80ce6557/manifest.json
[train-config]: https://github.com/niejunnan25/Openpi-PACE/blob/ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12/pace/src/openpi_pace/configs.py
[openpi-config]: https://github.com/niejunnan25/Openpi-PACE/blob/ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12/src/openpi/training/config.py
[loader]: https://github.com/niejunnan25/Openpi-PACE/blob/ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12/src/openpi/training/data_loader.py
[aloha-policy]: https://github.com/niejunnan25/Openpi-PACE/blob/ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12/src/openpi/policies/aloha_policy.py
[policy]: https://github.com/niejunnan25/Openpi-PACE/blob/ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12/src/openpi/policies/policy.py
[policy-config]: https://github.com/niejunnan25/Openpi-PACE/blob/ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12/src/openpi/policies/policy_config.py
[transforms]: https://github.com/niejunnan25/Openpi-PACE/blob/ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12/src/openpi/transforms.py
[sim]: https://github.com/niejunnan25/PACE/tree/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1
[clean]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/task_config/demo_clean.yml
[robot]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/envs/robot/robot.py
[base-task]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/envs/_base_task.py
[task]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/envs/handover_block.py
[steps]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/task_config/_eval_step_limit.yml
[rollout]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/pace/src/pace/rollout.py
[sim-robotwin]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/pace/src/pace/robotwin.py
[evaluate]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/pace/scripts/evaluate.py
[core]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/pace/src/pace/core.py
[preprocess]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/policy/pi05/scripts/process_data.py
[convert]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/policy/pi05/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
[install]: https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/script/_install.sh
