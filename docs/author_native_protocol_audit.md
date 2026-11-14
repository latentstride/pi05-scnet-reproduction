# 作者原生与项目受保护入口

两条路径使用同一冻结 RoboTwin `handover_block` 资产。模型仓库为 `niejunnan25/pi05-robotwin`，checkpoint `17ef545270cf26ed81169893b60f845f80ce6557`，norm stats SHA256 为 `8a1f114a8873dfa0788c7f512851ee3063224ef2b8af270421076a2e4e544960`。PACE `fd2e624cc19325cc9b20a620c82d51c0d8e25cc1`，Openpi-PACE runtime `f4e52e6de4cc487a11585a3a7414d610b2883dae`。这是单场景复现协议。作者发布的整组评测中，实际随机流、首帧、首块和成功轨迹仍无逐值金标。

| 维度 | `author_native` | `protected` |
| --- | --- | --- |
| 入口 | `scripts/g2_author_native_entry.py`，把作者环境直接交给作者 `run_episode` | `scripts/g2_protected_entry.py` 标记独立模式后，以同一进程执行原 `g2_author_h50_trial.py` |
| 场景和输入 | 缓存分片 0 首条 seed 100000，作者 `load_environment → setup_demo → restore_cached_scene_state → sample_instruction`，作者 unseen 抽样和原观测编码 | 相同场景，既有复位比对、tick 0 实测接触、关节和几何准入 |
| 语言与 RNG | 复现 Python seed 100000 和单场模型 key(0)，原先实际指令为 `Grab the red block using the left arm.` | 同一重建；H50 旧共享服务协议已记录 RNG 哈希 |
| 动作 | 上游 `run_episode → env.take_action(qpos) → TOPP / 原回退 / 原夹爪限幅 → PhysX`，消费 50 行或由原生成功/800 动作上限提前结束 | 原始动作同样进入原生 `take_action`，但由项目包装器在每步前后准入。该 H50 包装器保留原 `hold` 和安全停止，不启用项目预载、接近修正、重写、裁剪或通用恢复 |
| 项目额外保护 | 执行链不实例化项目的 `AuthorBackend`、FCL 候选检查器、接触跟踪器或跟踪门禁 | 维持原 FCL 2 mm、候选和实测几何、夹爪/接触和跟踪保护，凭证不足时停止。ARX 的连续性和恢复仍保留在原跨本体保护链 |
| 记录和终止 | 请求、原始块、查询边界、RNG 和终止记录，作者 `close_env`。GPU 资源和进程组由外层监督，资源截止单列 | 保留既有逐步日志、实际接触、停止凭证、隔离与资源分类 |

作者原生入口仅接受新签发且绑定 Job ID 的 `author_native` 授权。历史 B/C 的授权不适用。入口不创建分配。入口使用既有 `run_deployment_stage` 隔离进程组、实施批准的硬墙钟并回收，同时保存宿主退出证据。未来部署仍须独立冻结发布包并验证 GPU 资源监控。当前未批准实际作业。CPU 验收复用冻结的 Headless 适配，仅替代渲染与未参与 qpos 的 CuRobo 启动，保留原生 PhysX、TOPP、关节和夹爪。它用于已保存动作的反事实，不用于验证实时视觉推理。

用实际固定版本的 PACE rollout 执行了协议测试，确认原 50 行逐值进入环境，第二次观测在动作 50，作者成功信号可提前终止，未成功时由原逻辑耗尽 800 动作。真实 CPU A/B 回放另见[对照报告](g2_cpu_counterfactual.md)。[原 rollout](https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/pace/src/pace/rollout.py#L8-L53) · [原生控制](https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/envs/_base_task.py#L1479-L1664) · [协议测试日志](../research/g2/final-system-review/protocol-tests.txt)
