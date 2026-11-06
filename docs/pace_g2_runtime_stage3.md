# G2 第三轮：运行入口与双本体 CPU 验收

2026-11-05。已实现统一真实任务入口，完成本地 85 项回归和 SCNet CPU 几何、资产及导入核验。未申请或提交 GPU，尚未执行真实渲染任务的动态闭环。

## 统一任务入口

新增 `g2_run_task.py` 与 `g2_task_runtime.py`，复用固定 PACE checkpoint 服务，不启动新的模型进程。

- 新 episode 创建新场景，按固定源码 SHA-256 验证 checkout，载入原生 Aloha 或 ARX 配置。原生 Aloha 保留上游 qpos/TOPP 路径；ARX 使用已经实现的双臂关节路径与 FCL 检查。
- 场景创建/reset 也在 episode 的异常记录范围内。setup 失败仍输出结果；结束或异常后关闭视频、环境和 session。
- `g2_native_backend.py` 保留原生 take_action 行为，通过 scene.step 转发器记录实际物理步、关节与夹爪状态、驱动目标和 TCP 误差，失败时 hold，关闭时恢复原 scene。
- ARX 初始化的 warmup 绕过新增与上游一致的 200 点夹爪插值，避免 setup 内的 together_open_gripper 调用被缺失接口阻断。初始化夹爪使用显式每指标定，阻止原始负端点直接下发。
- ARX 正式下发前重新读取 actual qpos 与基座，拒绝观测后改变的起点。hold 只读取状态，不依赖相机。双臂同步下发及物理步回调仍由 ARXBackend 完成。
- 专家几何回放直接核验 ARX actual 状态，不再依赖策略输入侧的 Aloha 代理 IK；策略条件仍完整执行代理状态映射。

当前入口只接受 diagnostic 数据；创建真实 renderer/scene 还要求已有 `cpf-pace-g2` 分配。CPU 的 `--inspect-only` 在创建渲染器前结束。没有作业申请或重排功能。

## 夹爪与 reset 配置

`g2_prepare_config.py` 从固定资产派生端点，替代 Mock 的人为 0.04 m 开端点：

| 本体 | 配置原始闭合/张开值 | 派生可执行闭合/张开值 |
| --- | --- | --- |
| Aloha | −0.01 / 0.045 m | 0 / 0.045 m |
| ARX | −0.01 / 0.045 m | 0 / 0.044 m |

规则为逐指求配置端点与 URDF prismatic 限位的交集，保留原值、限位、规则、最终值和 URDF SHA-256。它是名义机械几何标定，**接触动力学尚未验收**。Aloha 原生基准的控制语义仍沿用 G1；上述源端点用于实际开度解释及代理几何。

配置与证据位于 `research/g2/diagnostic-configs-20261105/`，包括两本体 × 窗口 1/4、端点证据和一份明确的 reset fixture。500 动作/600 秒仅为诊断入口上限，不是已批准或冻结的正式实验预算。

`g2_initial.py` 从 actual articulation 与 actor 读取初态，并按独立的物体、基座、关节和夹爪状态核验 reset。fixture 使用 G1 作业 126905534 记录的物体/目标/桌面位姿，机器人关节显式设为 home；属于可复查的 reset 干预，不声称 seed 相同就复现了物理初态。RoboTwin Actor 没有 set_pose；实现已核对其包装 Entity API。

## 实网格碰撞规则差异及修复

最初的严格 FCL 路径将 Aloha home 末端目标映射到 ARX 后，16 组 IK 候选全部拒绝。逐项诊断表明，命中的是每个 ARX articulation 内的 `base_link ↔ link2`，并非双臂互碰。

固定 ARX 资产 `curobo_tmp.yml` 的 self_collision_ignore 已包含这一对。现在通过该文件的 SHA-256：

`07c7faf92f6c9c7e3b7c370cb4487494d236935910cf6d89333f89f0b56c0ddb`

导入原生忽略规则，仅用于**同一个已登记 articulation** 的内部连杆对。跨 articulation、其他物体以及未登记连杆对仍拒绝。原始命中、按规则忽略的命中和最终阻断分别保留，真实入口写入 collision-checks.jsonl；没有修改全局 AllowedCollisionMatrix，以免同名 ARX 连杆造成跨臂豁免。对象/连杆身份字段依据 [MPlib 0.2.1 API](https://motion-planning-lib.readthedocs.io/stable/reference/collision_detection/index.html)。

无忽略规则时的失败证据 `fixture-collision-details-r3.json` 保留于 `research/g2/scnet-stage3-20261105/`；修复后的证据另建目录，没有覆写失败结果。该规则只解决本体自碰撞定义差异，不豁免手指与任务物体的抓取接触。

## SCNet CPU 最终结果

源码包本地/远端 SHA-256 一致：

`409a9ab12acc4a3820cb677e6769d7bd499bd2194595b930e62b57c370bd5d3e`

远端开发目录：`tools/g2-runtime-stage3-final-20261105/`。这是诊断开发包，不是 G2 GPU 发布包。对应源码归档在 `results/g2/stage3-20261105/runtime-source.zip`。

CPU 夹具加载真实两本体碰撞网格，按固定任务定义构造地面、墙、桌面/四腿、物块及目标；没有渲染器，没有 physics step。它用于几何/reset 核验，未调用完整 PACE setup_demo。

| 项目 | 结果 |
| --- | --- |
| 独立读取两本体的物体、目标、桌面位姿 | 一致 |
| Aloha / ARX 各自初始碰撞状态 | 均无阻断碰撞 |
| Aloha → ARX IK | 双臂通过，位置最大约 0.399 mm，姿态最大约 0.0105° |
| ARX → Aloha 代理 IK | 双臂通过，位置最大约 0.050 mm，姿态最大约 0.0011° |
| Aloha 目标对应的 ARX 连接路径 | 148 tick，149 次碰撞查询 |
| ARX 目标对应的 Aloha 连接路径 | 174 tick，175 次碰撞查询 |
| 导入原生 self-ignore 后重叠双臂 | 仍检测到互碰 |
| 导入原生 self-ignore 后加入障碍物 | 仍检测到环境碰撞 |
| 实际容器 Aloha / ARX 入口只读预检 | 退出码均 0 |
| 独立 Aloha 代理碰撞世界 | 实际任务 actor 几何复制及 refresh 后障碍物检出通过，ARX actual qpos 未变 |

路径 tick 为 0.02 秒，属于离线规划时间网格，实际控制调用和物理步均为 0。这些结果没有产生任务成功率。

原始 JSON、轨迹 NPZ、CPU 日志及入口导入日志在 `research/g2/scnet-stage3-final-20261105/`。第一轮 add_ground 默认创建视觉材质的 CPU 失败也已留存；修复为 render=False 后复验通过。

## 本地集成与当前门槛

完整回归 **85 项通过**（G2 44 项、原有 41 项），见 `results/g2/cpu-stage3-final-20261105/tests.log`。覆盖原生完整 episode 的逐物理步记录、两本体/两窗口 session、setup 异常留证、专家代理 IK 独立性、初始化夹爪、reset 比对、共享 articulation 的原子碰撞查询，以及自碰撞规则不影响互碰/环境检查。13 个 Mock episode 和现有汇总器也通过。

当前仅有一份已核验的 CPU reset 候选，未分配正式 pair_id。正式任务初态仍需在实际任务 session 中回读验证；不能把这份 CPU 几何夹具当作已完成动态验收。

运行配置保持 `contact_mode=strict_no_contact_exemptions`。因此抓取接触、持物碰撞模型、交接约束及释放事件仍是下一道门槛；当前模式不能用于声称完整迁移任务已通过。之后还需冻结独立试次、正式 run_lock、显存/耗时预算和唯一 Slurm 入口，才能请求一次综合 GPU 执行授权。
