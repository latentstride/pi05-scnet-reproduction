# RoboTwin PACE 阶段 6–7 评测协议 v1

首个任务为 `handover_block`。本版固定研究对照、纳入规则和统计口径。正式运行前仍须采集并验收初态、验证 ARX 执行链、冻结实际配置及试次清单。本次交付的离线自测全部使用合成记录。

## 动作来源和研究对照

| 动作来源 | 条件 | 动作生成与执行 |
| --- | --- | --- |
| `source_policy` | `a_aloha_native` | Aloha-AgileX 原生 PACE。按原生观测接口闭环查询，执行绝对双臂 qpos 和夹爪目标。 |
| `target_policy` | `b_arx_adapted` | ARX-X5 使用同一个冻结的 PACE checkpoint 和归一化统计。ARX 实测观测经 Aloha 代理状态映射，策略输出经源 FK、任务系工具位姿传递、ARX IK、双臂联合检查和重定时后执行。 |
| `expert_replay` | 独立专家对照 | 固定示范轨迹，按 `expert_representation` 区分 `expert_qpos` 与 `expert_ee`。按实际下发接口填写 `execution`。跨本体 qpos 来源走源 FK，映射后才生成目标关节命令。 |

策略主对照使用同一任务提示、场景、成功判据、任务时限与策略块消费规则，比较原生系统和适配系统的端到端成功率。相机布局、可达域和控制链随本体变化，主对照的差值涵盖这些变化。权重候选沿用阶段 5 的 `pi05_robotwin_handover_block`、HF revision `17ef545270cf26ed81169893b60f845f80ce6557`、EMA step 30,000。正式清单须记录经过完整校验的权重清单和归一化文件 SHA-256。专家条件的 checkpoint 为 null，专家来源、原始示范 revision、episode/帧索引、参考位姿定义和重定时器记录在冻结配置及原始日志。 [阶段 5 运行计划](pace_runtime_plan.md) · [控制适配规范](robotwin_control_adapter_spec.md)

每次策略查询使用该试次的新鲜观测。阶段 5 前缀烟测、假后端、合成数据和 LIBERO 指标均独立归档。正式统计按 `action_source` 与条件分表，分别报告 Aloha 策略、ARX 策略和专家回放成功率。

## 初态采集与配对

1. 在 reset 和固定的稳定等待结束后、下发首个动作前，保存同一物理 tick 的快照。记录 seed、两侧实际 articulation 根位姿（Aloha 可共用）、两侧实际六关节、实际夹爪、任务系在世界系的位姿、所有任务相关物体在任务系的位姿、障碍和场景配置。以物体稳定 ID 对齐。位姿统一为 `[x,y,z,qw,qx,qy,qz]`，位置以 m 计，四元数为单位 WXYZ。快照、实际关节名、夹爪和姿态校准证据存入 `episode_artifact` 指向的试次目录。
2. 在计分前，用固定采样序列建立候选初态与 `pair_id`。对两本体的抓取、交接、放置候选关键位姿做双臂可达及碰撞预检，保存各自结果、边界、后端、资产、阈值和报告 ID。通过共同工作域后锁定入组名单。工作域之外的候选和原因单列；正式名单内运行时遇到的新不可达进入失败统计。
3. 每次 reset 都与冻结的目标初态复核。用共同任务坐标系比较物体，位置差 ≤1 mm、SO(3) 角距 ≤0.1°。同本体消融还要求同 reset 夹具、同场景状态，实测夹爪归一化开度偏差 ≤0.01，实际基座与任务坐标使用同一位姿容差，关节最大绝对偏差 ≤0.001 rad。跨本体使用各自锁定的基座与初始关节状态。seed 只作追溯，允许同一对使用不同 seed。
4. 仅在初态、环境配置和共同工作域检查通过后标记 `matched`。不符时停止并记 `initial_state_mismatch`；reset 或日志基础设施失败导致无法核验时，保留预定试次，记 `unverified` 和对应失败。汇总器复查配对双方的初态与公共场景；冻结目标和工作域报告由运行侧校验。

## 终止分类与统计分母

每个预定 episode 一条记录。首个导致停止的事件确定唯一 `outcome`，上游异常即使未推进控制也写一条终止记录。可恢复事件另外进入原始日志和计数。超时按触发环节归类，并保留原始错误码、阶段、步号和证据文件。

| outcome | 判据 |
| --- | --- |
| `success` | 固定 RoboTwin 任务判据在预定步数或仿真时限内成立。 |
| `task_unsuccessful` | 执行到预算终点仍未达到任务判据。 |
| `initial_state_mismatch` | 初态复核未通过。 |
| `environment_error`, `interrupted` | 资产、reset、渲染或仿真基础设施故障；作业中断单列。 |
| `ik_unreachable`, `ik_failed` | 有几何/后端明确证据才记不可达；有限搜索无解或 IK 超时归入 `ik_failed`。 |
| `planning_failed`, `joint_limit` | 碰撞、无有效同步路径或规划超时；位置、速度、加速度等越限独立归类。 |
| `adapter_error`, `control_error` | 观测/动作适配、标定或同步故障；控制器/执行后端错误独立归类。 |
| `policy_failed` | 策略调用失败、超时或返回无效输出。 |

- 条件成功率使用 `success / 冻结试次总数`，环境错误、初态不符和中断仍计入分母，并列各类终止次数。正式汇总要求每个锁定试次都有记录。原始试次 ID 不复用，后续复跑使用新的冻结清单。
- 配对差值为两边均核验初态的完整配对上 `success_B − success_A` 的平均值。条件按 ID 字典序确定 A/B，主对照即 ARX 减 Aloha。报告配对覆盖、缺失条件、未纳入数和各方单边成功数。初态已核验后的环境故障按失败计入该差值。

## 指标采集

| 指标 | 口径 |
| --- | --- |
| 轨迹位置和姿态 | 记录同任务系、同工具点 U、同控制 tick、左右身份一致的已执行目标与实测 EE 对。位置用欧氏距离，逐 episode 求 RMSE 和线性插值 P95；姿态用四元数最短旋转角，求均值和 P95（度）。只纳入有对齐目标和实测值的样本，记录样本数。原生 qpos 目标经冻结的 FK 转成 U。汇总列是各 episode 指标的等权均值，注明参考定义 ID 和覆盖数。 |
| 规划、越限 | 分别累计失败的规划请求 / 规划请求、越限的同步候选检查 / 同步候选检查。一次请求指一次可独立判定结果的规划器调用；候选检查任一侧任一维越限计一次。原生直控无规划时规划分母为零。 |
| 夹爪 | 采集两侧实测归一化开度（0 闭、1 开），按冻结标定和阈值与期望状态逐侧逐 tick 比较，记录不符数、样本数及最终左右开度。保存接触/持物证据到原始日志。 |
| 推理和控制耗时 | 单调墙钟计每次尝试（含失败）。推理记录客户端发请求到响应或失败的端到端时间；控制记录后端提交到返回的时间。IK/规划耗时另存原始日志。汇总按调用次数加权；冻结是否将首轮编译计入计时，主比较使用同一热身规则。 |
| 执行窗口 | 记录配置窗口上限（动作行）、实际窗口数、选择的动作行数、完成的动作行数及仿真秒数。停止时记录未执行后缀，分别保留控制调用和物理步数。PACE 输出 50 行是模型时域，窗口是每次查询最多消费的行数。 |

未采集用 null，实际观测到零则写零。分母为零的汇总比值留空。保留原始逐 tick 记录和 metric 覆盖数，以便重算并审查提前终止造成的采样差异。

## 单变量消融和运行冻结

- **优先执行窗口消融。** 在同一已验收的执行链上比较每次消费 1 行和 4 行，保持 checkpoint、映射、初态、任务预算和模型时域相同。运行时先验证两档窗口均可安全执行。
- **条件式专家表示消融。** 只有同一示范的 qpos 路径经 FK 后与 EE 路径可在同一任务系和工具点对齐，才比较 `expert_qpos` / `expert_ee`。其余初态、夹爪事件、后端、重定时规则固定。历史帧周期未知时按几何路径重定时，保存原始帧索引。
- **条件式映射消融。** 仅比较一个经过验收的输入映射选项或一个输出映射选项，其余链路固定。现阶段没有已验证可替换路径，暂不入组。跨本体关节索引直连和混合微调均排除。

正式运行前，保存并评审一份 `run_lock.json`，其中 `schema_version` 为 1，`episodes` 为每个预定试次的 `episode_id, task, seed, comparison_id, pair_id, condition, environment` 原样投影。还需在锁中固定并引用内容哈希对应的初态/任务提示与成功判据、配置、日志规则、采样顺序、样本数、控制/任务时限、重试上限、容差、热身以及分析脚本版本。每个正式记录引用锁文件原始字节的 SHA-256。汇总器核对文件哈希、清单投影和完整性。实际样本数与初态名单待仿真采样后冻结。正式运行门槛包括图形验收、Aloha 真实 checkpoint 完整任务循环，以及 ARX 观测/控制、限位、夹爪与失败处置通过运行验证。

## 离线接口

`evaluation_schema.json` 是单条 episode 的 JSON Schema Draft 2020-12。JSON 输入为 episode 数组。CSV 输入顶层字段一列，**每格均为 JSON 字面量**，例如字符串字段包含 JSON 双引号，缺失值写 `null`，对象字段写 JSON 对象后由标准 CSV 转义。多文件输入的 episode ID 在同一运行锁下唯一。各输出按 `--data-kind` 隔离并标记数据来源。

Python 3.10+，需要 `jsonschema>=4.18,<5`。运行示例中的输入均是调用方提供的记录文件。

```bash
python scripts/summarize_experiments.py mock.json mock.csv --data-kind synthetic --output /tmp/pace-mock
python scripts/summarize_experiments.py formal.json --data-kind formal --run-lock run_lock.json --output /tmp/pace-formal
python -m unittest discover -s tests -p test_summarize_experiments.py -v
```

输出目录须为新目录。包含 `condition_summary.csv`、`paired_summary.csv`、`pair_index.csv`、`failure_index.csv` 及保留全部字段的 `episodes.json`。失败索引用 episode ID 关联完整初态、环境版本、checkpoint 和执行策略。
