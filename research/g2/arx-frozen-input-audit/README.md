# 冻结 π₀.₅ 在 ARX-X5 上的输入与抓取审计

两项 GPU 作业共 17 次归档请求的可复核输入链为 **INPUT_ADAPTER_VERIFIED**。两块原始策略动作在对应回放中为 **POLICY_ACTION_FAILURE**。训练样本的真实分布、历史服务端瞬时张量及当前控制链在新状态上的策略响应仍为 **INSUFFICIENT_EVIDENCE**。本轮没有确认输入或动作接口缺陷，代码改动限于离线审计与证据核验。生产控制、模型、任务判据和物理限制沿用原版。GPU 新作业与新模型推理均为 0。

## 实际请求

| 审计项 | 实际值与核验 |
|---|---|
| RGB 布局与相机身份 | 两项作业 51 个真实数组均为 uint8 CHW，3×240×320。通过 PACE 参考编码与客户端回灌逐值核验。将实测方块和机器人关节代入固定相机安装的投影，全部可见框残差 ≤1.385 px。Q2 head/left/right 红色像素数为 3,331 / 36,290 / 0。左腕框 [127,47,319,239] 与投影差 0.151 px。右腕画面内无块，几何也预测出画。 |
| 模型图像字段 | 保存的 cam_high、cam_left_wrist、cam_right_wrist 分别进入 base_0_rgb、left_wrist_0_rgb、right_wrist_0_rgb，三路 mask 为 true。固定服务源码对实际保存数组做 CHW→HWC、双线性等比 resize 到 224×168、上下各补黑 28 px，结果为 224×224×3。Observation.from_dict 将 uint8 映射至 [−1,1]。归档无原始运行时模型图像张量。 |
| 状态数值 | 左 6 关节 rad、左夹爪开度，接右 6 关节 rad、右夹爪开度。两项首查均为 [0,0,0,0,0,0,1,0,0,0,0,0,0,1]。Q2 左 7 维为 [−0.04201,2.09739,0.96895,1.08291,1.27402,0.04496,0.99583]，左两指实测为 43.633/44.000 mm，右开度为 0.99802。17 个原请求的状态逐值复算一致。由 ARX 实测关节到虚拟 Aloha TCP 的最大位置/角度差为 0.1635 mm / 0.02218°。基座快照、关节名字、左右世界变换及完整原数值见 JSON。 |
| 服务、提示与归一化 | 两项提示均为 “Pick up the red block with the left arm, hand it to the right arm, and place it on the target.”。客户端发送 images、state、prompt，CPU 用真实输入完成 MsgPack 往返。任务 checkpoint 固定为 17ef545270cf26ed81169893b60f845f80ce6557，norm 文件为 assets/robotwin/norm_stats.json，两个启动回执各匹配 28 项文件。原配置关闭通用 Aloha 重编码，执行一次分位数归一化。Q2 左夹爪的分位数归一化值为 0.992054。Q1/Q2 固定 tokenizer 在 CPU 重建 72/79 个非填充 token，均低于 200。首查两维夹爪分位值 1.000398 对应上界 bin 255，未发生越界索引。实际服务端解包字节与张量未保存。 |
| 输出和执行窗口 | 原数组均为 50×14。训练的 delta 掩码只覆盖 12 个臂关节，服务反归一化后仅一次恢复绝对关节，夹爪维保留绝对开度。CPU 数值往返最大误差 6.7e−16。127 个历史批准动作与保存原行的 TCP 及夹爪请求吻合。127999421 的 Q1 消费 36 行、丢弃 14 行，Q2 消费 1 行、丢弃 49 行。每行的实测物理步数独立于 50 行预测时域。 |
| 历史实跑版本 | 两项发布为 b7202f8，实际配置文件 SHA 与发布清单一致，运行时 canonical SHA 与 setup 回执一致。两个 policy RNG 为 20261107/20261108，场景 seed 均为 1。检查过的历史适配、捕获、模型服务和归一化代码与当前相同。d8bfbc6 已改变 g2_arx_direct 的接触重规划、g2_remaining_episode 的计划日志和 g2_robotwin_backend 的物理诊断。 |

逐项源码路径、原始值、断言和缺失项见 [input_adapter_audit.json](input_adapter_audit.json)。[固定参考训练协议](../../../docs/pi05_robotwin_protocol_audit.md)保留了训练数据确切字节缺失的限制。渲染实测图像与真实训练图像的统计等价性仍未确定。

## 逐行抓取几何及物理核对

[policy_grasp_geometry.csv](policy_grasp_geometry.csv) 按查询与 1-based 行号记录 100 个原始 14 维动作、两臂 TCP/相对方块位姿、两指请求与指节链接原点、真实 Aloha/ARX 网格在实测起始开度及动作请求开度的重叠深度，以及匹配到的 GPU 实测和既有 CPU 反事实。位姿采用 xyz 米及 wxyz，深度和横移为 mm。网格对照固定各自查询时的方块，IK 结果是独立端点诊断。CSV 的空物理字段表示该行没有对应的记录。

| 观察 | 第 1 次查询 | 第 2 次查询 |
|---|---|---|
| 原查询起始方块上的首次左指端点重叠 | Aloha、ARX 均为第 36 行，起始开度深度 0.025/1.983 mm。第 37 行 ARX 9.896 mm。 | ARX 第 7 行 0.901 mm，Aloha 第 8 行 2.516 mm。第 8 行 ARX 4.516 mm。 |
| GPU 实测接触 | 第 36 行 tick 5,783，左单指，原行开度 0.99657，方块平面移动约 0.015 mm。 | 此查询始于已有左单指接触的快照。第 1 行后 tick 5,790 又记录到单指接触，随后重观测。 |
| 归档左夹爪时序 | 第 49 行才低于 0.95，最小 0.83638，整块没有 ≤0.5。 | 第 21 行低于 0.95，第 27 行首次 ≤0.5，第 35 行已近闭合。 |
| 当前控制器的 CPU 证据 | 复用从第 36 行前实测快照开始的 53 步回放，同 tick 复现左 link8 单指接触并停在重观测边界。 | 新增从真实 Q2 起点复建的第 1 行接口回放，推进 1 步至 tick 5,784，产生左单指接触和有效新请求。另外复用从独立 CPU 前缀后继续的旧尾段，第 8/9/10 行先后横推超过 1/5/10 mm，第 27 行结束实测开度为 0.492，方块已横移 139.4 mm。 |

Q1 第 36 行目标 TCP 横向偏差 24.16 mm，实测 23.97 mm，历史批准 IK 残差 0.091 mm，实际 TCP 对插值指令的跟踪误差 1.66 mm。Q2 第 20 行的沿工具 X 距离已接近零，但 ARX 横向仍偏约 26 mm。第 27 行闭合时偏差仍为约 26.1 mm。当前回放均未稳定持物或抬升 50 mm。旧尾段实际推进 6,330 步，有 1,099 步单指接触，双指接触 0。

已成功的 ARX 原生 episode 1、7 仅用作抓取阶段和控制能否保持双指接触的正对照，本轮未重跑。GPU seed 1 的方块初始平面坐标约 (−74.4, 6.8) mm，原生 episode 1 为 (−166.2, 171.3) mm。首次接触姿态也有差异。抓取几何只按各自物块及 TCP 比较。原生左右接触、持物和 34.67–40.29 mm 抬升已由既有 [CPU 对照](../arx-native-grasp-control/)验证。

CPU 证据支持该归档失败的归因，尚缺当前控制代码在真实新观测下的冻结策略输出，无法据此宣布可重复的稳定抓取或整任务成功。若开展新的诊断性在线评测，依赖可复核的新状态、实际发送三路 RGB 和状态、模型返回的原始 50 行，以及逐步双指接触、持物和 ≥50 mm 抬升证据。候选原因逐项见 [根因矩阵](root_cause_matrix.md)。

## 复现

在仓库根执行。现有证据校验仅需 Python/NumPy。完整输入审计用既有 CPU 依赖 NumPy、Pillow、PyYAML、openpi_client msgpack 和可选 SentencePiece。固定上游源文件使用 JSON 中列出的 URL 与哈希。下载时按 JSON 键名保存。可选 tokenizer 文件通过既有 checkpoint 资产获取并核对 SHA-256。两个 GPU 完整归档来自历史试次目录，公开证据包含所有 17 个请求和动作数组，以及携带原始行号的事件节选。

```bash
D=research/g2/arx-frozen-input-audit
PREVIOUS=research/g2/arx-stage6-concentrated-20261114
export PYTHONPATH="$PWD/scripts:$PWD/research/g2/final-system-review"
export CUDA_VISIBLE_DEVICES=""
python "$D/verify.py"
python research/g2/arx-native-grasp-control/verify.py
python "$PREVIOUS/verify_physics.py"
python "$D/input_audit.py" "$GPU_ARCHIVE" "$ASSETS" "$ASSET_AUDIT" "$PINNED_SOURCES" \
  "$CPU_OUTPUT/input_adapter_audit.json" --model-files "$CHECKPOINT_ASSETS"
python "$D/grasp_geometry.py" "$GPU_ARCHIVE" "$D/evidence" "$ASSETS" "$ASSET_AUDIT" "$PREVIOUS" \
  "$CPU_OUTPUT/policy_grasp_geometry.csv"
```

逐行端点复跑沿用当前 g2_arx_endpoint_cpu.py，在 CPU PhysX/FCL 环境分别对 evidence/inputs/q1.json 与 q2.json 传入原 ASSET_AUDIT、ASSETS、新输出目录及 --rows 1 … 50。Q2 首行接口复跑用以下命令。速度与 PhysX 求解器接触缓存未归档，场景恢复以实测位置和姿态为起点。接口回放使用 CPU 替代 RGB。

```bash
python "$PREVIOUS/production_cpu.py" "$D/evidence/inputs/q2.json" "$ASSET_AUDIT" "$ASSETS" \
  "$CPU_OUTPUT/q2-physical" --first-row 1 --mode interface
```

[证据目录](evidence/)包含 SHA-256、矩阵端点和新增 Q2 物理回执。首行回放在第一次 CPU 接触边界停下。真实 Q2 请求由上一块的单指接触触发，恢复的 CPU 场景只在第一次步进后重建接触流形，CPU 的 1 步与历史 GPU 的 7 步属于不同重启条件。旧尾段的较大物块位移来自另一个明确标注的反事实起点。

定向回归共 30 项通过，既有两组 CPU 对照证据校验和本目录自动验证均通过。
