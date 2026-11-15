# ARX-X5 首次左手抓取的 CPU PhysX 对照

基于 `acd8568830188576f912c2ae83c28ea3013c951d` 的直接控制器、Aloha Fixed H50 三条真实成功记录和 ARX G1–G3 的 GPU 物理与策略原件。对照从 G1 作业 `128047752` 第 2 查同一时刻的实测快照开始（tick 5767，首次左单指接触），三组共用同一个 50×14 策略块。动作文件 SHA-256 为 `75a202211a64ef701077cc1109fa9a024615d52bd7c0affcf0e2840b7c37e5e0`。快照重建了关节、指状态、物体位姿与资产，历史证据缺少动态速度和 PhysX 接触缓存。CPU 对照适合检验抓取局部行为，无法逐物理步复刻在线运行。

## 偏心来源

两套左手在 link6 上的 TCP 固定变换相同，平移为 TCP 基准轴前方 120 mm，旋转为绕 X 轴 π。Aloha 原日志的 TCP 姿态再乘一次绕 X 轴 π 才与适配器 TCP 同约定。按真实 URDF 和 STL 复算，两套指尖碰撞网格的 SHA-256 分别相同。装配和行程仍有差异。

| 碰撞网格接触面指标（TCP 局部坐标） | Aloha-AgileX | ARX-X5 |
| --- | ---: | ---: |
| 两侧内表面中心均值 X / Y / Z（mm） | 28.070 / 0.0015 / 1.300 | 30.070 / 0.0020 / 1.440 |
| 张开内表面 Y（mm） | −44.999、45.002 | −44.402、44.406 |
| 闭合内表面 Y（mm） | 0.001417、0.001583 | −0.401583、0.405583 |
| 内表面 X 区间（mm） | 20.570–35.570 | 22.570–37.570 |
| 两指各自的行程（mm） | 45 | 44 |

| 真实记录 | 首次有效左指接触 tick / 开度或指令 | 首次双指 tick / 左爪指令 | 双指发生时或归档闭爪目标的物块横向位置（夹持中心坐标，mm） |
| --- | --- | --- | ---: |
| Aloha 127589584 | 4791 / 0.9995 | 5823 / 0.593 | −6.71 |
| Aloha 127589587 | 4501 / 0.9991 | 5360 / 0.552 | +5.64 |
| Aloha 127589590 | 5264 / 0.7077 | 5306 / 0.563 | +3.04 |
| ARX G1 | 5767 / 0.9958 | 无 | +26.19 |
| ARX G2 | 5784 / 0.9959 | 无 | +26.07 |
| ARX G3 | 4855 / 0.9963 | 无 | +45.17 |

Aloha 的夹爪列取当步策略指令。ARX 的首次接触列取实测夹爪开度，最后一列取第 2 查起点物块位姿相对于策略闭爪目标的夹持中心。G2、G3 在碰撞门禁处停止，未执行这些闭爪目标。第一触碰之前 100 ms 的 TCP 位移（各自物块的局部坐标）和接触面的完整变换均见 [geometry.json](geometry.json)，双指接触由每指非零冲量核实。场景对照使用物块相对坐标，Aloha 和 ARX 初态与观测不同，数据支持定位偏差所处的环节，无法只凭这些试次判断模型为何预测该目标。

两个 TCP 的横向装配差仅 0.0005 mm。沿前向的 2 mm 装配差会影响边缘触碰，但无法解释第 2 查原始策略目标已有的 26–45 mm 横偏。对已经获解的在线行，重算 FK/IK 的最大位置残差为 G1 0.291、G2 0.259、G3 0.290 mm。G1 开爪接近时物块已横移最多 49.68 mm，第 26 行实测 TCP 到原目标仍只差 0.40 mm。主要证据指向归档策略目标偏心及接近与闭爪不同步。G4 是单独的右臂碰撞问题。

## 实验代码

- A 使用 `DirectAdapter.policy_row` 和现有真碰撞安全分段。B 从相同原始关节动作计算任务坐标 TCP，用两本体碰撞网格的内表面定义夹持中心，用 `oracle_object_pose` 对正，保留每行原始夹爪命令和姿态。C 使用同一几何模块，先沿当前接近轴后退，再横向对准和接近；在实际开度、物体重叠范围和双侧净空满足条件后，使用本块的首个有效闭爪请求。只有实际左右指同属左手持物后才尝试抬升，升幅取自归档块的左 TCP 垂直意图。
- 使用现有关节限位、速度限制、真实网格碰撞、2 mm 指块门禁、夹爪限力和实测接触逻辑。模块只由 CPU 入口显式启用。原策略数组保存在 [g1-q2.json](g1-q2.json)。
- A/B/C 都是归档动作反事实。修正改变物块或手臂状态后，后续编排仍来自这一个归档块。任何延伸到在线任务的结论都要求在新的真实 RGB 和状态上重新推理。

## CPU PhysX 实测

采样周期 4 ms，以至少两指都有原接触追踪器要求的真实冲量为双指接触。闭爪前位移统计至左指实测归一化开度首次 ≤0.95。稳持时间要求原追踪器保持左手持物且双指持续接触，抬升取该状态下物块相对本组起点的最高增量。

| 对照 | 物理步 | 闭爪前最大横移（mm） | 左手双指接触（步） | 最长连续持物（s） | 持物抬升（mm） | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A 原映射 | 3,530 | 38.0272 | 0 | 0 | 0 | FAIL |
| B 夹持几何对准 | 2,812 | 0.0030 | 963 | 3.844 | 1.057 | FAIL（抬升） |
| C 几何对准和接触感知闭爪 | 1,247 | 0.0034 | 483 | 1.924 | 76.992 | PASS |

A 在第 14 行的候选指块穿透处停止，实测左爪开度始终 ≥0.9958，累计 1,185 个单指接触步。GPU 原件同样在近全开时推块，闭爪前原始记录最高 49.68 mm。B 使用原行序，保留了闭爪请求对应的时序，实际形成左手持物，但原第 35 行请求再开爪且右手没有接管，触发原有的供体释放门禁。C 在先退后对准时保留原始接近姿态，闭爪前的残余横偏为 0.580 mm、最小单侧净空 12.061 mm。它用第 26 行的 0.4962 左爪请求完成闭合，以实测双指持物决定进入抬升，并从归档策略块选取 79.693 mm 的左 TCP 垂直意图。C 结束时仍由左手双指持物。三组的指块 2 mm 门禁、ARX 实体自碰撞和夹爪限力全程启用。

用同一 CPU 引擎重跑 ARX 原生 episode 1，推进 2,023 个物理步，有 1,448 步双指接触，抬升 40.295 mm。该正对照用了已归档的原生动作，按既有局部回放停止规则在抬升后结束。场景初始物块 XY 约为 (−166.2, 171.3) mm，G1 约为 (−74.4, 6.8) mm，适合确认环境能再现有效抓取。正对照未与 A/B/C 合并计算。逐步记录、失败细节与哈希见 [evidence/](evidence/) 和 [results.json](results.json)，实际加载的控制代码与发布包成员哈希见 [run_manifest.json](run_manifest.json)。

## 从 CPU 抓取到在线验证

用 G1 第 2 查真实头部 RGB、已知相机标定和桌面高度拟合桌上直立方块，得到平面位置误差 **0.319 mm**、沿当前抓取横轴误差 **0.0024 mm**，最大拟合边缘误差 0.243 px。估算只使用归档 RGB 和已知几何，PhysX 方块真值仅用于事后比较。该结果覆盖一帧可见且未裁切的方块，遮挡、离桌或物块倾倒的姿态尚未验证。[vision.json](vision.json)

下一次完整在线测试的最小改动是在第一次左手抓取范围提供经验证的相机位姿输入，接入相同对准、实测接触和限力闭爪逻辑。若视野或几何假设失效，需重观测或安全停止。适配改变实测状态后，立即丢弃归档尾段，用新 RGB 和机器人状态重新请求冻结模型。C 通过了约定的局部抓取验收。新状态下的策略输出、在线抬升、交接及放置均无本轮在线实测。GPU 作业数为 0。

## 复现

从仓库根目录运行。数据解析和汇总使用 Python 3.11+，CPU 实验复用已装 SAPIEN 3.0.0b1、MPLib 和 NumPy 的 PhysX 环境。`ASSETS` 指向固定 RoboTwin 资产根，`ASSET_AUDIT` 来自实际 G1 发布包，`ARX_ARCHIVE` 与 `ALOHA_ARCHIVE` 分别指向 [ARX 原始归档](../arx-online-batch-20261115/results/README.md) 与 [Aloha 原始归档](../../../docs/g2_aloha_six_results_20261111.md) 的解包目录。视觉复核还需要 SciPy、PyYAML 和 [固定相机内参](https://raw.githubusercontent.com/niejunnan25/PACE/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/task_config/_camera_config.yml)，路径设为 `CAMERA_INTRINSICS`。

```bash
D=research/g2/arx-grasp-adaptation
N=research/g2/arx-native-grasp-control
P=research/g2/arx-stage6-concentrated-20261114
export PYTHONPATH="scripts:research/g2/final-system-review${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES= ROBOTWIN_GPU=0
OUT=$(mktemp -d)
mkdir "$OUT/evidence"
python "$D/prepare.py" "$ARX_ARCHIVE" "$OUT/g1-q2.json"
python "$D/geometry_audit.py" "$OUT/g1-q2.json" "$ASSET_AUDIT" "$ASSETS" \
  "$ALOHA_ARCHIVE" "$ARX_ARCHIVE" "$OUT/geometry.json"
python "$P/production_cpu.py" "$OUT/g1-q2.json" "$ASSET_AUDIT" "$ASSETS" \
  "$OUT/evidence/A" --first-row 1 --mode archived_tail
for mode in B C; do
  python "$D/run.py" "$OUT/g1-q2.json" "$ASSET_AUDIT" "$ASSETS" \
    "$OUT/evidence/$mode" --mode "$mode" --timeout 160
done
python "$N/replay.py" "$N/inputs/episode01.json" "$ASSET_AUDIT" "$ASSETS" \
  "$OUT/evidence/expert" --mode native
gzip -n "$OUT/evidence/A/events.jsonl" "$OUT/evidence/expert/events.jsonl"
python "$D/summarize.py" "$OUT/evidence" "$OUT/g1-q2.json" "$OUT/results.json"
RAW="$ARX_ARCHIVE/pace-g2-standalone-arx-128047752/trials/unified-seed1-arx-full-w4/episode"
python "$D/vision_check.py" "$OUT/g1-q2.json" "$RAW/observation-00002.npz" \
  "$ASSETS/embodiments/ARX-X5/config.yml" "$CAMERA_INTRINSICS" "$OUT/vision.json"
```
