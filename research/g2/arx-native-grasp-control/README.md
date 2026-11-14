# ARX 原生抓取与冻结策略动作的 CPU 对照

基线为 `d8bfbc6` 的直接控制器和 `bb59269` 的原生专家归档。复用原生 ARX episode 1、7 的真实物体初态、规划缓存与物理轨迹；单独使用作业 `127999421` 的原始动作及首次接触前快照。GPU 场景的物块初始位置约为 (−74.4, 6.8) mm，ARX episode 1 为 (−166.2, 171.3) mm。两者虽然都标为 seed 1，物理初态不同。

## 原生正对照

从每条记录的专家闭爪起点恢复实测双臂关节、指位置和物块位姿。按原始存档采样点重建关节和归一化夹爪请求，交给生产 ARX 限速、限力、预测碰撞与实测接触检查。两种 CPU 路径使用同一输入：ARX 关节直接输入，以及 ARX TCP → Aloha IK 表示 → 正常策略适配 → ARX IK。各自运行至持物且抬升超过 20 mm。四次均保持双指抓取并抬升。原生缓存通过归一化请求接口重定时，原生演示的 4 ms 动作时序与夹爪驱动力不要求逐值复制。两条输入分别 29、30 行；纯 FK/IK 往返均有解，最大目标差 0.382 mm / 0.0034°。实际运行结果和物理事件见 [evidence](evidence/)。

| ARX 示范 | 原生关节直入抬升 | Aloha 往返抬升 | 双指接触物理步（直入 / 往返） | 首次接触相对步（直入 / 往返） |
|---|---:|---:|---:|---:|
| episode 1 | 40.29 mm | 40.29 mm | 1,448 / 1,388 | 564 / 565 |
| episode 7 | 34.65 mm | 34.67 mm | 1,448 / 1,413 | 565 / 570 |

四次均在相同的夹爪动作和抬升阶段形成持物。往返的执行终点最大位置差分别为 0.278 mm 和 0.257 mm。提前达到抬升截止条件后停止，物理步差也包含重定时差异。

## 首次单指接触

- 作业 `127999421` 第 1 次查询第 36 行，原始日志第 5,858 行，tick **5,783**。左爪请求 **0.996572**，实测两指 **43.633 / 44.000 mm**，已接近全开。首个 50 行块中左爪没有归一化请求 ≤0.5。左臂 IK 残差 **0.091 mm**，实测 TCP 对插值指令误差 **1.664 mm**，对原动作行终点误差 **2.417 mm**。作业 `127999407` 在第 36 行规划拒绝前无实测指块接触。
- 将方块中心表示在同一定义的 TCP 坐标中，两条原生轨迹首次接触的横向分量为 **0.424 / 0.990 mm**，策略原目标为 **24.159 mm**，实际为 **23.974 mm**。专家接触处的闭爪请求分别为 **0.503 / 0.518**。策略和专家的 TCP 姿态也不同，该横向差仅用于比较各自抓取几何，不能直接用作坐标补偿。见 [坐标对照](geometry.json)。
- 冻结第 36 行起点的物块，评估同名义 TCP 下真实指尖网格。第 35 行两个本体都无重叠。第 36 行对应左手第 2 指的端点深度为 Aloha **0.015 mm**、ARX **2.012 mm**。这是固定端点预测。原始动作第 36 行从 GPU tick 5,730 的实测快照开始用 CPU 物理执行，推进 **53 步**，在同一 tick **5,783** 复现左 **link8** 单指冲量约 **0.000253 N·s**。该步物块横移约 **0.014 mm**，随后停在重新观测边界。见 [网格对照](evidence/policy-mesh/receipt.json) 和 [物理回执](evidence/policy-at-first-contact/receipt.json)。
- 可测的大幅推移出现在已有第 2 次查询离线反事实。第 8 行首次超过 **1 mm**，第 9 行超过 **5 mm**，第 10 行超过 **10 mm**，阈值首次越过时均为左单指接触；对应的模型夹爪请求接近 1。第 27 行首次请求 ≤0.5，结束时实际开度降到 **0.492**，方块已横移 **139.4 mm**。该尾段总计 1,099 个单指接触步、0 个双指接触步。该结果使用从 CPU 回放前 7 行得到的快照和旧的整块动作，适用范围是归档反事实。[位移汇总](q2-displacement.json)

证据支持该归档策略在还未对准、夹爪仍开着时进入接触，ARX 指尖几何放大了端点干涉。已执行路径的 IK 和跟踪误差量级不足以解释约 24 mm 的夹爪横向偏差。两条正对照证明这一版本的映射和控制可把可执行的抓取目标保持到稳定持物。现有证据未证明通用 TCP 或夹爪接口缺陷，改动限于对照工具、物理留证和报告。原始归档动作未形成稳定抓持。完整在线闭环仍缺少能在真实接触反馈下完成对准和有效闭爪、形成稳定抓持的策略动作。

## 复现

在仓库根目录设置 `ASSETS`（实际 RoboTwin 资产）、`CPU_OUTPUT`（新目录）及 CPU Python 环境。复用现有 SAPIEN PhysX CPU 和 MPLib，关闭 CUDA 可见设备。快照复建只恢复归档的实测位置和物块位姿，未归档的速度与求解器接触缓存不作恢复。RGB 是 CPU 固定输入。

```bash
R=research/g2/arx-native-grasp-control
AUDIT=research/robotwin100_course/sources/embodiment_config_audit.json
export PYTHONPATH="$PWD/scripts:$PWD/research/g2/final-system-review"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$CPU_OUTPUT"
for ep in episode01 episode07; do
  for mode in native roundtrip; do
    python "$R/replay.py" "$R/inputs/$ep.json" "$AUDIT" "$ASSETS" \
      "$CPU_OUTPUT/verify-$ep-$mode" --mode "$mode"
  done
done
python "$R/replay.py" "$R/inputs/first-push.json" "$AUDIT" "$ASSETS" \
  "$CPU_OUTPUT/policy-at-first-contact" --mode policy
python scripts/g2_arx_endpoint_cpu.py "$R/inputs/first-push.json" "$AUDIT" "$ASSETS" \
  "$CPU_OUTPUT/policy-mesh" --rows 35 36
python "$R/verify.py"
python "$R/analyze_tail.py" research/g2/arx-stage6-concentrated-20261114 "$CPU_OUTPUT/q2-displacement.json"
```

`prepare.py PREPARED GPU_ARCHIVE OUTPUT` 可从原生准备后的缓存和完整 GPU 归档重建公开输入。回放日志校验使用解压后的 SHA-256。生产控制脚本与 CPU 实际执行文件核对见 [控制源码 SHA-256](evidence/executed-control-sha256.txt)。原有安全阈值、FK/IK、夹具参数及模型未改。GPU 提交数为 0。

定向回归共 44 项通过。已有五组 CPU 物理日志、两组网格端点、两组重观测和实体碰撞停止负例也通过原验证脚本。
