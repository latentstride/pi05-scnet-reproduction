# 离线 ARX 分析与课堂图表

本目录记录历史证据和生成方法。新试次使用同一个分析脚本，在独立目录输出。原发布包、配置、控制器和结果原件按历史哈希保留。

| 图表 | 数据口径 |
|---|---|
| [三场景 H50/PACE 比较](figures/aloha-three-scenes.pdf) | 全部六条原生实跑，每场景每方法 1 次。点标示完成动作、真实模型查询、物理步与任务结果，G6 800 动作到限。 |
| [G3/G6 交接窗口](figures/aloha-g3-g6-handover.pdf) | 横轴对齐各自首次双手四指同时接触。显示前 2 秒至后 13 秒内的每个原始物理步。块高相对复位高度，速度为有符号垂直速度，开度为两指实体的实测距离。0/1/2 指接触根据有接触点的物理记录计数。 |
| [两本体专家轨迹统计](figures/expert-trajectory-statistics.pdf) | 每点 1 条轨迹，全部 100 条。黑线为每本体的轨迹中位数。重复状态分子为同一 episode 内精确相等的 14 维状态重复行，分母为该 episode 帧数。 |
| [双向 FK/IK 可解率和残差](figures/expert-cross-embodiment-ik.pdf) | 可解率分母为所有单臂目标。位置与姿态曲线为有解子集的经验累积分布，每个合格目标等权。未解目标保留在可解率分母。 |

G3/G6 的原始 18,700/42,341 条物理记录包含 tick 0。图仅显示指定交接窗口，窗口内不抽点。窗口之外的首触、完整轨迹终止和任务结果仍在原汇总。三对首请求哈希与采样键相同，首块输出均不相同，差异不足以支持对 PACE 窗口的因果归因。右侧双指仍接触、左侧首次失去双指接触时，G3 方块高于起始 157.85 mm，垂直速度 +0.0010 m/s。G6 高于起始 125.00 mm，垂直速度 −0.6259 m/s，之后跌至起始高度以下 40.22 mm，仅有桌面接触。专家数据无显式时间戳，帧数不用于推断运行时长。IK 几何筛查使用名义夹具，未计入碰撞、接触或时序。Aloha → ARX 为 28,268/28,268（100%），ARX → Aloha 为 19,273/19,360（99.55%）。图表均为描述性统计，无假设检验或置信区间。[原生六项回执](../../../docs/g2_aloha_six_results_20261111.md) · [专家全量口径](../../robotwin100_course/REPORT.md)

## ARX 日志验证

- [历史策略 127510363](arx-127510363.json) 提取 138 个执行动作、36 次请求、16,160 步，活动均在接近阶段。第 35 次请求的源代理前向门禁截断前缀，第 36 次请求已发出但规划未获准。终止例证是源代理 `fl_link4 / fl_link6`，4 个连续性通过的 IK 候选碰撞，另 12 个候选先被连续性排除。目标碰撞审计 22,978 次调用无阻断，源代理 189 次调用中 15 次有阻断。作业正常收尾，任务失败原因为规划拒绝。
- [历史专家 127147943 seed 1](arx-127147943-expert-seed1.json) 依据实测保持阶段确认抓取、共持、右手接管与释放，最终使用原生任务成功回执确认放置。查询数 0，保留专家来源标签。
- 已归档的投影 CPU 接口事件用于测试新字段关联。两段均识别出执行 4 行、弃用 46 行并关联后续同 episode、时钟前进的新请求。该接口证据来自 CPU 与模型服务替身。测试还覆盖硬截止留下的半行日志以及普通坏行拒绝。

分析器分别记录已执行动作、已请求前缀、每次查询保存的模型动作、响应校验、计划与物理步。投影后请求记录源时钟关联、双臂与物体字段、保存的 RGB 及其格式校验。只有明确的执行回执才填写逐查询完成动作数。接触、稳定抓取、交接和放置分别采用实测接触追踪器、共持后右手持有和原生成功回执。候选碰撞审计、实测碰撞诊断和终止回执保持不同来源。文件缺失保留为空，硬截止时保留最后一个结果快照和日志计数的差异。

## 复现

从仓库根目录运行。使用 Python 3.12、NumPy 2.5.4 和 Matplotlib 3.10.8。图表路径参数为新目录，分析输出为新文件。

```sh
python -B scripts/analyze_g2_arx_result.py "$ARX_RAW_RUN" /tmp/arx-summary.json
# 历史策略对照可另指定审计日志目录和已发布的文件哈希。
python -B scripts/analyze_g2_arx_result.py "$ARX_127510363" /tmp/arx-127510363.json \
  --collision-dir "$ARX_COLLISION_DIRECTORY" \
  --expected-hashes research/g2/job-127510363/result.json
PYTHONPATH=scripts:tests python -B -m unittest -v test_analyze_g2_arx_result

python -B research/g2/course-analysis/plot_course_results.py /tmp/course-figures \
  --font '/System/Library/Fonts/STHeiti Medium.ttc'
# 加 --raw-root "$RAW_ROOT" 从 G3/G6 原 physics.jsonl 重建展示窗口并校验完整文件哈希。
```

若试次还未创建 `trials/`，可通过 `--episode unified-seed1-arx-full-w4` 提取独立作业的启动和终止信息。`--collision-dir` 为可选覆盖，默认读取试次 episode 内的两份碰撞审计。输入日志、每个已找到的动作与观测文件及回执的哈希写入分析 JSON。原始日志在对应 SCNet 结果目录，原始专家日志在 `job-127147943-full-evidence.tar.gz` 的 `trials/unified-seed1-arx-expert-repaired`。

`g3-g6-window.csv.gz` 存储交接图的逐步实测列，`figure-data.json` 记录输入 SHA-256、窗口锚点与专家统计，图同时提供 PDF 和 300 dpi PNG。用于课堂投屏的图幅采用横向版式。复核包括已知回执计数、窗口连续性、FK/IK 残差阈值、中文字形、版面和 PDF/PNG 渲染。
