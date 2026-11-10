# CPU PhysX 反事实 · 作者 H50 首块

## 实验设计

场景为作者缓存分片 0 的首项 `100000`，相同 Aloha URDF、碰撞网格、原生 `setup_demo` 和 2,500 个稳定步。两组在独立进程复位，使用作业 `127510335` 的原始 50×14 动作，SHA256 `e1ac8bd5119a1cb24e9761d301aa971ec6e11706eae715be468c5cc4babe14e8`，提示为 `Grab the red block using the left arm.`。同一 Headless CPU 适配只替换渲染与未参与 qpos 的 CuRobo 初始化。对照变量是项目保护包装器。没有重新预测，第二次推理调用前停止读取档案。模拟器中的逐步记录器只在 PhysX 返回后采集状态，未改变任何驱动或判据。[回放脚本](../research/g2/final-system-review/counterfactual.py)

| 结果 | `author_native` | `protected` |
| --- | --- | --- |
| 首块 | 完成 **50/50**，5,369 个物理步 | 完成 **42/50**，第 43 行中途在 tick 4746 候选拒绝 |
| 停止原因 | 上游在 50 动作后到达下一查询边界，缺少第二个保存的模型块 | 原 2 mm 保护生效，`SAFETY_STOP / controlled_candidate_stop` |
| 首个实测带冲量的指—块接触 | tick **4760**，第 43 行，左侧 `fl_link7 / box` | 无 |
| 指—块接触最小实测 separation | **−0.03056 mm** | 正分离，保护前无实际指—块负分离 |
| 指—块单点最大冲量 | **0.0007021 N·s** | 0 |
| 方块最大位移 | **27.62 mm**，首块结束仍偏移 **24.33 mm** | **0.00198 mm** |
| 抓持与任务 | 无任一侧双指承载，原成功谓词为假 | 无抓持，原成功谓词为假 |

两组的 tick 0 复位记录和一直到 tick 4746 的逐步状态一致。实测关节位置/速度、方块位姿、线速度和角速度的逐元素最大差均为 **0**。保护侧的拒绝诊断与原 GPU 失败记录逐值一致，hold、场景关闭和上下文退出通过。两条 CPU 进程都正常退出。

原生首块内，178 个 tick 有可量测的指—块冲量，122 个 tick 有负 separation。方块最大线速度 **0.03449 m/s**、最大角速度 **0.3898 rad/s**、相对初态最大转角 **13.21°**。记录保存接触极值的法线和位置、tick 4746/4747 的实测相对位姿、动作 42–50 终点的方块状态和夹爪开度。[CPU 对照摘要](../research/g2/final-system-review/cpu-counterfactual.json)

## 如何解释 2.052 mm

2.052 mm 是门禁在 tick 4746 用下一驱动目标做几何前瞻得到的穿透。两种 CPU 路径当时实测左指与方块仍有约 1.913 mm 间隙，冲量为零。原生在下一子步仍有正间隙，随后出现真实、单指、带冲量的接触。PhysX 在首块结束前没有表现出方块弹飞或毫米级实测指—块穿透，物块仍受桌面支撑，但被开爪推离起点约 2–3 cm。模型首块始终没有明显闭爪请求，实际夹爪也未形成双指承载。

**门禁确实提前阻止了一段 PhysX 能继续执行的接触。** 此处实测响应可解释为受约束的单指推块，现有证据不足以认定它是有用的抓取接触，也不足以将原门禁判成几何误报。候选重叠、动力学求解后的实际接触和不重观测继续消费旧动作是三种不同证据。原生跑完 50 行只证明这份保存动作在本次 CPU 条件下可执行至下一观测，完整交接结果为 **UNKNOWN**。[既有 FCL / TOPP 逐值复算](g2_author_h50_root_cause_127510335.md)

## 复核

使用已固定的 CPU SAPIEN 3.0.0b1、MPLib 0.2.1、PACE 及候选包，`$chunk` 指向归档原始 `chunk-00.npy`。同一解释器中不要混用两种模式。

```sh
audit=research/g2/final-system-review
for mode in author_native protected; do
  CUDA_VISIBLE_DEVICES= PYTHONPATH="$release/scripts:scripts:$OPENPI_CLIENT_SRC" \
    python3 -B "$audit/counterfactual.py" "$mode" "$pinned_pace" "$release" \
    "$chunk" "$new_output/$mode"
done
python3 -B "$audit/summarize_counterfactual.py" "$new_output"
```

真实物理日志包含每个 tick 的方块姿态、速度、手指相对姿态、实测开度、接触点与冲量。仓库提供可复核摘要、代表点和原文件 SHA256。原始 SCNet 证据保留，未上传私有路径或原始视频。
