# 受保护的作者 Aloha Fixed H50 诊断入口

**准入结论为 `NOT_READY`。** 本轮以 PR #5 的 `b3ac1ea` 为基线。独立入口和现场凭证不会改动原 G2 或十项统一配置。现行资源封套尚不能证明允许完成作者 800 动作上限，因此固定预算内的截断必须单列。

## 执行与安全

`g2_author_native_h50.author_native_h50()` 使用 PACE `fd2e624`、缓存分片 0 首场景 `100000` 和原生 `setup_demo`，核对资产、关节、物体和基座读回。不会加载项目锁定夹具。使用作者的 unseen 生成器，固定本次 Python 种子并记录实际 prompt。原生编码保留 14 维状态的 dtype。沿用冻结 checkpoint 启动凭证、模型清单与 norm stats。独立服务协议可与原 G2 客户端共用同一个模型服务。它在本场景开始时恢复公开默认 JAX key(0)，保存实际 pre/post key、由固定 split 得到的采样 key、查询序号和请求哈希。公开论文批次当时的随机流仍未知。

作者 `pace.rollout.run_episode` 负责取块、逐行调用和重新查询。入口记录原始 50×14 响应和行哈希，以原值调用 `take_action(qpos)`。原生 TOPP、夹爪插值和驱动限幅保持原样。额外安全层只允许或拒绝，保留 0.01 的夹爪超限拒绝边界和原臂关节限位。入口不做额外剪裁、预载、端点修正、重查或恢复。动作、观测时序或原生完成条件偏离时记 `PROTOCOL_DEVIATION`。原生谓词成功为 `SUCCESS`，无安全/资源干预且耗尽 800 动作为 `TASK_INCOMPLETE`，物理步或墙钟用尽为 `RESOURCE_CENSORED`。

复位末尾的原生稳定检查执行真实物理步。随后直接读 `sapien.Scene.get_contacts()`，将测量来源、episode、tick、实际关节位置/速度、物体与基座状态和接触哈希绑定到诊断 tick 0。成功读到的空列表和读失败有不同语义。读取与 hold 校验不推进物理，候选拒绝前后的接触哈希及实测状态须一致。原有实际几何、关节、抓持和跟踪检查仍生效。证明还要求 hold 的位置、手指及零驱动速度读回，close、场景上下文退出，以及由宿主验证的进程组回收与服务存活。下一试次须独立复位和准入。危险或凭证缺失触发整场停止。

## CPU 验收

| 验收用例 | 实测结果 |
| --- | --- |
| 原生复位 | 场景 100000 通过锁定状态和质量比较。作者完成 2,500 个稳定检查步后，诊断 tick 0 读取 33 个真实 PhysX 接触流形。实际 prompt 为 `Grab the red block using the left arm.` |
| H50 与中途拒绝 | 查询起点为动作 0 和 50，对应 tick 0 和 1,123。第 51 行完成后在 tick 1,147 注入拒绝。两臂共 104 次 TOPP 调用均返回原生轨迹，其中含被拒绝动作的两次规划。已完成动作每次 9–27 子步。原始左爪请求保留 1.001，原生驱动上限约 0.045 m。候选停止凭证完整。 |
| 首步拒绝 | 已完成动作和物理步均为 0。停止凭证的状态及接触哈希等于初始真实采样，hold、close 和进程组回收均通过。 |
| 缺少当前采样凭证 | 实测场景中注入丢失接触时钟，hold 和 close 仍完成。没有候选停止凭证，继续门禁拒绝。 |
| 危险分类 | 完成 1 个真实物理步后注入实际危险标记，原生动作尚未完成。结果为 `actual_safety_stop`，未签发候选停止凭证，继续门禁拒绝。 |
| 回归 | 281 项 Python 测试及 13 项合成 episode 通过。全量回归使用本地已有的冻结专家回放数据。WebSocket 测试覆盖同一端口的作者 RNG 回执串联和旧 G2 客户端兼容，使用键及策略替身。 |

各故障用例使用独立进程及新的场景。收据验证使用实际进程组回收结果，服务存活条件由 CPU 替身提供。原生现场 GPU 服务的退出顺序尚待核对。 [CPU 实测回执](../research/g2/author-native-h50/cpu-physics-e2e.json) · [运行日志](../research/g2/author-native-h50/cpu-physics-e2e.log) · [回归摘要](../research/g2/author-native-h50/regression-summary.json)

实测 CPU 场景使用无渲染 PhysX、碰撞 URDF、相机占位像素和固定模型响应。它复用了原生稳定流程、MPLib/TOPP、夹爪控制、PhysX 接触及 FCL 门禁。故障注入验证分类与停止凭证，注入的危险标签不代表仿真中实际发生了碰撞。模型和 GPU 渲染未运行。

## 800 动作资源核算

| 项目 | 已有真实证据与边界 |
| --- | --- |
| 每个动作的物理步 | `127147943` 六条原生策略记录有 243 个完整动作，单动作 19–178 步，各试次均值 82.53–132.88。部分末动作排除。`127192740` 已完成 27 动作，40–178 步、均值 132.41，包含项目接近修正开销。历史记录缺少逐臂 TOPP 回退标记，分支比例为 `UNKNOWN`。 |
| 原生 TOPP | 每臂异常或空输出回退到 50 子步，仅夹爪继续插值。两臂均回退恰好 50 步，若所有动作如此则为 40,000 步。原生成功可中断单个动作，正常 TOPP 轨迹长度可超过 50。没有作者 800 动作的已测上界。 |
| 物理步与墙钟 | 25,000 步允许 800 动作平均 31.25 步。按历史各试次均值做敏感性估计为约 66,026–106,308 步，控制时间约 807–1,459 秒。计入各原生试次的历史墙钟余量（复位、渲染及未单列开销）和 16 次暖查询后，条件估计为 1,105–1,781 秒，另计共享启动与整场清理。现有软限 1,150 秒、硬限 1,200 秒。上述历史外推只说明截断风险，不能当作场景 100000 的耗时范围或上界。 |
| 共享服务与分配 | 两次历史日志的权重校验、导入、设备初始化和恢复合计约 297–330 秒，首次编译及推理约 71–73 秒。历史六项动态准入合计约 1,357 秒，包含相应的查询等待。计整场时间时应将首次推理只算一次。原封套 7 小时，试次硬限总额 16,800 秒，图形/启动/统计/清理额度 4,200 秒，另有 4,200 秒预留。资源建议只涉及未来评审，未改现行额度。 |

Vulkan/EGL、真实三路 RGB、模型加载、实际 JAX RNG 收据与 GPU 进程退出仍属现场验证项。当前预算是否容纳 800 动作仍为 `UNKNOWN`。原 10 项、既有结果、检查点和安全阈值保持原样。

**唯一建议**。下一轮人工评审先确定这条诊断的终点与预算。沿用现行封套时，只验收安全允许下的窗口消费与任务进展，截止记为 `RESOURCE_CENSORED`。若验收要求覆盖完整 800 动作，则须在同一次分配内为该试次单独核定物理步、墙钟和清理余量，再判断是否能替换既有槽位。保持原安全阈值。

## 复现

在已有固定 PACE 资产、Linux CPU SAPIEN 3.0.0b1、MPLib 0.2.1 和冻结 OpenPI client 的环境中，依次使用新输出目录运行四个独立进程。入口 callable 与现场服务扩展见[执行源码](../scripts/g2_author_native_h50.py)和[服务源码](../scripts/g2_author_native_h50_service.py)。CPU 使用旧候选的只读资产与安全模块，并覆盖候选停止及清理模块为当前版本，实际输入差异核验见[版本及哈希](../research/g2/author-native-h50/runtime-provenance.json)。

```bash
export PYTHONPATH="scripts:tests:$OPENPI_CLIENT_SRC"
export CUDA_VISIBLE_DEVICES=
for test_case in mid tick-zero missing actual; do
  python scripts/g2_author_native_h50_cpu.py \
    --pace "$pinned_pace_with_assets" --release "$frozen_release" \
    --reference research/g2/author-h50-preflight/cpu-reset-readback.json \
    --output "$new_output_root/$test_case" --case "$test_case"
done
python -m unittest -v test_g2_author_native_h50
python scripts/g2_author_native_h50_budget.py \
  --job-127147943 "$saved_job_127147943" --job-127192740 "$saved_job_127192740" \
  --published-result research/g2/candidate-stop-127192740/result-127192740.json \
  --output "$new_budget_json"
```
