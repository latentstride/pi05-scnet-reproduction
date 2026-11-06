# G2 专家物体稳定初态 CPU 复现

日期：2026-11-06。三条官方示范的生成位姿已在 Aloha、ARX 两个独立 CPU 场景中完成物理稳定，共六个场景、每个生成物块后2500步。已导出实际物体位姿、机器人基座、实际双臂关节和夹爪标量。三组物体配对一致，新增四项验收/拒绝测试通过；完整116项回归通过。

## 已取得证据

最终目录 `research/g2/expert-settle-source-mass-20261106/`：

- `settling.json`：六个场景的完整实际快照、物体速度、质量、接触点/分离距离/冲量、ARX物理过滤规则。
- `N-robot/snapshot.json`、`settle_tail.npy`：逐场景快照和最后200步物块实际pose，后者不是插值或理论平面高度。
- `candidates.json`：三个独立seed的候选，物体配对检查通过，逐条保留正式实验排除原因，formal_pair_id=null。
- `episodeN_projection.png`、`image_consistency.json`：以稳定后的实际物体pose投影到官方首帧。
- `executed-script.py`、`config.json`、`input.json`：实际运行输入；脚本与本地 `scripts/g2_settle_expert_cpu.py` 的SHA一致。
- `evidence.zip`：远端压缩包回读SHA `d7bfb7bb8c0cecc137bfccf6a9ea3ef155ee048d581cb47b8e0666dc40203c0c`，已本地复算一致。后续新增候选/投影/测试文件单独记录在checksums.json。

|episode / seed|稳定后实际物块 xyz (m)|跨本体物体差|末200步位置变化|物块投影bbox最大边界差|
|---|---|---|---|---|
|0 / 0|(-.173311859,.197931752,.839999974)|0|0|3.989px|
|1 / 1|(-.074376896,.006847504,.839999974)|0|0|1.385px|
|2 / 2|(-.196634576,.155283988,.840000153)|0|0|1.665px|

两场景的三种物体（物块、目标区、桌子）pose逐值相同；末尾线速度和角速度为0，动态物块质量0.72000003kg，末帧接触只连接物块与桌面，无机器人支撑物块。目标区投影边界差2.03–2.40px。投影只能说明图像一致性，不代表三维精度达到该像素换算值。

## 构造与适用范围

CPU SAPIEN 3，物理周期4ms，无renderer、模型、CUDA或GPU申请。使用固定审计的机器人URDF与真实碰撞网格；去除视觉元素以支持无渲染CPU。任务物理构造按固定官方 `create_actor.py`：桌面10cm厚、四腿10cm截面、桌面z=.74；物块半尺寸(.03,.03,.1)、默认密度；静态目标半尺寸(.05,.05,.005)；摩擦(.5,.5)、恢复0。机器人links按base task设为1kg。源码及哈希在 `research/g2/expert-source-20261105/manifest.json`。

先前一轮使用URDF原质量，证据保留在 `expert-settle-20261106/`；统一link质量重跑后，三个物体pose均未改变。最终采用source-mass目录。首次容器overlay启动失败日志留在远端，改用现有sandbox的 `--writable` 后成功；未提交Slurm作业。

这批是**有实际物理测量的重建快照**。归档原始物体pose仍未发现；数据生成commit未知。机器人使用已声明基座、显式打开夹爪/零关节初始化、1000/200位置驱动、300步预稳定；没有逐条复刻原任务的夹爪插值与其被动力补偿，所以实际机器人状态不能标成原始专家初态。ARX沿用已审计同实例原生self-ignore的PhysX约定；原始Aloha未额外放宽过滤。

实测机器人关节会偏离零驱动目标，例如Aloha左臂第三关节约−.045854rad、ARX约−.045073rad。这些实际值已记录，没有拿目标0替代。两类机器人初始状态仍有自接触记录；物体配对通过不等于机器人几何检查或完整任务可达性通过。

## 后续门槛

目前三个候选的正式排除原因明确保留：完整任务共同可达性未检查、机器人初始自接触需审查、原专家实际机器人初态未恢复、当前采用显式初始化。正式实验需选择并冻结可复现reset约定，再完成抓取—交接—放置轨迹几何检查、实际相机观测、专家回放和预算/发布门槛。本轮没有放行正式schema、run_lock或GPU入口。

## 复验

`scripts/g2_settle_expert_cpu.py` 在现有SCNet CPU容器中运行，参数为审计资产根目录、asset-audit、config、input、fresh output。外围240秒超时，成功退出0。所有远端写入均在用户 `$HOME/code_cpf/pi05` 内。

```sh
PYTHONPATH=scripts <CPU_TEST_ENV>/bin/python scripts/g2_settled_candidates.py --input research/g2/expert-settle-source-mass-20261106/settling.json --output /private/tmp/g2-settled-candidates.json
<CPU_TEST_ENV>/bin/python scripts/g2_check_expert_projection.py --settled research/g2/expert-settle-source-mass-20261106/settling.json
PYTHONPATH=scripts <CPU_TEST_ENV>/bin/python -m unittest discover -s tests -p test_g2_settled_candidates.py -v
```

新增拒绝验收覆盖：两本体物块位置相差2mm、末帧仍有速度、机器人接触支撑物块。均被拒绝，不因真实三组样本通过而放宽阈值。

## 后续复核：初态网格碰撞已通过

2026-11-06 对同六个实际初态完成FCL检查全部clear，解除原先“自接触需审查”的排除项。PhysX正分离接触候选与网格穿透已经区分。完整854帧IK通过，但两本体的末段驱动目标均有手指—托盘静态重叠，仍待真实专家物理回放核验，详见 [全路径筛查](pace_g2_workspace_stage6.md)。最新候选在 `research/g2/expert-collision-compare-20261106/candidates.json`。

## 质量参数更正

后续查明原`Actor`构造函数会把动态物块质量覆盖为.01kg。本报告历史场景保留了primitive密度默认.72kg，故其“source-mass”指robot links的1kg对齐，不能据此称物块质量也与原任务一致。历史证据不改写；新的CPU构建器默认已改为.01kg，三条10g原生专家物理回放成功并与归档EE核对通过。详见 [原生专家回放验收](pace_g2_native_expert_replay.md)。
