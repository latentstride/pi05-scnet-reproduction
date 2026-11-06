# G2 第二轮 CPU 验收 · 2026-11-05

本轮完成远端源码与实网格碰撞路径核验、ARX 控制桥接实现、阶段调度及 schema 导出。G2 仍处于 A–D 开发阶段，没有申请或提交 GPU。

## 已通过的实际 CPU 路径

通过用户已登录的 Chrome E-Shell 连接 SCNet login08；外部 SSH 本轮仍超时。仅在项目 `$HOME/code_cpf/pi05` 内新增 `tools/g2-cpu-stage-20261105/` 和 `results/g2/` 下的 CPU 脚本、日志与结果。

固定 PACE 为 `fd2e624cc19325cc9b20a620c82d51c0d8e25cc1`，机器人、规划器、基础任务、handover_block 和任务配置五份实际文件逐一与 HEAD blob 比较，全部一致。回读源码及哈希在 `research/g2/runtime-source/`；安装版 MPlib 源码与 ARX 配置在 `research/g2/installed-source/`。没有改动固定上游文件、LIBERO 基线、共享环境或其他作业。

实际容器使用 SAPIEN 3.0.0b1、MPlib 0.2.1。CPU 场景加载两份 ARX 的完整碰撞几何，仅删除 visual/material，保留原 URDF 的全部 9 个 collision 元素，并对每个 collision mesh 记录 SHA-256：

| 检查 | 结果 |
| --- | --- |
| 原 URDF SHA-256 | 与既有资产审计一致 |
| 名义中立双臂状态 | 无碰撞 |
| 大障碍物覆盖机器人 | 检测到环境碰撞 |
| 双臂根基座重叠 | 检测到互碰 |
| 联合关节目标 +0.002 rad | 生成 6 tick 同步轨迹 |
| 路径碰撞检查 | 8 次调用，正常完成 |
| 速度与加速度 | 满足本次 CPU 夹具施加的 0.5 rad/s、1 rad/s² |
| 检查前后实际 qpos | 未变 |
| GPU / 渲染器 / 物理步 | 均未使用 |

这些速度、加速度是本次合成 CPU 测试的命令边界，不是 ARX 额定参数或正式实验参数。场景也不是完整 handover_block；未包含抓取、持物、交接及释放的接触约束。

原始 CPU 结果、日志与固定源码核验在 `research/g2/scnet-cpu-20261105/`。远端结果为 `results/g2/collision-path-cpu-20261105/`。第一轮只做碰撞的原始结果另存 `research/g2/collision-cpu-20261105.json`。

## 新增代码与接口验收

- `g2_collision.py`：从实际 SAPIEN 场景构造同时包含两臂的 FCL 世界；按每个实例的关节名改动规划模型，检查后恢复模型状态；不修改仿真实际 qpos。每个动作前缀前需 refresh 物体与基座。该实际调用路径已在上述 CPU 场景运行。
- `g2_robotwin_backend.py`：ARX 配置构造、作用域内的 CuRobo warmup 绕过、双臂同 tick 下发、原生每指关节目标、实际状态采集、TCP 跟踪误差、hold/close。六臂关节和两指按名字查找；同一物理步前提交左右全部目标。夹爪直接接收经过校验的每指坐标，避开原生 set_gripper 的二次映射；同时维护原任务成功判据读取的归一化命令字段。越限/非法轨迹在任何下发前拒绝。此桥接仅经本地接口替身测试，尚未执行真实 ARX 动态闭环。
- `g2_capture.py`：实际 articulation 标识加入 per_scene_id，以区分同名 ARX 实例；状态采集可独立于三路图像运行，避免在每个物理子步强制渲染。
- `g2_orchestrator.py`：动态验收、Aloha、ARX、消融、专家和统计六阶段，共享单调时钟总预算。模型仅启动一次，模型启动失败不重启；ARX/消融/专家依赖动态验收，Aloha 与统计可独立保留。真实子进程测试覆盖硬超时、共享总时限、SIGTERM 摘要、进程组清理和依赖跳过。当前仅接受 synthetic/diagnostic 计划，拒绝 formal；没有作业申请逻辑。
- `g2_export.py`：把实际 episode 证据转为既有 evaluation_schema.json 全字段，校验 episode、条件、动作窗口与数据身份。未知夹爪/限位/时间指标保留 null；超时保留 timeout code，并按其阶段归入现有 outcome。当前拒绝 formal 导出，避免诊断记录冒充正式实验。
- `g2_mock_acceptance.py`：持久化两本体窗口 1/4、专家回放及失败路径共 13 个合成 episode；输出 schema 记录后直接调用现有 summarize_experiments.py，生成条件汇总、两个配对差值表与失败索引。A/B 名称固定源本体在前、ARX 在后，差值为 ARX − Aloha。这些配对和成功均为合成接口证据。

## 测试与证据

完整本地回归 **73 项通过**，其中 G2 32 项、原有 41 项；见 `results/g2/cpu-next-20261105/tests.log`。本轮持久化的 13 个 episode、评测 JSON、汇总 CSV 和 SHA-256 在 `results/g2/next-20261105/episodes/`。Mock 成功数仅验证执行分支，不构成真实任务成功率。

执行器的控制调用数来自后端显式回报；未知或未返回的调用计数保持未知。动作 completed 数仅统计完整返回的前缀，部分执行须结合逐步日志复查。硬中断后的未终结 result 仍需外层恢复索引，导出器不会把 running 状态自动转换为成功或失败。

## 进入发布准备前仍需完成

1. 将场景 reset、ARXBackend、适配器及模型客户端接成真实完整 episode 入口；原生 Aloha 的完整执行入口也需接入这套日志。桥接实现和 CPU 碰撞通过不能代替该集成。
2. 固定接触工具、夹爪端点及命令边界的正式实验配置；完整 handover_block 的自碰/环境/持物/交接接触许可与事件停留仍须设计和验证。现有 FCL checker 没有偷偷豁免手指与目标物体碰撞。
3. 导出并核验两个本体的真实初态快照、独立物体位置和共同工作域。已回读专家缓存前 20 条候选记录；尚未分配正式 pair_id，也未冻结正式试次。
4. 将阶段调度接入最终只读发布、GPU 显存监测、阶段产物验收和硬中断恢复；补齐正式 schema/run_lock 门槛。当前调度器的通过只表示子进程退出情况，不自行认定任务成功。
5. 根据真实控制成本、模型和 ARX 并发显存核算样本数与唯一作业时限，再生成 run_lock.json 与唯一 pace_g2.slurm。随后才进入单独 GPU 授权步骤。

当前没有 G2-E 作业授权，也没有可提交的 G2 Slurm 脚本。
