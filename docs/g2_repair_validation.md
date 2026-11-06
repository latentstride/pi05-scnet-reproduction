离线修复版本为 offline-repair-20261106-r1。旧源码保存于 research/g2/offline-repair-20261106-r1/before，原始r2运行发布及127054052结果保留。当前开发源码中的修复按以下范围实施。

| 文件 | 修复 | 对应验证 |
|---|---|---|
| g2_locked_task.py | 诊断配置导出diagnostic-record，正式资格需显式声明 | 实际误标记录回归，原配置False保持诊断 |
| g2_locked_score.py | 按声明种类读取记录，继续核对完整锁定名单和缺失项 | 8条实际记录与缺失拒绝测试 |
| g2_single_statistics.py | 核对监督器评分资格，输出任务成功计数；旧formal标签须显式离线重分类 | 原始配置/记录矛盾复现，诊断0/8汇总 |
| g2_orchestrator.py、g2_single_launch.py | 增加execution_complete和独立evaluation摘要接线 | 原监督器回归与全量CPU预检 |
| g2_adapter.py | 明确source/target世界，保留失败前工具目标、状态、种子、全部候选及连续性拒绝 | 四个真实事件修复前后静态复现 |
| g2_contact.py | 指扫描拒绝记录实际深度、阈值、起始与请求开度 | 两个真实指—物块事件重建 |
| g2_capture.py、g2_episode.py | 在策略边界记录真实任务对象位姿，缺失对象保持未知 | 任务坐标变换、无对象后端与原有rollout测试 |

新对象留证只影响未来运行，历史Aloha缺失的物块轨迹仍为未知。新增候选信息也没有假称原始日志已经保存全部细节。此次重建结果与原记录分开保存。

实际回归结果为158项全部通过，包含原有149项和9项修复回归。13个合成episode全部通过。最终完整输出目录为 results/g2/offline-repair-127054052-final2。新测试文件 tests/test_g2_failure_repair.py 明确依赖本地真实选取证据与静态复现JSON，数据缺失时会报错，不会使用Mock替代真实事件复验。

CPU场景验证包括四个真实ARX事件的冻结源码版本、诊断修复版本、扩展种子版本与重新计算原始策略FK版本。四次拒绝分类和碰撞拒绝对数保持16/16/4/16。source事件另有12对连续性拒绝。原始动作重新经Aloha FK得到的目标与三个planning事件原请求逐值相同，最大矩阵差为0。12个专家接近段目标与整条关节路径通过，原速度、加速度、关节限位、接触和碰撞规则全部沿用。

750份Aloha原生计划全部逐值复现。静态回放均为0物理步。原始127054052的2,712个文件已在分析前后两次核对SHA-256，差异为0。模型启动回执中的28项文件摘要与固定manifest一致，八份配置与锁中摘要一致。ContactLimits类的AST与修复前一致，QuinticMotion、基础碰撞分类和限力控制源码未改。G1相关运行逻辑与LIBERO冻结结果记录的摘要复核通过，详见 baseline-integrity.json。

实际旧记录已通过 --legacy-formal-records 参数在新目录生成诊断汇总。新汇总为diagnostic、formal_score_eligible=false、8条终态记录齐全、成功数0，结果分布为4次task_unsuccessful、3次planning_failed、1次adapter_error。原始formal-record.json没有被覆盖。对照时除data_kind外逐字段相同。

语法检查覆盖所有新增/修改Python。新增分析代码、统计模块和测试通过Ruff检查，修改的既有模块通过Ruff F类静态检查。新增代码已格式化。修复后的代码快照、补丁和证据文件清单由SHA256SUMS及修复清单绑定。该清单包含实际读取的选取证据、配置、静态结果和测试摘要，原始大数据的逐文件摘要另见原始inventory.json。

复现本地完整验收需固定OpenPI客户端源码及完整研究数据。

```bash
G2_PYTHON=/path/to/cpu/python \
OPENPI_CLIENT_SRC=/path/to/Openpi-PACE/packages/openpi-client/src \
bash scripts/g2_cpu_preflight.sh results/g2/new-offline-check
```

只读分析入口包括g2_failure_inventory.py、g2_analyze_saved_actions.py、g2_replay_native_prefixes.py、g2_static_failure_replay.py和g2_native_target_audit_cpu.py，均支持 --help。静态FCL入口需要已经具备权限的SAPIEN/MPLib CPU环境及固定资产。输入指向原始或验哈希后的证据，输出必须使用新目录。SCNet未执行重型动力学重放，几何检查采用单线程与硬超时。

离线转换和分析入口另有输入目录保护。旧记录转换拒绝把输出写入原运行目录，静态FCL入口拒绝把输出写入release、assets或选取证据目录。该保护已有实际旧记录依赖路径回归。
