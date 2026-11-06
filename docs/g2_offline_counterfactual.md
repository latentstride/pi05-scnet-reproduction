CPU 反事实使用保存的真实策略输出和已验证官方 Aloha 专家的关节目标。本轮没有产生新的 π₀.₅ 输出。

Aloha 的动作变换复核在冻结 native_prefix 上运行，原窗口1/4的750份已保存计划逐值相同。原有左右臂顺序、绝对qpos、归一化夹爪、有界截取均保留。固定动作块内的未来闭爪保留情况如下。

| 原试次 | 原窗口中的闭爪块 | 假设16行且满足原截取规则 | 假设32行且满足原截取规则 |
|---|---:|---:|---:|
| seed0-A-native-w1 | 0/300 | 35/299 | 270/299 |
| seed0-A-native-w4 | 22/75 | 28/75 | 67/75 |
| seed1-A-native-w1 | 0/300 | 168/297 | 272/297 |
| seed1-A-native-w4 | 0/75 | 11/73 | 66/73 |

16/32行只是假设读取保存数组后的截取统计。部分动作块的后缀超过原夹爪0.01截取包络，已按原规则排除。现有执行器只支持窗口1/4，这些假设没有生成新运行配置或安全可执行计划，更没有改变原试次。扩大窗口会改变后续观测和模型响应，闭环成功仍为 GPU_DEFERRED。

ARX 已完成三个独立 CPU 版本。r1 使用冻结控制源码复现四个失败端点；r2 使用仅增加诊断信息的修复版本，失败结局一致，并加入专家整段路径检查；r3 增加有界额外种子诊断。原始目标仍4/4拒绝、专家接近段目标与路径仍12/12通过。结果分别保存于 offline-repair-20261106-r1/static-replay.json、static-replay-127054052-r2/replay.json 和 static-replay-127054052-r3/replay.json。

专家对照按 episode0/1 的官方 Aloha trace 提取0、25、100 tick关节目标，经原Aloha FK与基座/工具变换形成任务空间目标，再调用ARX IK。没有把 ARX 专家关节值传给 Aloha 适配器。原始完整 trace 的哈希与采样索引在 expert-controls/provenance.json 中。当前反事实只评价各 captured approach state 下的端点和关节插值，不评价持物动力学、全任务成功或模型新的闭环行为。

QuinticMotion 检查所有采样关节和指几何，保持原速度、加速度、采样步长、接触阈值和碰撞策略。12条对照路径包含2–904个控制点，仍未推进物理。四个策略失败在端点 IK 筛选已发生，因此没有通过更换绕障搜索器回避端点问题。

Aloha 额外自碰撞复核采用原全身 URDF，但只构建机器人几何，不加入任务对象，检查初态、首次50mm误差处和末态的实际/驱动关节。初态四次全clear。seed0-w1末驱动目标、seed1-w1末实测/目标、seed1-w4末实测/目标均有重叠，seed0-w4末态均clear。该结果与真实轨迹偏离并存，但缺少完整物理接触证据，不能将所有跟踪偏差归为同一实现错误。

复现入口为 scripts/g2_static_failure_replay.py、scripts/g2_native_target_audit_cpu.py、scripts/g2_analyze_saved_actions.py。前两者需要已有 SAPIEN3.0.0b1/MPLib0.2.1 CPU环境和固定资产。分析脚本明确读取原数据目录，输出至新目录。SCNet运行设置单线程、总超时240/90秒，无scene.step、无GPU或模型启动。原始日志、视频和控制输出均保留。

r4进一步读取三个planning事件对应的原始action文件，重新计算源Aloha FK，得到的工具矩阵与原请求最大差为0，之后的目标IK仍得到相同拒绝。source observation事件没有新的模型动作，直接从实际ARX观测重建工具位姿。最终入口g2_replay_native_prefixes.py也独立完成750份计划复现，结果与初版审计一致。
