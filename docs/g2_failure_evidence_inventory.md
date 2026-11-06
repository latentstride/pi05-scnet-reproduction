作业 127054052 的证据清单已在 2026-11-06 只读核验。原目录为 <PROJECT_ROOT>/results/pace-g2-127054052，本轮输出均位于另建的 results/g2/ 与本地 research/g2/ 目录。

共 2,712 个原始文件、419,842,365 字节。逐文件 SHA-256、大小和 mtime_ns 保存于 research/g2/failure-audit-127054052-r1/inventory.json。没有以 GitHub 汇总代替原始事件。

| 试次 | 动作块 | RGB观测 | 原生计划 | ARX计划 | 必备文件缺失 |
|---|---:|---:|---:|---:|---|
| seed0-A-native-w1 | 300 | 300 | 300 | 0 | 无 |
| seed0-A-native-w4 | 75 | 75 | 75 | 0 | 无 |
| seed0-B-arx-w1 | 26 | 26 | 0 | 25 | 无 |
| seed0-B-arx-w4 | 7 | 7 | 0 | 6 | 无 |
| seed1-A-native-w1 | 300 | 300 | 300 | 0 | 无 |
| seed1-A-native-w4 | 75 | 75 | 75 | 0 | 无 |
| seed1-B-arx-w1 | 56 | 57 | 0 | 56 | 无 |
| seed1-B-arx-w4 | 1 | 1 | 0 | 0 | 无 |

八个试次的 result.json、events.jsonl、head.mp4、initial-readback.json、setup.json、formal-record.json 均存在。四个 ARX 各有 target collision 与 proxy collision 两份审计日志。Aloha 原生路径没有生成这两类规划日志，这与运行代码一致。seed1-B-arx-w1 有 57 次观测、56 次模型输出，最后一次在观测适配时拒绝。seed1-B-arx-w4 没有计划文件，因为首个策略目标在计划保存前拒绝。

已保存 840 个模型动作块、841 个 RGB 观测文件及 837 个计划文件。Aloha 750 份原生计划全部通过冻结 native_prefix 逻辑逐值复现。动作块形状均为 50×14，数值有限。这里的完整性核验包含所有原始文件的摘要，选取包仅用于取回分析，原始全量轨迹仍保留在 SCNet。

来源链使用 r2 冻结包，manifest SHA-256 为 dbcb8d40ca88074751229302a0b35797698fa85cd56f43dc1c1eff4de963c09c，run lock SHA-256 为 49281639cfcc7375e8f37be556b8c0d9349aeab1ebc969d1d7ca5638be759d67。八份配置按锁中的逐文件哈希复核。模型启动为 ready、设备 cuda:0、revision 17ef545270cf26ed81169893b60f845f80ce6557，28 个参数与配套文件摘要绑定固定模型清单。没有启动新模型或修改权重、统计。

事件时序逐行检查 sim_s，八个试次均无逆序物理时间。监督器记录同一模型进程启动一次，六阶段执行完毕。八条结果仍为四次 task_unsuccessful、三次 planning_failed、一次 adapter_error。

Aloha 原始 executed_step 没有保存物块位姿或物理接触对，snapshot 也没有对象轨迹。因而无法从历史日志精确恢复每个时刻的持物关系、物块高度、法线和接触冲量。原视频存在，已取回 47 个抽样帧与四张最终帧并检查。抽样帧不能排除未采样时刻的短暂接触。ARX 保存的 contact_state 则覆盖物块位姿、持有状态和双指接触，四次失败均处于 approach、无已确认持有者。

选取包包括带原始行号的非物理步事件、定间隔及最后两步控制状态、末端原始动作/计划、全部试次结果和启动记录。选取包不是全量日志的替代品。transfer.zip SHA-256 为 294eae2fa78d91ec984000074c68b83346507d7cc8025d034526dfc9685269ee。后续复核原始全目录摘要的结果单独保存于 original-integrity-after.json。
