# ARX 在线 TCP 投影与 GPU 准入

**`READY_FOR_GPU`，本轮 GPU 提交数 0。** 现有 ARX 闭环已能在有界 TCP 投影动作后重新采集实测状态，经源代理 IK 发出下一次策略请求，再对新响应执行原规划或同 tick 安全停止。CPU 试验使用实际 PhysX 关节、夹爪和物体读回，三路 RGB 和模型服务使用显式测试替身。GPU 实验将使用现有三相机和冻结 π₀.₅ 服务完成新状态推理。[CPU 回执](../research/g2/arx-projection-online/cpu-summary.json) · [请求与动作日志摘录](../research/g2/arx-projection-online/cpu-exchanges.json)

## 触发与安全行为

`task_tcp_projection=bounded_task_tcp_projection_v1` 只在独立 ARX 候选启用。原双臂规划和源代理前向检查都通过时直接执行原计划。目标规划候选或源代理几何拒绝时，只有实测状态仍处于无指物接触的 `approach`、原保护允许且恢复额度未耗尽，才进入投影。初始实测几何、关节或夹爪指令越界、同步错误以及接触期拒绝均进入既有停止处理。

搜索按任务系三个坐标轴的正负方向尝试 4、8、11.5 mm，最多 18 候选，单次搜索上限 120 秒并受试次剩余时间约束。右 TCP、姿态和夹爪意图沿用原目标。偏离包络为 12 mm/1°，右 TCP 保持原 IK 残差约束。原关节限位、速度和加速度、源代理连续性与几何、目标完整轨迹碰撞、接触屏障、实测跟踪和有限力夹爪检查均照常执行。运行时根据当前原计划和实测状态重新搜索。

执行修正前缀后整块剩余行作废。下一轮通过 `ARXBackend.capture()` 在同一物理时刻采集三相机 RGB、双臂实测关节和夹爪、物块与目标位姿。推进后的时间戳与源代理重映射通过后，原客户端发送新请求。回执保存修正来源时间戳、所选偏移、拒绝原因、弃用行数、后续实测状态和响应。修正后首次新响应若被原门禁拒绝，会在原 `RuntimeReset` 路径执行 hold 和 close，候选拒绝留存同 tick 证明。缺少新状态或源代理映射失败会在发送请求前停止。

实验方法为**冻结 π₀.₅ 加动作级受约束重定向**。checkpoint、norm stats、提示、RNG seed `20261107` 和任务成功谓词与作业 `127510363` 相同。

## 定向 CPU 结果

| 独立历史快照 | 旧放行 | 修正后首块 | 修正后第二次请求 | 第二响应及停止结果 |
| --- | ---: | --- | ---: | --- |
| 第 35 次，tick 15,855 | 2/4 | 4 行，495 物理步，弃用 46 行 | tick 16,350 | 测试服务根据新状态生成响应，通过原规划再执行 4 行，共推进 530 步 |
| 第 36 次，tick 16,160 | 0/4 | 4 行，610 物理步，弃用 46 行 | tick 16,770 | 新测试响应的越界夹爪指令被拒，hold、close、同 tick 停止证明通过 |

每条回放由对应的历史快照独立重建，历史速度和接触求解器缓存未归档，因此按零初速度重启。第 35、36 次分别搜索到 +X 4.0、11.5 mm。两段通过生产 `run_condition`、`RecoveryCoordinator`、`ARXBackend` 和 `open_policy`。实际回环 WebSocket 收到三路图像和重映射后的新 14 维状态，响应与原块逐值不同。第 35 次第二响应是由当次请求状态构造的测试命令。CPU 测试无冻结模型推理调用，也未测量抓持成功率。

23 项本地定向回归通过，覆盖原计划直通、接触或放置后拒绝投影、实测几何拒绝、过期观测、重映射失败和新动作拒绝。第 36 次 CPU PhysX 测试验证越界响应在新增控制步前安全停止。

## 单作业配置

| 配置项 | 值 |
| --- | --- |
| 任务 | RoboTwin `handover_block`，ARX-X5，历史 seed 1 夹具，W4，最多 300 策略动作 |
| 资源 | `ksagnormal01/gnode39`，1×RTX 4090 24 GB，4 CPU，32 GiB，90 分钟 |
| 流程 | 一个作业、一个模型服务、原动态准入、一个独立试次、最多四次恢复 |
| 时限 | 图形 300 s，启动及准入 1,680 s，试次软／硬限 2,550／2,600 s，统计及清理各 120 s，宿主预留 360 s，剩余 220 s |
| 费用上限 | ¥5.40，按提交门禁上限 ¥3.60/卡时 × 1.5 卡时 |
| 授权状态 | 待授权，已提交 0 项 |

SCNet 已完成只读分区、节点、账号关联、QoS、磁盘与重复任务预检，完整候选通过 89 个成员校验。Slurm 控制器中的历史作业记录已过期。提交助手仅为此投影候选从已验证的 `127510363` 授权回执和 Slurm 账务记录解析账号与 QoS，仍对当前分区、节点、账号、并发和磁盘执行原检查。实际提交时需要当次授权及通过原报价和余额门禁。[包配置与哈希](../research/g2/arx-projection-online/package.json) · [SCNet 预检](../research/g2/arx-projection-online/platform-preflight.json) · [历史官方报价回执](../research/g2/standalone-review/scnet-quote.json)

构建使用与作业 `127510363` 对应的原单作业归档，保留原 Slurm 脚本、模型入口和控制后端。新包更新 ARX 试次配置与其哈希绑定，移除仅由作者模式消费且绑定旧计划字节的 H50 覆盖声明。源文件哈希见[清单](../research/g2/arx-projection-online/source-sha256.json)。

```sh
python3 -B scripts/g2_standalone_package.py arx . \
  research/g2/standalone-review/g2-standalone-arx.tar.gz \
  /path/to/stage/g2-standalone-arx.tar.gz --arx-projection
```

在原无 GPU PhysX 环境中，将解包后的 `scripts`、本仓库 `scripts`、`research/g2/final-system-review` 和原 OpenPI 客户端加入 `PYTHONPATH`，复核两次独立快照。

```sh
CUDA_VISIBLE_DEVICES= python -B research/g2/arx-projection-online/replay.py \
  "$arx_release" "$pinned_pace/assets" "$job127510363" \
  research/g2/aloha-to-arx-20261111/arx-input-snapshots.json "$fresh_cpu_output"
```
