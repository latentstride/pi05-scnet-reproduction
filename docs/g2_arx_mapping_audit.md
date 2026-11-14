# ARX seed1 映射、代理自碰撞与窗口审计

## 直接阻断

**CONFIRMED。** `127510363` 的目标 ARX 没有记录到危险碰撞。第 35 次请求在 tick 15855 的第 3 个候选端点上遇到源代理 `fl_link4 / fl_link6` 自碰撞，只执行前两行。第 36 次请求在 tick 16160 成功构造真实新观测并取得模型响应，随后动作规划不能通过源代理前向准入。生产回执的 16 对 IK 候选中，12 对先被连续性筛除，另外 4 对都发生源代理自碰撞，最大接触深度约 0.370–0.390 mm。第 36 次没有推进物理步。终止类别为 `planning_failed`。[真实作业回执](../research/g2/job-127510363/result.json)

**CONFIRMED。** CPU 从两个真实 tick 的 ARX 关节、夹爪、物块及源代理请求状态重建场景，调用冻结 `RecoveryCoordinator.action_plan → checked_prefix / _upward_plan → source_admissible → solve_pair`。第 35 次同样只批准两行，第 36 次批准零行，复现 4 对碰撞与 12 对连续性拒绝。独立把两组所有原记录候选送回 CPU FCL，两组各 12 对曾被连续性拒绝的候选也都碰撞。对三个实际失败的源目标另用额外起始种子在原残差、关节、连续性和碰撞约束下检查 **36、40、44 对**，未找到可用解。此审计没有推进物理或启动模型，有限搜索不能证明所有 IK 分支或改变接近路径后都不可行。[复算数据](../research/g2/final-system-review/arx-replay.json)

**几何身份。** 锁定 URDF 中 `fl_link4` 经 `fl_joint4` 接到 `fl_link3`，`fl_link6` 经 `fl_joint6` 接到 `fl_link5`。两者非直接邻接，链接分别有 `link4.STL`、`link6.STL` 的非空碰撞形状。原始 URDF 哈希、两个网格哈希及零碰撞局部变换已留证。源代理用保留碰撞形状的 URDF 创建，无目标 ARX 的同 articulation 白名单。该组合有理由做自碰撞检测，重叠可按固定资产复现。实物连杆公差及是否存在满足全部原约束的另一条跨本体后继路径仍为 **UNKNOWN**。[代理模型构造](https://github.com/latentstride/pi05-scnet-reproduction/blob/fae63f80ec65ce6121f486f0eac57ea4bdb16724/scripts/g2_proxy_collision.py#L9-L58) · [候选证据和资产哈希](../research/g2/final-system-review/arx-replay.json)

## 动作窗口

夹爪低于或等于 0.5 只作离线描述。0 为闭，1 为开。保留数表示至少有一次低开度预测落在该前缀，不能推断实际接触。

| 数据 | W4 左/右 | W26 左/右 | W31 左/右 | H50 左/右 |
| --- | ---: | ---: | ---: | ---: |
| 历史 `127147943` ARX seed1，39 块 | 0/0 | 2/0 | 33/0 | 37/4 |
| `127510363` 第 35、36 块 | 0/0 | 0/0 | 2/0 | 2/1 |

第 35 块左首次低开度在第 27 行，右无。第 36 块左在第 28 行，右在第 48 行。W4 从未选到这些行，且第 35 块第 3 行就被几何截断，第 36 块未执行任何一行。W26、W31 和 H50 都包含这些已出现阻断的前缀。当前生产 ARX adapter 明确只接受 1–4 行，长窗口是对保存动作的离线前缀统计。按同一状态、原约束和顺序延长块，不会让第 35 块自动跳过先前候选。换窗口后会改变实际查询状态，长期闭环变化是 **UNKNOWN**。[离线窗口证据](../research/g2/final-system-review/arx-archive-comparison.json) · [前缀约束](https://github.com/latentstride/pi05-scnet-reproduction/blob/fae63f80ec65ce6121f486f0eac57ea4bdb16724/scripts/g2_adapter.py#L358-L417)

## 专家参照及优先级

已有 `127147943` 的成功 ARX seed1 专家试次。复核初始读回，机器人基座、任务框架、方块/目标/桌、夹具哈希、双臂实测关节及夹爪六组字段与本次策略均逐值相同。专家在 tick 7012 左持、18149 双持、20105 右持、27267 释放后达到原成功谓词。它证明该**初态的目标本体任务**可行，专家路径不使用策略的源代理观测。以物块坐标下双 TCP 位置对齐，最近的专家接近样本仍相差左 68.4 mm、右 204.4 mm，姿态差 14.2°、6.7°，不构成策略 tick 16160 的匹配状态金标。作者缓存场景 100000 的匹配专家轨迹仍 **UNKNOWN**。[专家对照数据](../research/g2/final-system-review/arx-archive-comparison.json)

因此，当前 ARX 的直接瓶颈是**策略请求的后继无法通过源代理几何和连续性准入**。当前有证据说明 W4 没消费后段闭爪，有证据说明有限 IK 解集不够，也有证据说明保护按设计阻止了不可表示的后继。没有证据能把任务失败单独归于窗口、模型训练或 IK 求解器。继续跨本体研究有同初态专家可行性基础，但需原生源本体闭环参照，并找到一个 CPU 可复核、满足原约束、实际通往抓持的 ARX 后继，才有理由再排 GPU。

```sh
audit=research/g2/final-system-review
CUDA_VISIBLE_DEVICES= PYTHONPATH="$arx_release/scripts:scripts" python3 -B \
  "$audit/arx_replay.py" "$job127510363" "$arx_release" "$pinned_pace/assets" \
  research/g2/job-127510363/result.json "$new_cpu_dir"
python3 -B "$audit/arx_archive_comparison.py" "$job127510363" "$job127147943"
```
