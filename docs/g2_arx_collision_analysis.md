四次 ARX 都在 IK 候选端点筛选阶段拒绝。QuinticMotion 尚未收到被拒绝的 waypoint，现有记录没有证据将这四次终止归为关节插值路径中间碰撞。

| 试次 | 动作数 | 世界与调用链 | 碰撞拒绝对数 | 具体拒绝 |
|---|---:|---|---:|---|
| seed0-B-arx-w1 | 25 | Adapter.actions → solve_pair → target ContactWorld.ik_clear → project_fingers | 16 | 候选左指1在起始开度与物块相交 |
| seed0-B-arx-w4 | 24 | Adapter.actions → solve_pair → target ContactWorld.ik_clear → project_fingers | 16 | 候选左指1在起始开度与物块相交 |
| seed1-B-arx-w1 | 56 | Adapter.observe → solve_pair → source ContactWorld.ik_clear → proxy checker | 4 | 虚拟 Aloha fl_link4–fl_link6 自碰撞 |
| seed1-B-arx-w4 | 0 | Adapter.actions → solve_pair → target ContactWorld.ik_clear → target checker | 16 | 实际 ARX 右臂 link2–link5 自碰撞 |

两个 seed0 事件的“start”指在候选 arm q 固定后，从实测指开度开始扫描的几何端点，不能解释成当前真实机械臂已经穿透物块。CPU 重建中，w1 左指1的起始深度为 5.680498 mm，w4 为 2.752330 mm，均超过原有 2 mm 规则。实测开度分别为 43.881156/43.888327 mm，对应请求为 43.930907/43.990006 mm，指令仍接近全开。正常闭合被物块阻挡的投影规则没有被关闭，这里是开爪下的候选臂目标已压入物块。

seed1-w1 发生在一次真实观测映射中。当前 ARX 状态转换为虚拟 Aloha 工具位姿时，四个满足连续性限制的候选均在虚拟左腕 fl_link4–fl_link6 相交，原记录深度约 0.009292–0.009989 mm。实际 ARX 当前关节并未被这些虚拟候选替换。原第 56 次动作之后的观测已保存，下一次策略推理和动作并未发生。

seed1-w4 的初态门槛与首次 proxy 映射已通过，第一份实际模型动作之后目标 IK 才拒绝。原始 object _18 是右臂，候选 link2–link5 深度约 0.016843–0.020853 mm。CPU 重建名称为 arx_right_15，按角色、link 和几何对应，编号差异来自独立场景实体分配。该事件支持首个策略目标在所枚举分支上不可执行，不支持初态检查漏掉同一实测碰撞。

每个事件最后一个 snapshot、最后一次成功 policy_input、两步末端控制状态、请求 task_tools、IK seed、指开度与候选 q 均已保存。逐值资料位于 static-replay-127054052-r3/replay.json。该文件保留原事件行号，并保存重建后的全部候选 verdict、两侧 IK 解数量、连续性拒绝及当前实测指开度。四个时刻的 contact_state 均为 approach、holders=[]，物块仍接近各自初始位置，没有已确认抓持或另一只臂携物的前提。

重建使用实际命名关节、实测基座、固定 URDF、原接触阈值和原始 request，0 物理步、无渲染、无模型。四个原失败全部复现，原保存的 8/8/4/8 个候选也全部拒绝。原始候选摘要最多保存8项，碰撞审计和修复后的重建记录补齐全部候选信息。

额外种子搜索没有修改生产候选算法。原始工具容差、关节界限和 source 连续性仍适用。四次分别检查70、65、104、225对额外候选，没有发现通过者。这是有界搜索结果，不是全局无解证明。没有充分证据将这些拒绝改为豁免、扩大容差或更换规划器。

同样规则下，源 Aloha 专家第0、25、100物理tick的工具目标，经明确 FK 变换后进入目标 ARX IK 和 QuinticMotion。12/12目标与完整关节路径均通过，最大速度不超过原0.5 rad/s。这些控制点属于专家接近段，不能据此宣布全交接路径已通过生产门禁。它们说明当前场景和适配路径仍存在安全可达目标。

候选计数需区分全部分支对与通过连续性限制的分支对。原 solver 每臂最多从四组种子产出四个解，ranked product 有16对，其中包含不同绕角表示及相近解。seed1-w1 有12对先因连续性拒绝，余下4对因碰撞拒绝。首次快速尝试还会重复第一对，故碰撞调用次数与表中拒绝对数不完全相同。
