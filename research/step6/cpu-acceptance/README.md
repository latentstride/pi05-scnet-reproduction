# CPU 验收 · 2026-11-16（北京时间）

**`STEP6_PARTIAL_READY`**。在已有 SCNet CPU 容器中，用真实 PACE/RoboTwin `handover_block`、SAPIEN 3.0.0b1 / PhysX、mplib 0.2.1 和 ARX-X5 URDF 执行。原生任务的初始化、重置、块质量（10 g）及成功判定均保留。CPU 入口只移除呈现组件和 URDF 的视觉节点；原碰撞网格、关节、物理参数与安全实现保留。

| 验收项 | 实测 |
| --- | --- |
| 独立重置和配对 | seed 0–4，10 个独立物理场景，5 对初态匹配；块位置随场景 seed 变化。CPU 相机矩阵不可用。 |
| 合法响应、执行和回读 | 每场两次真实 loopback WebSocket 交换，共 20 次。首个合法窗口后在 tick 210（0.84 s）重新构造请求。全部试次均记录实际 qpos 变化和更新后的命令。合法组每场 8 行、270 tick，`ACTION_LIMIT`。 |
| 原安全门禁拒绝 | 另一组每场已运行首个 K4。第二次响应首行的左臂关节 1 为 99 rad，被原 `bounds` 拒绝为 `position_limit`；五次负例均无额外物理步，`SAFETY_STOP`。 |
| 安全、记录与分析 | 共 60 行、2,400 个真实 PhysX 步。每条均有回执、接触/关节轨迹及原始请求和响应。离线重算与回执一致，对 20 个请求、20 个响应及 10 个握手核验原始消息哈希。7 个定向单测通过。 |
| 三路真实 RGB 和视频 | `GPU_UNVERIFIED`。CPU 容器渲染初始化探针超时，无可用渲染设备。WebSocket 测试用了明确标记的图像协议 fixture；真实相机未采样，未生成 MP4。 |
| 模型与研究问题 | 无 π₀.₅ GPU 推理、无实际 LoRA 权重、无正式成功率。两种替身只测试协议及控制，正式分母为空。 |

[验收记录](status.json) · [轨迹验收摘要](verification.json) · [逐 seed 配对表](pairs.csv) · [统计输出](summary.json) · [CPU 控制台日志](cpu.log) · [10 条试次完整记录包](fixture-evidence.zip)

解压 ZIP 后有 `pair/`，可独立重放核对。命令从第六步 checkout 根目录运行；验证脚本需要 NumPy、Step 5 的数据模块依赖和 OpenPI client：

```bash
python -m scripts.step6_arx.verify_cpu /absolute/extracted/pair
python -m scripts.step6_arx.analyze /absolute/extracted/pair --output /absolute/new-summary
```
