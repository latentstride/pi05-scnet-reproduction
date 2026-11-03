# 阶段三：离线观测适配与校验

2026-11-03｜**完成 A1–A6，六条官方仿真示范均通过 CPU 离线 schema 校验。** 模型输入数据准备可用；未执行模型推理或仿真回放。

|本体|episode0|episode1|episode2|结果|
|---|---:|---:|---:|---|
|Aloha-AgileX|295|277|282|3/3 通过|
|ARX-X5|195|192|194|3/3 通过|

- **A1–A3**：[数据契约](data_contract.md) 固定 revision、字段、单位、关节顺序及数据缺项。[适配器](../../scripts/observation_adapter.py) 输出原始目标语义的 float32 `state[14]`、三路 CHW uint8 RGB、原 episode `seen[0]` 指令；语言路径、SHA-256 和索引保存在独立 provenance。没有生成 next-frame 标签、actual_qpos 或策略动作。
- **A4**：[校验结果](observation_adapter_validation.json) 核对六个 HDF5 与语言文件 SHA-256；检查全部 1,435 帧的字段长度、数值有限性、14 维状态、夹爪范围及重复值、四元数范数、左右臂拼接和固定配置关节顺序。每条首/中/尾帧运行完整适配器，新增 54 张三视角解码。其余图像合法性复用相同 HDF5 哈希与固定 decoder 的既有 5,740 帧解码证据，未重新进行全量解码。完整观测可由 `EpisodeAdapter.observations()` 按帧惰性生成；本轮新生成的是 18 个完整观测，未声称逐帧重新生成所有图像。
- **A5**：[4 项独立测试](pace_transform_validation.json) 通过：adapt_to_pi=False、三相机布局、14 维动作输出、关节 delta/绝对目标往返与夹爪不变、源统计 SHA-256、归一化可逆及重复归一化检出、坏 schema 拒绝。只执行固定上游源码的 AST 选定纯变换定义；Normalize/Unnormalize 直接测试数值方法，不导入模型/JAX/tokenizer、不加载参数。未验收完整策略运行时。
- **A6**：[样例图目录](stage3_visualization/) 导出 Aloha episode0 的 0/147/294 帧三视角、首尾原图、全部 295 帧的 14 维关节目标与双臂 xyz/四元数曲线（PDF/PNG/SVG），附原始绘图 CSV 和哈希 provenance。横轴仅用 frame index；没有真实秒数。已视觉检查版面与曲线。

复验（在仓库根目录，已有抽样 HDF5/metadata 位于 `research/robotwin/data_probe`）：

```sh
python -m pip install -r research/robotwin/stage3_requirements.txt
python scripts/validate_observation_adapter.py
python scripts/test_pace_transforms.py
python scripts/visualize_observation_episode.py
```

边界：state 来源是 **recorded_joint_target**，不是 actual_qpos；关节物理名称依据固定配置，HDF5 没有名称表；timestamp/fps/dt 缺失，频率为 null；末端工具点/FK、原始时序、跨本体源统计的控制效果和 GPU 图形验收均不由本阶段推断。阶段三未使用 GPU、未下载模型权重。
