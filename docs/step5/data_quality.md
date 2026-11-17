# ARX-X5 数据转换与质量审计

原始数据来自 RoboTwin2.0 `3dc3b798668feb99ac61cc9086d84cbcc3d79186` 的 `handover_block` ARX-X5 clean 归档。50 个 HDF5 的成员字节数和 ZIP CRC32 全通过。逐条 SHA-256、字段、夹爪首次闭合位置、相邻重复目标、先验拆分和训练统计见[转换审计](data_audit.json)。派生 HDF5 为 380,290,351 字节，SHA-256 为 `19da908478bd36cbf3e2c0b507479d53db6c178bfd07d16cff7a04ee07fd72ca`，已在 SCNet 个人项目目录按该哈希验证。归档原件与派生 HDF5 均保存在项目数据区。 [数据归档来源](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/3dc3b798668feb99ac61cc9086d84cbcc3d79186)

| 拆分 | Episode 数 | 记录帧数 | 完整 H50 窗口 |
| --- | ---: | ---: | ---: |
| Train | 40 | 7,736 | 5,736 |
| Validation | 5 | 978 | 728 |
| Test | 5 | 966 | 716 |

按 `SHA256("step5:42:<episode>")` 排序后分配 40/5/5，组内按 ID 排序。Validation 为 `7,18,23,31,43`，Test 为 `6,29,33,38,48`。每条有 188–201 帧。窗口和归一化均按拆分隔离，统计只用 Train。

## 观测、标签与时序

- 手臂由左右两个 ARX 实例的 `joint1..joint6` 构成，单位 rad。14 维的顺序为左 6 关节、左夹爪、右 6 关节、右夹爪。向量与四个原始分列逐值相等。归一化夹爪命令为 0 闭合、1 张开，50 条每侧均覆盖两个端点。归档 `endpose/*_gripper` 是按 drive target 算出的标量，与此批 `joint_action` 夹爪标量逐帧相等。 [既有关节与夹爪契约](../../research/robotwin/data_contract.md)
- 固定 RoboTwin 的 `get_obs()` 在采样时读相机与 `get_*_arm_jointState()`。后者的手臂列来自 `get_drive_target()`，实际 qpos 由独立的 articulation `get_qpos()` 路径读取。专家控制在物理步推进后按条件存帧，并在动作边界额外保存。实验用同帧 RGB 与命令 `q[t]` 作条件，用 `q[t+1:t+51]` 作后续保存帧目标。末尾不足 50 帧的查询剔除；不填充、不跨 episode、不移除重复帧。已发现 2,297 对相邻命令完全相同。原始 HDF5 缺少物理时间戳、帧率、速度和单独的未来命令流，报告中的 t 是采样顺序索引。 [RoboTwin 采样源码](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/_base_task.py) [机器人关节读取](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/robot/robot.py)
- 同帧三路 240×320 RGB 为 head、left、right，映射到 OpenPI 的 base 和左右 wrist 槽位。三路相机矩阵首轴对齐且数值有限。旧 JPEG 本身按 legacy 规则交换 R/B，转换器采用 PIL 解码后应用匹配的通道次序，保留原 JPEG 字节。29,040 张图与已归档 XPolicyLab codec 解码逐像素相等。第三视角、点云和相机矩阵不参与训练。 [参考 codec](../../research/robotwin/stage3_sources/decode_image_bit.py)
- 取回归档中 50 份指令 JSON，ZIP 读取时校验 CRC，共验证 seen 5,000 条、unseen 5,000 条均为非空字符串。监督使用既有 G2-E 的固定任务级英文指令 “Pick up the red block with the left arm, hand it to the right arm, and place it on the target.”。它描述共同任务，与任何侧车条目均不逐字相同。各 episode 的侧车哈希和样例见[对照审计](data_context.json)。本轮估计的是固定指令任务表现。 [既有固定提示](../g2e_gpu_plan.md)

## 命令与物理关节差异

独立比对既有 50 条 ARX 原生 SAPIEN CPU 回放中的 `frame_ticks`、`native_drive`、`actual_qpos` 和夹爪请求。9,680 个保存帧的手臂驱动目标和两个夹爪请求与原始 HDF5 **逐值一致**。对应回放物理关节相对驱动目标的绝对差，手臂中位数 0.000571 rad、P95 0.004759 rad、P99 0.053879 rad、最大 0.149182 rad，RMS 0.009534 rad。四个物理夹爪关节与各自 drive target 的最大绝对差为 0.042122 m，涉及接触受阻和夹爪机构驱动。量值描述已有 CPU 回放；原 HDF5 实际 qpos 仍缺失。由回放生成的时钟和 qpos 未进入训练，夹爪物理位移也未替换归档的无量纲命令。 [逐集对照](data_context.json) [CPU 专家回放证据](https://github.com/latentstride/pi05-scnet-reproduction/blob/b56085401017c965af95cef70f543120ae5a4dd8/research/g2/native-expert-100/expert_replay_100_summary.md)

## 数值链和复现

以原始 float32 命令 `q[t]` 为 anchor，仅 12 个手臂坐标做 `target[t+k] - q[t]`，每个 k 共用同一 anchor。两夹爪保留绝对量。状态和混合动作分别计算训练集的逐维 `q01,q99`。前向为 `2*(x-q01)/(q99-q01+1e-6)-1`，不裁剪；推理先截取模型的前 14 列，逆归一化，然后仅对手臂加回原始 anchor。OpenPI 先将 14 维归一化状态离散化成文本 token，再补 18 列零得到网络张量，解析完整输出仍为 50×14。原生控制接口仅把夹爪裁到 [0,1]；关节保持原值，由共用的既有安全控制检查。离线误差统计使用裁剪前值。

全量目标 delta 往返最大绝对误差 2.3842e-7 rad，且夹爪绝对值守恒。包含分位数归一化和绝对还原的 7,180 个完整窗口全部通过 1e-6 逐值容差。两次独立转换的 HDF5 和审计 JSON 均字节一致。转换及复验命令从源码 checkout 根目录执行，所有输入和结果路径显式给出。

```bash
python -m scripts.step5_arx.data \
  --raw /absolute/folder/arx-x5 --zip-index "$PWD/docs/step5/arx_zip_index.json" \
  --output /absolute/folder/arx-drive-target-v1.h5 --report /absolute/folder/data_audit.json

python -m scripts.step5_arx.audit_context \
  --raw /absolute/folder/arx-x5 --replay /absolute/replay/results/arx-x5 \
  --metadata /absolute/folder/instructions --decoder "$PWD/research/robotwin/stage3_sources/decode_image_bit.py" \
  --output /absolute/folder/data_context.json

STEP5_ARX_DATASET=/absolute/folder/arx-drive-target-v1.h5 \
  python -m unittest discover -s tests -p 'test_step5_arx_data.py' -v
```
