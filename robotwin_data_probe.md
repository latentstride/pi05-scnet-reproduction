# RoboTwin 两种本体数据抽样与动作空间核验

日期：2026-11-01。任务：`handover_block`。本体：Aloha-AgileX、ARX-X5。

**两种本体各 3 条官方 HDF5 示范已读取，静态数值检查通过，并在 SCNet 完成独立 CPU 复验。** 数据没有显式时间戳，也没有单独的动作命令字段；时间重建、运动学对应和仿真回放仍需验证。本次未使用 GPU，未修改 LIBERO 基线环境。

## 1. 数据来源与抽样

使用官方 `TianxingChen/RoboTwin2.0`，固定 revision 为 `3dc3b798668feb99ac61cc9086d84cbcc3d79186`。不是第三方 LeRobot 转换数据，也不是真机采集：这里的“真实轨迹”指实际发布的仿真专家示范文件。

| 本体 | 官方归档 | 归档大小（bytes） | HDF5 数量 | 本次抽样 |
| --- | --- | --- | --- | --- |
| Aloha-AgileX | `dataset/handover_block/aloha-agilex_clean_50.zip` | 546,033,461 | 50 | episode0、1、2 |
| ARX-X5 | `dataset/handover_block/arx-x5_clean_50.zip` | 332,395,804 | 50 | episode0、1、2 |

从 ZIP 中按 HTTP Range 提取所选成员，主体数据传输约 52.25 MB，没有下载两个完整归档。还读取了 seed、场景摘要和所选 episode 的语言 JSON。每个提取成员通过 ZIP CRC，保存 SHA-256；SCNet 上传后重新校验了六个 HDF5 和十个元数据文件。完整 ZIP 的 LFS SHA-256 仅作来源元数据，**未声称已下载并校验整个 ZIP**。详见 [样本清单](research/robotwin/sample_manifest.json) 和 [抽样脚本](scripts/sample_robotwin_archives.py)。

官方任务为左臂抓红色方块、交接到右臂并放到蓝垫。同任务页面公布的数据生成成功率为 Aloha-AgileX 83%、ARX-X5 81%、Franka-Panda 0%，因此首组改用 Aloha-AgileX＋ARX-X5；这些是官方生成统计，不是本次实测成功率。[官方任务与生成统计](https://robotwin-platform.github.io/doc/tasks/handover_block.html)、[固定数据目录](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/3dc3b798668feb99ac61cc9086d84cbcc3d79186/dataset/handover_block)

## 2. 实际数据形状

`T` 为 episode 的记录帧数。

| 字段 | 两本体共同形状 | 含义与检查 |
| --- | --- | --- |
| `joint_action/left_arm`、`right_arm` | `(T,6)`，float64 | 每臂六个关节的记录值；不能当作关节速度 |
| `joint_action/vector` | `(T,14)`，float64 | 左六关节＋左夹爪＋右六关节＋右夹爪；与分字段拼接完全一致 |
| `endpose/left_endpose`、`right_endpose` | `(T,7)`，float64 | xyz＋四元数；不是六维位姿增量 |
| 两组 `left_gripper`、`right_gripper` | `(T,)`，float64 | joint_action 和 endpose 中对应值完全一致，范围均为 `[0,1]` |
| 重建的双臂末端＋夹爪向量 | `(T,16)` | 每臂 xyz＋四元数＋夹爪，由上述字段拼接；文件中没有独立该字段 |
| 相机 `rgb` | `(T,)`，定宽字节串 | 压缩图像；解码后为 `(240,320,3)` uint8 |
| `intrinsic_cv` | `(T,3,3)` | 相机内参 |
| `extrinsic_cv` | `(T,3,4)` | CV 相机外参 |
| `cam2world_gl` | `(T,4,4)` | GL 相机到世界变换 |
| `pointcloud` | `(T,0)` | 空数据，不视为可用点云 |
| timestamp / fps / dt | 未发现 | 全部 dataset 和根属性均未给出显式采样时间 |

`joint_action` 是字段名称，不能据此断言它就是独立的控制命令。本样本没有单独 `action`、desired target、qvel 或每步执行时间字段。采用“下一帧状态为动作”的转换会生成新的标签，必须单独记录；转换标签不等于原始控制命令。

| 本体 | episode0 | episode1 | episode2 | seed 文件前 3 项 |
| --- | --- | --- | --- | --- |
| Aloha-AgileX | 295 帧 | 277 帧 | 282 帧 | 0、1、2 |
| ARX-X5 | 195 帧 | 192 帧 | 194 帧 | 0、1、2 |

两种本体都有 head、left、right 三路相机。Aloha 另有 `observation/front_camera`，ARX 另有顶层 `third_view_rgb`，二者不是同名结构，不能无条件打包成相同四相机接口。

## 3. 静态检查结果

| 检查项 | 结果 | 说明 |
| --- | --- | --- |
| 六条文件的 SHA-256 | 通过 | 本地与 SCNet 一致 |
| episode 内各字段长度 | 通过 | 都与本 episode 的 T 对齐 |
| 所有数值字段有限值 | 通过 | 无 NaN/Inf |
| 14 维向量与分字段拼接 | 通过 | 最大绝对差 0 |
| 夹爪范围与重复字段一致性 | 通过 | 0—1；最大绝对差 0 |
| 四元数单位范数 | 通过 | 六条轨迹两臂最大误差约 `8.88e-16` |
| 图像读取 | 通过 | 全部 5,740 个相机帧解码成功，240×320×3，均非恒定图像 |
| 相机 CV/GL 转换一致性 | 通过静态检查 | `extrinsic_cv × cam2world_gl` 与 `diag(1,-1,-1,1)` 的最大元素误差约 `3.95e-7` |
| 时间戳及真实采样周期 | 缺失，未通过 | 不能直接按 30/50 Hz 还原执行时刻 |
| FK、IK、关节限位、轨迹回放 | 尚未执行 | 数值可读不证明物理可执行 |

图像调用固定 RoboTwin 源码 `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f` 的官方解码器，Git blob 为 `32ac399ef585a1ee6dea34eb35cb53c2814007b4`。样本全部未带新格式 `XPL-RGB1` 标记，按官方旧版路径解码，没有额外交换 RGB/BGR。[官方解码器](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/data/decode_image_bit.py)

完整 schema、数值范围、首帧状态、四元数及相机检查数值见 [probe_results.json](research/robotwin/probe_results.json)，可复验脚本见 [probe_robotwin_data.py](scripts/probe_robotwin_data.py)。

## 4. 坐标、单位与动作语义

四元数排列 `[qw,qx,qy,qz]`、世界系绝对末端目标及夹爪 0 闭合/1 张开的含义来自官方源码；范数检查本身无法判别四元数排列或参考系。关节旋转量按仿真惯例为 rad，xyz 按仿真长度约定为 m，HDF5 本身没有单位属性，后续须结合固定资产和运动学复核，不能宣称已独立标定单位。

官方末端读取包含 `global_trans_matrix`、`delta_matrix` 和工具偏移。`get_ee_pose` 与 TCP、URDF 原始末端链节并非同一工具点；本体配置和该数据的采集源码需匹配。本次尚未取得归档对应的完整采集环境版本及两种本体配置，不把“已读当前固定源码”写成“历史数据已通过 FK 对应验证”。[位姿与工具点源码](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/robot/robot.py)

同一零关节向量不代表同一末端位姿：两种本体 episode0 首帧关节向量均为两组六个 0 加夹爪 1，但左末端 xyz 分别约为 `[-0.297924,-0.313804,0.941999]`、`[-0.299500,-0.252299,0.940493]`。这直接说明跨本体不能复制 qpos 作为同一物理目标。

`qpos` 控制接收关节位置目标，`ee` 接收每臂 xyz＋四元数＋夹爪，执行还经过规划/时间参数化。观测采样、规划目标与物理积分步分开记录；数据转换器硬编码的 fps 不作为原始频率证据。[官方控制接口](https://robotwin-platform.github.io/doc/usage/control-robot.html)、[动作执行源码](https://github.com/RoboTwin-Platform/RoboTwin/blob/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f/envs/_base_task.py)

## 5. 对回放和后续实验的影响

seed 文件及场景摘要已读取，但本次 `scene_info.json` 的对应 `info` 为空，未提供完整物体位姿快照。seed 相同也不能证明两本体随机初始化完全一致。后续回放需固定采集版本/资产，重建并核对实际物体初态、工具点及采样时刻；若时间不可恢复，应明确采用几何路径和重新定时的回放，不能称为原始时序重放。

本轮验收为：共同任务两本体数据入口存在、各三条示范可读、图像与状态结构及部分坐标数值一致性通过。时间戳缺失已经查明，属于后续回放的待解决条件。不能据此勾选 GPU 图形能力、专家回放成功率或固定权重 π₀.₅ 自主控制。

保持现有 [图形 Preflight](docs/robotwin_graphics_preflight.md)。下一步先补齐采集版本、工具点/运动学和时间语义证据，再准备一次独立环境中的短时 GPU 验收；仍分别记录 Vulkan、SAPIEN、CuRobo。现有 `pi05_libero` 的双臂模型兼容性另行验收，不混用专家回放与 VLA 成功率。

## 6. 文件与执行位置

SCNet 原始样本、seed/场景/语言元数据、哈希及 CPU 复验 JSON 放在个人项目的 `research/robotwin/data_probe/`。本机独立 CPU 环境完成官方图像解码和完整静态检查；SCNet 沿用既有环境仅做只读数据复验，没有安装新包。公开仓库保存报告、提取/核验脚本、样本清单和数值结果，不上传整条 HDF5。

补读官方本体配置文件的网络操作未执行：自动审批服务因额度限制未能完成审查，这不是对动作安全性的否定；本报告将相应的资产/FK 验证保留为待完成项。
