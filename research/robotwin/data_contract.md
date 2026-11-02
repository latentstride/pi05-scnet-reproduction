# handover_block 离线观测契约 v1

日期：2026-11-03。范围为已抽样的官方仿真专家示范，两本体各 episode0–2；不是实机数据。数据仓库 `TianxingChen/RoboTwin2.0`，revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`。逐成员 SHA-256 以 `sample_manifest.json` 为准，完整 ZIP 未校验。模型统计资产与数据 revision 分开管理。

## 原始字段与顺序

两本体共有下列字段，T 为记录帧数，全部沿首轴对齐。

|字段|形状/类型|语义|
|---|---|---|
|joint_action/left_arm、right_arm|T×6 float64|记录的关节 drive target，非实际 qpos|
|joint_action/left_gripper、right_gripper|T float64|归一化夹爪目标，0 闭合、1 张开，无量纲|
|joint_action/vector|T×14 float64|左六关节、左夹爪、右六关节、右夹爪；须与分字段逐值一致|
|endpose/left_endpose、right_endpose|T×7 float64|记录的末端位姿 xyz、qw qx qy qz|
|endpose/left_gripper、right_gripper|T float64|与 joint_action 中夹爪一致|
|observation/{head,left,right}_camera/rgb|T 定宽字节串|压缩 JPEG，官方解码结果 HWC 240×320×3 uint8 RGB|
|各相机 intrinsic_cv / extrinsic_cv / cam2world_gl|T×3×3 / T×3×4 / T×4×4 float32|内参、世界到 CV 相机、GL 相机到世界|
|pointcloud|T×0|空，不作为可用点云|

|本体|左臂索引 0–5|右臂索引 7–12|额外视角|帧数|
|---|---|---|---|---|
|aloha-agilex|fl_joint1…fl_joint6|fr_joint1…fr_joint6|observation/front_camera/rgb，及其相机矩阵|295 / 277 / 282|
|arx-x5|左实例 joint1…joint6|右实例 joint1…joint6|third_view_rgb|195 / 192 / 194|

左右指机器人左右臂，不按图像左右判断；夹爪索引为 6、13。关节名来自同 revision 的 `embodiment_assets/embodiments/*/config.yml`，校验向量拼接和固定配置顺序；HDF5 不含关节名称，不能仅靠数值证明采集器物理接线。旋转关节采用 rad；末端位置采用仿真 m，四元数无量纲。HDF5 未存单位属性，单位依据已审计源码/资产约定。末端记录是世界参考下经 RoboTwin 方向矩阵和工具定义转换的位姿，不可直接等同 URDF link/TCP；契约不重新标定坐标，不声明轨迹 FK 已通过。CV/GL 轴变换为 diag(1,-1,-1,1)。

## 策略输入及责任边界

每帧仅返回 `state`、`images`、`prompt`。state 为原始 vector 的 float32 副本，shape=(14,)，语义仍是 recorded_joint_target；images 将 head/left/right 分别映射 cam_high/cam_left_wrist/cam_right_wrist，HWC→CHW，保留 uint8 和原分辨率；不增加 cam_low，不消费额外视角。

解码采用归档官方 decoder，Git blob `32ac399ef585a1ee6dea34eb35cb53c2814007b4`，源 revision `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`。旧 JPEG 不额外交换颜色通道。后续 AlohaInputs 将 CHW 转回 HWC；模型 transforms 承担 resize_with_pad(224,224)、padding/tokenization。适配器不 resize、不归一化、不做 joint delta，不加载模型。

语言来自同 episode 的 `metadata/episodeN.json`，确定性选 `seen[0]` → prompt（英文原文，不翻译）；允许显式选 split/index，并在独立 provenance 中保留相对路径、SHA-256、split、index、原文和数据 revision。HDF5 内没有语言，缺失或空文本应报错，不用任务名填补。

## 数据概念隔离

|名称|本阶段行为|
|---|---|
|recorded_joint_target|原始 joint_action/vector；作为 PACE 兼容 state 来源|
|recorded_endpose|原始 endpose 两臂；只用于验证和可视化|
|actual_qpos|缺失；不以 drive target 冒充|
|derived_next_state_action|不生成；若将来生成必须记录 frame t→t+1、末帧处理和标签版本|
|policy_predicted_action|本阶段没有；未来对外为 14 维绝对关节目标，内部关节 delta 与其有别|

无独立 action、qvel、timestamp、fps、dt、完整物体初态快照；帧序号不是物理时钟。报告时间频率为 null，不用转换器常量伪造 30/50 Hz。有效 schema 仅说明输入准备可用，不证明仿真回放或模型推理成功。

## 统计链

PACE 配置 adapt_to_pi=False，use_delta_joint_actions=True。输入 AlohaInputs → 关节 delta（训练有 actions 时，掩码 6真/1假/6真/1假）→ 源统计 Normalize → 模型 transforms。输出模型 transforms → Unnormalize → AbsoluteActions（加回原始当前 state 的关节部分，夹爪保持绝对量）→ AlohaOutputs 截取 14 维。adapter 保留物理值，归一化仅在策略变换层一次。统计来自源 handover_block/assets/robotwin/norm_stats.json，不用 ARX 样本重估；跨本体统计适用性属于后续实验问题。

源版本：Openpi-PACE checkpoint manifest 指向 ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12；已有审计文件来自 f4e52e6de4cc487a11585a3a7414d610b2883dae。测试固定前者的 transforms.py，并记录已有配置/策略文件 blob；不把版本不同隐藏为同一源码。
