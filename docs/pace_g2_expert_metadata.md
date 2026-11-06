# G2 官方专家时间与初态追查

日期：2026-11-06。已找到官方专家缓存，完成三条 Aloha 示范的逐值对照，并重建时间轴和初始生成位姿。稳定后实际物体位姿仍未在归档中发现；不能据此宣称正式配对初态验收通过。

## 数据及来源

官方数据集 `TianxingChen/RoboTwin2.0`，固定 revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`，成员来自 `dataset/handover_block/aloha-agilex_clean_50.zip`。已有 HDF5、seed、scene_info、语言和 embodiment 配置本轮重新检查；新增 `_traj_data/episode{0,1,2}.pkl`。仅通过 HTTP Range 接收 585735 字节，9 次请求；完整 ZIP 未下载、未做全包 SHA 校验，成员 ZIP CRC 和 SHA-256 已核验。来源清单见 `research/g2/expert-cache-uncached-20261106/provenance.json`。

PKL 只有左右规划路径列表（7 和 4 段），每段包含 status、position、velocity。受限反序列化仅允许三个 NumPy 构造符，拒绝其他 global。HDF5 没有 timestamp/fps/dt、物体位姿或 actual qpos；scene_info 仅有空杂物、空纹理和空 info。完整 ZIP 成员目录也未发现其他快照成员。首帧 RGB 和逐帧相机内外参确实存在。

源码固定到 RoboTwin `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`，保存于 `research/g2/expert-source-20261105/`，补取了相机和采集脚本，逐文件哈希见 manifest。**数据包未声明生成源码 commit**，因此以下时间/位姿标为源码重建，不能改写为原始测量。三个完整动作序列和图像投影提供独立一致性证据。

## 时间重建结果

源码 scene timestep 为 1/250 秒，clean save_freq=15。每段先保存，再在控制下标 0、15、30…物理推进后保存，段末再保存。夹爪先插值 200 步，再保持终点 100 步（base task 的 alpha=0.5 padding）。同时运动按原始左右归一化进度调度。不能直接用帧编号乘 60ms。

|episode|seed|帧数|重建物理步|重建时长|全帧14通道最大差|
|---|---|---|---|---|---|
|0|0|295|3945|15.780s|0|
|1|1|277|3655|14.620s|0|
|2|2|282|3764|15.056s|0|

三条全部逐帧、逐通道精确一致。普通帧间隔 15 tick=60ms，边界含 0、1 等其他间隔；0 tick 表示同一物理时刻重复采样。生成了 `episodeN_derived_timing.npz`：physics_tick、time_s、recorded_joint_target。时间原点为第一条保存帧，未包含 setup、稳定或规划墙钟。回放时须保留停留长度、处理同 tick 记录，不对零时间差计算速度。这里没有证明实际 qpos 或动力学回放成功。

## 物体初态追查

根据 clean 配置和 source，使用 episode 对应 seed 重建生成位姿。随机调用顺序包括一次 table-height uniform，即使范围为零也消耗随机数；两个静态相机各调用 randn(3) 和 uniform，即使关闭随机位移仍消耗随机数。之后才是物块 xyz、三个旋转角、目标区 xyz。直接把 seed 的前两个随机数当物块 xy 会得到错误位置。

|episode|物块生成 xyz (m)|目标区生成 xyz (m)|物块/目标投影边界最大差|
|---|---|---|---|
|0|(-.173312,.197931,.842)|(.113069,.151011,.741)|4.18 / 2.40 px|
|1|(-.074377,.006847,.842)|(.129715,.190037,.741)|1.35 / 2.03 px|
|2|(-.196635,.155283,.842)|(.217800,.192699,.741)|1.90 / 2.21 px|

四元数和完整数值保存在 `episodeN_initial_candidates.json`。用 HDF5 首帧内外参投影实际长方体八角，与原始色块 bbox 比较，记录到 `image_consistency.json`。黄色框为物块，青色为目标区。episode0、2 的物块顶部超出图像边界，比较时裁剪到图像范围。该比较是二维位置一致性检查，不能据 bbox 数值推出三维位姿精度。原始旧 JPEG 通过 PIL 查看会红蓝互换；检查脚本按官方 decoder 的 legacy 语义恢复 RGB，原始 JPEG 字节另存且未改动。

**剩余差距是稳定后的实际快照。** 源码生成物块后先运行 2000+500 步稳定，再开始保存帧。动态物块会受重力/接触影响，生成 z=.842 不等于首帧实际 z。下一步可用这些已获图像/标定/生成参数建立 CPU 场景并导出稳定后 pose，再核对多视角图像；必须分别记录 Aloha 和 ARX 的实际物体、基座、qpos/夹爪，之后才能冻结 pair_id。当前不把同 seed 或上述生成位姿当成已验收物理配对。

## 复验

使用项目临时 Python 环境（numpy1.26.4、h5py3.11.0；投影预览另需 Pillow11.3.0）：

```sh
PYTHONPATH=scripts /private/tmp/pi05-g1-tests/bin/python -m unittest discover -s tests -p test_g2_expert_metadata.py -v
/private/tmp/pi05-g1-tests/bin/python scripts/g2_audit_expert_metadata.py --cache research/g2/expert-cache-uncached-20261106 --output /private/tmp/g2-expert-audit-fresh
/private/tmp/pi05-g1-tests/bin/python scripts/g2_check_expert_projection.py
```

三个新增测试通过：三条真实示范全量匹配、非白名单 pickle 拒绝、seed 重建不污染全局 RNG。结果目录为 `research/g2/expert-metadata-20261106/`。本轮没有申请 GPU 或提交作业。

## 2026-11-06 CPU 稳定复现补充

现已完成三组生成位姿、两本体共六个CPU场景的2500步稳定测量，导出了新的实际重建快照。物块稳定z约.840m，三组跨本体物体pose逐值一致，末200步位置变化和末速度均0。原归档快照仍缺；当前初始化约定与原任务夹爪/被动补偿存在差异，不能标为原专家完整机器人初态。见 [稳定初态CPU复现](pace_g2_settled_initials.md)。
