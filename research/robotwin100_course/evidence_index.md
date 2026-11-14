# 图表与证据索引

| PPT 页 | 数据、指标与口径 | 来源与复算 |
|---|---|---|
| 1–4 | 两种官方专家画面，文献比较，开源模型与冻结权重 | 固定数据 revision 的 episode0 head_camera；references.md |
| 5 | 每本体 50 个轨迹长度。点为各轨迹，箱线图显示中位数及四分位范围，单位帧 | raw HDF5 / joint_action/vector 行数；trajectory_quality.csv；figures.py |
| 6–8 | 两本体帧数、重复状态与完整重复帧，全量 23,814 帧；文件、记录与图像三个不同分母 | summary.json、embodiment_summary.csv、field_inventory.csv、anomaly_index.json；analyze100.py |
| 9 | 源目标 → 源 FK/TCP → 名义公共坐标 → 目标 IK | sources/mapping_source_lock.json、mapping_config.json、sapien_fk_smoke.json、URDF 和配置哈希 |
| 10 | 全量 47,628 单臂目标，有解 CDF 以每方向合格样本为分母，87 个未解留在全表 | mapping_summary_full.json、ik_results_full.csv、mapping_full/；map100.py、verify_results.py |
| 11 | LIBERO 15/15，历史策略 0/8、修复专家 2/2，最新 1 项运行、9 项未运行 | sources/libero_evaluation_results.csv、固定 PR #5 作业记录和根因报告 |
| 12 | H50 seed1 原持续闭爪与 4 mm 预载，单分支 n=1，首次左持后时间 / s、相对漂移与驱动 / mm | sources/contact_curves.csv、原公开片段、g2_final_concentrated_cpu_review.md；figures.py |
| 13–14 | 真实 Dashboard 截图、运行记录、实证结果与贡献预留 | figures/dashboard_*.png、logs/dashboard-tests.json、Vibe_Coding.md、演讲备注.md |

原始证据和变换版本详见 data/download_manifest.json、sources/mapping_source_lock.json 与 PR #5 的固定提交。姿态四元数使用 WXYZ。轨迹曲线按原始记录索引，CPU 干预图才以物理 tick × 0.004 s 表示时间。框图按列说明数据与算法的衔接。所有统计为描述性汇总，连续记录不作为独立任务试次。

附带的课程指南在本机文件名为《课程学期大作业指南.pdf》。已重新读取其中的两人分工、10–15 分钟、七个数据分析步骤、AI 记录和贡献要求。老师的四项口头要求来自本轮任务及此前文字，分别在第 3、4、12、13 页呈现。
