# RoboTwin 100 条专家轨迹 CPU 离线分析

浏览器双击 [dashboard/index.html](dashboard/index.html) 即可离线使用。网页需要旁边的 ECharts、data.js 和预览图目录。课堂 PPT 与发布压缩包在 output/，14 页预览在 previews/。本地原始 HDF5 在 raw/，可用固定版本下载脚本独立取得。

- [中文报告](REPORT.md)
- [证据索引](evidence_index.md)
- [参考文献](references.md)
- [运行与调试记录](Vibe_Coding.md)
- [100 条轨迹质量表](data/trajectory_quality.csv)
- [全部单臂 IK 尝试](data/ik_results_full.csv)
- [独立校验](logs/independent-verification.json) 和 [浏览器实测](logs/dashboard-tests.json)

在本目录运行。需要 Python 3.12、NumPy、h5py、Pillow 和 Matplotlib。绘图默认使用本机中文字体，可按系统替换。原始 HDF5 已具备时可以直接从 analyze100.py 开始。

```sh
python -m pip install -r requirements.txt
python code/download100.py
python code/analyze100.py
IK_SCOPE=full python code/map100.py --frames 0
python code/build_metrics.py
python code/figures.py
python code/test_pipeline.py
python code/verify_results.py
```

网络使用系统 CA 的环境可设置 ROBOTWIN_CA_BUNDLE 为受信任的 PEM。下载日志按成员持久化，重新运行会以 ZIP CRC 及长度检查原件并复用。已生成的 episode 和 mapping 输出可续跑。复现实验更改求解参数时使用新的独立输出目录，避免使用旧的逐轨迹缓存。

PPT 生成使用 Codex 运行时的 @oai/artifact-tool。浏览器验证使用 Playwright 和本地 Chrome，运行脚本为 build/test_dashboard.mjs。当前输出保留在本地，GitHub 版本包括复现代码、图表、主要结果与离线网页。下载清单和索引支持核对原件。

全部步骤使用 CPU。模型作业结果和 CPU 动力学是已归档证据，按原各自分母报告。
