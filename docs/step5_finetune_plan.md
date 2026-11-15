# 第五步 · π₀.₅ ARX-X5 原生 LoRA 监督微调

任务为 RoboTwin `handover_block`。从 PACE 的 `17ef545270cf26ed81169893b60f845f80ce6557` Orbax 权重初始化，训练中只更新 LoRA。原生输入使用 ARX 的三路 RGB 和当前驱动目标，原生输出是 50 个绝对关节与夹爪目标。实现、验收与既有 LIBERO、Aloha、投影基线和安全控制分别保存在独立路径。

- [数据转换与质量报告](step5/data_quality.md)记录拆分、时序、语言、颜色及命令与物理状态对照。
- [LoRA 配置与训练准入](step5/admission.md)记录参数、checkpoint、CPU 与 GPU 的实测边界。
- [实验矩阵、资源与恢复](step5/experiments.md)给出基线对照和单作业执行方式。
- [训练入口](../scripts/step5_arx/train.py)、[策略服务](../scripts/step5_arx/serve.py)与[单作业 Slurm 脚本](../scripts/step5_arx/train.slurm)仅用于第五步。

本轮进行 HDF5 转换、CPU 测试与运行环境核验，未提交 Slurm 作业、启动计费 GPU 或下载新大权重。Aloha 原生 Fixed H50 为 3/3、PACE 为 2/3。既有 ARX 作业 `128103655` 为 0/2，尚未形成稳定双指抓持。这些历史结果继续保留，独立于第五步冻结原生 ARX 对照。新策略的首个 GPU 更新、20 步过拟合、正式训练及闭环表现均待实测。 [Aloha 原生回执](https://github.com/latentstride/pi05-scnet-reproduction/blob/acd8568830188576f912c2ae83c28ea3013c951d/docs/g2_aloha_six_results_20261111.md)
