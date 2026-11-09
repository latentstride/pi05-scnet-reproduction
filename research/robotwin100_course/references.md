# 参考文献与开源代码

访问核对日期：2026-11-09（北京时间）。技术论断优先论文与作者仓库。固定commit优先于浮动main。

[1] Open X-Embodiment Collaboration. **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**. arXiv:2310.08864, 2023 / ICRA 2024. https://arxiv.org/abs/2310.08864 。多机器人数据与正迁移。并非本项目冻结模型FK/IK迁移的直接验证。

[2] Black K, et al. **π₀.₅: a Vision-Language-Action Model with Open-World Generalization**. CoRL 2025, PMLR 305:17–40. https://proceedings.mlr.press/v305/black25a.html ，https://arxiv.org/abs/2504.16054 。异构共训练、开放世界泛化。

[3] **RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation**. arXiv:2506.18088, 2025. https://arxiv.org/abs/2506.18088 。官方代码 https://github.com/RoboTwin-Platform/RoboTwin ，任务 https://robotwin-platform.github.io/doc/tasks/handover_block.html 。论文平台规模不能替代本项目实际样本量。

[4] Nie J, Li J, Zhang J, et al. **PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking**. arXiv:2606.00537v1, 2026. https://arxiv.org/html/2606.00537v1 。执行窗口控制、方法适用边界。本文未声称复现PACE论文总体成功率，也未把本项目K31称作PACE自适应窗口。

[5] Physical Intelligence. **OpenPI**. https://github.com/Physical-Intelligence/openpi 。LIBERO项目固定版本215abfb217dbac7d5f1273282331b9b1866c0479。

[6] 作者开源实现 PACE @ fd2e624cc19325cc9b20a620c82d51c0d8e25cc1：https://github.com/niejunnan25/PACE/tree/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1 。Openpi-PACE服务 f4e52e6de4cc487a11585a3a7414d610b2883dae；训练来源 ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12，二者角色分开。

[7] RoboTwin任务模型卡：https://huggingface.co/niejunnan25/pi05-robotwin/resolve/17ef545270cf26ed81169893b60f845f80ce6557/README.md 。官方HDF5没有固定时间戳，不把转换器声明的50 fps当作原始采样间隔。

[8] 官方数据：https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/3dc3b798668feb99ac61cc9086d84cbcc3d79186/dataset/handover_block 。本项目抽样的clean_50归档不能认作checkpoint训练集逐条字节相同。

[9] 项目仓库：https://github.com/latentstride/pi05-scnet-reproduction 。Draft PR [#2](https://github.com/latentstride/pi05-scnet-reproduction/pull/2)、[#3](https://github.com/latentstride/pi05-scnet-reproduction/pull/3)、[#4](https://github.com/latentstride/pi05-scnet-reproduction/pull/4)、[#5](https://github.com/latentstride/pi05-scnet-reproduction/pull/5)。PR5读取固定686c86c8dc838dd06df744aaf763db14a872aa66。

[10] Zhao T Z, Kumar V, Levine S, Finn C. **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**. Robotics: Science and Systems, 2023. arXiv:2304.13705. https://arxiv.org/abs/2304.13705 。提出Action Chunking with Transformers（ACT），为动作分块与执行时序提供直接相关背景。它的控制实验不等于本项目冻结π₀.₅与ARX迁移验证。

本轮重新取得的一级来源快照：`sources/oxe_paper.html`、`robotwin_paper.html`、`pace_paper.html`、`act_paper.html`。π₀.₅全文从项目原始PDF重新提取，见pi05_paper_extracted.txt；部署架构同时对照固定官方源码openpi_pi0.py。文献表中的“证据适用边界”是对本项目使用范围的判断，不是声称原论文从未研究相关领域问题。

## 本轮新增证据

- [固定 RoboTwin 数据版本](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/3dc3b798668feb99ac61cc9086d84cbcc3d79186/dataset/handover_block)
- [Draft PR #5](https://github.com/latentstride/pi05-scnet-reproduction/pull/5)。
- [127192740 结项和安全退出审计（固定提交 0467a96）](https://github.com/latentstride/pi05-scnet-reproduction/blob/0467a96258a2e3982196ae928a153ed85bddce8d/docs/g2_candidate_stop_127192740.md)
- [后续 CPU 归因审计（固定提交 ccdf121）](https://github.com/latentstride/pi05-scnet-reproduction/blob/ccdf1210327d6a69190bf1b88a9dcfbcb02a96f3/docs/g2_second_query_root_cause_127192740.md)
