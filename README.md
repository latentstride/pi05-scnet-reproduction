# π₀.₅ 官方复现与异构数据接口研究

G2 当前结果（2026-11-06）见 [作业 127054052 报告](docs/pace_g2_run_127054052.md)。八次真实策略试验均未成功，Aloha 四次到达 300 动作上限，ARX 四次因碰撞筛选相关错误提前终止。实现、测试和配置已归档，后续继续分析失败原因。

**阶段一已通过功能性验收，进入阶段二数据集调研。** 使用官方 `pi05_libero` checkpoint，完成真实 LIBERO 观测推理、完整闭环和三个任务各五轮评测，15/15 成功，轨迹与视频核验通过。这是受限功能性基线。

- [验收结果](acceptance_report.md)、[逐轮结果](evaluation_results.csv)、[评测报告](evaluation_report.md)
- [基线冻结记录](baseline_freeze.md)：固定第一阶段版本与证据，不重复安装或重跑相同 15 轮。
- [数据集比较与选型](dataset_comparison.md)：RoboTwin 为主候选，保留 DROID、BridgeData V2、ALOHA Sim 的早期接口调研。
- [RoboTwin 可行性审计与实验方案](robotwin_feasibility.md)：优先做同任务跨本体的专家轨迹和 qpos/ee 控制适配，区分回放与模型自主控制。
- [SCNet Vulkan/graphics 实测预检查](docs/robotwin_graphics_preflight.md)：登录节点和现有容器已检查，GPU 图形透传仍待目标节点验证。
- [独立 RoboTwin 研究环境](docs/robotwin_environment.md)：镜像已建立，273 个包兼容、六个原生扩展及 CPU 最小物理验收通过，LIBERO 基线保留。
- [SAPIEN 静态 FK 与关节坐标系验证](docs/robotwin_sapien_fk_smoke.md)：两种本体 9 个快照、12 次末端和 219 次关节坐标系检查通过。按用户要求暂不执行 GPU 验收，不下载 PACE 权重。

已完成 handover_block 的 Aloha-AgileX 与 ARX-X5 各 3 条官方示范读取、完整静态检查及 SCNet CPU 复验，见 [数据抽样报告](robotwin_data_probe.md)。发现原始 HDF5 没有显式时间戳或单独动作命令字段，回放前须解决时间与工具点语义。当前不新增 GPU 实验；现有 `pi05_libero` 的双臂模型兼容性独立验收。

## 基线版本

- [OpenPI](https://github.com/Physical-Intelligence/openpi)：`215abfb217dbac7d5f1273282331b9b1866c0479`
- LIBERO：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- 模型：`gs://openpi-assets/checkpoints/pi05_libero`
- Python 3.11.18，JAX/JAXLIB 0.5.3，PyTorch 2.7.1，NumPy 1.26.4。

模型加载 12.42 秒，首次推理含编译 42.24 秒，相同输入第二次推理 0.101 秒；见 [推理验证](docs/inference_validation.md)。历史进度见 [environment.md](environment.md)，最终验收状态以验收报告为准。

`scripts/` 保存部署与验收脚本，`docs/官方复现工作清单.md` 为原始清单。运行源码、环境、模型、容器、缓存、日志及原始结果统一放在项目所属人的 SCNet 个人目录。公开文档以 `<PROJECT_ROOT>` 表示部署位置；原始视频、轨迹、模型和平台内部信息不纳入公开仓库。
