# π₀.₅ 官方复现与异构数据接口研究

**阶段一已通过功能性验收，进入阶段二数据集调研。** 使用官方 `pi05_libero` checkpoint，完成真实 LIBERO 观测推理、完整闭环和三个任务各五轮评测，15/15 成功，轨迹与视频核验通过。这是受限功能性基线。

- [验收结果](acceptance_report.md)、[逐轮结果](evaluation_results.csv)、[评测报告](evaluation_report.md)
- [基线冻结记录](baseline_freeze.md)：固定第一阶段版本与证据，不重复安装或重跑相同 15 轮。
- [数据集比较与选型](dataset_comparison.md)：DROID、BridgeData V2、ALOHA Sim 的本体、观测、状态、动作、坐标系及仿真兼容性。

当前不新增 GPU 实验。下一步先以 CPU 小样本核对数据版本和接口。Panda 上的成功不代表现有 checkpoint 可直接控制 WidowX 或 ALOHA；格式接入、控制器兼容和策略有效性分别核验。

## 基线版本

- [OpenPI](https://github.com/Physical-Intelligence/openpi)：`215abfb217dbac7d5f1273282331b9b1866c0479`
- LIBERO：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- 模型：`gs://openpi-assets/checkpoints/pi05_libero`
- Python 3.11.18，JAX/JAXLIB 0.5.3，PyTorch 2.7.1，NumPy 1.26.4。

模型加载 12.42 秒，首次推理含编译 42.24 秒，相同输入第二次推理 0.101 秒；见 [推理验证](docs/inference_validation.md)。历史进度见 [environment.md](environment.md)，最终验收状态以验收报告为准。

`scripts/` 保存部署与验收脚本，`docs/官方复现工作清单.md` 为原始清单。运行源码、环境、模型、容器、缓存、日志及原始结果统一放在项目所属人的 SCNet 个人目录。公开文档以 `<PROJECT_ROOT>` 表示部署位置；原始视频、轨迹、模型和平台内部信息不纳入公开仓库。
