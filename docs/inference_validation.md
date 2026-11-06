# π₀.₅ 真实 LIBERO 观测推理验证

日期：2026-11-01。环境与一次推理的阶段目标已完成。

固定 OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`，使用官方 `pi05_libero` checkpoint。权重 16 个文件全部通过 CRC32C 校验。

输入来自真实 `libero_spatial` 场景的任务 0、初始状态 0，采集 seed 为 7。任务指令为 “pick up the black bowl between the plate and the ramekin and place it on the plate”。初始化后发送 10 步官方等待动作，读取相机图像与机器人状态。两路图像按官方客户端规则旋转 180°、补边缩放到 224×224，类型 uint8；状态为 8 维。

| 检查项 | 结果 |
| --- | --- |
| 作业退出状态 | COMPLETED，退出码 0:0 |
| 模型加载 | 12.4185 秒 |
| 首次推理 | 42.2381 秒，含首次编译开销 |
| 相同输入第二次推理 | 0.100954 秒 |
| 动作形状 | 10×7 |
| 动作最小值 / 最大值 | -1.0004114283711911 / 0.9544639779864547 |
| 数值检查 | 全部有限，无 NaN / Inf |
| 输入与动作归档 | `results/inference/observation_and_actions.npz` |
| 完整报告 | `results/inference/report.json` |

输出取自官方 `create_trained_policy(...).infer(...)['actions']`，通过配置的输出变换生成。以上是同一观测的两次实测耗时，不代表长期平均延迟。

主环境版本：Python 3.11.18、JAX/JAXLIB 0.5.3、PyTorch 2.7.1、NumPy 1.26.4。所有部署资产与原始输出保存在项目所属人的个人目录；公开记录不包含共享账号路径、节点、驱动或账单信息。

本次完成了真实观测到动作的推理验证。尚未执行预测动作的完整闭环 episode，也未完成 15 轮受限评测。

