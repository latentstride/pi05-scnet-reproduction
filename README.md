# π₀.₅ 官方推理复现

在 SCNet 部署 [OpenPI](https://github.com/Physical-Intelligence/openpi)，使用官方 `pi05_libero` checkpoint，逐步完成推理验证和 LIBERO 闭环基线。

当前阶段：主环境 202 个包兼容检查及 OpenPI 核心导入已通过；官方权重 16 个文件全部校验通过。LIBERO 仿真环境继续配置中，GPU 作业排队，模型加载和推理尚未验收。当前机时预算上限为 20 元，先验证环境及一次推理。后续 15 轮评测另行安排。

## 版本

- OpenPI：`215abfb217dbac7d5f1273282331b9b1866c0479`
- LIBERO：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- ALOHA：`d1dc83afd89ded4379851257fe5d85632d31d5ec`
- 模型：`gs://openpi-assets/checkpoints/pi05_libero`
- uv：0.13.6；Python：3.11.18

## 文件

- `docs/官方复现工作清单.md`：原始工作清单。
- `environment.md`：已实测环境、限制和当前进度。
- `scripts/`：部署与检查脚本。脚本的实际执行状态见环境记录。

源码、模型、容器、缓存和实验输出保存在 SCNet 持久化目录，不纳入本仓库。所有结果必须完成实际运行后再记录；空白验收项不代表已完成。

官方说明：[安装](https://github.com/Physical-Intelligence/openpi#installation)、[LIBERO 评测](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)。
