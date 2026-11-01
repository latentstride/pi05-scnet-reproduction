# π₀.₅ 官方推理复现

使用 [OpenPI](https://github.com/Physical-Intelligence/openpi) 和官方 `pi05_libero` checkpoint，验证推理及 LIBERO 仿真基线。

**环境部署与真实 LIBERO 观测推理已通过。** 主环境 202 个包兼容检查及核心导入通过；官方权重 16 个文件校验通过；独立 LIBERO 环境 125 个包兼容检查、场景加载与真实观测采集通过。模型生成 10×7 动作序列，全部有限。完整闭环及 15 轮受限评测已完成：三个任务各 5 轮，15/15 成功，轨迹与视频均通过核验。详见 [评测报告](evaluation_report.md) 和 [逐轮结果](evaluation_results.csv)。

| 实测项目 | 结果 |
| --- | --- |
| 模型加载 | 12.42 秒 |
| 首次推理（含编译） | 42.24 秒 |
| 相同输入第二次推理 | 0.101 秒 |

详见 [推理验证记录](docs/inference_validation.md) 与 [环境记录](environment.md)。

账号由团队共用：源码、环境、Python 安装、权重、容器、缓存、日志和结果归入项目所属人的个人目录。公共文档使用 `<PROJECT_ROOT>`，实际平台路径与原始运行记录保存在私有部署笔记。

- OpenPI：`215abfb217dbac7d5f1273282331b9b1866c0479`
- LIBERO：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- ALOHA：`d1dc83afd89ded4379851257fe5d85632d31d5ec`
- 模型：`gs://openpi-assets/checkpoints/pi05_libero`

`scripts/` 保存部署和验收脚本，`docs/官方复现工作清单.md` 为原始清单。模型、容器、缓存和原始实验输出不上传此仓库。
