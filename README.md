# π₀.₅ 官方推理复现

使用 [OpenPI](https://github.com/Physical-Intelligence/openpi) 和官方 `pi05_libero` checkpoint，验证推理及 LIBERO 仿真基线。

主环境 202 个包兼容检查及核心导入通过，迁移至个人项目目录后复验通过。官方权重 16 个文件校验通过，tokenizer 离线编码通过。独立 LIBERO 环境 125 个包兼容检查通过，已成功采集真实场景图像与机器人状态。模型推理、完整闭环及 15 轮评测尚未完成。

账号由团队共用：源码、环境、Python 安装、权重、容器、缓存、日志和结果必须放在项目所属人的个人目录。公共文档使用 `<PROJECT_ROOT>`，具体平台路径和运行记录留在私有部署笔记。

- OpenPI：`215abfb217dbac7d5f1273282331b9b1866c0479`
- LIBERO：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- ALOHA：`d1dc83afd89ded4379851257fe5d85632d31d5ec`
- 模型：`gs://openpi-assets/checkpoints/pi05_libero`

`environment.md` 记录验证状态，`scripts/` 保存部署和验收脚本，`docs/官方复现工作清单.md` 为原始清单。模型、容器、缓存和原始实验输出不上传此仓库。
