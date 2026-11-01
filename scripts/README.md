# 部署脚本

在 SCNet 项目根目录按以下顺序执行：

1. `bash scripts/fetch_source.sh`
2. `python scripts/fetch_submodules.py`
3. `bash scripts/pull_container.sh`
4. 官方安装方式：`bash scripts/install_openpi.sh`。当前 CUDA 基础镜像没有 Git，该方式已实测失败。
5. 针对此镜像使用 `bash scripts/install_locked.sh`：从官方 `uv.lock` 导出依赖，LeRobot 使用官方锁定的同一 commit 源码归档，再调用 `install_inside.sh`。此备用路径仍需实际安装验收。
6. `python scripts/download_checkpoint.py`：官方权重断点续传，完成后检查所有文件字节数。

模型加载、核心导入、GPU 和 LIBERO 验收状态详见根目录 `environment.md`。
