# 部署脚本

在 SCNet 项目根目录运行，按 fetch_source.sh → fetch_submodules.py → pull_container.sh → install_openpi.sh 顺序执行。download_checkpoint.py 支持官方权重断点续传。

当前 install_openpi.sh 的原始官方安装路径在 CUDA 基础镜像内遇到 Git 缺失，尚未通过依赖和导入验收。详见 environment.md。
