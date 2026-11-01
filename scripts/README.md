# 部署脚本

在 SCNet 项目根目录执行。当前实测结果见根目录 `environment.md`。

1. 先运行 `python3 -m pip install --target tools/uv uv==0.13.6` 安装独立 uv。然后 `bash scripts/fetch_source.sh`，`python scripts/fetch_submodules.py`：下载固定 commit 官方源码及精确子模块归档。
2. `bash scripts/install_locked.sh --prepare-only`：导出官方锁定运行依赖和下载精确 LeRobot 源码。
3. `bash scripts/install_devel.sh`：下载官方 CUDA devel 镜像，解压持久化 sandbox，安装主环境。`run_container.sh` 使用该 sandbox，并设置持久化模型缓存及节点临时字节码缓存。
4. `bash scripts/validate_core.sh`：检查全部依赖和 `pi05_libero` 核心配置导入。主环境已通过。
5. `python scripts/download_ranges.py`：分段获取官方权重，避免长连接中断；下载完成后运行 `PYTHONPATH=tools/checksum python scripts/verify_checkpoint.py` 校验官方 CRC32C / MD5。校验依赖可用 `tools/uv/bin/uv pip install --python python3 --target tools/checksum google-crc32c==1.9.0` 安装。
6. `bash scripts/install_render_libs.sh`：从 Ubuntu 官方源下载图形库，仅解包到项目目录。
7. `bash scripts/install_libero.sh`：安装独立 Python 3.8 LIBERO 环境；仿真和实际渲染仍待验收。
8. `sbatch scripts/inference.slurm`：仅提交一次。1 GPU、30 分钟硬上限，先采集真实 LIBERO 观测，再执行两次相同输入推理，分别记录首次编译和后续耗时。需先确认环境验收、GPU 单价与预算。

`inference_test.py` 也支持不传 `--observation` 使用官方合成示例；这种输出不能计为真实 LIBERO 观测推理或闭环验证。`capture_libero.py` 沿用固定版本官方客户端的 180° 图像旋转、resize-with-pad 和状态拼接规则。

`install_openpi.sh`、`pull_container.sh` 和 `install_locked.sh` 中的 runtime 路径保留用于记录最初方案；在当前集群上应使用 devel sandbox。
