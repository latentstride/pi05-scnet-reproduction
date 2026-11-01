# SCNet 环境记录

检查日期：2026-11-01（北京时间）。

| 项目 | 已观察结果 |
| --- | --- |
| 区域 | 华东一区（昆山） |
| 登录节点 | login08；不是已分配的 GPU 节点 |
| 系统 | CentOS 7，glibc 2.17 |
| 原有 Python / Git | 3.10.14 / 1.8.3.1 |
| Slurm GPU 分区 | ksagnormal01，RTX4090-PCIE-24GB-LS |
| 共享盘 | df 显示总计 450 GB、剩余约 168 GB；个人配额未核实 |
| 项目目录 | SCNet 用户主目录下 `project/pi05` |
| 容器运行时 | `/public/software/apps/apptainer/1.3.4/bin/apptainer`，版本命令成功 |
| 新装 uv | 0.13.6，独立安装至 `tools/uv`，版本命令成功 |
| 新装 Python | uv 报告已安装 CPython 3.11.18；容器内运行待验证 |
| CUDA 容器 | NVIDIA 官方 `nvcr.io/nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04`，SIF 约 2.0 GB，构建成功 |

## 网络与部署选择

GitHub 网站直连及 `git clone --recurse-submodules` 超时。GitHub API 和 codeload 可访问，已下载固定 commit 的官方源码，并记录归档 SHA256。子模块通过 Git tree API 读取精确 gitlink SHA，再下载官方归档；这种方式不提供完整 Git 历史，不能声称完成了原生 git submodule 初始化。

PyPI 可访问，uv 安装成功。原有 conda 环境未修改。

CentOS 7 的宿主机环境与官方 Ubuntu 运行环境不同，因此使用 Ubuntu 22.04 容器提供运行库。第一次容器拉取在 180 秒检查超时后中断，复用缓存重新构建后生成了约 2.0 GB 的 SIF。已实际执行容器内 `cat /etc/os-release`，返回 Ubuntu 22.04.3 LTS。当前运行时没有 squashfuse，每次执行会将 SIF 解压至临时 sandbox；日志中的这个提示不等于执行失败。

LIBERO 和 ALOHA 的精确版本归档均已下载并解压。官方 checkpoint 清单共 16 个文件，合计 12,439,085,481 字节，清单没有分页；已启动断点续传下载，但尚未完成，不能进行模型加载验收。

容器 SHA256：`c6690adbbcccd0c19a31b6d25fe22dcb9e28984b27d700fb7dda7e03a11454a7`。

官方 `uv sync --frozen` 已实测因镜像内缺少 Git 而失败。备用安装保留官方锁文件的依赖版本，以精确 commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` 的 LeRobot 源码归档代替 Git 下载；已导出依赖并启动备用安装，导入验收尚未通过。

第一次权重传输下载了部分文件后遇到 curl 56 连接中断，已启动断点续传重试。尚未有完整权重验收结果。

最后检查：续传也以 curl 56 结束，权重仍不完整；CUDA devel 镜像流程仍在后台运行。共享文件系统剩余约 146 GB。未提交新的 GPU 作业。

备用安装解析了 199 个依赖，但在构建 `evdev` 时因缺少 Linux 头文件而失败。已将 SIF 转为可写 sandbox，尝试在容器内补齐系统依赖；集群拒绝 fakeroot：当前用户不在 `/etc/subuid` 中，且运行时没有可用的 fakeroot 支持。未修改宿主机权限或安装系统包。

当前阻碍是 runtime 镜像缺少编译依赖，而此节点无法在容器内通过 fakeroot 安装系统包。已启动预装编译工具和头文件的官方 NVIDIA CUDA devel 镜像方案：`scripts/install_devel.sh`，远端日志为 `logs/install_devel.log`，结果尚未验收。本次仍未达到 OpenPI 环境验收标准。

## 验收状态

- [x] 登录节点、存储、调度器和软件运行时初查
- [x] 固定 OpenPI 源码版本并下载归档
- [x] uv 安装与版本检查
- [x] 完成并验证 Ubuntu CUDA 容器（操作系统执行验证；GPU 透传待验证）
- [ ] 完成官方依赖安装及核心模块导入
- [ ] 确认 GPU 单价，按 20 元总上限设置作业时间
- [ ] 验证计算节点的驱动、GPU、JAX CUDA 兼容性
- [ ] 下载官方权重并验证一次推理
- [ ] LIBERO 真实 observation、闭环与 15 轮评测

本次尚未提交新的 GPU 作业。账号中原有 Gaussian 实验的排队作业不属于本项目。
