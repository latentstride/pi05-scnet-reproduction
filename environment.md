# SCNet 环境记录

检查日期：2026-11-01（北京时间）。

## 已验证

| 项目 | 结果 |
| --- | --- |
| 区域 / 分区 | 华东一区（昆山）/ `ksagnormal01` |
| 分区标注 GPU | RTX4090-PCIE-24GB-LS；实际分配卡、驱动待作业运行核验 |
| GPU 单价 | 当日平台生效订单显示 3.60 元 / 卡时 |
| 本次预算 | 总上限 20 元，先完成环境和一次推理 |
| 宿主系统 | CentOS 7，glibc 2.17；登录节点无分配 GPU |
| 持久化目录 | SCNet 用户主目录下 `project/pi05` |
| 共享盘 | 最近 `df` 剩余约 115 GB；不是个人配额证明 |
| Apptainer | `/public/software/apps/apptainer/1.3.4/bin/apptainer` |
| 工作容器 | NVIDIA 官方 `cuda:12.2.2-cudnn8-devel-ubuntu22.04`，持久化 sandbox |
| uv / 主环境 Python | 0.13.6 / CPython 3.11.18 |
| 主环境 JAX / PyTorch | 0.5.3 / 2.7.1+cu126 |
| 主环境依赖检查 | 202 个包，`All installed packages are compatible` |
| OpenPI 核心导入 | `CORE_IMPORT_OK pi05_libero 0.5.3 2.7.1+cu126` |
| 官方 checkpoint | 16 个文件，12,439,085,481 字节；全部 CRC32C 验证通过，带 MD5 的 10 个文件亦通过 |
| tokenizer | `big_vision/paligemma_tokenizer.model`，4,264,023 字节，已缓存 |
| LIBERO 独立 Python | CPython 3.8.20，环境 `.venv-libero`，依赖安装及仿真验收进行中 |

核心导入通过不等于 CUDA、模型加载或推理通过。当前尚无 GPU 推理成功记录。

## 固定源码

- OpenPI：`215abfb217dbac7d5f1273282331b9b1866c0479`
- LIBERO：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- ALOHA：`d1dc83afd89ded4379851257fe5d85632d31d5ec`
- LeRobot：`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`
- checkpoint：`gs://openpi-assets/checkpoints/pi05_libero`

GitHub 直连与 Git 克隆超时，API 和 codeload 可用。使用官方固定 commit 归档，并通过 Git tree API 读取精确子模块 gitlink；未创建完整 Git 历史，未完成原生 git submodule 初始化。

## 可用的安装路径

官方 `uv sync --frozen` 在基础镜像内因缺 Git 失败。改为从同一官方 `uv.lock` 导出版本固定的运行依赖，LeRobot 使用锁定 commit 的官方源码归档，安装 199 个依赖及 3 个 editable 项目。主环境依赖检查与核心导入均已通过。

runtime 镜像缺编译头文件，构建 `evdev` 失败；已改用预装工具链的官方 devel 镜像。当前账号无 fakeroot/subuid 支持，没有修改宿主机权限或原有 conda 环境。共享文件系统的只读 sandbox overlay 不可用，因此运行时使用 `--writable --cleanenv --bind /public:/public`。

将 SIF 一次解压为 `containers/pi05-devel` 后重复使用，避免每次执行重新解包。字节码缓存位于节点 `/tmp`，避免共享盘上大量小文件写入。

主环境 OpenCV 额外检查发现缺少 `libGL.so.1`。正在通过 Ubuntu 官方源下载图形运行库，再解包至项目的 `sysroot/root`；不在宿主机安装系统包。实际渲染尚待验收。

## 验收命令与记录

从 SCNet 项目根目录运行：

```bash
bash scripts/validate_core.sh
# 已通过，日志 logs/core_validation.log
# 版本快照 logs/installed_packages.txt

bash scripts/install_libero.sh
# 独立仿真环境；尚未完成验收

# GPU 分配后使用 --nv；设置由 run_container.sh 统一处理
PI05_GPU=1 bash scripts/run_container.sh .venv/bin/python -c 'import jax; print(jax.devices())'

sbatch scripts/inference.slurm
# 1 GPU，30 分钟硬上限；先采集真实 LIBERO observation，再加载模型与推理
```

checkpoint 验证记录保存在 SCNet 的 `results/checkpoint_verification.json`，末行日志为 `ALL_CHECKPOINT_HASHES_VERIFIED 16`。权重、容器、缓存、原始观测和私有平台日志不上传公开仓库。

GPU 探测作业 `126524136` 已提交，当前仍为 `PD (Priority)`，限制 3 分钟，按 GPU 单价计算最多 0.18 元。推理作业脚本限制 30 分钟，按 GPU 单价计算最多 1.80 元；平台其他计费项目以实际账单为准。作业预检给出远期排程估计，尚不能确认本次何时分配 GPU。

## 待验收

- [x] 固定源码、官方锁定运行依赖及核心导入
- [x] 完整官方权重、CRC32C / MD5 校验、tokenizer 缓存
- [x] 核实 GPU 单价并准备限时作业
- [ ] GPU 驱动、JAX CUDA 识别与计算
- [ ] LIBERO 依赖完整性、场景加载、图像和状态读取
- [ ] 模型加载、真实观测推理与动作数值检查
- [ ] 策略服务通信、完整闭环 episode、15 轮受限评测

原清单中的闭环和 15 轮评测尚未完成；本次先完成用户限定的环境与一次推理。
