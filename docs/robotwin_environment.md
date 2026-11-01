# RoboTwin 独立研究环境

构建日期：2026-11-02。用途：SAPIEN 运动学排错、后续 Vulkan 离屏渲染与 CuRobo 最小规划。本文记录独立依赖环境，不改变 LIBERO 冻结基线，不加载 PACE 权重。

状态：构建中；最终状态以 CPU 验证 JSON 和下方验收记录为准。

## 环境布局

在个人项目根目录内创建 `research/robotwin/robotwin-cu122/`，格式为 **Apptainer sandbox 研究镜像**，不是修改或复制后继续共用 LIBERO 的虚拟环境。基础镜像是既有只读 CUDA 12.2.2 cuDNN8 devel / Ubuntu 22.04 SIF；创建了新的可写文件树。

镜像内部使用 `/opt/robotwin/`：Python 安装在 `python/`，虚拟环境在 `venv/`，源码在 `src/`，原生系统库在 `sysroot/`，缓存及编译临时文件也在该目录。它们在宿主物理上均位于个人 `research/robotwin/robotwin-cu122/` 下面，不使用课题组共享 conda 或 LIBERO `.venv`。

SCNet 当前已验证的容器模式为 `--writable --cleanenv`，运行入口统一使用 [run_robotwin_env.sh](../scripts/run_robotwin_env.sh)。镜像依赖 NVIDIA 宿主驱动通过后续 `--nv` 注入；镜像内不安装或修改宿主驱动。

## 固定依赖与源码

| 项目 | 固定值 |
| --- | --- |
| Python | 3.10.19 |
| PyTorch / torchvision | 2.4.1+cu121 / 0.19.1+cu121 |
| SAPIEN | 3.0.0b1 |
| CuRobo | 0.7.8；提交 `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` |
| PyTorch3D | 0.7.8；提交 `75ebeeaea0908c5527e7b1e305fbc7681382db47`（上游标签 `V0.7.8`） |
| NumPy / SciPy | 1.26.4 / 1.10.1 |
| MPLib / Open3D / Warp | 0.2.1 / 0.18.0 / 1.12.0 |
| RoboTwin | `13c3c47ff4312dd62484bcd51be034af55c062d1` |
| CUDA 扩展目标 | `8.9+PTX`，对应此前使用的 RTX 4090；更换 GPU 架构需另行核验或重编译 |

Python 与 PyTorch 是不同组件；采用上述版本是为了匹配所选 RoboTwin 源码依赖。系统原生依赖从 Ubuntu Jammy 软件仓库获取，在镜像内的私有 sysroot 解包，包括 Vulkan loader、`vulkaninfo`、EGL/GLVND 和 X11/Wayland 库。NVIDIA ICD 发现配置已准备，但该配置本身不证明设备或 graphics 透传可用。

实际导入检查发现基础 CUDA 镜像还缺少 `libbsd0`/`libmd0` 和 Open3D 使用的 LLVM libc++ 运行库，已在私有 sysroot 补齐，并加入安装脚本。SAPIEN URDF loader 按上游安装流程显式使用 UTF-8。运行入口将 OpenMP、OpenBLAS、MKL 线程数固定为 1，控制共享登录节点上的 CPU 使用。

来源：[RoboTwin 固定依赖](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/requirements.txt)、[RoboTwin 安装脚本](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/_install.sh)、[PyTorch3D 发布记录](https://github.com/facebookresearch/pytorch3d/releases/tag/V0.7.8)、[CuRobo 构建源码](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/setup.py)。精确约束和来源见 [constraints.txt](../research/robotwin/env-build/constraints.txt)、[source_manifest.json](../research/robotwin/env-build/source_manifest.json)。

## 使用和复现

从个人项目根目录执行：

```bash
# 使用已有研究镜像；这两条命令不申请 GPU。
bash scripts/run_robotwin_env.sh python -c 'import sapien, torch; print(torch.__version__)'
bash scripts/run_robotwin_env.sh vulkaninfo --help

# 在同一目录复建独立镜像（安装/编译耗时，不需要 GPU）。
bash scripts/build_robotwin_env.sh
```

运行入口的工作目录为镜像内 `/opt/robotwin/src/RoboTwin`。显式执行项目脚本时使用项目的绝对路径；Python 位于 `/opt/robotwin/venv/bin/python`。

完整依赖安装器为 [robotwin_env_inside.sh](../scripts/robotwin_env_inside.sh)，验证器为 [validate_robotwin_env.py](../scripts/validate_robotwin_env.py)。构建只取源码和软件依赖，未下载 RoboTwin 全量任务资产或 PACE 模型参数；加载真实任务仍需对应资产。

## 验收记录

| 检查 | 当前结果 |
| --- | --- |
| Python 包兼容检查 | 已通过；当前已装 272 个包 |
| CuRobo 五个 CUDA 扩展 | 编译、安装及实际导入通过 |
| SAPIEN CPU PhysX 场景 | 已通过创建、步进 |
| Vulkan loader | `ctypes.CDLL('libvulkan.so.1')` 已通过 |
| Open3D / MPLib | 实际导入通过 |
| PyTorch3D | CUDA/CPU 扩展编译中，尚未验收 |
| 完整验证 JSON、依赖冻结清单 | 等待 PyTorch3D 安装完成后生成 |

GPU 可见性、NVIDIA Vulkan 枚举、SAPIEN RGB 渲染和 CuRobo 实际 CUDA 规划属于后续短时 GPU 验收，本次不标为通过。既有 GPU Slurm 模板还需与本镜像的可写运行方式、库路径及设备选择核对后再提交。

CuRobo 0.7.8 的 `MotionGen` 在类方法默认参数中调用 `Pose.from_list()`，导入模块时便尝试创建 CUDA tensor。登录节点实际导入到此处报 `Found no NVIDIA driver`；因此 CPU 验证单独导入五个已编译 CUDA 扩展，并检查配置文件，`MotionGen` 模块及其运行行为留待 GPU 分配后验收。这项边界记录在验证 JSON 中，未通过更改上游实现或模拟 GPU 来跳过。
