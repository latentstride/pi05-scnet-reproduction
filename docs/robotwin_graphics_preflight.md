# RoboTwin 图形运行条件预检查

检查日期：2026-11-01。结论：**当前尚未通过 RoboTwin 图形验收，不能以 LIBERO/CUDA 已通过代替。** 本次没有新增 GPU 作业或 GPU 机时费用，也没有安装依赖或修改既有基线容器。

## 实测结果及适用范围

此次在 SCNet 登录节点和现有 π₀.₅ CUDA 容器执行只读命令。队列查询成功，当前没有本项目正在运行的 GPU 作业；会话没有 Slurm 作业分配，也没有 NVIDIA GPU 设备。下面的失败仅描述该登录节点和当前容器，不代表 SCNet 所有 GPU 节点均不支持 Vulkan。

| 检查 | 实测结果 | 判断 |
| --- | --- | --- |
| 登录节点 Vulkan 工具 | 存在 `/usr/bin/vulkaninfo`；该旧版本不支持 `--summary`，已改用无参数调用 | 工具存在不代表驱动可用 |
| 登录节点 Vulkan 实例 | 无参数调用退出码 255：`Cannot create Vulkan instance`，`VK_ERROR_INCOMPATIBLE_DRIVER` | 当前节点未能创建实例 |
| 登录节点驱动发现 | 动态库缓存有 `libvulkan.so.1`；标准 ICD 目录未发现 JSON，未发现 NVIDIA 图形库缓存项 | 有 loader，缺少当前可用 NVIDIA Vulkan 的证据 |
| 现有容器启动 | Apptainer 1.3.4；只读模式遇 overlay 挂载错误，按既有 `--writable --cleanenv` 模式执行只读清单命令成功 | 容器能启动；这不是 Vulkan 通过 |
| 容器 Vulkan 工具/库 | `vulkaninfo` 不在 PATH；镜像标准路径未发现工具、`libvulkan.so*` 或 ICD JSON，库缓存未列出所查 NVIDIA 图形库 | 现有 CUDA 容器尚未配齐图形探测条件 |
| 容器能力变量 | `NVIDIA_DRIVER_CAPABILITIES=compute,utility` | 镜像默认值未列 graphics；该变量不决定传统 `--nv` 的全部绑定，不能单凭它判定权限 |
| Apptainer GPU 路径 | 既有运行脚本使用传统 `--nv`；安装的库清单包含 EGL/GLX/NVIDIA 图形库名称；帮助中有 `--nvccli` | 库清单和选项存在，不证明 GPU 节点实际可挂载/可用 |
| NVIDIA Container CLI | 登录节点 PATH 未找到 `nvidia-container-cli` | 尚未证明可使用 `--nvccli` 路线；不是全平台缺失结论 |
| GPU 节点和容器内 NVIDIA Vulkan | 本次无 GPU 分配，未进行 `--nv` GPU 实测 | **待验证，不能标为通过或平台不支持** |

容器清单在无 GPU 分配、未启用 `--nv` 的条件下采集，描述镜像及当前运行环境，不描述计算节点注入宿主驱动后的最终状态。`--writable` 沿用当前平台已可运行的容器模式，执行内容仅查询工具、环境变量和库；没有向容器安装或写入文件。

原始 `preflight.json`、命令日志及读取到的库清单保存在 SCNet 个人项目 `research/robotwin/preflight-20261101/`。公开仓库只保存本摘要，不公开共享账号、节点和平台内部路径。

## graphics 的验证方法

RoboTwin 文档对 Docker 要求 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`，并要求用 `vulkaninfo` 检查。NVIDIA 文档说明 graphics 对应 OpenGL/Vulkan 驱动能力；compute/utility 分别不能代替它。[RoboTwin 安装说明](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)、[NVIDIA 驱动能力说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)

当前使用 Apptainer，不能照抄 Docker 环境变量后就宣布权限已开通。传统 `--nv` 按宿主库清单绑定库，`--nvccli` 是另一条依赖 NVIDIA Container CLI 的路径。需在目标 GPU 节点核对设备、驱动库、Vulkan loader、ICD 清单和实际枚举结果；不修改平台管理员配置或课题组共享环境。[Apptainer 1.3 GPU 文档](https://apptainer.org/docs/user/1.3/gpu.html)

`nvidia-smi` 的正常退出只计作管理接口检查；`vulkaninfo` 只有识别到实际 NVIDIA GPU 才计入下一阶段，软件设备 llvmpipe/lavapipe 不计作 NVIDIA 图形通过。无显示器时，surface/display 警告本身不应与 Vulkan 实例或 GPU 枚举失败混淆。

## 继续租用之前的条件

先确认平台目标 GPU 规格/镜像支持 NVIDIA Vulkan、图形驱动透传及所需容器运行方式。当前没有可复用的 GPU 分配，因此无法在零新增 GPU 机时条件下完成计算节点实测；本次停在此处，没有为补齐检查自行提交付费作业。

有可用且已授权的目标节点分配后，按顺序验收：宿主机 Vulkan 枚举 NVIDIA GPU；独立研究容器内枚举同一目标 GPU、确认所需图形队列和扩展；SAPIEN 最小离屏场景生成有效 RGB 图像；CuRobo 单独做最小规划验证。图形和 CUDA 规划分别记录结果，全部通过后再安装完整任务环境或跑回放。后续图形依赖放在独立研究容器/目录，保留 LIBERO 冻结基线。
