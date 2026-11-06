# RoboTwin SCNet 线上核查

2026-11-03，北京时间。仅查询现有账本、节点和文件，复查已部署脚本和本地测试。GPU 验收仍未通过。

| 作业 | 节点 | Slurm 分配 | 运行 | 原始失败 |
| --- | --- | --- | --- | --- |
| 126598641 | gnode95 | 1 GPU、4 CPU、16 GB | 2 秒 | NVIDIA 查询返回了卡和驱动 535.146.02。宿主 `vulkaninfo` 命令缺失，退出 127。 |
| 126663358 | gnode97 | 1 GPU、4 CPU、16 GB | 1 秒 | `nvidia-smi -i <作业脚本选出的编号>` 返回 `No devices were found`，退出 6。 |

两次作业的 `AllocTRES` 和各自 batch 步都记录了 `gres/gpu=1`。第二次没有运行无筛选的 `nvidia-smi -L`，也没有记录运行时 `SLURM_JOB_GPUS`、`CUDA_VISIBLE_DEVICES` 或可见设备文件，因此历史数据无法区分编号映射问题、设备隔离问题和节点驱动故障。当前两个节点都登记了 8 张 RTX 4090，但节点状态快照仅能说明当前调度登记状态。Slurm 配置使用 `task/cgroup`，当前账户未取得计算节点作业内的具体设备隔离规则。查询时本项目验收队列中没有活动作业。

第一台节点的 535.146.02 驱动高于 CUDA 12.2 Update 2 文档所列 Linux 配套版本 535.104.05。第二台节点的实际驱动未由这次失败日志取得。这个版本比对仅覆盖 CUDA 的基础兼容性。 [NVIDIA CUDA 12.2 发布说明](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-toolkit-release-notes/index.html)

SCNet 登录节点能看到 NVIDIA CUDA 12.1 和 12.2 的模块目录。平台说明 GPU 设备应在计算节点检查。Apptainer `--nv` 提供 CUDA 设备和基础驱动绑定，Vulkan 验收仍要取得目标节点真实的 NVIDIA ICD 和 graphics 库。SCNet 的公开图形桌面说明也没有给出这个批处理队列中 Apptainer 离屏 Vulkan 的可用性。 [SCNet GPU 使用说明](https://www.scnet.cn/help/docs/mainsite/hpc/cmd/skills/) · [Apptainer GPU 支持](https://apptainer.org/docs/user/1.3/gpu.html)

修订 `7ec62bb` 的 SCNet 文件 SHA-256 为 `e081b109747bd686d45e817249125bbc5114a5e3517e0fac31ec7a978f3fdc7c`。在线 shell 语法和随镜像携带的 `vulkaninfo --help` 启动检查通过。本地模拟验证了 CUDA 单卡 PCI 选择、零卡/多卡拒绝和初始化错误记录，本轮再次核对 Vulkan UUID 相同通过、不同或缺失时拒绝。Vulkan 驱动枚举、容器透传、SAPIEN RGB 和 CuRobo 规划均缺少实机结果。

## 再分配 GPU 之前需要平台确认的证据

- 请 SCNet 支持根据作业 `126663358` 核查 gnode97 当时的 GRES 设备映射、作业内 `/dev/nvidia*` 访问权限和 NVIDIA 驱动状态。需要同一单卡作业环境内无筛选的 `nvidia-smi -L`，以及 CUDA 可见设备数量和 PCI/UUID 作为对照。
- 请确认昆山 RTX 4090 队列是否提供 NVIDIA Vulkan ICD、对应 graphics 驱动库，以及 Apptainer 作业内离屏 Vulkan 的支持方式。计算设备和图形能力分别确认。
- 当前验收脚本在图形阶段失败时会停止，因而同次分配无法得到后续 CuRobo 结果。若获准再实测，应先把计算和图形验收拆成同一分配中的独立检查，各自保留退出码和日志。

SCNet 原始审计日志位于个人项目目录的 `research/robotwin/env-build/logs/gpu-accounting-audit-20261103.txt`、`gpu-node-audit-20261103.txt` 和 `gpu-online-audit-20261103.txt`。本文件包含作业和节点信息，留在个人工作区。

## 作业 126711883 实机复测（2026-11-03）

SCNet 历史作业记录：22:30:27 提交，22:39:25 结束，状态失败，运行 4 分 29 秒（北京时间）。按单卡 3.60 元/小时估算约 0.27 元，以账单为准。此次复测使用 `0425156`，未再提交作业。

| 阶段 | 退出码 | 日志证据 |
| --- | --- | --- |
| CUDA 分配识别 | 通过 | CUDA 得到了单卡 PCI/UUID，作业继续进入规划和图形检查。 |
| CuRobo | 124 | `motion.warmup` 的轨迹种子输入为 `[28, 6]`，`get_seed_set` 要求三维或四维并抛出 `ValueError`；外层达到 240 秒时限，没有生成规划轨迹。 |
| NVIDIA 图形驱动发现 | 0 | 找到宿主 ICD 和驱动库，生成 38 条库绑定。 |
| 宿主 Vulkan | 0 | Vulkan UUID 核验通过宿主侧检查。 |
| 容器 Vulkan | 1 | Vulkan loader 无法从绑定的 `libGLX_nvidia.so.0` 获取 `vkCreateInstance`，报 `ERROR_INCOMPATIBLE_DRIVER`。具体动态库依赖或初始化失败原因尚未定位。 |
| Vulkan UUID | 1 | 容器未枚举分配到的同一 NVIDIA UUID。 |
| SAPIEN 渲染 | 1 | `import sapien` 的 `_ensure_egl_icd` 找不到 `/usr/share/glvnd/egl_vendor.d`，未生成 RGB。 |

原始日志保存在 `$HOME/code_cpf/pi05/research/robotwin/graphics-126711883/`，以 `check_status.tsv`、`curobo.log`、`container_vulkan.txt`、`vulkan_identity.log` 和 `sapien.log` 为准。这次已有宿主图形通过证据，前文要求先确认宿主图形的建议应以本节为准。当前应暂停重跑：先修复 EGL 配置和 CuRobo 预热，查清容器 ICD 依赖，再决定是否需要付费复测。

## 旧 GLX 报错的离线核对

对照 [Khronos Vulkan Loader 源码](https://github.com/KhronosGroup/Vulkan-Loader/blob/v1.3.204/loader/loader.c#L1312-L1323)，旧日志对应 ICD 已被打开、加载器已找到 `vk_icdGetInstanceProcAddr`，但请求 `vkCreateInstance` 时得到空指针。镜像内已核实存在 `libX11.so.6`、`libGLX.so.0` 和 `libEGL.so.1`，不能把问题直接归结为镜像缺少 X11。旧作业保留的是绑定路径，实际 NVIDIA GLX/EGL 库及对应 `glcore`、`eglcore` 在登录节点不可读。缺少当时的动态链接跟踪，无法判断具体是间接依赖、驱动初始化还是设备访问。静态链接依赖分析需要同版本驱动文件，可以离线执行。已排队的 `126712735` 验证 EGL 路线，不能反推旧 GLX 故障的确切原因。

## 原始证据与 CPU 复核

原样归档 `container_vulkan.txt`、`sapien.log`、完整 `curobo.log`、`driver_binds.tsv`、`check_status.tsv` 和当时的宿主及容器 ICD JSON。所有条目通过 SHA-256 校验。`container_icd_linker.log` 在该作业中不存在，已在 `inventory.txt` 标记。SCNet 归档位于 `$HOME/code_cpf/pi05/logs/robotwin-evidence-126711883.tar.gz`，内附实装源码行号、`readelf` 输出和 RoboTwin 规划器调用位置。

- **SAPIEN**。直接检查 sandbox，`/usr/share/glvnd/egl_vendor.d` 和 `/etc/glvnd/egl_vendor.d` 都不存在。实装的 SAPIEN wheel 自带 `vulkan_library/10_nvidia.json`，指向 `libEGL_nvidia.so.0`。实装源码的静态回归用模拟成功的 `nvidia-smi` 进入 EGL 检查，再验证显式设置 `__EGL_VENDOR_LIBRARY_FILENAMES` 会在枚举目录前返回。验证覆盖导入钩子的分支，实际驱动解析与渲染仍待验证。[SAPIEN 上游问题 #256](https://github.com/haosulab/SAPIEN/issues/256)
- **CuRobo**。原测试的 `warmup(..., parallel_finetune=False)` 在 `motion_gen.py:1935` 调用 `plan_single`，并开启 `enable_finetune_trajopt`。`motion_gen.py:3506` 令首轮轨迹优化只返回最佳解，`trajopt.py:1459` 通过单个索引取 `result.raw_action[idx]`，得到去掉种子维的 `[28,6]`。`motion_gen.py:3516` 将它原样复制，在第 3541 行传给微调求解器，最终由 `trajopt.py:1659` 拒绝。用固定源码中的 `get_seed_set` 在本地做 CPU 形状回归，复现二维拒绝，验证补一个维度为 `[1,28,6]` 后通过且轨迹元素完全相同。这只验证单批单种子的形状修复候选。未修改已安装 CuRobo，RoboTwin 自身的 `warmup` 与笛卡尔 `plan_single` 仍在验收范围内。[固定版 MotionGen](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/src/curobo/wrap/reacher/motion_gen.py#L3498-L3548) · [固定版 TrajOpt](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/src/curobo/wrap/reacher/trajopt.py#L1454-L1460)
- **容器图形库**。`readelf -d` 已检查镜像内 `libGLX`、`libX11`、`libEGL` 和 Vulkan loader 的 SONAME 及直接依赖。旧 ICD JSON 与绑定清单可用于复核库路径。sandbox 中的 NVIDIA 库目标在解绑后只剩零字节占位文件，登录节点无法读取对应宿主源文件，因此不能对当时实际绑定的 NVIDIA 库执行 `ldd -r` 或确认加载的版本组合。下一版脚本在容器 Vulkan 失败时单独保存链接诊断。
- **日志保留**。本地以假命令执行当前 Slurm 脚本的实际阶段封装和汇总逻辑。CuRobo 超时和容器 Vulkan 失败仍能保留原始阶段输出、容器链接诊断和 `acceptance.json`。缺少驱动时下游状态记为 `skipped`，已执行阶段的结果仍然保存。
