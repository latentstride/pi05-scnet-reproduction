# RoboTwin GPU 图形与规划验收

独立研究镜像已建成，SAPIEN 静态 FK 已通过。首次 GPU 验收在宿主 Vulkan 工具启动阶段退出，第三步未通过。镜像与 FK 证据见 [环境记录](robotwin_environment.md) 和 [FK 报告](robotwin_sapien_fk_smoke.md)。

| 检查 | 首次结果 |
| --- | --- |
| NVIDIA GPU 可见性 | 通过，进入下一阶段 |
| 宿主 Vulkan | 作业环境无法启动 `vulkaninfo`，报 `No such file or directory` |
| Apptainer Vulkan | 待测 |
| SAPIEN 离屏 RGB | 待测 |
| CuRobo UR5e 最小规划 | 待测 |

提交前只在登录节点核实了宿主工具的可用性。计算节点的作业环境需要可用的探针。修订版复用研究镜像中的 `vulkaninfo` 和对应的动态加载器，在 Apptainer 外以宿主进程运行，并指向计算节点实际安装的 NVIDIA ICD 和驱动库。容器检查单独运行，两侧均需识别调度器分配的同一 GPU UUID。`bash scripts/robotwin_graphics.slurm --host-tool-check` 用于在登录节点验证工具启动。该预检覆盖可执行文件与依赖加载，GPU 驱动枚举与离屏渲染仍需实测。

[Slurm 脚本](../scripts/robotwin_graphics.slurm) 设定单卡和 10 分钟硬时限。[验收程序](../scripts/robotwin_gpu_smoke.py) 要求渲染图像随场景变化，UR5e 轨迹含有效运动并达到目标。UR5e 用例覆盖 CuRobo 基本规划能力，目标本体的资产与规划另行验收。后续付费作业按次确认。
