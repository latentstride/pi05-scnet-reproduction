# RoboTwin GPU 图形与规划验收

独立研究镜像已建成，SAPIEN 静态 FK 已通过。首次 GPU 验收在宿主 Vulkan 工具启动阶段退出，第三步未通过。镜像与 FK 证据见 [环境记录](robotwin_environment.md) 和 [FK 报告](robotwin_sapien_fk_smoke.md)。

| 检查 | 首次结果 |
| --- | --- |
| NVIDIA GPU 可见性 | 通过，进入下一阶段 |
| 宿主 Vulkan | 作业环境无法启动 `vulkaninfo`，报 `No such file or directory` |
| Apptainer Vulkan | 待测 |
| SAPIEN 离屏 RGB | 待测 |
| CuRobo UR5e 最小规划 | 待测 |

提交前只在登录节点核实了宿主工具的可用性。计算节点的作业环境需要可用的探针。复测前应确认计算节点上的可执行路径，或预先准备可在宿主运行的独立探针。用宿主实际的 NVIDIA 驱动与 ICD 验证分配设备，再在容器内验证相同设备。

[Slurm 脚本](../scripts/robotwin_graphics.slurm) 设定单卡和 10 分钟硬时限。[验收程序](../scripts/robotwin_gpu_smoke.py) 要求渲染图像随场景变化，UR5e 轨迹含有效运动并达到目标。UR5e 用例覆盖 CuRobo 基本规划能力，目标本体的资产与规划另行验收。后续付费作业按次确认。
