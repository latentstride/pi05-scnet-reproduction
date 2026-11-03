# RoboTwin 图形与规划复测结果

GPU 验收未通过。CUDA 单卡识别通过，宿主 Vulkan 枚举到了同一块分配的 NVIDIA GPU。容器渲染和运动规划仍未验收。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| NVIDIA 图形驱动发现 | 通过 | 找到宿主 NVIDIA ICD 和驱动库 |
| 宿主 Vulkan | 通过 | 后续设备核验通过了宿主侧 CUDA UUID 检查 |
| Apptainer Vulkan | 失败（1） | Vulkan loader 未能从绑定的 `libGLX_nvidia.so.0` 取得 `vkCreateInstance`，报告 `ERROR_INCOMPATIBLE_DRIVER` |
| 容器 Vulkan UUID | 失败（1） | 容器未枚举 CUDA 分配卡的 UUID |
| SAPIEN 离屏渲染 | 失败（1） | 导入时找不到 `/usr/share/glvnd/egl_vendor.d`，未生成 RGB |
| CuRobo 运动规划 | 失败（124） | 预热在轨迹种子处抛出 `ValueError`：输入为 `[28, 6]`，接口要求三维或四维；外层达到超时，未生成规划轨迹 |

容器 Vulkan 与 CuRobo 是两个独立问题。Vulkan 报错还不足以确定驱动库是缺少依赖、混用了版本还是因其他原因初始化失败。再次分配 GPU 前，应先用现有容器检查 EGL 目录及 vendor JSON，验证 NVIDIA ICD 的依赖与加载顺序，并修正 CuRobo 预热输入及阶段日志。NVIDIA 记录了在缺少 X11 客户端库时可使用 EGL ICD；这是待核对的候选路径。[NVIDIA 驱动组件说明](https://download.nvidia.com/XFree86/Linux-x86_64/535.146.02/README/installedcomponents.html)

本次未加载策略权重，也未开展任务回放。
