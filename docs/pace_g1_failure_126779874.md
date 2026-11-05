# G1 作业 126779874 失败复盘

2026-11-05，北京时间。作业于 08:26:09 启动，08:26:14 退出，运行 5 秒，Slurm 为 FAILED、ExitCode=1:0。没有重新提交作业。

## 已有证据

- 固定发布包和 CPU 双侧通过凭证通过校验。
- CUDA 与 nvidia-smi 均识别到分配的 RTX 4090。
- 图形驱动发现阶段报 `NVIDIA ICD library missing from host linker cache`。
- G1.1、G1.2、G1.3 全部为 skipped。没有恢复模型，没有生成真实三相机，也没有下发 Aloha 动作。
- 本次节点为 gnode65；上次图形验收成功的 126712735 在 gnode39。上次成功记录的宿主 ICD 使用 GLX 库，容器则使用 NVIDIA EGL。

原脚本在原 ICD 库解析失败时立即断言退出，尚未检查后续要求的 NVIDIA EGL 库，也没有在退出前保存当前节点的完整 ICD 配置和链接器缓存。因此只能确认原 ICD 库解析失败；无法从现有日志证明 gnode65 的 EGL 是否可用，或断言该节点完全缺少图形驱动。

## 免费修订

将驱动发现提取到 `scripts/robotwin_graphics_driver.py`，在失败前保存 `driver_discovery.json` 和 `ldconfig.txt`。保留原 ICD 能解析时的行为；原 ICD 库无法解析而已安装的 NVIDIA EGL 可解析时，显式生成 EGL ICD，再由现有宿主/容器 Vulkan 枚举和分配卡 UUID 校验决定是否通过。缺少 NVIDIA EGL 或 NVIDIA ICD 时仍失败，不用软件渲染代替。

四种 CPU 回归场景通过：原 GLX/EGL 布局、原 ICD 库缺失但 NVIDIA EGL 可用、EGL 缺失、无 NVIDIA ICD。原有 13 项 PACE/G1 测试也通过。该结果验证代码分支及日志保存，不代表 gnode65 的 GPU 渲染已通过。

原运行发布包、原失败目录和旧日志保持不变。原 CPU 凭证不适用于修改后的发布包。新版本仍需完成发布与免费门槛；在再次明确授权之前，不提交任何 GPU 作业。后续 G1 仍只允许一个综合作业，内部顺序执行三个阶段。

完整原始证据保留在 SCNet 项目目录 `results/pace-g1-126779874/`，并取回本地 `research/g1/failed-126779874/`。
