# 短时 GPU 图形与规划验收准备

日期：2026-11-02。状态：**独立研究镜像与 CPU 依赖验收已完成，SAPIEN 静态 FK 检查通过；GPU 验收尚未提交，按用户要求暂不执行第三步。** 见 [环境记录](robotwin_environment.md) 与 [FK 报告](robotwin_sapien_fk_smoke.md)。GPU 模板仍需与实际镜像入口、设备和计费规则核对，不能直接视为已可提交。PACE 大权重继续暂缓。

## 作业范围

模板 [robotwin_graphics.slurm](../scripts/robotwin_graphics.slurm)：单 GPU、4 CPU、16 GB 内存、Slurm 硬时限10分钟。未指定分区/计费资源，提交前必须核对 SCNet 可用分区、目标 GPU、当前价格、原20元总预算的实际余额及最小计费粒度。不能用“10分钟”推定费用；本轮只准备，不申请资源。

必须从 `$HOME/code_cpf/pi05` 提交。研究镜像必须放在个人根目录的 `research/robotwin/` 内，由 `ROBOTWIN_IMAGE` 指定；`ROBOTWIN_PYTHON` 是镜像内部 Python 路径。未设置参数或镜像不存在会退出。现有 LIBERO 镜像不作为默认值。

| 顺序 | 检查与通过证据 | 超时 |
| --- | --- | --- |
| 1 | 宿主 `nvidia-smi` 导出 GPU UUID/型号 | 20秒 |
| 2 | 宿主 `vulkaninfo` 正常完成并列出分配的 NVIDIA GPU UUID | 45秒 |
| 3 | Apptainer 内 `vulkaninfo` 列出与宿主同一 GPU UUID；软件渲染器不算通过 | 45秒 |
| 4 | SAPIEN 离屏小场景输出两张160×120 RGB；移动红盒后图像至少100像素发生改变 | 150秒 |
| 5 | CuRobo 最小 UR5e 运动规划返回 success，产生多步且有限的关节轨迹 | 240秒 |

每一步失败即退出，并写入 `FAILED`；全部通过才产生 `acceptance.json`。Vulkan 旧工具若不导出 UUID，会保守失败并保存原始日志，需在申请前准备合适工具，不能以名称相似替代设备身份。作业可能看到多张物理 GPU，但容器 Vulkan 必须与管理接口所列 NVIDIA UUID 相交；仍需结合调度器分配和 CUDA 可见设备人工核对，不把整节点设备枚举当作额外使用授权。

## 不占 GPU 时先准备的条件

- 独立镜像内准备 Vulkan loader、`vulkaninfo`、NVIDIA ICD 发现路径；宿主驱动库由 Apptainer `--nv` 绑定。可显式绑定已审核的 ICD 目录，但不修改管理员或共享账号配置。
- 匹配 RoboTwin stable_2.0 的依赖已安装：`sapien==3.0.0b1`、`torch==2.4.1`、CuRobo `v0.7.8`，NumPy、Pillow 等核心依赖 CPU 导入通过；镜像与实际包版本已记录。
- CuRobo 五个 CUDA 扩展及 PyTorch3D 扩展已在 CPU 节点提前编译，内置 UR5e 配置已检查。运行缓存位于研究镜像 `/opt/robotwin/cache/`，物理上属于个人项目目录；构建缓存已清理。付费作业内不安排下载或安装；若目标 GPU 需要重编译，应先记录并调整方案，不自动续租。
- 日志、RGB 和规划轨迹保存在个人 `research/robotwin/graphics-<JOB_ID>/`；提交前先创建项目 `logs/`。硬件和环境验收不加载 π₀.₅，不下载 RoboTwin 全量资产。

渲染/规划程序见 [robotwin_gpu_smoke.py](../scripts/robotwin_gpu_smoke.py)。本地已做 Python 语法及 Slurm shell 语法检查；Mac 上没有运行 SAPIEN/CuRobo，运行兼容性仍待目标环境验证。UR5e 测试仅证明 CuRobo 软件栈可规划，不验收 Aloha/ARX 的机器人资产和碰撞模型。

来源：[RoboTwin 安装要求](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)、[固定 requirements](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/requirements.txt)、[CuRobo 版本安装脚本](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/_install.sh)、[SAPIEN 相机 API](https://sapien-sim.github.io/docs/user_guide/rendering/camera.html)、[CuRobo MotionGen 示例](https://nvlabs.github.io/curobo/v0.7.6/get_started/2a_python_examples.html#motion-generation)。规划示例依据0.7.6文档编写，目标0.7.8的实际 API 兼容性尚未运行确认。
