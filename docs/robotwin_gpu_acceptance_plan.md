# 短时 GPU 图形与规划验收准备

日期：2026-11-02。状态：**独立研究镜像与 CPU 依赖验收已完成，SAPIEN 静态 FK 检查通过；用户逐次确认费用后，已提交单次 GPU 验收作业 `126598641`，当前排队，尚无 GPU 验收结果。** 见 [环境记录](robotwin_environment.md) 与 [FK 报告](robotwin_sapien_fk_smoke.md)。GPU 模板已按实际镜像和安装版本的 API 修订并通过静态检查；实际 GPU 运行能力尚未验收。PACE 大权重继续暂缓。

## 作业范围

模板 [robotwin_graphics.slurm](../scripts/robotwin_graphics.slurm)：`ksagnormal01` 分区、RTX 4090 24 GB、单 GPU、4 CPU、16 GB 内存，Slurm 硬时限10分钟。已通过 SCNet 订单页核对 GPU 单价为 **3.60元/卡时**；按实际运行10分钟估算 **0.60元**。平台账单为最终费用依据。此前本项目 Slurm GPU 作业累计28分22秒，按该单价估算约1.70元，并非课题组全部用量或正式账单。用户确认“开始”后，已提交单次作业 `126598641`。首次状态查询为 PENDING（Priority），运行时间0；调度器暂未给出预计启动时间。尚未产生验收日志和结果。

**费用授权规则：用户要求任何产生费用的操作逐次确认。原20元预算不构成自动提交授权。** 本次单卡、最多10分钟及预计0.60元的费用已获用户明确确认；若资源、价格或计费方式变化，应重新确认。排队不计入实际运行时间，不安排自动续租或未经确认的第二次作业。

必须从 `$HOME/code_cpf/pi05` 提交。模板固定使用独立镜像 `research/robotwin/robotwin-cu122`、内部 `/opt/robotwin/venv/bin/python`，并要求 `BUILD_COMPLETE` 存在。现有 LIBERO 镜像不参与本验收。
| 顺序 | 检查与通过证据 | 超时 |
| --- | --- | --- |
| 1 | 宿主 `nvidia-smi` 导出 GPU UUID/型号 | 20秒 |
| 2 | 宿主 `vulkaninfo` 正常完成并列出分配的 NVIDIA GPU UUID | 45秒 |
| 3 | Apptainer 内 `vulkaninfo` 列出与宿主同一 GPU UUID；软件渲染器不算通过 | 45秒 |
| 4 | SAPIEN 离屏小场景输出两张160×120 RGB；移动红盒后图像至少100像素发生改变 | 150秒 |
| 5 | CuRobo 最小 UR5e 运动规划返回 success，产生多步且有限的关节轨迹 | 240秒 |

每一步失败即退出，并写入 `FAILED`；全部通过才产生 `acceptance.json`。Vulkan 旧工具若不导出 UUID，会保守失败并保存原始日志，需在申请前准备合适工具，不能以名称相似替代设备身份。从 Slurm 分配字段选出唯一 GPU，宿主和容器 Vulkan 都必须枚举其确切 UUID；容器显式传入该 UUID 作为 CUDA_VISIBLE_DEVICES。SAPIEN 设备及渲染器 PCI 地址必须与该分配匹配，CuRobo 要求 CUDA 仅可见一张 GPU。

## 不占 GPU 时先准备的条件

- 独立镜像内准备 Vulkan loader、`vulkaninfo`、NVIDIA ICD 发现路径；使用 Apptainer `--nv`，并将宿主实际安装的 NVIDIA 图形库只读绑定到个人镜像内的专用路径。容器 ICD 复制真实宿主配置并改写库路径，保留其 API 版本，不凭空生成驱动能力；不修改管理员或共享账号配置。
- 匹配 RoboTwin stable_2.0 的依赖已安装：`sapien==3.0.0b1`、`torch==2.4.1`、CuRobo `v0.7.8`，NumPy、Pillow 等核心依赖 CPU 导入通过；镜像与实际包版本已记录。
- CuRobo 五个 CUDA 扩展及 PyTorch3D 扩展已在 CPU 节点提前编译，内置 UR5e 配置已检查。运行缓存位于研究镜像 `/opt/robotwin/cache/`，物理上属于个人项目目录；构建缓存已清理。付费作业内不安排下载或安装；若目标 GPU 需要重编译，应先记录并调整方案，不自动续租。
- 日志、RGB 和规划轨迹保存在个人 `research/robotwin/graphics-<JOB_ID>/`；提交前先创建项目 `logs/`。硬件和环境验收不加载 π₀.₅，不下载 RoboTwin 全量资产。

渲染/规划程序见 [robotwin_gpu_smoke.py](../scripts/robotwin_gpu_smoke.py)。本地已做 Python、Slurm shell 及三个内嵌 Python 块的语法检查；目标环境 CPU 检查确认 `Device.is_cuda()`、`can_render()` 为方法，`pci_string` 为属性。Mac 上没有运行 SAPIEN/CuRobo，运行兼容性仍待目标环境验证。UR5e 测试仅证明 CuRobo 软件栈可规划，不验收 Aloha/ARX 的机器人资产和碰撞模型。

来源：[RoboTwin 安装要求](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)、[固定 requirements](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/requirements.txt)、[CuRobo 版本安装脚本](https://github.com/RoboTwin-Platform/RoboTwin/blob/13c3c47ff4312dd62484bcd51be034af55c062d1/script/_install.sh)、[SAPIEN 相机 API](https://sapien-sim.github.io/docs/user_guide/rendering/camera.html)、[CuRobo MotionGen 示例](https://nvlabs.github.io/curobo/v0.7.6/get_started/2a_python_examples.html#motion-generation)。规划参数已对照 [CuRobo v0.7.8 固定源码](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/examples/motion_gen_example.py) 核对，使用 UR5e retract 配置附近的小幅运动；要求末端位置误差不超过0.01米、姿态误差不超过0.1弧度。实际 GPU 运行仍待验收。计费依据：[SCNet 计费规则](https://www1.scnet.cn/help/docs/mainsite/hpc/advanced-guide/billing-rule/)。

## 提交前验证记录

- 2026-11-02：`sbatch --test-only` 接受作业参数；该命令只预检，不分配资源。
- 正式作业：`126598641`，`cpf-rw-graphics`；脚本硬时限 `00:10:00`。
- 已核对 SCNet 与本地脚本 SHA-256 一致；同步包约182 KiB，未新增模型或全量资产下载。
- 原始提交编号与预检日志保存在个人目录 `research/robotwin/env-build/logs/`。
