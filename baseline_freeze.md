# 第一阶段基线冻结记录

冻结日期：2026-11-01。冻结名称：`pi05-libero-functional-2026-11-01`。

第一阶段功能性验收通过：真实 LIBERO 观测推理、完整闭环，以及 spatial 任务 0、1、2 各 5 轮，15/15 成功。固定点为公开仓库提交 [`4f87a9085d9f8a2e1bcc14840443aeb5c14db63a`](https://github.com/latentstride/pi05-scnet-reproduction/tree/4f87a9085d9f8a2e1bcc14840443aeb5c14db63a)，后续 main 的资料更新不改变该冻结点。

## 固定版本与设置

| 项目 | 固定值 |
| --- | --- |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LIBERO | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| ALOHA 子模块 | `d1dc83afd89ded4379851257fe5d85632d31d5ec` |
| LeRobot 依赖 | `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |
| checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` |
| 主环境 | Python 3.11.18，JAX/JAXLIB 0.5.3，PyTorch 2.7.1，NumPy 1.26.4 |
| LIBERO 环境 | Python 3.8.20，robosuite 1.4.1 |
| 容器 | NVIDIA CUDA 12.2.2 cuDNN8 devel / Ubuntu 22.04；Apptainer 1.3.4 |
| 评测 | libero_spatial，task 0/1/2，initial_state 0—4，seed=7 |
| 控制 | 20 Hz；等待 10 步，每 5 步重新推理，最多 220 个预测控制步 |
| 输入输出 | 8 维状态、双路图像；horizon=10，单步动作 7 维 |

既有权重验收：16 个文件、12,439,085,481 字节，全部 CRC32C 通过，具备 MD5 的 10 个文件亦通过。本次沿用原验收，没有重新下载或重新校验大文件。

## 固定证据

- [验收报告](https://github.com/latentstride/pi05-scnet-reproduction/blob/4f87a9085d9f8a2e1bcc14840443aeb5c14db63a/acceptance_report.md)
- [逐轮结果](https://github.com/latentstride/pi05-scnet-reproduction/blob/4f87a9085d9f8a2e1bcc14840443aeb5c14db63a/evaluation_results.csv)
- [评测报告](https://github.com/latentstride/pi05-scnet-reproduction/blob/4f87a9085d9f8a2e1bcc14840443aeb5c14db63a/evaluation_report.md)
- [控制接口](https://github.com/latentstride/pi05-scnet-reproduction/blob/4f87a9085d9f8a2e1bcc14840443aeb5c14db63a/action_space_notes.md)
- [环境说明](https://github.com/latentstride/pi05-scnet-reproduction/blob/4f87a9085d9f8a2e1bcc14840443aeb5c14db63a/environment.md)
- [机器可读清单](baseline_manifest.json)：来源提交、Git blob SHA 及已核对摘要文件的 SHA-256。

原始轨迹、15 个视频、配置及核验 JSON 保留在 SCNet 个人项目原评测目录。此次冻结采用固定源码/证据提交和摘要清单，没有设置远端只读权限，也没有新建完整虚拟环境或容器快照。

## 后续使用

保留已验收环境、权重、脚本及结果；不重复安装，不重跑相同 15 轮。第二阶段资料见 [dataset_comparison.md](dataset_comparison.md)。后续依赖、数据转换和仿真环境使用独立研究目录；新增文件继续放在项目所属人的个人目录，不修改团队共享环境。

15/15 是三个任务各五个初态的功能性结果，不是完整 LIBERO benchmark，也不证明 `pi05_libero` 可直接控制其他本体。新增权重、动作接口或任务环境均另记实验版本和评测结果；本阶段不新增 GPU 作业。
