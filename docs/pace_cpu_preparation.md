# PACE handover_block CPU 准备验收

2026-11-04 在 SCNet 昆山完成。项目根目录为 `$HOME/code_cpf/pi05`。

| 验收项 | 结果 |
| --- | --- |
| PACE 源码 | `niejunnan25/PACE@fd2e624cc19325cc9b20a620c82d51c0d8e25cc1`，跟踪文件干净 |
| OpenPI-PACE 源码 | `niejunnan25/Openpi-PACE@f4e52e6de4cc487a11585a3a7414d610b2883dae`，跟踪文件干净。安装按固定 gitlink 使用 Aloha 和 LIBERO 子模块归档 |
| 仿真资产 | `TianxingChen/RoboTwin2.0@3dc3b798668feb99ac61cc9086d84cbcc3d79186` 的背景、机器人本体和物体归档通过发布 SHA-256 校验并解包。三个 seed 各 100 条场景通过兼容性检查。Aloha URDF、SRDF、30 个 mesh、两份 cuRobo 配置和任务指令通过静态检查 |
| Checkpoint | `niejunnan25/pi05-robotwin@17ef545270cf26ed81169893b60f845f80ce6557`。权重目录为 12,441,552,737 字节，含 tokenizer 共 12,445,816,760 字节，28 个文件全部通过 SHA-256 复验 |
| 独立模型环境 | Ubuntu 22.04 Apptainer 容器内的 `tools/pace-model-venv`。Python 3.11.18，207 个包通过依赖检查。JAX 0.5.3 枚举 CPU，成功加载 `pi05_robotwin_handover_block`，动作窗口为 50 |

安装从已核验的 `uv.lock` 导出依赖，显式选择锁内的 CPython 3.11 Linux PyAV wheel。固定版本的 Gemma 模块在导入时需要 pytest，安装脚本也从同一锁文件补齐 pytest 和相关依赖。下载脚本的 `--endpoint` 支持可达的镜像，两个入口均使用固定 revision、长度和 SHA-256 清单。

SCNet 项目内的 checkpoint 位于 `cache/pace-handover/`，tokenizer 安装到 `cache/openpi-pace/big_vision/paligemma_tokenizer.model`。验收日志位于 `logs/pace-prepare/`，其中 `checkpoint-sha256sum.log`、`handover-assets.log` 和 `model-env-final.log` 分别记录全量校验、任务资产检查和 CPU 配置导入。复现步骤见 [PACE 阶段五运行计划](../pace_runtime_plan.md)。

本轮提交 GPU 作业数为 0。checkpoint 恢复、真实策略推理和仿真闭环属于下一阶段。
