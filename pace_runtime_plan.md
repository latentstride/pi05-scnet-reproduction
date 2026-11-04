# PACE 阶段五运行计划

2026-11-03 · 单任务 `handover_block` · 原生双臂 Aloha-AgileX

本阶段验收范围为 CPU 接口、控制循环和文件校验。图形 Gate 通过后才执行大权重下载、JAX 初始化及仿真闭环。所有本阶段脚本输出均为诊断记录，`score_eligible=false`，与正式成功率汇总分离。

## C1 · 版本和兼容矩阵

| 层 | 固定版本或资产 | 兼容证据与待验项 |
| --- | --- | --- |
| PACE 仿真适配器及随仓 RoboTwin 源码 | `niejunnan25/PACE@fd2e624cc19325cc9b20a620c82d51c0d8e25cc1` | 使用该提交自身的 `envs/`、`task_config/` 和 `pace/`。上游缓存校验 base task、任务、相机和 clean 配置的哈希，通过后才加载第一条固定场景。 |
| RoboTwin stable_2.0 环境基点 | `RoboTwin-Platform/RoboTwin@13c3c47ff4312dd62484bcd51be034af55c062d1` | PACE 文档指定的 stable_2.0 基点，也是现有图形研究环境的依赖基点。任务运行使用上一行的 PACE 源树，避免混用普通 stable_2.0 checkout 与 PACE 缓存。 |
| OpenPI-PACE 模型服务 | `niejunnan25/Openpi-PACE@f4e52e6de4cc487a11585a3a7414d610b2883dae`，固定 `uv.lock` 和 gitlinks | 审计过的运行配置为 `pi05_robotwin_handover_block`，`adapt_to_pi=False`，动作时域 50。HF 清单记录的训练代码为 `ba9876639cb7a4bf0c7db00ec39eeb9c8efd8b12`。固定服务版本与训练版本的实际恢复兼容性留待首次加载验证。 |
| HF 权重和任务归一化 | `niejunnan25/pi05-robotwin@17ef545270cf26ed81169893b60f845f80ce6557` | EMA step 30,000，Orbax OCDBT，资产 ID `robotwin`。使用 `handover_block/assets/robotwin/norm_stats.json`，3,300 bytes，SHA-256 `8a1f114a8873dfa0788c7f512851ee3063224ef2b8af270421076a2e4e544960`。 |
| LIBERO | [独立冻结记录](baseline_freeze.md) | 两套 Python 环境、配置、权重及结果路径分别管理。 |

来源见 [PACE 固定版本与仿真说明](https://github.com/niejunnan25/PACE/tree/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1#simulation-evaluation)、[RoboTwin 固定源码](https://github.com/RoboTwin-Platform/RoboTwin/tree/13c3c47ff4312dd62484bcd51be034af55c062d1)、[OpenPI-PACE 配置](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/pace/src/openpi_pace/configs.py)、[HF 发布清单](https://huggingface.co/niejunnan25/pi05-robotwin/blob/17ef545270cf26ed81169893b60f845f80ce6557/manifest.json)。

## C2–C3 · 接口与闭环

- WebSocket 在连接后先传服务元数据，后续每条二进制消息采用固定 OpenPI 客户端的 NumPy MessagePack codec。服务元数据携带 `protocol=1`、`task=handover_block`、`backend=fake|checkpoint`，客户端校验三项。消息限制 16 MiB，连接和收包超时默认 30 秒。文本错误帧、超时、断线、坏形状和非有限值终止该次运行。
- 观测为 `state float[14]`、非空 `prompt` 和三路 `uint8 RGB CHW`。头相机及左右腕分别映射到 `cam_high`、`cam_left_wrist`、`cam_right_wrist`。状态顺序为左臂六关节、左夹爪、右臂六关节、右夹爪。模型返回有限浮点 `actions[50,14]`，为同顺序的绝对关节及夹爪目标。checkpoint 的归一化、delta 还原及输出切片由上游策略加载器处理。
- 原生模式使用 PACE 专家缓存中 `handover_block`、评估 seed 分片 0 的第一条初态。固定提示词随 `summary.json` 保存。使用 `demo_clean`、原生双臂 Aloha、D435 与 `qpos` 控制。默认至多 8 个控制步、2 次推理，每次消费最多 4 行，未执行的后缀在下次查询时丢弃。CLI 硬上限分别为 20 步、5 次查询、8 行。环境任务判据、步数和查询上限任一触发即停止。这是固定前缀烟测。
- 每次运行使用新目录。保存 `steps.csv`（下发目标）、`queries.csv`（推理耗时和实际执行行数）、`chunk_*.npy`（原始完整块）、`head.mp4`（每控制步一帧、10 FPS 播放）、`summary.json`（计数、场景来源与退出状态）。观测失败前可能只有部分文件。捕获异常和 SIGINT/SIGTERM 后清理连接、视频与环境，写入 `aborted` 并返回非零。对原生渲染或 CUDA 调用的硬挂起，另设作业级时限。

### CPU 启动与断言

本地 CPU 已通过 8 项断言，覆盖正常往返、坏包、超时、异常退出、前缀缓存、视频产物和小文件续传校验。

从项目根目录执行。提前准备上述固定提交的 OpenPI-PACE 源树，此处仅需要其中的小型客户端包。

```bash
export OPENPI_ROOT=/absolute/path/to/Openpi-PACE
export PYTHONPATH="$OPENPI_ROOT/packages/openpi-client/src"
python -m pip install -r scripts/pace_smoke_requirements.txt
python -m unittest discover -s tests -p test_pace_runtime.py -v

# 两个终端使用同样的 PYTHONPATH。
python scripts/pace_model_smoke.py --port 8000
python scripts/pace_client_smoke.py --output results/pace5/cpu-fake
```

## C4 · 下载与校验

[下载清单](pace_download_manifest.json) 从固定 HF 清单提取。任务目录 12,441,552,737 bytes，另有共享 tokenizer 4,264,023 bytes，合计 12,445,816,760 bytes。清单逐文件固定相对路径、长度和 SHA-256。

```bash
# 默认离线打印计划。Python >=3.11。
python scripts/pace_download.py

# 图形 Gate 通过并确认存储条件后显式执行。
python scripts/pace_download.py --download --dest "$PI05_ROOT/cache/pace-handover"
python scripts/pace_download.py --verify --dest "$PI05_ROOT/cache/pace-handover"
export OPENPI_DATA_HOME="$PI05_ROOT/cache/openpi-pace"
mkdir -p "$OPENPI_DATA_HOME/big_vision"
cp "$PI05_ROOT/cache/pace-handover/tokenizer.model" \
   "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
```

下载器只请求清单中的固定 revision URL。传输中断后保留同目录 `.part`，再次运行按字节续传。通过大小和 SHA-256 后原子改名。完整文件校验失败时，人工确认并移走对应坏文件或坏 `.part` 后重试。

## C5 · 图形与推理隔离

| 进程 | 镜像 / 环境 | 硬件与验收 |
| --- | --- | --- |
| RoboTwin 仿真客户端 | 已构建 `robotwin-cu122` Apptainer sandbox，基础镜像 `nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04`。Python 3.10.19，SAPIEN 3.0.0b1，PyTorch 2.4.1+cu121，CuRobo v0.7.8 固定提交 `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`，扩展目标 8.9+PTX。 | 独立验收宿主驱动、Vulkan ICD、三路离屏 RGB、CuRobo 和 Aloha 任务资产。依赖证据见 [环境记录](docs/robotwin_environment.md) 和 [图形 Gate](docs/robotwin_gpu_acceptance_plan.md)。 |
| PACE 模型服务 | 计划使用独立 Linux x86_64 Python 3.11 环境，以 OpenPI-PACE 提交内的 `uv.lock` 固定 JAX/JAXLIB 与 CUDA 12 插件 0.5.3、Orbax 0.11.13。锁文件 SHA-256 为 `5e3a9a0a12d9a6048afea5591f4520c98585499cbd4a8343dcabfe2aaed94e3d`。首轮采用独立 venv。如需容器，以独立 Ubuntu 22.04 Linux x86_64 镜像为构建目标，首跑前记录镜像摘要和宿主驱动。 | 单独验收 CUDA device、Orbax 恢复、tokenizer、首次编译、显存峰值和有限动作输出。将模型进程放在自己的 venv 或独立容器内。 |

首选同一节点的独立进程与同一网络命名空间，使用 `ws://127.0.0.1:8000`。模型服务只绑定回环地址。同一 GPU 上的模型和图形进程有独立显存分配，需实测合计峰值。多卡时可分别映射设备。分节点时使用受控隧道把客户端回环端口转发到模型节点，沿用同一 WebSocket 协议。传输只含观测与动作数组，两个环境无需共享 JAX、CuRobo 或 SAPIEN 包。

通过图形检查后，先运行原生假策略烟测。SCNet 的 RoboTwin 容器已有 NumPy 和 imageio-ffmpeg；用该容器的 Python 将 `msgpack==1.1.1`、`websockets==15.0.1` 安装到项目的 `tools/pace-smoke-deps`。源代码和资产使用 [CPU 准备步骤](docs/pace_cpu_preparation.md) 固定的版本。获得单次 GPU 作业授权后，从项目根目录提交：

```bash
sbatch --job-name=cpf-pace-native-fake --output=logs/pace-native-%j.out \
  scripts/robotwin_graphics.slurm --pace-native
```

作业校验容器 Vulkan 枚举到所分配的 GPU，然后在同一节点启动不带 checkpoint 参数的 WebSocket 服务。场景取专家缓存 split 0 的第 0 条，用 PACE 的指令生成器选取固定的 `unseen` 任务指令。策略收到的 14 维向量沿用 PACE 的训练约定：左臂六个关节驱动目标、左夹爪、右臂六个关节驱动目标、右夹爪。另记录仿真实测关节角，以便检查控制后的物理状态。假策略仅用于接口烟测。

结果位于 `results/pace-native-<job-id>/rollout/`，包含视频、初始三路 RGB、每次观测的关节及帧摘要、完整动作块、查询及动作日志、运行摘要。作业退出前回读这些产物，核对动作、观测、计数、停止原因和视频帧数；校验成功才写出 `artifact-verification.json` 中的通过结果。单路图像在保持姿态时可以不变，三路全程逐字节不变则无法通过原生验收。若作业失败，查看同目录下的 `FAILED`、`pace.log` 和 `fake-policy.log`。

随后用核验过的模型目录和 tokenizer，按以下方式启动真实模型烟测。这里的 `$PACE_ROOT` 是含所需 RoboTwin 资产的固定 PACE checkout。`$PI05_ROOT`、`$PACE_ROOT`、`$OPENPI_ROOT` 需要在对应进程或容器中可读。

```bash
# 模型节点。用固定 checkout 和 submodule gitlinks 创建独立环境。
(cd "$OPENPI_ROOT" && GIT_LFS_SKIP_SMUDGE=1 uv sync --locked --python 3.11)
# 在该 venv 中，按实际设备编号执行。
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python "$PI05_ROOT/scripts/pace_model_smoke.py" --openpi-root "$OPENPI_ROOT" \
  --download-dir "$PI05_ROOT/cache/pace-handover"

# 仿真节点或同节点的仿真环境，先装入 smoke 依赖并暴露 openpi-client 源路径。
python "$PI05_ROOT/scripts/pace_client_smoke.py" --pace-root "$PACE_ROOT" \
  --backend checkpoint --timeout 180 --output "$PI05_ROOT/results/pace5/native-checkpoint"
```

通过标准为固定初态下收到有效 `(50,14)`、推进有界的 `qpos` 控制循环并生成诊断文件。首次加载和首轮编译时限以实际作业资源为准。
