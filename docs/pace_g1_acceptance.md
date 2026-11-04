# PACE × RoboTwin G1 一次性验收

G1.1、G1.2、G1.3 由同一份 [Slurm 入口](../scripts/pace_g1.slurm) 顺序执行。G1.2 使用 G1.1 保存的真实三相机与关节观测；G1.3 使用 G1.2 同一服务进程和与 G1.1 一致的初态。流程不修改冻结的 LIBERO 基线。

## 免费预检

复用现有 Ubuntu 22.04 模型容器、RoboTwin 容器、固定 PACE/OpenPI-PACE checkout 及已下载资产。RoboTwin 容器 Python 3.10 使用项目目录 `tools/pace-smoke-deps` 中的固定 msgpack、WebSockets wheel；两容器需能执行 Git。模型容器从 `tools/pace-git/bin` 查找项目自备 Git。

模型使用已有 `containers/openpi-ubuntu` sandbox，以 `--writable` 启动以兼容 SCNet 的挂载方式。该目录内需存在空的 `public/` 挂载点，供项目路径的 `/public:/public` 绑定使用；这不重建模型环境，也不修改冻结基线。修复挂载配置后可只复验此前失败的模型侧，保留同一发布包已通过的仿真侧记录。

先从已提交的 Git 版本打包一份只读发布目录。以下命令只运行 CPU 检查；本地打包后也可将同一目录传到 SCNet 的项目路径。

```bash
revision=$(git -C "$REPO" rev-parse HEAD)
release="$PWD/tools/pace-smoke-$revision"
preflight="$PWD/results/pace-preflight-$revision"
bash "$REPO/scripts/prepare_pace_smoke_release.sh" "$REPO" "$revision" "$release"
bash "$release/pace_g1_preflight.sh" "$preflight"
```

预检要求实际检查固定源码、场景缓存、Aloha 的 URDF/SRDF、mesh、规划器配置及任务指令，并在 RoboTwin 容器内运行一次真实 WebSocket 合成闭环。模型容器单独核对 Git 版本、子模块 gitlink、28 个模型文件、归一化统计、缓存 tokenizer 的 SHA-256、OpenPI 配置和 CPU JAX。GPU 模型启动使用该预检的文件身份记录，并重新核对较小的 tokenizer 和归一化文件。

固定版 CuRobo 的 `MotionGen` 在导入时创建 CUDA tensor。只有栈精确包含 `motion_gen.py` 和 PyTorch CUDA 初始化、且异常明确为登录节点无 NVIDIA 驱动时，CPU 预检才把此项记为 GPU 待验。CPU 进程临时提供一个一经构造就会报错的规划器符号，仅用于完成其余模块的导入；GPU 作业使用未修改的真实规划器。其他导入错误均失败。阶段耗时见 `phases.jsonl`，模块耗时见 `importtime.log`，超过 30 秒的阶段定期将调用栈写入 `stacks.log`。

## 待审批作业

建议使用 `ksagnormal01` 的单张 RTX 4090（24 GiB）、4 CPU、32 GiB 主机内存，硬时限 50 分钟。各阶段预算：G1.1 12 分钟；G1.2 模型就绪 13 分钟、首次推理 7 分钟；G1.3 12 分钟。其余 6 分钟供图形检查、报告与清理。按历史单卡 3.60 元/卡时估算，满时约 3.00 元；实际价格和内存资源计费以批准时的 SCNet 报价为准。

仅在两个 CPU 预检均通过且用户明确批准后，提交唯一的综合作业：

```bash
sbatch "$release/pace_g1.slurm" "$release" "$preflight"
```

作业核对发布清单和预检凭证，在同一张卡上复用此前验证的 Vulkan/EGL 设置。模型容器禁用 JAX 显存预分配，保存 JAX 编译日志与首次推理时间；执行期间采样显存。统一报告包含各阶段开始、结束、退出码、成功/失败/跳过，以及唯一模型服务的进程标识。真实场景初始化、三相机渲染、Orbax GPU 恢复、首次编译、并发显存峰值及真实闭环仍需这一批次最终验证。
