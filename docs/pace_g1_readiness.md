# G1 提交前验收记录

历史记录：2026-11-04，北京时间。该版本随后获批提交的 `126779874` 已在图形前置检查失败。当前修订进度见 [2026-11-05 免费预检](pace_g1_preflight_20261105.md)；下文是旧发布包的预检证据，不能用作新发布包的提交凭证。

## 旧作业

先核对项目名称、所有者和工作目录，确认 126767492 仍为 PENDING 后，于 13:22:38 取消。Slurm 账本为 CANCELLED，运行时间 00:00:00，Start=None，AllocTRES 为空。没有取消其他作业。

## 已通过的免费检查

| 项目 | 结果 |
| --- | --- |
| 固定 PACE 源码与专家缓存 | 源码跟踪文件检查、缓存兼容哈希通过；split 0 第 0 条，scene seed 100000 |
| Aloha 原生软件与资产 | 任务模块、MPlib、30 个 mesh、URDF/SRDF、规划配置和官方任务指令通过 |
| CuRobo | 精确识别 MotionGen 在模块导入时创建 CUDA tensor 的无驱动异常；仅此项延期到 GPU，未记作真实规划通过 |
| 仿真容器通信 | Fake Policy 的真实 WebSocket 通信、2 次查询、8 步合成控制和 9 帧视频回读通过；未将合成图像记为真实三相机 |
| 独立模型容器 | 项目自备 Git、固定 OpenPI-PACE、两个子模块 gitlink 及文件存在性检查通过 |
| 模型文件 | 28 文件、tokenizer、归一化统计 SHA-256 复验通过，checkpoint revision 为 17ef545270cf26ed81169893b60f845f80ce6557 |
| 模型 CPU 导入 | OpenPI 完整导入、JAX 0.5.3、CPU 设备 TFRT_CPU_0，通过 |
| 本地测试 | 13 项通过，涵盖有序执行、单模型进程复用、场景加载失败、服务失败、超时、坏动作、显存不足及异常清理 |
| 冻结 LIBERO | 冻结记录、模型清单、结果和评测源码的跟踪文件比对未发生改动，未重新运行基线 |

此前定位并修复了模型容器 overlay 挂载、缺少 /public 挂载点及诊断采样方式问题。失败日志保留。最终模型文件校验耗时约 134 秒，OpenPI CPU 导入约 526 秒，JAX CPU 设备检查约 8 秒。先前退出码 139 的准确根因未确认；更换为 Python 线程快照后，本次完整导入和设备检查成功。

## 固定版本与证据

运行发布包提交：`1f66d619e65200f0dde82da7b6a7f3a27500c560`。GitHub `main` 的综合流程和诊断代码已同步；运行代码修订见 `30784ff`。发布包是从本地 Git 提交导出的只读文件集，按内容哈希核验，不依赖排队时修改工作目录文件。

发布包 SHA256SUMS 文件的 SHA-256：

```text
7c199b8a2174be37b74d2d1f9efef0d872fc672315c17cbb9c238fe736f2dd12
```

SCNet 项目目录内：

- 发布目录：`tools/pace-smoke-1f66d619e65200f0dde82da7b6a7f3a27500c560/`
- 免费预检：`results/pace-preflight-1f66d619e65200f0dde82da7b6a7f3a27500c560/`
- `exit-codes.tsv`：simulator=0，model=0。
- `PASSED_G1_RELEASE_SHA256`：与上述发布包哈希一致。
- 模块导入计时、线程快照、失败记录和成功产物均保留；精简证据已取回本地 `research/g1/`。

离线集成报告确认：G1.1 结束后才进入 G1.2；G1.2 与 G1.3 使用同一个模型进程，启动次数为 1；同一初始观测路径和服务 URI 由固定启动计划传递。这是 CPU 模拟流程证据，尚不是三阶段真实 GPU 验收。

## 待审批的唯一作业

入口为 [scripts/pace_g1.slurm](../scripts/pace_g1.slurm)，内部依次运行 G1.1 → G1.2 → G1.3，不含第二次提交、自动重排或自动重试。复用之前通过的 NVIDIA EGL/Vulkan 配置，并将阶段结果统一保存为 `g1_acceptance.json`。

拟申请单张 RTX 4090、4 CPU、32 GiB 主机内存，硬时限 50 分钟。内部预算为仿真 Fake Policy 12 分钟、模型就绪 13 分钟、首次查询 7 分钟、真实短闭环 12 分钟，另留 6 分钟用于图形前置检查和清理。按此前核对的 3.60 元/卡时，GPU 满时部分约 3.00 元；实际平台报价及附加计费须在授权时确认。

仍需在这一个批次验证：真实 SAPIEN 场景与三相机、CuRobo CUDA 初始化、Orbax GPU 恢复、首次 JAX 编译、模型与仿真并发显存、真实 Aloha qpos 执行。显存门槛、超时和失败记录已具备，不能保证这些尚未实测的项目一定成功。

批准后仅拟执行以下一条提交命令；目前没有执行：

```bash
sbatch "$HOME/code_cpf/pi05/tools/pace-smoke-1f66d619e65200f0dde82da7b6a7f3a27500c560/pace_g1.slurm" "$HOME/code_cpf/pi05/tools/pace-smoke-1f66d619e65200f0dde82da7b6a7f3a27500c560" "$HOME/code_cpf/pi05/results/pace-preflight-1f66d619e65200f0dde82da7b6a7f3a27500c560"
```
