# G1 修订发布包免费预检（2026-11-05）

最终发布包 `02ff7dd` 已通过完整 SCNet CPU 预检，当前等待用户明确批准提交综合 G1。旧包记录保留为历史证据。GitHub main 的运行脚本提交为 `69181a6`，SCNet 使用下文记录的只读发布包。

## 本次修复及审查

- 驱动发现：保留原 ICD 可解析时的行为；仅当已安装的 NVIDIA EGL 可用时生成备用 ICD，仍要求后续宿主/容器 Vulkan 和分配卡 UUID 验证。发现失败前保存 ICD 与链接器缓存证据。
- 就绪日志：用二进制逐行读取，保留未完成的行，避免非 UTF-8 输出或分段写入误报模型启动失败。
- 权重后端：将误拼写的 `model.safensors` 检查改为上游实际使用的 `model.safetensors`，避免意外切换到 PyTorch 恢复路径。
- 异常 ICD：配置结构错误进入明确诊断路径，保存错误证据。
- 发布预检：驱动发现和三阶段监督回归随只读发布包打包，在 SCNet 宿主 Python 运行；这项失败时不生成完整通过凭证。

审查了观测编码、动作顺序与范围、WebSocket 关闭、超时清理、G1 阶段依赖、显存门槛和 Slurm 调用关系。入口仍是单个 `pace_g1.slurm`，内部通过 Bash 调用图形前置检查后运行监督程序；没有内部 `sbatch/srun/salloc`、自动重排或第二个模型恢复。冻结 LIBERO 跟踪文件未改动，未重新运行基线。

## 历史发布包 9d92996 的结果

| 检查 | 结果 |
| --- | --- |
| 本地 G1 相关回归 | 20 项通过 |
| SCNet 宿主驱动与监督回归 | 10 项通过 |
| 本地/远端发布文件 SHA-256 | 通过 |
| 8 个 Bash/Slurm 脚本语法 | 通过 |
| 28 文件、tokenizer、归一化统计 SHA-256 | 首轮 177.655 秒，复验 105.228 秒，均通过 |
| GitHub main 8 个修改文件 | 与本地修订逐个比对一致 |
| 模型 CPU 完整导入与设备检查 | 首轮在 OpenPI 导入超时。复验通过，导入 378.430 秒，JAX CPU 设备检查 5.685 秒 |
| 仿真 CPU 导入、资产与合成闭环 | 两轮均通过，复验回读 2 次查询、8 个控制步、9 帧视频 |
| 完整发布包门槛凭证 | 复验三项退出码均为 0，凭证与发布清单 SHA-256 一致 |

回归覆盖正常有序执行、单模型进程跨阶段复用、场景初始化异常、模型服务异常、坏动作、超时、显存不足、资源清理，以及缺失库、错误 ICD、日志编码和分段写入。

## 历史发布包 9d92996 的版本及证据

- 运行发布提交：`9d92996cd68be16a3fdffc52e86941eebff5887f`（本地 Git 导出），代码及测试同步到 GitHub main `207d75ca1818706bee8d27a2ba86a2b0ff21edb9`。
- 发布清单 SHA-256：`982b7339c84a4429993bb371b2b2379f2b04ec3da2732ee8ddc36ff3364ae6a7`。
- SCNet 发布目录：`tools/pace-smoke-9d92996/`。
- SCNet 首轮预检为 `results/pace-preflight-9d92996/`，通过凭证在 `results/pace-preflight-9d92996-recheck/`。
- 本地首轮证据为 `research/g1/preflight-9d92996/`，复验证据为 `research/g1/preflight-9d92996-recheck/`。

首轮状态为宿主 0、仿真 0、模型 124。模型达到 1200 秒硬上限。调用栈和导入计时显示 OpenPI 间接加载了训练数据依赖，期间有 PyAV、PyArrow 等模块加载进展。根因尚不能确定。恢复 SSH 后，用新目录复验同一个只读发布包，沿用固定源码、容器及预检时限。保留两轮日志。

## 尚需实机验证

本次 OpenPI 导入耗时约 6 分 18 秒。GPU 模型就绪阶段的 13 分钟还需覆盖设备识别和 Orbax 恢复。首轮导入超时说明耗时仍可能波动，GPU 阶段的余量以作业实测为准。

失败作业 `126779874` 所在节点是否具备可加载的 NVIDIA EGL 库仍未知；本次驱动回归使用模拟文件布局，不能证明该节点图形环境正常。真实场景/三相机、CuRobo CUDA 导入与构造、Orbax GPU 恢复、首次 JAX 编译和模型/仿真并发显存均需在未来获批的同一个 G1 分配中验证。

资源配置未提高：1 GPU、4 CPU、32 GiB、50 分钟；按此前历史 GPU 单价满时 GPU 部分估算约 3 元，实际报价待授权时核对。本次未获新的提交授权；旧版本的提交入口已经由最终修订的入口替代。


## 最终修订 02ff7dd

本次实际修复包含分阶段模型启动监督、统一作业截止时间、模型 SIGTERM 落盘，以及更完整的节点驱动诊断。运行代码提交为 `02ff7dd0fa8108c2e70fe990dfe3576982139807`。发布清单哈希为 `13e61dcc82282099af93dc8168947a88d7e0f2a71dd5cf9b837dab678f22b57d`。本地 G1 回归 24 项通过。SCNet 最终包宿主回归 13 项通过，宿主、仿真、模型退出码均为 0。新通过凭证与上述清单哈希一致。模型文件校验耗时 109.918 秒，OpenPI 导入 347.686 秒，JAX CPU 检查 5.260 秒。合成通信产物回读为 2 次查询、8 步和 9 帧。完整证据位于 `research/g1/preflight-02ff7dd/`。

SCNet 最终发布目录为 `tools/pace-smoke-02ff7dd/`，完整 CPU 预检目录为 `results/pace-preflight-02ff7dd/`。只有该目录的最终通过凭证可用于这个版本。

| 阶段 | 单项上限 | 超时证据 |
| --- | --- | --- |
| 模型容器与 Python 启动 | 180 秒 | model-server.log 的启动标记，以及 G1.2-exception.log 的 container_startup 失败阶段 |
| 模型文件身份与小文件复核 | 180 秒 | 模型 phases.jsonl 与 stacks.log |
| OpenPI 完整导入 | 1200 秒 | 模型阶段调用栈，明确记录 openpi_import |
| JAX GPU 设备初始化 | 120 秒 | jax_device 阶段事件与调用栈 |
| Orbax 恢复和策略构建 | 600 秒 | orbax_restore_and_policy_create 阶段事件、失败摘要 |
| 首次 JAX 编译及真实观测推理 | 客户端 390 秒，父进程 420 秒 | first_jax_compile_and_inference、query-phases.jsonl、query-status.json 和 query-stacks.log |

模型整体就绪上限为 1800 秒。单项上限共享作业总预算，不能相加推定为总运行时间。三个 G1 阶段共用一次资源分配、一个模型服务和作业启动后第 49 分钟的执行截止时间，第 50 分钟为 Slurm 硬上限。监督程序将每阶段实际可用时间写入 `effective_timeouts_s`。超时会终止该阶段、保存异常并跳过依赖它的步骤。

首次 JAX 编译发生在第一次真实观测推理中，日志阶段同时覆盖编译和推理。单独的编译器耗时由 JAX 编译日志记录，阶段总耗时不等同于纯编译耗时。

驱动发现现在保存节点名、系统版本、NVIDIA 驱动版本、相关环境变量、原始 ICD 文档、解析结果，以及 ldconfig 标准输出、错误输出和退出码。绝对 ICD 路径、按 soname 查找、GLX 无法解析时检查已安装 NVIDIA EGL 等分支均有 CPU 回归。`gnode65` 的真实 EGL 能力仍未确认。

SCNet 只读查询确认 `gnode39` 属于 `ksagnormal01`，其 GPU 型号为 RTX 4090，平台的 sbatch 支持 `--nodelist`。因此拟提交脚本加入 `#SBATCH --nodelist=gnode39`，限定到之前完成图形验收的节点。该限制会缩小调度范围，可能增加等待时间。实际作业准入仍以用户授权提交后的调度回执为准。本次只查询节点和 sbatch 帮助，没有提交调度请求。

最终拟提交入口只允许下面这一条命令，从个人 SCNet 项目根目录执行。完整 CPU 预检已通过，等待用户明确批准后方可执行。

```bash
sbatch "$HOME/code_cpf/pi05/tools/pace-smoke-02ff7dd/pace_g1.slurm" "$HOME/code_cpf/pi05/tools/pace-smoke-02ff7dd" "$HOME/code_cpf/pi05/results/pace-preflight-02ff7dd"
```
