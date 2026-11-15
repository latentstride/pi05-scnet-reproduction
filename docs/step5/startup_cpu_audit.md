# ARX LoRA 启动链路审查

## 两次实跑失败

两次提交都使用 `ksagnormal01`、一张 `RTX4090-PCIE-24GB-LS`、8 CPU、64 GiB 主存、6 小时时限和显式 `code_cpf/pi05` 工作目录。历史提交命令、调度器回执、完整 stdout 和 Python traceback 位于项目根的 `logs/step5-arx-20261115/` 和 `logs/step5-arx-20261115-r1/`。`submission-intent.txt` 与各自历史验收的 `tested-command.txt` 逐字节相同。两个发布目录的 `train.py`、`model.py`、`env.sh`、`train.slurm` 均与各自归档相符。

| 作业 | 发布和退出位置 | Slurm 结果与进度 |
| --- | --- | --- |
| `128113950` | `4ea21ee` 的 `run_fit` 第 291–295 行。两个独立 JIT 的输出在第 295 行进入 `assert_allclose`。左值为 LoRA loss，右值为原生冻结 loss。两个调用使用同一个 `Observation`、FP32 `[1,50,32]` 动作目标及 PRNG key。 | `FAILED 1:0`，5 分 32 秒。LoRA/冻结 loss 约 0.11087/0.11028，相对差 0.535%，超过原容差。JAX allocator 达 7.24 GiB，尚无反向。 |
| `128115085` | `2ebd1e7` 的 `main:417 → run_fit:312 → check_initial_loss:89 → loss.lower → jax._src.stages.make_args_info → OpenPI 类型检查`。JAX 用 `ArgInfo` 叶子重建 `Observation`，Jaxtyping/Beartype 在 `images` 上拒绝该类型。 | `FAILED 1:0`，4 分 24 秒。51 个底座数组同源，10 个 A 有限且非零、10 个 B 全零。JAX allocator 达 7.05 GiB，尚无初始 loss、反向或 checkpoint。 |

第一作业的两个 JAX CUDA 编译缓存分别写于 21:23:52（LoRA）和 21:24:17（冻结）。解码后的 BF16 HLO 中，冻结 Q/KV/out 投影使用 Triton，LoRA 对应 GEMM 使用 cuBLAS，包含底座和 A 的拼接权重。两图的底座算子形状、归约与舍入路径不同。逐层数值贡献仍未实测。第二作业在 JAX AOT 参数元数据阶段退出，没有找到该次门禁落盘的 CUDA executable。 [128113950 的断言](https://github.com/latentstride/pi05-scnet-reproduction/blob/4ea21eead9ecabfd13fcdce0f9a62d82b1145103/scripts/step5_arx/train.py#L288-L297) [128115085 的 AOT 调用](https://github.com/latentstride/pi05-scnet-reproduction/blob/2ebd1e70eebdd3c45f4fbc54ac8c11cf92a4e1ce/scripts/step5_arx/train.py#L85-L103) [CUDA HLO 对照](admission.md#作业-128113950-的初始化损失)

| 原始证据 | SHA-256 |
| --- | --- |
| 作业 `128113950` stdout | `23a17f4e0511f017221a298597ad532acd68e33ae0422802d5d8c58d48711861` |
| 作业 `128115085` stdout | `6bb71cc74eef481fed93dfdc53a98bd5dce9b54581702c81c0af1ed4504bc711` |
| 作业 `128113950` 发布包 | `cb088097280218bf59adac728f5efb918cc92c3c2c239604ead063e02c051e85` |
| 作业 `128115085` 发布包 | `26b9db4a2c51985c802e75daae85b5145e5eeb4521159e22fe59e8f36427bf43` |
| 作业 `128113950` 提交命令 | `5649eddf39f99d2b41abcafeeba2ae70eb32b7be4a92e6a60eccc3b7eb986468` |
| 作业 `128115085` 提交命令 | `2c1fb5e3b3c776ceacaa3e234916518cc8fef74ea26272df748750d440fc9359` |
| SCNet CPU 原问题复现 `results/step5-arginfo-repro.wOARR4/reproduce.log` | `ee9a9467fe686c8ca45006b87f4444ac2fc80f4887d2e9c7ffc480237f5c621e` |

SCNet CPU 原问题复现用历史发布的 `env.sh cpu` 及 `2ebd1e7` 的生产 `check_initial_loss(flow_loss(...))`，以 `Dataset.read("train", 24, 3)` 和 `NativePolicy.batch()` 构造实际 `Observation`。模型使用全参数树和 5 项 Orbax BF16 恢复探针，其余以 shape/dtype 占位。`.lower()` 复现同一 `images`/`ArgInfo` 异常，复现脚本核对错误签名后退出 0。

## 修复与训练语义

AOT 对照把 `Observation` 的 PyTree 叶子作为外层编译输入，并在 `jit` 内按原 PyTree 重建 OpenPI `Observation`。`flow_loss` 仍执行原 `nnx.merge`、`pi.eval()` 及 `compute_loss(..., train=False)`。`optimizer_step` 仍执行 `pi.train()`、真实图像增强、`jax.value_and_grad`、Optax 裁剪与 AdamW。A/B、冻结权重、观测、动作和随机 key 仍为动态输入。全量加载检查两个图共用的 51 个底座数组及 10 对 A/B 的初始化值。

门禁用同一已编译 LoRA 图评估种子初始化和参数全零对照，也用这张图及同一评测噪声计算 20 步后的 loss。初始记录含两种 LoRA loss、冻结 loss、绝对与相对差异、容差、阈值、有限性和判定。保留 `rtol=1e-3`、`atol=1e-5` 和 20 步相对下降 1% 的阈值。冻结 loss 差异用于诊断，任一 loss 非有限仍停止。若在初始评估或单步结束时已收到停止请求，入口返回 75。非有限过拟合结果也会写入失败产物。

## SCNet CPU 验收方法

固定栈为 OpenPI-PACE `f4e52e6de4cc487a11585a3a7414d610b2883dae`，checkpoint 仓库版本 `17ef545270cf26ed81169893b60f845f80ce6557`，Apptainer 1.3.4 和 Python 3.11.18。实际项目根为 `/public/home/acvkpnrzir/code_cpf/pi05`，容器为其下的 `containers/openpi-ubuntu`，CPU 和 GPU 使用同一 venv、固定代码树和资产缓存。CPU 启动器强制 `JAX_PLATFORMS=cpu`；未调用调度器提交接口。

| 身份 | SHA-256 |
| --- | --- |
| 转换后的 HDF5 | `19da908478bd36cbf3e2c0b507479d53db6c178bfd07d16cff7a04ee07fd72ca` |
| 数据审计 JSON | `2282b0f6fe09d9fa4861fa3d875edc28def2d5c33a6a4746463e93c1cd1d8936` |
| 28 文件 checkpoint 清单 | `835d60676236e69276032f7bc1ea4837eb6fcda6d4f3dbab2043965063a3d97b` |
| Orbax 根 `manifest.ocdbt` | `585e745b3c42a97e11cf797f4a6c2af39e0bfc8a7f208ab8aa969110b88aef4f` |
| PaliGemma tokenizer | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |

首次修订的 CPU 回归（`results/step5-arx-startup-cpu.YyJBDj/cpu-validation.log`）退出 1。数据的 5 项测试通过，新增验收错误地要求宿主侧 token 数组为 `int32`。固定 tokenizer 用 `np.asarray(tokens)` 得到宿主侧 `int64`，JAX 关闭 x64 时按 `int32` 进入模型。修复后的验收记录两种 dtype，保留既有数据转换和训练输入。该轮没有生成 CPU 收据。

第二次候选的 BF16 小模型回归退出 1，记录位于 `results/step5-arx-startup-cpu.cU7lh1/cpu-validation.log`。同图门禁和正常 JIT 都执行到了实际模型，观察到 LoRA loss 约 1.57e18、冻结 loss 约 1.30，随后也出现非有限 loss 和梯度。保留这些失败记录。小模型改用 FP32 计算来验证实际更新及恢复机制，完整模型的 BF16 配置保持原样。

完整模型使用参数树和 dtype 约束、代表 Orbax 张量与真实 ARX 样本。它覆盖 AOT `.lower()`、冻结和 LoRA 评测、两种采样图、`train=True` 梯度及更新树，不实体化 3B 级底座或执行其模型计算。数值验收用小型 OpenPI `Pi0` 的同一 `pi05` 路径，保留 257,152 词表、三路 224×224 图、200 token、真实增强、50×32 目标、生产优化器与 loss。测试构造器将 Gemma 两分支缩为宽 32、1 层、MLP 64、2 个 query head、1 个 KV head、head dim 16、LoRA rank 2，视觉采用 SigLIP `mu/32`，底座为随机初始化后转 BF16，执行机制验收使用 FP32 计算，LoRA 保持 FP32。真实图像和动作来自同一批次。20 步循环两次执行实际梯度更新，并对测试中的最终比较值分别注入 0.5% 与 2% 来覆盖拒绝、放行和恢复分支。这个控制测试不能用于评定完整模型的过拟合表现。


### 完整模型结构级结果

`7da802f` 对应的 SCNet 运行完成，退出 0。校验了全部 28 个 checkpoint 文件和 tokenizer 的长度与 SHA-256。生产加载器对上 51 个冻结张量和 20 个 LoRA 张量，CPU 实际恢复其中 5 个 BF16 张量。LoRA 全树使用 FP32，AdamW 状态有 40 个 moments 和 1 个计数器。用同一真实输入完成生产 AOT `.lower()`、冻结评测、两种采样抽象输出及 `train=True` 的梯度与 Optax 更新追踪。训练返回的 LoRA 和 Optax 叶子类型、形状及 dtype 均与初始化一致。完整 LoRA 和初始优化器状态的 Orbax 保存、恢复和跳过部分写入通过。此运行不调用完整模型的前反向数值内核。

| 实际 ARX 样本 `(episode=24, row=3)` | 形状 | dtype |
| --- | --- | --- |
| `Observation.images` 的三路 RGB | 各 `[1,224,224,3]` | `float32` |
| 三路 `image_masks`，均为真 | 各 `[1]` | `bool` |
| `state` | `[1,32]` | `float32` |
| `tokenized_prompt` | `[1,200]` | 宿主 `int64`，JAX `int32` |
| `tokenized_prompt_mask`，有效 74 个 token | `[1,200]` | `bool` |
| 动作目标，后 18 维为零 | `[1,50,32]` | `float32` |
| PRNG key，底层数据 `[0,42]` | 标量 | `key<fry>` |

OpenPI 类型为 `openpi.models.model.Observation`，Flax PyTree 为 9 个数组叶子，FAST 专用的两个字段均为 `None`。完整树和各字段是由 `Dataset.read → NativePolicy.batch → Observation.from_dict` 生成的，记录在 `results/step5-arx-full-abstract.crfvHw/arx-input-contract.json`，SHA-256 为 `6a899712467854d886db9a65d9bab8d429e65f0920a36d9810e857a039301141`。同目录 `audit.log` SHA-256 为 `72f542e83cc469df81618160cd2a610a46f76d5b054f1bfc53d48ff7e3bb4c3b`，`cpu-full-only.json` SHA-256 为 `4b2a1941702339b6b1b2bcb35b2d1d93647fa8321e8c369d36cc8d376beed9cb`。后者只记录结构验收范围。

```bash
root=$(cd ~/code_cpf/pi05 && pwd -P)
release="$root/tools/step5-arx-release-7da802f33d63cf2324a2137b29afb77f3682d776"
out="$root/results/step5-arx-full-abstract.crfvHw"
cd "$root"
timeout 850 bash "$release/scripts/step5_arx/env.sh" cpu "$root" -m scripts.step5_arx.train audit-cpu \
  --project-root "$root" --dataset "$root/data/step5-arx/arx-drive-target-v1.h5" \
  --data-audit "$release/docs/step5/data_audit.json" --cpu-receipt "$out/cpu-full-only.json"
```


### SCNet CPU 数值诊断

在同一个容器、venv 和 `env.sh cpu` 入口，用独立算式 `sum(A[:, :, None] * B[None, :, :], dtype=float64)` 检查 JAX 的小矩阵乘法。输入从 JAX 拷回逐值相同，NumPy BLAS 和 JAX 逐元素乘加在正常 FP32 误差内。`[347,32] @ [32,16]` 的参考输出最大绝对值为 23.589134；默认 JAX CPU `matmul` 此次最大值为 0，eager 与 JIT 的最大绝对误差都为 23.589134。另一些形状出现异常大值。关闭持久编译缓存仍复现。独立验证关闭 fast math、Eigen 多线程、MKL-DNN 或 CPU thunk 各选项，均未修复。独立固定 OpenPI Q/KV/out 和 MLP 的算子对照在 BF16 与 FP32 都出现异常，故小模型切换到 FP32 无法证明结果可靠。

可查阅 `results/step5-arx-cpu-xla-check.ZciNRg/check.log`（SHA-256 `77116529f91b9d15c1919142ce265d8409d4487d3a75226a1abe768cd8935ca4`）、`results/step5-arx-cpu-dot-triage.2PtI5E/default.log` 和 `results/step5-arx-cpu-op-microprobe.7mBW7V/probe.log`（SHA-256 `8be1466076545572206d2c75cc068b62a8067e5fa03e463f12c6cbf7ddc28ff5`）。这些数值定位到该运行环境的 JAX CPU 点积执行，尚未定位到具体库或指令。该诊断独立于 π₀.₅、LoRA 和 Orbax。 [OpenXLA CPU 后端调试选项](https://github.com/openxla/xla/blob/df971129bd82e381954da0185b534220e21798a4/xla/debug_options_flags.cc#L880-L945)


同一批矩阵在 SCNet 以仅进程生效的 `XLA_FLAGS=--xla_cpu_use_xnnpack=true` 串行重跑后，5 个形状与独立参考相符，最大误差 6.676e-6。该次退出 0，日志 `results/step5-arx-cpu-dot-triage.2PtI5E/xnnpack-serial.log` 的 SHA-256 为 `402d9fbeb8e51b9716c1e1bb977c29f822f6ca7938019fb63dbe8caeb9e0e294`。这验证了所测的 CPU 点积路径。GPU 发布未加入此设置。

最终源码的默认 FP32 回归和 XNNPACK 回归均启动了容器 Python，但 SSH 客户端在 1200 秒限时内未取到测试结果。XNNPACK 的实际回归执行脚本为 `results/step5-arx-startup-xnnpack.ojUrw5/cpu-xnnpack-review.sh`，日志为同目录 `cpu-validation.log`。脚本在调用生产模块前设置 `XLA_FLAGS=--xla_cpu_use_xnnpack=true`、`JAX_ENABLE_COMPILATION_CACHE=false`，仅在 CPU 进程内生效。最后可见的完整回归日志停在解释器和环境初始化输出，尚无本轮真实更新及恢复结果。

### 准入判定

**`CPU_NOT_READY`。** 真实输入的 AOT 问题已修复，完整模型结构审计通过。本轮尚无可信的小模型实际前反向、AdamW 更新和更新后 Orbax 恢复的完整往返，第三次 GPU 作业不进入最终提交审核。

| 结果 | SCNet 命令与证据 |
| --- | --- |
| 原始失败复现通过，退出 0 | 从真实项目根用历史 `2ebd1e7` 发布的 `env.sh cpu` 执行 `results/step5-arginfo-repro.wOARR4/reproduce.py` 的代码。日志保留旧的 `check_initial_loss` 完整 traceback。 |
| 发布包和 Bash 路径检查通过，退出 0 | `bash -n` 分别检查 `env.sh` 和 `train.slurm`。用 `bash "$out/step5-arx-startup-shell-review.sh" "$out" "$release"` 执行登录节点 Bash 4.2 回归。测试替身验证单卡和无显式掩码启动、拒绝多卡掩码、拒绝 staging 项目根，以及新任务和恢复路径收到 USR1 后退出 75。没有启动实际 CUDA 容器。 |
| 完整模型结构、恢复样本及初始状态 Orbax 通过，退出 0 | 上节 `audit-cpu` 命令及 `results/step5-arx-full-abstract.crfvHw/audit.log`。 |
| 缩小模型实际数值验收未通过 | 首个 BF16 小模型运行退出 1，11 项中 3 项报错。真实训练更新出现 `loss=15.075986862182617, grad=inf, change=nan`。无法完成更新后 Orbax 往返、20 步到正式训练的数值入口。拒绝非有限 loss 和非法更新指标的检查已实际触发。 |
| CPU 数学诊断完成 | `results/step5-arx-cpu-dot-triage.2PtI5E/probe.py` 用同一 `env.sh cpu` 执行，并输出每个矩阵形状的独立参考误差。诊断程序完成退出 0，各矩阵乘法的正确性由记录的差值判定。 |

SCNet 运行源码固定 `7da802f33d63cf2324a2137b29afb77f3682d776`，位于项目根的 `tools/step5-arx-release-7da802f33d63cf2324a2137b29afb77f3682d776/`。训练代码指纹 `76af52b077c5f2824965504c377f554001b9fec2f4d501236dd81c2d13857e50`，发布包 SHA-256 `6a821b30a1d4aca001b245b03c2f9bd5c14287b9b7d185c5961e7940cb5c6be4`。最终发布的 Bash 回归日志位于 `results/step5-arx-startup-cpu.HvMrwc/shell-review.log`，SHA-256 `73978c7005fe97c7772b0e1bcabcea9a017fb364d8212dd4b51fae07111bf8cf`。Bash 审计实际调用的 `train.slurm` 与发布归档一致，也已核对脚本委托到该发布的 `env.sh … -m scripts.step5_arx.train fit`。

旧收据 `69c89c7b93216414dd2adc1347242d61fa1380a2579c5b547656636323c272cf` 的代码指纹与新发布不符。新结构级记录已保存，发布目录尚无可用于 GPU 准入的 `cpu-receipt.json`。完整数值验证须由同一真实 ARX 批次和可靠的 SCNet CPU 后端重新运行，包括生产 `run_fit` 的训练和恢复路径。省略数值验收或使用旧收据会丢失这部分证据。


## CUDA 尚待验收

`GPU_UNVERIFIED` 涵盖当前发布上完整模型的 CUDA 同图初始数值、`train=True` 首次真实前反向、实际梯度和更新、4090 峰值、1% 过拟合门槛、400 步正式训练及 GPU 上的保存恢复。

现有 Slurm 串行入口在同一分配恢复完整权重、计算初始 loss、执行单步前反向及 AdamW 更新、记录显存、做 20 步小样本过拟合。前述门禁通过后才开始冻结验证和 400 步正式训练。异常会停止，日志会记录 JAX allocator。环境保持 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 和 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`。RTX 4090 首次梯度编译的临时 buffer、碎片、CUDA scratch/上下文开销及编译时长仍未知。若遇 OOM，应保留 stdout、阶段、JAX allocator、编译缓存和 Slurm 退出记录，判断失败发生于编译、前反向或之后的分配。若 20 步 loss 未降至少 1%，脚本记录固定评测 loss 与更新指标并停止。正式训练从最近完整 checkpoint 恢复，同时核对 AdamW 计数和保存步数。
