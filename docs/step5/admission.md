# LoRA 配置与训练准入

| 项目 | 状态 | 验收范围 |
| --- | --- | --- |
| 50 条轨迹及原生策略 I/O | **CPU 实测通过** | 40/5/5、真实数据集全部边界、29,040 张图逐像素对照、时序、动作往返及 CPU 回放命令对齐。 |
| 模型文件、运行时和 LoRA 参数图 | **SCNet CPU 实测通过** | 28 个既有文件的长度和 SHA-256，tokenizer 校验，固定源码和 CPU 设备，实作 NNX 图抽象形状审计。 |
| Orbax 与训练结构 | **SCNet CPU 实测通过** | 全树路径与形状、5 个 BF16 代表张量、真实训练批次、生产梯度/更新函数的抽象追踪。LoRA 与 AdamW moments 经 Orbax 保存和恢复后逐值相等。 |
| 全量模型实体化 | **既有 GPU 作业已通过加载** | 作业 `128113950` 在 RTX 4090 完成全量恢复，随后在初始损失断言停止。修订后的初始化断言待 GPU 实测。 |
| SCNet 项目根 | **CPU 实测通过** | 容器从显式项目根启动。用源码 staging 路径冒充项目根的调用已拒绝。该卷读回约 100 GiB 空闲。 |
| GPU 训练门禁 | **已实现，未执行** | 单步可测量更新、作业 `128113950` 在单步前退出，尚无训练、过拟合和训练显存结果。退出前 JAX allocator 峰值为 7.24 GiB。 |
| 线上原生对照 | **待实现仿真编排** | 已提供两种模型的策略接口与同一离线对照入口。原生仿真调用、配对场景锁及动态闭环尚待实施。 |

## 固定栈与优化器

使用 OpenPI-PACE `f4e52e6de4cc487a11585a3a7414d610b2883dae`。原 PACE 注册项对应 π₀.₅、PaliGemma 2B、动作专家 300m、离散状态输入、200 token、50×32 内部动作接口。公开 task checkpoint 提供 Orbax `params` 和源统计；第五步从相同权重初始化，初始化新的 AdamW 状态，记录 ARX train-only 统计。固定 `scripts/train.py` 的默认 loader 走 LeRobot 配置；新增入口复用同一 NNX π₀.₅、官方输入变换、flow-matching loss 和 Orbax 恢复器，以 HDF5 窗口加载数据。LIBERO、PyTorch 权重、原任务 EMA/优化器状态均不进入这个入口。 [OpenPI-PACE 固定配置](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/pace/src/openpi_pace/configs.py) [训练入口](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/scripts/train.py)

抽象实例化并检查实际 NNX 路径集合。20 个 LoRA 张量共 49,987,584 个参数，51 个底座张量共 3,353,433,872 个参数。两个分支各 18 层，LoRA 实际挂在 Q/KV/attention 输出投影和 MLP 的 gating/linear 投影。PaliGemma rank/alpha = 16/16，动作专家为 32/32。全部 vision、embedding、norm、base attention/MLP kernels 和 action/time projections 冻结；逐张量路径、shape、数量和 dtype 见[参数清单](parameter_layout.json)。上游 `get_freeze_filter()` 范围集中在 LLM 内。第五步以真实 LoRA 参数路径显式划分，只有这 20 个叶子进入微分与 Optax moments。 [Gemma LoRA 挂载](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/src/openpi/models/gemma.py) [上游 freeze filter](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/src/openpi/models/pi0_config.py)

底座 BF16、LoRA FP32。A 用固定 seed 的 N(0,0.01) 初始化，B 置零，使初始残差为零。优化器采用常数 3e-5、全局梯度裁剪 1.0、weight decay 0，无 EMA，batch 1。主模型继续预测 50×32，官方 flow loss 同时约束补零的 18 个动作维度。训练前比对 Orbax 与无 LoRA / 有 LoRA 模型的底座路径和形状，拒绝不匹配的结构。SCNet CPU 实测依赖为 JAX 0.5.3、Flax 0.10.2、Optax 0.2.4、Orbax 0.11.13 与 h5py 3.13.0。

11 月 15 日的收据与已验收发布保存在项目根下的 `results/step5-cpu-admission.rJvqzB/`，其中 `cpu-receipt.json` 是训练准入输入，`cpu.log` 保存该次 CPU 过程。收据 SHA-256 为 `8cca28981e2db38f6f43ca6b31531fe3f7ed0b7f9899f73432ae0cfc62e623cb`，对应 PR 的 `a24c5c1`。实测的冻结 Orbax 元数据含 51 个 FP32 张量，训练以 BF16 恢复底座。CPU 批次形状为 `[1,50,32]`，提示及状态合计 72 个有效 token，AdamW 有 41 个状态数组（40 个矩张量与 1 个计数器）。收据仅代表 CPU 通过；CPU 抽象追踪未执行全量模型计算核。

移除固定显存下限后的发布 `2242539` 已重新通过同一 SCNet CPU 验收。最新收据位于项目根下 `results/step5-cpu-4090.b5uXJy/cpu-receipt.json`，SHA-256 为 `81fada3cdbb2a1505a82aa6812810a5b4a70b31e288b923b10700e0d7b9418e6`。代码指纹与新发布一致。模拟 RTX 4090 准入检查通过，CPU 设备和过期收据仍被拒绝。该收据保留作 `2242539` 的历史记录。后续启动和恢复修订使用各自发布目录内的新 CPU 收据。实际显存峰值待 GPU 门禁。

## 作业 128113950 的初始化损失

作业使用 `4ea21ee`，完成 GPU 加载并执行两次真实前向，初始 LoRA 损失为 0.11087，原生冻结损失为 0.11028。日志保留的绝对差为 0.00058995，相对差为 0.00534957。两次调用用同一观测、动作、PRNG key 和 `train=False`。旧跨图断言先于梯度更新退出，未产生训练 checkpoint。原 Slurm 日志的 SHA-256 为 `23a17f4e0511f017221a298597ad532acd68e33ae0422802d5d8c58d48711861`。

失败作业只从 Orbax 恢复一次 BF16 底座，并向两个 NNX 状态绑定同一个字典。冻结图和 LoRA 图的非 LoRA 路径及形状一致。固定实现的 attention 和 MLP 把两级低秩乘积加到底座结果上，B 为零时该残差为零，初始化保留非零 A 供首步学习 B。修订在运行时核对逐路径数组引用和实际 A/B 数值。 [固定 LoRA 算子](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/src/openpi/models/lora.py)

从作业留下的两份 JAX 0.5.3 缓存中解压实际 CUDA 优化后 HLO，按算子元数据匹配同一底座投影。缓存写入时间与原调用顺序一致，LoRA 图比冻结图多 20 个入口张量。样例均产生 BF16 输出。Q 的合并输入在 HLO 中明确将 2048×2048 底座权重与 2048×128 的 A 权重拼成 2048×2176。

| 共享投影 | 原生冻结图 | LoRA 图 |
| --- | --- | --- |
| PaliGemma Q | Triton GEMM，内部矩阵 968×2048 | cuBLAS 合并 GEMM，矩阵 2176×968，含 8×16 个 LoRA A 输出 |
| PaliGemma KV | Triton GEMM，矩阵 512×968 | cuBLAS 合并 GEMM，矩阵 544×968，含 2×16 个 LoRA A 输出 |
| PaliGemma attention 输出 | Triton GEMM，内部矩阵 968×2048 | cuBLAS 合并 GEMM，矩阵 968×2064，含 16 个 LoRA A 输出 |

这里直接观察到了同一底座投影的 CUDA 算子选择和乘法形状变化。动态 A/B 进入编译参数，B 的运行值为零时仍执行 LoRA 图中不同的底座 GEMM。不同的归约与舍入路径可以使 BF16 前向出现数值差异。该分析确认了跨图比较的编译差异，尚未通过 CUDA 逐层重放量化各算子对 0.535% 的贡献。 [JAX 0.5.3 缓存格式](https://github.com/jax-ml/jax/blob/jax-v0.5.3/jax/_src/compilation_cache.py) [对应的 OpenXLA HLO 格式](https://github.com/openxla/xla/blob/df971129bd82e381954da0185b534220e21798a4/xla/service/hlo.proto)

| 原始压缩缓存 | SHA-256 |
| --- | --- |
| LoRA `jit_loss-b20c23c…-cache` | `73db1a728b31d97f8aeecc297b569271bc7ceaebe78936b1c184d1c608058499` |
| 冻结 `jit_loss-a1709b7…-cache` | `b60472e6e22cb296b915c42177e59909f2572df228d900108ed2280d967413e9` |

SCNet 在相同版本容器完成 8 项 CPU 回归。同一可执行图的对照保留原容差，独立冻结损失记录超过旧阈值的差异。人为非零 B 同时被参数检查和前向检查拒绝，A 全零也被拒绝。CPU 生产绑定检查覆盖两图 51 个相同底座引用，其中 5 个是实际恢复的 BF16 Orbax 张量，其余为形状占位；实际 LoRA B 全零 10 个，A 有限且非全零 10 个。新 CPU 收据在项目根下 `results/step5-lora-init-cpu.qdjocY/cpu-receipt.json`，SHA-256 为 `69c89c7b93216414dd2adc1347242d61fa1380a2579c5b547656636323c272cf`。修订后的 GPU 门禁尚未执行。

## GPU 串行门禁

1. 校验 CPU 收据、代码与数据哈希、容器依赖、绝对路径、磁盘及单个 CUDA 设备，记录 JAX allocator 容量后恢复冻结底座。核对两个模型共用同一套恢复后的底座数组，逐张量核查有限且非全零的 A 与全零 B。对固定样本、flow RNG 和 token，用同一个编译后的 LoRA 图比较种子初始化与 A/B 全零的损失，沿用 `rtol=1e-3`、`atol=1e-5`。独立编译的原生冻结损失写入 `initial_loss.json`，供跨图诊断。运行一轮真正前反向，要求 loss 有限、LoRA 梯度与更新范数非零，并记录 allocator 峰值。RTX 4090 24 GiB 可进入这项实测门禁。
2. 从原初始化重启，固定一个 train 窗口做 20 步过拟合。使用同一评测噪声和无增强输入比较前后 loss，要求 loss 有限且相对下降至少 1%。阈值是提前设定的诊断门槛，训练脚本遇失败退出并记录数据。
3. 从原初始化启动正式 400 步。固定 seed 派生窗口和 flow RNG，每 25 步保存，每 100 步离线验证。由验证 loss 与动作误差审阅下一轮是否调参。
4. 选定训练方案后运行独立的 `test` 命令，对同一测试集同时输出冻结与 LoRA 成对指标。HDF5 模式检查曾覆盖测试集边界，训练与 checkpoint 选择均只使用 Train/Validation。
