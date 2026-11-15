# LoRA 配置与训练准入

| 项目 | 状态 | 验收范围 |
| --- | --- | --- |
| 50 条轨迹及原生策略 I/O | **CPU 实测通过** | 40/5/5、真实数据集全部边界、29,040 张图逐像素对照、时序、动作往返及 CPU 回放命令对齐。 |
| 模型文件、运行时和 LoRA 参数图 | **SCNet CPU 实测通过** | 28 个既有文件的长度和 SHA-256，tokenizer 校验，固定源码和 CPU 设备，实作 NNX 图抽象形状审计。 |
| 全量模型实体化 | **GPU 待验收** | 登录节点整棵 CPU restore 进程退出 137。全量 GPU restore 在训练前由固定 dtype 的 OpenPI 恢复器执行。 |
| GPU 训练门禁 | **已实现，未执行** | 单步可测量更新、逐步 20 次拟合、正式训练、验证/测试误差、峰显存与恢复均无本轮 GPU 结果。 |
| 线上原生对照 | **待实现仿真编排** | 已提供两种模型的策略接口与同一离线对照入口。原生仿真调用、配对场景锁及动态闭环尚待实施。 |

## 固定栈与优化器

使用 OpenPI-PACE `f4e52e6de4cc487a11585a3a7414d610b2883dae`。原 PACE 注册项对应 π₀.₅、PaliGemma 2B、动作专家 300m、离散状态输入、200 token、50×32 内部动作接口。公开 task checkpoint 提供 Orbax `params` 和源统计；第五步从相同权重初始化，初始化新的 AdamW 状态，记录 ARX train-only 统计。固定 `scripts/train.py` 的默认 loader 走 LeRobot 配置；新增入口复用同一 NNX π₀.₅、官方输入变换、flow-matching loss 和 Orbax 恢复器，以 HDF5 窗口加载数据。LIBERO、PyTorch 权重、原任务 EMA/优化器状态均不进入这个入口。 [OpenPI-PACE 固定配置](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/pace/src/openpi_pace/configs.py) [训练入口](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/scripts/train.py)

抽象实例化并检查实际 NNX 路径集合。20 个 LoRA 张量共 49,987,584 个参数，51 个底座张量共 3,353,433,872 个参数。两个分支各 18 层，LoRA 实际挂在 Q/KV/attention 输出投影和 MLP 的 gating/linear 投影。PaliGemma rank/alpha = 16/16，动作专家为 32/32。全部 vision、embedding、norm、base attention/MLP kernels 和 action/time projections 冻结；逐张量路径、shape、数量和 dtype 见[参数清单](parameter_layout.json)。上游 `get_freeze_filter()` 范围集中在 LLM 内。第五步以真实 LoRA 参数路径显式划分，只有这 20 个叶子进入微分与 Optax moments。 [Gemma LoRA 挂载](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/src/openpi/models/gemma.py) [上游 freeze filter](https://github.com/niejunnan25/Openpi-PACE/blob/f4e52e6de4cc487a11585a3a7414d610b2883dae/src/openpi/models/pi0_config.py)

底座 BF16、LoRA FP32。A 用固定 seed 的 N(0,0.01) 初始化，B 置零，使初始残差为零。优化器采用常数 3e-5、全局梯度裁剪 1.0、weight decay 0，无 EMA，batch 1。主模型继续预测 50×32，官方 flow loss 同时约束补零的 18 个动作维度。训练前比对 Orbax 与无 LoRA / 有 LoRA 模型的底座路径和形状，拒绝不匹配的结构。

## GPU 串行门禁

1. 校验 CPU 收据、代码与数据哈希、容器依赖、绝对路径、磁盘及实际 JAX 设备内存，恢复冻结底座。对同一固定样本、flow 噪声和 token 条件，比较初始 LoRA 与冻结模型损失。运行一轮真正前反向，要求 loss 有限、LoRA 梯度与更新范数非零。
2. 从原初始化重启，固定一个 train 窗口做 20 步过拟合。使用同一评测噪声和无增强输入比较前后 loss，要求 loss 有限且相对下降至少 1%。阈值是提前设定的诊断门槛，训练脚本遇失败退出并记录数据。
3. 从原初始化启动正式 400 步。固定 seed 派生窗口和 flow RNG，每 25 步保存，每 100 步离线验证。由验证 loss 与动作误差审阅下一轮是否调参。
4. 选定训练方案后运行独立的 `test` 命令，对同一测试集同时输出冻结与 LoRA 成对指标。HDF5 模式检查曾覆盖测试集边界，训练与 checkpoint 选择均只使用 Train/Validation。
