# 第五步 RTX 4090 提交前回执

2026-11-16，昆山 SCNet。Draft PR #6 的运行代码提交为 `642e19625857c4d1d107d670b1075eadb7608a14`。PR 后续文档回执提交不改变运行代码、归档或准入身份。提交审核结论为 **`READY_FOR_USER_APPROVAL`**。CPU 数值状态仍为 **`CPU_NUMERIC_UNRESOLVED`**。这次仅完成提交前工程验收。未创建任何 Slurm 作业，未执行 `sbatch` 或 `sbatch --test-only`，未分配 GPU。

## 发布与输入

| 项目 | 实际值 |
| --- | --- |
| SCNet 项目根 | `/public/home/acvkpnrzir/code_cpf/pi05` |
| 发布目录 | `$root/tools/step5-arx-release-642e19625857c4d1d107d670b1075eadb7608a14` |
| 发布归档 SHA-256 | `7e9ce4c93e5ef544ae8b834c16d30b114ad6e6759fcb63f32e743261dacfe862` |
| 生产代码指纹 | `c2784083df74a3ad9a82f7f222acb548904b1211f5fb1ca847695b90452dd688` |
| ARX HDF5 SHA-256 | `19da908478bd36cbf3e2c0b507479d53db6c178bfd07d16cff7a04ee07fd72ca` |
| OpenPI-PACE / PACE checkpoint | `f4e52e6de4cc487a11585a3a7414d610b2883dae` / `17ef545270cf26ed81169893b60f845f80ce6557` |
| 审核及日志目录 | `$root/results/step5-gpu-review-642e196` |
| 待授权记录 | `$review/admission-review.json`，`authorization.state=PENDING`，SHA-256 `13b1413591ede2cad7ec2cba13a4bc18e0318a6b6249eecf4fa7cb8aee72d61a` |
| 运行目录 | `$root/results/step5-arx-4090-validation-642e196`，待创建 |

SCNet 解包前校验了归档，部署后用生产入口核对了实际文件内容、HDF5、OpenPI 源码及 tokenizer、权重快照、运行配置和容器身份。启动用 Apptainer 1.3.4、Python 3.11.18、JAX 0.5.3、Flax 0.10.2、Optax 0.2.4、Orbax 0.11.13。模型文件和依赖沿用既有安装。空闲空间实测 99 GiB，门禁要求 16 GiB。

历史证据保持原状态。早期两次 GPU 作业成功加载权重，分别在跨图初始损失断言和真实 `Observation` 的 AOT 元数据重建处退出。现在初始门禁用同一可执行图和原容差，真实观测以叶子数组传入 AOT，随后在图中重建。完整模型 CPU 结构追踪、一次真实小模型 LoRA 更新、51 个冻结叶子检查和 Orbax 恢复有既有回执。第二步 CPU 梯度不一致仍未解决。待授权记录绑定这些结构及数值证据的原文件 SHA-256。参见[两次启动失败与 CPU 验收](startup_cpu_audit.md)和[最后一次 CPU 定向诊断](gpu_submission_review.md)。

## 本轮实测

| 检查 | 节点与结果 |
| --- | --- |
| 本地语法，归档 Git PAX、远端校验与 Bash 语法 | `py_compile`、`git diff --check`、`bash -n`、SCNet `sha256sum -c` 退出 0。 |
| 生产 CPU 容器，9 项定向测试 | `login09`，退出 0，`targeted-tests.log` SHA-256 `4396c0503ab7b49d79a444a7bc6abae6471627581dc78b1aa2523cd69e2a45ec`。 |
| 一次性例外 | 临时 Ed25519 测试密钥的签名通过，待授权、缺签、字节篡改、过期身份、历史收据、资源或作业号不符均被拒绝。验证必须先预留，8 个并发消费请求只允许 1 个成功。 |
| 真实 ARX 输入 | 实际生产 HDF5 与 `NativePolicy.batch` 产生三相机、74 个有效 token 与 `[1,50,32]` target。轻量计算重现直接 `lower` 的 `ArgInfo` 错误，新入口在同一编译图的初始差为 0。独立冻结值的人工偏差只记录为诊断。 |
| GPU A/B/C 的 CPU 替身测试 | A 步 2 使用步 1 返回的 LoRA 和 AdamW 状态，计数为 1、2。B 次序为 1–20，阈值 1% 拒绝 0.5%。注入第二步非有限梯度、OOM 和冻结值变化均阻止正式训练并留下失败阶段。 |
| 正式训练与恢复 | 替身沿真实入口走完 400 步，使用真实 Orbax 成功保存和读取小状态。第 26 步收到停止请求，退出 75。继续时跳过未完成的第 50 步目录，从 26 恢复并执行第 27 步，最终保留第 400 步和四轮验证结果。 |
| Slurm 外壳测试 | `login07`，Bash 4.2.46，`scontrol` 与 Apptainer 使用替身，真实外壳传递作业 ID、分配记录和 GPU 可见设备，拒绝两个 GPU 和 staging 项目根。新运行及恢复的 USR1 协作停止均透传 75，临时文件清理。测试外层退出 0，脚本和输出见审核目录的 `step5-exception-shell-run.sh` 和 `step5-exception-shell-run.log`。 |
| 真实发布包待授权拒绝 | `login03`，`prepare-exception` 退出 0。同一待授权记录的 `preflight --reserve-submission` 退出 1，`PermissionError: CPU numerical exception has no explicit approval`。未进入调度器，未产生签名或训练目录。`pending-preflight.log` SHA-256 `d030e0d7fa5cbccfa6e1faa8a1870aa7258e66fab3914b804f1632970744893c`。 |

CPU 定向命令从项目根运行，`release`、`dataset` 取上表绝对路径。

```bash
cd "$root"
export APPTAINERENV_STEP5_ARX_DATASET="$dataset"
timeout -k 10s 480s bash "$release/scripts/step5_arx/env.sh" cpu "$root" \
  -m unittest discover -s "$release/tests" -p 'test_step5_[ag]*.py' -v
```

新作业的阶段日志为 `gpu_progress.json`、`initial_loss.json`、`gpu_stage_a_0001.json`、`gpu_stage_a_0002.json` 及 `gpu_overfit_*.json`。A 比对冻结张量的逐值哈希，B 以原固定样本和相对下降至少 1% 的门限验收。两者通过才生成 `gpu_gates.json` 并启动 C。C 每 25 步写完整 Orbax checkpoint，每 100 步验证。异常抛出时 `gpu_failure.json` 记录最近阶段、错误、allocator 统计和最近完整 checkpoint。进程遭硬杀时保留此前已写阶段和完整 checkpoint，以 Slurm 退出记录补足错误诊断。任何续跑都需要新作业授权并使用同一训练身份。

## 当前核价与资源

2026-11-16 登录后刷新了昆山共享 RTX 4090 商品页，当前 **¥3.60/卡时**。账号资源列表将该产品映射到 `ksagnormal01`。未提交的基础模板中选择该队列，`GPU/节点=1` 时每节点核数锁定为 8，默认内存显示 124.3516 GB。实际脚本只申请 **1 节点、1 task、1× RTX 4090 24 GiB、8 CPU、64 GiB RAM、6 小时**，batch size 1。资源申请没有节点独占。 [当前商品价](https://www.scnet.cn/ui/mall/detail/goods?type=base&goodsId=821418588551249920&goodsName=NVIDIA+4090+%E6%98%BE%E5%AD%9824GB+PCIE&purchasingOption=FORMAL&goodsType=SHARE&goodsResourceType=GPU&resourceType=SPECIAL&goodsResourceId=AMD+7543-64C-1T-8*4090-1&shopId=10001&common1=RESOURCE&id=AMD+7543-64C-1T-8*4090-1_FORMAL_SHARE&resource=RESOURCE&deliveryType=CLUSTER_QUEUE) [账号资源页](https://www.scnet.cn/ui/console/index.html#/my-product/basic-resources/computing-cesource) [队列基础模板](https://www.scnet.cn/ui/console/index.html#/job-submit)

以当前单价跑满 6 小时，计算费用上限 **¥21.60**。8 核在卡配额内，沿用现有存储空间，无新增存储配额采购。存储按账户配额单独计费。排队期间不算计算运行时长，最终金额以平台账单为准。本轮新增 GPU 费用 ¥0。 [核卡比说明](https://www.scnet.cn/help/docs/mainsite/hpc/noun-explanation/common-noun/index.html) [共享队列及存储计费规则](https://www1.scnet.cn/help/docs/mainsite/hpc/advanced-guide/billing-rule/)

## 待授权的提交命令

授权人确认此发布、目录、风险和预算后，将待审核记录另存为 `admission-approved.json`，在 `authorization` 写入 `state=APPROVED`、`approved_by`、`reference`。受信公钥部署到项目根下 `.step5-trust/gpu-approval-ed25519.pem`，由授权人控制写权限。授权人在保管私钥的环境中签署最终 JSON 原始字节。本轮只有未签名的待审核记录，无生产授权或作业号。恢复时为 `--resume` 生成独立的待审核记录和令牌。

```bash
openssl pkeyutl -sign -inkey "$approver_private_key" -rawin \
  -in admission-approved.json -out admission-approved.json.sig
```

以下命令供实际获批后使用。前置命令验证身份、签名及证据并原子预留，失败即结束。

```bash
root=$(cd ~/code_cpf/pi05 && pwd -P) || exit 1
release="$root/tools/step5-arx-release-642e19625857c4d1d107d670b1075eadb7608a14"
review="$root/results/step5-gpu-review-642e196"
dataset="$root/data/step5-arx/arx-drive-target-v1.h5"
audit="$release/docs/step5/data_audit.json"
receipt="$review/admission-approved.json"
run="$root/results/step5-arx-4090-validation-642e196"
cd "$root" || exit 1
bash "$release/scripts/step5_arx/env.sh" cpu "$root" -m scripts.step5_arx.train preflight \
  --project-root "$root" --dataset "$dataset" --data-audit "$audit" --cpu-receipt "$receipt" \
  --run "$run" --reserve-submission &&
sbatch --parsable --job-name=cpf-step5-arx-validation --partition=ksagnormal01 \
  --account=acvkpnrzir --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --gres=gpu:RTX4090-PCIE-24GB-LS:1 --time=06:00:00 --signal=B:USR1@300 --no-requeue \
  --comment=step5-arx:cb7b775b590d7bb84c2f27ec466eb643 --chdir="$root" --output="$review/slurm-%j.out" \
  "$release/scripts/step5_arx/train.slurm" "$root" "$release" "$dataset" "$audit" "$receipt" "$run"
```

CUDA 仍需实测修复后的完整 π₀.₅ AOT、两次真实 CUDA 反向、24 GiB 显存峰值、20 步 1% 门禁、正式训练吞吐和真实 Slurm 停止恢复。历史 CPU 异常仍可能出现在共享生产路径。测试替身只验证控制流、日志和恢复契约。

## 变更文件

- 运行入口与准入：`scripts/step5_arx/admission.py`、`scripts/step5_arx/env.sh`、`scripts/step5_arx/train.py`、`scripts/step5_arx/train.slurm`。
- 定向回归：`tests/test_step5_admission.py`、`tests/test_step5_gpu_gates.py`、`tests/test_step5_initial_loss.py`。
- 文档：`docs/step5/admission.md`、`docs/step5/experiments.md`、`docs/step5/gpu_preflight_release.md`。
