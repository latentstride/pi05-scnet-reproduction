# PR #7 单次 GPU 短窗验收

2026-11-18，北京时间。Step 5 作业 `128249645` 已完成 400 步，选定 `results/step5-arx-c21eec7/checkpoints/step-0400`。本短程入口现在绑定实际训练发布 `c21eec7` 及代码指纹 `1c81c04e06d86428b9a72fd9c435e7c5498e6b7cc6c5890ef1a666271be4ba27`。

## 短窗内容

作业使用一张 GPU，沿用 PR #5 的 CUDA 与 Vulkan 设备核对和 SAPIEN 渲染探针。冻结服务先启动并退出，随后在同一分配启动 LoRA 服务。两组使用同一个 seed 0 和策略 seed 42，各自复位并校对机器人、物体和相机初态。每组使用复位时真实三相机 RGB 发出一次 `query_index=0`，核对三张非均匀、互异的 240×320 RGB、首个有限的 50×14 动作张量及头相机 MP4。视频必须能解码至少两帧。原安全控制器最多接收该窗口的四行。规划阶段拒绝候选动作时，保留首轮推理和拒绝证据。实测安全异常使短窗失败。

入口为 `gpu_short.slurm`。每个阶段保存日志，`results/step6-gpu-short-<job-id>/acceptance.json` 记录结果和失败阶段。另有每组的模型身份、RGB 哈希、原始 WebSocket 消息、视频、物理轨迹、首轮请求耗时和已执行动作数。短窗结果在正式成功率之外。

调用期间核验训练发布指纹、固定数据、`gpu_progress.json` 的 `gpu_fit_complete` 与第 400 步、`selected.json`、所选及最终 Orbax checkpoint 的提交元数据。实际训练没有生成旧版 `gpu_gates.json`；本入口不要求或合成该文件。策略服务复用第五步的权重恢复和采样。

## 提交前检查

SCNet 已部署 `tools/step6-gpu-short-pr7-3b03054`，包含 PR #7 的兼容修复和固定 PR #5 运行文件。发布归档 SHA-256 为 `b7b4ab1869f34fbc738b13186581275a6f41b18dcec6daeec607cd4b3d203215`，清单 SHA-256 为 `fccc80bc73ed22bd990f1581cd22bd464ec0dcf1ddb2cd37eab50d957efd848f`；远端 `sha256sum -c`、Slurm 脚本 Bash 语法检查均通过。

本地 `py_compile`、checkpoint 定向测试、Slurm Bash 语法和 `git diff --check` 退出 0。SCNet 从真实项目根运行发布包 `preflight` 退出 0，返回 `CHECKPOINT_METADATA_VERIFIED`、`selected_step=400`、真实 checkpoint 路径、训练发布路径和训练身份 SHA-256 `4799f70b2e198674e3bd0e17444c8c078f7323f22f8e7b3eb65134fd515628ab`。真实 CUDA 权重恢复、三路 RGB 与首轮推理仍需在短程作业内验收。

## 启动命令和预算

以下命令先复核已完成的训练结果，再提交一个短程作业。作业脚本已经固定队列、节点和资源。本轮没有执行此命令中的 `sbatch`。

```bash
step6_root=$(cd "$HOME/code_cpf/pi05" && pwd -P) || exit 1
step6_bundle="$step6_root/tools/step6-gpu-short-pr7-3b03054"
step5_run="$step6_root/results/step5-arx-c21eec7"
cd "$step6_root" || exit 1

PYTHONPATH="$step6_bundle/suite" python3 -B -m scripts.step6_arx.gpu_short \
  preflight "$step6_root" "$step5_run" && \
sbatch --parsable --account="$(id -un)" --chdir="$step6_root" \
  --output="$step6_root/results/step6-gpu-short-%j.out" \
  "$step6_bundle/suite/scripts/step6_arx/gpu_short.slurm" "$step6_root" "$step6_bundle" "$step5_run"
```

`ksagnormal01 / gnode39`，1 节点、1 task、1× RTX 4090 24 GiB、8 CPU、64 GiB RAM。Slurm 硬时限 **1 小时 10 分钟**。图形探针限 5 分钟。每个服务启动限 17 分 30 秒，每组场景、请求和至多 K4 控制限 10 分 20 秒，每组证据检查限 1 分钟。首次请求允许单独等待最多 380 秒，仍受试次总时限约束。Slurm 提前 90 秒发出停止信号，保留阶段回执并清理子进程。

按 2026-11-16 第五步核验的昆山共享 RTX 4090 单价 **¥3.60/卡时**，跑满估算 **¥4.20**。沿用现有存储，计算费用以实际账单为准。 [SCNet 昆山算力市场](https://www.scnet.cn/ui/mall/search/resource?region=20091) [第五步核价记录](https://github.com/latentstride/pi05-scnet-reproduction/blob/2d2f74b4523317e1b22e6db6aeec8d1c81227012/docs/step5/gpu_preflight_release.md#当前核价与资源)
