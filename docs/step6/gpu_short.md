# PR #7 单次 GPU 短窗验收

2026-11-16，北京时间。当前状态为 **`CHECKPOINT_UNAVAILABLE`**。SCNet 上的历史训练作业 `128113950` 和 `128115085` 均为 `FAILED (1:0)`，各自检查点目录没有 `step-*`。新发布 `642e196` 对应的正式训练目录尚未创建，项目当前队列未见 Step 5 或 Step 6 作业。第五步训练发布与准入记录见 [PR #6 提交前回执](https://github.com/latentstride/pi05-scnet-reproduction/blob/2d2f74b4523317e1b22e6db6aeec8d1c81227012/docs/step5/gpu_preflight_release.md)。

## 短窗内容

作业使用一张 GPU，沿用 PR #5 的 CUDA 与 Vulkan 设备核对和 SAPIEN 渲染探针。冻结服务先启动并退出，随后在同一分配启动 LoRA 服务。两组使用同一个 seed 0 和策略 seed 42，各自复位并校对机器人、物体和相机初态。每组使用复位时真实三相机 RGB 发出一次 `query_index=0`，核对三张非均匀、互异的 240×320 RGB、首个有限的 50×14 动作张量及头相机 MP4。视频必须能解码至少两帧。原安全控制器最多接收该窗口的四行。规划阶段拒绝候选动作时，保留首轮推理和拒绝证据。实测安全异常使短窗失败。

入口为 `gpu_short.slurm`。每个阶段保存日志，`results/step6-gpu-short-<job-id>/acceptance.json` 记录结果和失败阶段。另有每组的模型身份、RGB 哈希、原始 WebSocket 消息、视频、物理轨迹、首轮请求耗时和已执行动作数。短窗结果在正式成功率之外。

调用期间核验训练发布指纹、固定数据、通过的第五步训练门禁、正式训练完成步及所选 Orbax checkpoint。策略服务复用第五步的权重恢复和采样。第五步的一次性训练授权限定于训练作业，推理入口直接校验训练产物和本次 CUDA 分配。

## 提交前检查

SCNet 已部署 `tools/step6-gpu-short-pr7-0498785`。运行代码提交为 `0498785f008737b167ecf5fe766166e0ca369edf`，包内包含固定 PR #5 运行文件。部署时 `sha256sum -c` 通过。清单 SHA-256 为 `533b44d00ca98605a03e7d7705c4593c2d79202769aa4450f9c72b7cca9bc23a`。发布归档 SHA-256 为 `58b1ffe613178a1fa8b391691aafcfcffc9eb8a8524c32050158f604cc67366e`。

9 项本地定向检查通过。SCNet 上的 Slurm 外壳和图形入口通过 Bash 语法检查，Vulkan 宿主工具通过 help 启动检查。PACE 固定 checkout 干净，D435 配置为 320×240。现有 RoboTwin CPU 容器用两帧测试图通过 MP4 编码及解码检查。已核验运行文件、现有容器路径、第五步发布指纹和 ARX 数据哈希。查询确认 `gnode39` 属于 `ksagnormal01` 并提供指定 GPU GRES。真实提交前命令因缺 `selected.json` 返回退出码 2，回执位于 `results/step6-gpu-short-preflight-0498785/preflight.json`。checkpoint 恢复、真实 RGB 和 GPU 首轮推理等待现场验收。

## 启动命令和预算

以下命令用于第五步产出有效选择和 checkpoint 后提交。当前执行到预检即结束。作业脚本已经固定队列、节点和资源。

```bash
step6_root=$(cd "$HOME/code_cpf/pi05" && pwd -P) || exit 1
step6_bundle="$step6_root/tools/step6-gpu-short-pr7-0498785"
step5_run="$step6_root/results/step5-arx-4090-validation-642e196"
cd "$step6_root" || exit 1

PYTHONPATH="$step6_bundle/suite" python3 -B -m scripts.step6_arx.gpu_short \
  preflight "$step6_root" "$step5_run" && \
sbatch --parsable --account="$(id -un)" --chdir="$step6_root" \
  --output="$step6_root/results/step6-gpu-short-%j.out" \
  "$step6_bundle/suite/scripts/step6_arx/gpu_short.slurm" "$step6_root" "$step6_bundle" "$step5_run"
```

`ksagnormal01 / gnode39`，1 节点、1 task、1× RTX 4090 24 GiB、8 CPU、64 GiB RAM。Slurm 硬时限 **1 小时 10 分钟**。图形探针限 5 分钟。每个服务启动限 17 分 30 秒，每组场景、请求和至多 K4 控制限 10 分 20 秒，每组证据检查限 1 分钟。首次请求允许单独等待最多 380 秒，仍受试次总时限约束。Slurm 提前 90 秒发出停止信号，保留阶段回执并清理子进程。

按 2026-11-16 第五步核验的昆山共享 RTX 4090 单价 **¥3.60/卡时**，跑满估算 **¥4.20**。沿用现有存储，计算费用以实际账单为准。 [SCNet 昆山算力市场](https://www.scnet.cn/ui/mall/search/resource?region=20091) [第五步核价记录](https://github.com/latentstride/pi05-scnet-reproduction/blob/2d2f74b4523317e1b22e6db6aeec8d1c81227012/docs/step5/gpu_preflight_release.md#当前核价与资源)
