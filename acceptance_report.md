# π₀.₅ 环境与闭环验收结果

验收日期：2026-11-01。结论：环境部署、真实观测推理、完整闭环和 15 轮受限评测均通过。

| 验收项 | 实测证据 | 结果 |
| --- | --- | --- |
| 固定源码版本 | OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`；精确 LIBERO 子模块归档 | 通过 |
| 主环境依赖及导入 | 202 个包兼容检查通过；`CORE_IMPORT_OK pi05_libero` | 通过 |
| 官方模型资源 | checkpoint 共 16 个文件，全部 CRC32C 校验通过；带 MD5 的文件亦校验通过；tokenizer 离线编码成功 | 通过 |
| GPU 推理 | JAX CUDA 识别成功，官方策略加载并基于真实 LIBERO 观测生成动作 | 通过 |
| LIBERO 环境 | 独立环境 125 个包兼容检查通过；离屏渲染、图像与状态采集成功 | 通过 |
| 策略通信与闭环 | 官方 WebSocket 服务与客户端连接；实际执行预测动作并获得新观测 | 通过 |
| 15 轮评测 | 3 个任务，各 5 个初始状态，15/15 成功 | 通过 |
| 轨迹与视频 | 15 个组合齐全；状态与动作全部有限；数组维度和视频帧数匹配 | 通过 |
| 退出状态 | COMPLETED，ExitCode 0:0 | 通过 |

## 推理实测

- 模型：官方 `pi05_libero` checkpoint。
- 输入：两路 224×224×3 uint8 图像、8 维状态及任务指令，来自真实 LIBERO observation。
- 输出：10×7 动作序列，全部有限，无 NaN / Inf。
- 模型加载 12.4185 秒；首次推理 42.2381 秒（含首次编译），相同观测第二次推理 0.100954 秒。
- 主环境：Python 3.11.18、JAX/JAXLIB 0.5.3、PyTorch 2.7.1、NumPy 1.26.4。

## 受限闭环评测

libero_spatial 任务 0、1、2，各初始状态 0—4，seed=7。每轮等待 10 步，每 5 步重新推理，最多 220 个预测控制步。作业总运行时间 17 分 4 秒。

| 任务 | 完成轮数 | 成功数 | 控制步数（按初始状态 0—4） |
| --- | --- | --- | --- |
| 0 | 5 | 5 | 103、105、83、79、73 |
| 1 | 5 | 5 | 110、97、107、100、113 |
| 2 | 5 | 5 | 127、93、96、93、101 |
| 合计 | 15 | 15 | 全部在 220 步以内成功 |

验收程序确认状态数组为 `(控制步数+1,8)`、执行动作数组为 `(控制步数,7)`、预测动作块为 `(调用次数,10,7)`，数值均有限，每个视频的帧数等于控制步数加一。

## 证据文件

- [推理验证记录](docs/inference_validation.md)
- [逐轮评测结果](evaluation_results.csv)
- [评测报告](evaluation_report.md)
- [状态、动作与控制接口说明](action_space_notes.md)
- 远端本轮输出：`<PROJECT_ROOT>/results/evaluation-<JOB_ID>/`，包括 `summary.json`、`artifact_validation.json`、`evaluation_config.json`、`evaluation_results.csv`、`evaluation_report.md`、`episodes/`、`videos/`、`success_rates.png` 和 `success_rates.pdf`。
- 运行日志与原始轨迹、视频保存在项目所属人的个人目录；公开仓库保留代码和结果摘要。

该结果证明当前配置可运行，是 3 个任务各 5 轮的功能性基线，不代表官方完整 LIBERO benchmark。首轮推理记录中的 `closed_loop_completed=false` 描述的是当时单次推理阶段；后续闭环与 15 轮评测已经完成，以本验收结果为准。
