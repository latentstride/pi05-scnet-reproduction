# G2 七项启动失败与复发防护

11 月 10 日至 11 日，七项均在计算节点 `gnode39` 以 `FAILED / 1:0` 退出。七份原始 `result.json`、分配回执、日志摘要、文件清单与哈希均已从 SCNet 读取。[现场证据](../research/g2/parallel-startup-failure/seven-jobs.json)

| 作业 | Job ID | 耗时 | 最后成功阶段 | 第一个致命失败 | 置信度 |
| --- | --- | ---: | --- | --- | --- |
| G1 | 127556961 | 28 秒 | Slurm 分配与启动回执 | 图形 shell 项目目录校验 | CONFIRMED |
| G2 | 127556962 | 50 秒 | 分配查询超时一次后恢复，启动回执落盘 | 图形 shell 项目目录校验 | CONFIRMED |
| G3 | 127556963 | 50 秒 | Slurm 分配与启动回执 | 图形 shell 项目目录校验 | CONFIRMED |
| G4 | 127556966 | 8 秒 | Slurm 分配与启动回执 | 图形 shell 项目目录校验 | CONFIRMED |
| G5 | 127556967 | 7 秒 | Slurm 分配与启动回执 | 图形 shell 项目目录校验 | CONFIRMED |
| G6 | 127556975 | 14 秒 | Slurm 分配与启动回执 | 图形 shell 项目目录校验 | CONFIRMED |
| G7 | 127556977 | 13 秒 | Slurm 分配与启动回执 | 图形 shell 项目目录校验 | CONFIRMED |

所有作业通过了 SBATCH shell、Python、包校验、授权校验和分配查询。七份结果写进了调用 sbatch 的 `stage/final-verify.YF2YZR/results/`，而非实验提交回执指向的项目 `results/`。现场保存的七份 `scontrol` 输出均显示 `WorkDir` 是项目根目录，说明 `--chdir` 已生效。旧 SBATCH 脚本随后把 `SLURM_SUBMIT_DIR` 当作 Python 的项目根目录。该变量是调用 sbatch 的目录。旧图形脚本也先切到该目录，接着要求当前目录等于真实项目根。这个检查早于创建图形目录、安装图形失败 trap 和发现 CUDA。现场七份 `graphics.log` 仅有 locale 警告，均缺少 GPU 可见性文件、图形阶段标记和模型日志。用冻结包里的原图形脚本重放这两个目录的组合，得到同样的提前退出。七项均为 `INFRASTRUCTURE_FAILURE`，模型加载与首次推理均未到达。[Slurm 环境变量定义](https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES)

七项退场后，现行控制器已无法用 Job ID 重新返回详情。实际运行时保存的七份分配快照仍可读，与 sacct 及授权使用记录一致。

G2 的第一次 `scontrol` 超时与该目录错误是两个事件。它有完整的后续分配回执。其他作业各自的查询耗时为 UNKNOWN。日志也未覆盖本批 GPU 驱动和渲染能力，不能据此判断驱动状态。

## 修复及验证

- 批量提交明确传入 `G2_PROJECT_ROOT`，批处理 shell 在 Python 之前核对 `pwd -P`。并行包内的图形脚本也使用这个显式根目录。历史压缩包和记录保留。
- shell、Python 引导、包校验、授权、分配、图形和模型阶段输出带时间的事件。运行目录创建后，异常同时保存在结果 JSON。分配查询只对短时传输或配置问题进行最多三次读取，超时分别为 15、20、25 秒，间隔各 2 秒。Job ID、真实分配、节点、资源、计时与授权仍校验。
- `g2_parallel_status.py` 每次查询立即输出，七项都终态后退出。连续两次工具错误退出并给出错误，执行没有后台轮询。
- 19 项 CPU 定向回归通过，包含归档原图形脚本故障复现、显式工作目录隔离、Python 启动错误、查询上限、原部署回归、批次释放顺序和监控终止。[测试结果](../research/g2/parallel-startup-failure/cpu-tests.txt) · [重新打包的 SHA256 清单](../research/g2/parallel-startup-failure/rebuilt-batch.json)

## 两级准入与计算节点探针

**当前状态为 NOT_READY。** 本轮没有获准创建新的 GPU 作业，修复后的计算节点探针尚未执行。真实 CUDA、Vulkan 和容器入口属于待测项。已在 SCNet 登录节点 login04、Python 3.10.14 运行新版 shell 的无分配负例，确认当场记录缺失 Job ID、失败阶段及退出码 1。[登录节点日志](../research/g2/parallel-startup-failure/login-node-smoke.txt)

一级使用一个经另行授权的 10 分钟 `gnode39` 启动探针，同一 batch shell、包校验、Job ID 授权、实际分配、目录和端口检查。探针检查七个解包 manifest、冻结权重文件元数据、norm/tokenizer、场景资产、CUDA 和 Vulkan，执行原图形入口及模型/仿真容器导入，保存真实阶段、退出和进程组凭证。探针不做策略推理。使用同一 Slurm 脚本，以 `--time=00:10:00 --time-min=00:10:00` 覆盖正式时限，工作目录和 `G2_PROJECT_ROOT` 同时绑定项目根。设 `G2_STARTUP_PROBE_BATCH` 为私有 staging 中的 `batch.json`。先 hold，取得真实 Job ID 后签发专用授权，保持普通授权的包、Job ID、报价和资源校验，并加上 `probe_only=true`、`probe_wall_s=600` 以及对应批次 SHA；校验授权后再 release。产物为 `results/pace-g2-startup-probes/<JobID>/result.json`。

二级正式批次使用新的包和独立费用授权。`--execute --probe-job-id <JobID>` 在创建任何正式作业前读取探针结果和真实 Slurm 作业及批处理终态，校验同节点、GPU 分配、七份包 manifest、阶段和清理。七份正式授权再逐项绑定该探针回执哈希。包可提前生成；采用探针通过后才创建正式作业的保守路径，保留七项原有的 hold、逐项绑定、统一 release 流程。模型作业间无依赖，原七项失败记录继续保留。

构建候选，使用一个新的目录，不覆盖历史发布包：

```bash
python3 -B scripts/g2_parallel_package.py . \
  research/g2/standalone-review/g2-standalone-author.tar.gz \
  research/g2/parallel-batch "$FRESH_PACKAGE_DIRECTORY"
```
