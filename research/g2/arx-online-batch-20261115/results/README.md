# 原始结果归档

本批次无已提交作业。获得正式 Job ID 后，将完整 `results/pace-g2-standalone-arx-<JOB_ID>/` （含原始 RGB、动作、物理事件、视频和回执）保存到独立归档目录，同时保留 `sacct`、提交阶段 stdout/stderr、发布包与 SHA-256。运行缓存无需归档。

在 `submission_receipts.json` 中填写实际 Job ID、目录、任务 episode、调度终态、退出码；`result_directory` 相对传给分析器的结果根目录。然后在仓库根执行：

```bash
D=research/g2/arx-online-batch-20261115
python "$D/analyze.py" /path/to/results "$D/submission_receipts.json" --hash-files > trial-summary.json
```

分析器读取原任务判据、动作与查询计数以及接触追踪。双指接触要求同一只手两个独立指尖。稳定持物以原物理跟踪器确认的 holder 为准。50 mm 抬升基于真实初态物块高度。交接要求左手持物、双手共同持物、右手接管的顺序。单独列出资源中断及证据不足的试次，成功率分母只计有完整终局任务判据的已评分分配。视频路径和证据文件 SHA-256 写入 JSON。
