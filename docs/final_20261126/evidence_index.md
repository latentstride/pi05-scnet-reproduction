# 最终报告证据索引

全部数值来自已保存产物，本轮离线取回/整理不启动GPU。公开副本仅脱敏部署绝对路径；来源与归档双哈希见[source-provenance](../../research/final_20261126/source-provenance.json)。

|主张/内容|归档证据|
|---|---|
|7/10完整成功、10/10抓起、8/10交接|[comparison.json](../../research/final_20261126/comparison.json)、[原始十场receipt](../../research/final_20261126/raw-ten-seed-receipts.json)、[十场CSV](../../research/final_20261126/ten-seed-results.csv)|
|统一600步/K对照|[comparison.md](../../research/final_20261126/comparison.md)|
|官方容差/张爪谓词|[实际任务源码摘录与哈希](../../research/final_20261126/official-success-predicate.txt)、[原生setup源哈希](../../research/final_20261126/native-setup-seed0.json)|
|EMA身份与Adam2000|[model-metadata](../../research/final_20261126/model-metadata.json)、[training-final](../../research/final_20261126/training-final.json)、[ema-checkpoint](../../research/final_20261126/ema-checkpoint.json)|
|训练配置与参数分区|[training-identity](../../research/final_20261126/training-identity.json)、[trainable-parameters](../../research/final_20261126/trainable-parameters.json)|
|150Train独立seed/原划分/新统计|[dataset150](../../research/final_20261126/dataset150.json)、[norm_stats](../../research/final_20261126/norm_stats.json)、[collection-manifest](../../research/final_20261126/collection-manifest.json)|
|RGB通道/专家批采完成|[color-check](../../research/final_20261126/color-check.json)、[collection-complete](../../research/final_20261126/collection-complete.json)|
|23.69→0.009364mm初始化修复|[原始A/B](../../research/final_20261126/historical/initial-gravity-counterfactual.json)、[修后仍失败](../../research/final_20261126/historical/initial-gravity-physical-result.json)|
|解码与裁剪并未推迟该次闭爪|[action13解码](../../research/final_20261126/historical/action13-decoding.json)、[定向归因](../../research/final_20261126/historical/action13-attribution.md)|
|1nm误杀、8.36μm目标微增|[原源码与前后数值](../../research/final_20261126/historical/randomized-data-and-donor-audit.txt)|
|随机化扩容与clean收口的不利结果|[arx200报告](../../research/final_20261126/historical/arx200-pause.txt)、[Val1000](../../research/final_20261126/historical/arx200-val1000.json)、[Val1500](../../research/final_20261126/historical/arx200-val1500.json)、[两场实测](../../research/final_20261126/historical/arx200-physical1500.json)、[clean收口报告](../../research/final_20261126/historical/clean-stage2.txt)、[四场实测](../../research/final_20261126/historical/clean-stage2-physical.json)|
|原生statejitter交接几何问题|[离线几何报告](../../research/final_20261126/historical/native-statejitter-geometry.md)、[数值](../../research/final_20261126/historical/native-statejitter-geometry.json)|
|190clean四节点400步历史成绩|[原已核实报告](../../research/final_20261126/historical/clean190-results.md)|
|旧Job128329279/128382169与专家执行链|[抓取失败](../../research/final_20261126/historical/grasp-failure-128329279.md)、[采样结果](../../research/final_20261126/historical/grasp-sampling-128382169.md)、[定向归因](../../research/final_20261126/historical/directed-attribution-20261119.md)|
|GPU释放与自动关机|[resource-closeout](../../research/final_20261126/resource-closeout.json)、[完成/早停凭证](../../research/final_20261126/completion.json)|
|代表视频来源/哈希|[媒体manifest](../../media/final-arx/manifest.json)|

历史报告中的“当前/下一步”保留为当时记录，最终状态以本次报告与原生receipt为准。完整H5、checkpoint、每tick物理日志未复制进Git；部署记录用`<PROJECT_ROOT>`表示个人项目根目录，不包含平台凭证。结果JSON中的0.2秒等指标来自读日志统计，不是额外在线停止护栏。
