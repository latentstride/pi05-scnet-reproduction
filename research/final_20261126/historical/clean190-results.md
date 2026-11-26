# clean190 本轮结果核实

2026-11-25 通过 SCNet 山东四区文件管理只读核对原始 final-report.md、shutdown-accepted.json、shutdown-response.json；未重新开机。

- 原始结果目录：<PROJECT_ROOT>/results/step5-arx-clean190-auto-20261125-v1
- final-report.md：结束状态 finished，新 clean 示范150/150；最终节点2500；四节点原生评测共20场。
- final-report.md / complete.json 修改时间：北京时间2026-11-25 07:49:33。
- shutdown-accepted.json 修改时间：北京时间2026-11-25 07:49:35（悉尼10:49:35）。内容为 notebook_id=2123408832426430466、requested=true。
- shutdown-response.json：code="0"、message="success"、attempt=0；同一秒落盘。
- 控制台刷新后该实例显示已关机。07:49:35是平台接受关机请求的时间，不是另行核实的物理停机完成或账单结算精确秒。

|节点|seed|抓起≥30mm|最大稳定净抬升mm|右手独立持物秒|放置|官方成功|
|---:|---:|---|---:|---:|---|---|
|1000|0|True|506.081|0.000|False|False|
|1000|1|True|447.597|0.000|False|False|
|1000|2|True|397.975|0.000|False|False|
|1000|3|True|366.855|0.000|False|False|
|1000|4|False|21.184|0.000|False|False|
|1500|0|True|188.965|0.000|False|False|
|1500|1|True|197.870|0.000|False|False|
|1500|2|True|243.835|0.000|False|False|
|1500|3|True|173.975|0.000|False|False|
|1500|4|True|67.431|17.180|False|False|
|2000|0|True|359.032|0.000|False|False|
|2000|1|False|0.000|0.000|False|False|
|2000|2|True|366.178|0.000|False|False|
|2000|3|True|237.927|0.000|False|False|
|2000|4|True|219.774|0.248|False|False|
|2500|0|True|206.517|0.000|False|False|
|2500|1|True|256.025|9.640|False|False|
|2500|2|True|253.743|0.056|False|False|
|2500|3|True|243.885|23.276|False|False|
|2500|4|True|219.006|19.236|False|False|

step2500实现5/5抓起，三个seed右手独立持物9.640–23.276秒；seed2的0.056秒仅为短暂持物记录。所有节点均无放置或官方完整成功。未凭这些汇总数字诊断失败因果；没有启动新训练。
