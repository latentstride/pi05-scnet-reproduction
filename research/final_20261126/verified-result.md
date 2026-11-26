# 新150-clean EMA 原生评测：核实完成

只读核对山东四区持久化原始结果，未重新启动 A800。平台该 Notebook 当前为 Terminated，GPU 释放记录为 0 MiB、无 CUDA 进程，关机请求 code=0/success。

最佳模型：EMA150/checkpoints/step-2000，EMA decay=0.99，K=15；从原 PACE 基座在新150条clean上微调。第2000步 seed0–4 官方3/5，按既定条件早停，没有训练第3000步或执行其条件性K8。seed5–9补测4/5。合计抓起10/10，右手独立持物≥0.2秒8/10，官方完整成功7/10；全部从各自原始seed初态原生闭环执行，动作上限600。

|seed|稳定双指净抬升mm|最长右手独立持物s|官方check_success|
|---:|---:|---:|---|
|0|171.747|20.012|true|
|1|171.117|20.984|true|
|2|172.515|18.616|true|
|3|161.081|18.448|false|
|4|165.635|0.000|false|
|5|171.169|19.504|true|
|6|212.796|0.000|false|
|7|166.506|19.196|true|
|8|168.949|19.436|true|
|9|164.709|19.556|true|

七条官方成功seed为0、1、2、5、7、8、9，均在273–291行动作内结束，原始receipt同时记录native_success=true、native_eval_success=true、stop_reason=official_success，汇总placed_observed=true。seed3已右手独立持物18.448s，终点目标功能点差[-35.712,15.799,0.00048]mm，官方失败；seed4和6没有达到右手独立接稳，均600动作结束。这里只报告现象，不据终点坐标强行归因具体策略机制。

统一seed0–4口径（原生qpos/TOPP、600动作、无G2）的官方成功率：混合基线step0600/K15为0/5；旧190clean step2500/K4为0/5、K15为2/5、K25为1/5；新150clean EMA2000/K15为3/5。全表和逐seed功能点误差见comparison.md、comparison.json；视频保持在远端各seed/head.mp4。

新结果属于π₀.₅经ARX示范微调后驱动ARX完成任务的成绩；冻结Aloha→ARX路线单独评价。多个训练配置与K同时改变，因此收益不能仅归于EMA或某一个改动。
