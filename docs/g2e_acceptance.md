# G2-E 准备阶段验收

G2-E 的预测块统计、跟踪分析、原生执行协议、CPU 验收和单次 GPU 方案已完成。真实模型下的长窗口动力学与原生完整任务成功仍为 **GPU_DEFERRED**；没有执行资源申请或修改既有作业。

| 交付/检查 | 结果与证据 |
|---|---|
| 全部首次闭爪分布 | 840块，Aloha750/ARX90，左右手各51类，包含整块无闭爪；PNG与PDF已绘制并目视验收 |
| 数据选窗 | 按四组原生轨迹分别达到50%/90%条件覆盖率的最小公共K，选择26/31；1/4作为对照 |
| 截断率 | 原生K1=97.25%、K4=96.98%、K26=25.55%、K31=7.01%，分母728个预测过左闭爪的块 |
| 限位与裁剪重放 | 全K1…50扫描；750个原生旧plan逐元素复现；K26/31各744/750合规，6个保留拒绝 |
| Aloha跟踪 | 全部72,123物理步；12关节RMSE/P95/max、左右TCP误差、首次超限原事件；明确历史夹爪drive/接触缺失 |
| 完整CPU回归 | 最终174项测试通过，13个合成episode通过；`results/g2/g2e-20261106-r2-cpu/tests.log` 与 `episodes/mock-acceptance.json` |
| 真几何静态检查 | 两种锁定原生初态、两组托盘侵入负例均符合预期，零物理步、无模型/GPU；static-guard-r2.json |
| 宿主入口 | 三阶段native-only监督、单模型启动、真实进程组清理/超时/模型失败测试、8个锁定命令、bash语法与argv dry-run通过 |
| 远端准入CPU检查 | 8个配置/初态绑定、源与资产、28个模型文件绑定先前完整SHA回执均通过；remote-preflight-final.json |
| 原始证据保护 | 2,712个原作业文件重新SHA-256全部一致；原冻结release manifest仍为dbcb8d40ca88074751229302a0b35797698fa85cd56f43dc1c1eff4de963c09c |
| GPU审批 | awaiting_user_approval；cpu_release_validated=true、gpu_release_ready=false；未提交 |

详细结果：[分布与截断分析](g2e_window_analysis.md)、[跟踪分析](g2e_tracking_analysis.md)、[执行协议](g2e_execution_protocol.md)、[GPU方案](g2e_gpu_plan.md)。图的字体、源数据/统计口径、自动QA误报解释与人工检查位于 `research/g2/g2e-20261106-r1/figure-source/QA.md`，最终字体smoke为PASS。

准备包本地与SCNet目录分别为 `research/g2/g2e-20261106-r1/release/`、`tools/g2e-release-20261106-r1/`。最终manifest SHA-256：`42c5f51650d7be602bef143101bbda485c0ec42090c752febab2b80fb8bb21a1`。run_lock SHA-256：`f054d1495ec048ca9b0035eab110a1ce092390d63c7bbb3b5eeea39833aa61f0`。运行Python生成的`__pycache__`不进入发布清单；82个有意发布文件逐项校验。源变更前副本在before/，CPU修复基线与原冻结数据均保留。

GPU审批的具体范围为：**单作业、1GPU、最多8次Aloha原生完整任务、3小时30分钟分配上限；历史0.06元/分钟估算上限12.60元**，详细分阶段预算与早停规则见方案。当前费率和余额在未来实际提交前核对；没有授权追加作业、时长或ARX预算。

本轮只对CPU工程验收与可复现准备负责：真实模型加载和首次编译、新协议原生TOPP实际推进、真实接触闭合/交接/放置、GPU节点上的吞吐和显存余量均在同一个待审批综合作业内验证。冻结场景的seed0首块K31于索引26拒绝，说明长开环确有候选碰撞风险；它不是物理仿真失败或成功结论。新K1/K4也采用同一安全监督，避免把安全协议差异归因于窗口。

只有原任务成功谓词及完整实测物理阶段证据共同成立，才进入后续ARX资源评审；本作业无ARX自动分支。试次完成、评分成功、完整阶段证据分别报告，基础设施中止与未运行试次均不补算。

提交更新（2026-11-06 22:14，北京时间）：用户已明确批准，唯一作业 **127081074** 已提交并在gnode39运行，TimeLimit与TimeMin均3小时30分钟。图形门禁通过，模型服务启动1次并进入OpenPI导入；完整任务结果尚未产生。授权后准入信息变更对应manifest `fb24a4c727a45945ebe46faaa036f26837c1ab79c3e4d7590433c9f84ef6bfe5`，代码及run_lock未变。当前订单3.60元/卡时，账户可用余额足够，单次上限12.60元。回执见 `research/g2/g2e-20261106-r1/submission/`。此授权已使用，不重复提交。
