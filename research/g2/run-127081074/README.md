# 127081074 G2-E结果证据

本目录公开的是一次已结束GPU作业的诊断结果：COMPLETED/0:0、8个预定试次均有真实结果、完整任务成功0/8。startup的4动作入口不计入8试次。阶段passed不能替代任务成功。完整验收报告在../../../docs/g2e_run_127081074.md。

analysis-evidence.zip（约1.18MB）是精简分析归档：164个原始action-*.npy（50×14）、164个plan-*.npy（K长度的选择前缀），以及全事件流计算出的请求/跟踪汇总、首次持有及拒绝上下文。没有模型权重、图像、视频或完整1GB物理事件流。原始运行完整642文件保留于研究工作区，remote-file-inventory.json只描述其清单，retrieved字段指原始元数据取回范围，不指每项都公开上传。

本目录的文档与文本证据已将部署绝对路径替换为<PROJECT_ROOT>；transfer-verification.json及远端inventory的原SHA指原始取回文件/归档，不能当作脱敏文本的公开文件SHA。publication-manifest.json给出本次公开上传文件SHA。分析脚本保留原字节，远端脚本只是追溯已有事件的读取程序，不是作业提交入口。

复查（仓库根目录，Python需numpy）：

```sh
cd research/g2/run-127081074
python3 -m zipfile -e analysis-evidence.zip .
python3 analyze_local_results.py
```

输出audit-summary.json及derived/new-prediction-closure-analysis.json。该分析会重写自身衍生输出，不会修改原始Numpy、调用模型或操作GPU。逐项断言8结果、164原预测/前缀、512完整动作及42320物理步。公开端可用这些数据重算预测截断统计；逐物理步跟踪汇总依赖归档内的全事件流衍生结果，原始全事件流未随库发布，不能声称从公开小归档重跑完整物理分析。

K31最后的近闭合命令微增触发张爪保护，需CPU复查语义，不能直接说模型主动释放。所有碰撞/限位/接触阈值及冻结运行代码未改变。原生完整任务仍失败，统一剩余方案submission_allowed=false，本轮未提交、修改或取消任何GPU作业。
