# 最终核心代码快照

本目录为实际实验核心脚本快照（仅规范末尾空行，逻辑原样）：prepare150/dataset150、状态扰动/六阶段采样、全视觉与动作分区、原生Flow训练/EMA、EMA推理、原生qpos评测和只读指标汇总。源/归档SHA见`research/final_20261126/source-provenance.json`。

运行依赖部署端固定OpenPI `f4e52e6de4cc487a11585a3a7414d610b2883dae`、PACE `fd2e624cc19325cc9b20a620c82d51c0d8e25cc1`、原`step5-execution-aligned`的`NativePolicy/NativeAdapter`、数据读取、serve.Session、lifecycle.write与资产；本目录是核心代码存证，不是包含权重/数据/所有部署依赖的一键安装包。Notebook关机守护和平台私有会话没有进入归档。

训练输入state采用与专家标签一致的drive target；12臂关节增量目标相对同一state，夹爪为绝对标量0闭/1开，模型补到32维，输出恢复14维后交原生qpos。最终评测没有套用冻结Aloha视觉或动作重定向。

`sampling_six.py`文件顶层历史说明仍写half，其实际batch分支和manifest为80/20，训练身份JSON也为80/20；归档保持运行逻辑，报告以实际分支为准。`prepare150`依赖已完成190clean容器，仅提取新150Train并原样保留旧Val/Test；不在归档步骤重新执行。

本地仅做语法与离线结果核对，不启动这些GPU入口。正常复核请运行仓库根目录的`python scripts/verify_final_archive.py`。
