# 仓库系统审计

审计日期：2026-11-07。基准为 `latentstride/pi05-scnet-reproduction` 的main提交 `d64b0e4feba82c4a97dbbfc8c6b518c67743a0f1`。本次属于保守维护：不修改算法、控制器、GPU入口或历史实验数据。

## 仓库健康状况

证据与代码已较完整地归档，但阶段导航、入口语义和版本追溯需要整理。Stage 1为15/15受限功能基线；G1为8步接口验收；G2和G2-E各自真实策略结果均0/8，不能从CPU回归或作业COMPLETED推出完整任务通过。

本次基准含1023个跟踪文件、17951061字节：根目录20、docs58、scripts162、tests32、research751。静态审计覆盖235个重要脚本（包含研究快照及根目录推理脚本），233份既有Python源码无语法错误。逐文件用途、阶段、使用证据、调用者/依赖关联、测试关联与active/experimental/historical/unknown建议见[脚本索引](script_inventory.md)及[审计数据](repository_audit_data.json)。静态关联不等于执行覆盖率。

## Git与工作区

- 开始时远端分支只有main和`codex/pace-stage-6-7-evaluation`，没有Tag和开放PR。`chore/repository-cleanup`当时不存在，已从上述最新main建立，未覆盖既有同名工作。
- GitHub API核对main共166项Commit元数据，最早为`3db400a0`，最新为`d64b0e4`。来源checkout为浅克隆，本地只见14项日志；未把浅克隆日志当完整历史。历史摘要在审计JSON内。
- 原发布checkout有4个已修改文件及171个未跟踪文件，状态与diff已在本地单独记录，原样保留。另一个研究工作目录内待完成的G2 CPU修改同样保留，不复制到维护分支。
- 使用独立干净工作副本完成维护。未删除Commit/Tag/分支，未运行hard reset或强推，未改写历史，未向main提交，不修改GitHub设置。
- 本轮新增维护工具不导入项目依赖、不执行资源命令；候选修复、GPU测试和安装/下载脚本未运行。

## 已确认的风险（按严重程度）

|级别|发现|影响 / 处置|
|---|---|---|
|P1 · 研究准入|G2/G2-E完整任务未通过；旧剩余方案以K1/K4原生完整成功作为全部ARX入场条件，而已有K1/K4无实际左闭爪证据|原方案可能跳过有价值的ARX诊断。已在首页和实验索引标明研究待修订；本维护PR不修改算法、名单、评分或门槛|
|P1 · 入口误用|多个Slurm/宿主监督/发布入口共存，历史“已授权”“待审批”措辞分别对应不同冻结包|不能直接拿仓库某个脚本或旧授权申请资源。维护索引区分库、规划器、历史入口及实际发布；未执行任何资源动作|
|P2 · 可复现性|完整188项回归依赖不在公开库的专家/物理轨迹、固定OpenPI客户端及集群环境|轻量99项通过不能声称公开库全部188项通过；未补造数据、未安装大型环境或模型|
|P2 · 信息边界|标准文本扫描发现42处部署/用户路径模式，分布于既有证据文本；无所配置的token/私钥头模式命中|本次不复制路径值、不改写原证据，审计JSON只记文件和行号。未来公开证据需核对脱敏。没有据正则零命中宣称完整密钥审计；归档内容未作全量秘密扫描|
|P2 · 版本追溯|报告中的本地发布SHA、manifest SHA、run lock及GitHub证据提交常混用；G1运行标识02ff7dd未出现在本次main历史中|实验索引分列并保留unknown，不把“发布报告里的SHA”自动称为已核验远端执行Commit|
|P2 · 测试关联|若干安装/提交/诊断入口没有直接静态测试关联，动态字符串调用与外部部署依赖也不能由AST完全解析|脚本索引标unknown；不靠运行真实GPU命令补覆盖。后续针对关键入口另做隔离CPU参数/异常测试|
|P3 · 结构|136组精确重复内容；monitor有无后缀版及v2/v3/v4/v5，release也保存v5副本|冻结副本具有哈希与追溯用途，不能据重复就删除或据编号大小选生产入口。已列索引，全部保留|
|P3 · 导航|scripts/pace_runtime_plan.md的5个相对链接失效；首页混合不同时间的阶段状态|只修正文档链接并重组导航，保留所有历史报告的事实和数据|
|P3 · Git卫生|原忽略规则缺少本地工具缓存、临时输出、若干权重/原始数据扩展及常见私钥文件|增加忽略规则；不解除跟踪、不删除历史文件，不忽略所有JSON/CSV或所有ZIP|

## 大文件与二进制

基准中超过1MiB的文件只有：

|文件|字节|判断|
|---|---:|---|
|research/g2/failure-audit-127054052-r1/action-audit.json|1247450|精简预测审计JSON；已有来源，保留|
|research/g2/run-127081074/analysis-evidence.zip|1177487|已有精简分析包；不得据此扩大原始数据上传范围，保留|

还存在既有G1证据tar.gz、MP4及研究图像。它们没有在本轮重新上传、解压执行、移动或删除。未发现跟踪的模型权重文件；这是文件名/体积与树清单检查结论，不是归档内部内容的完整证明。新增.gitignore不能删除已提交的二进制；迁移制品存储只能另提方案。

## 本轮修改

|文件|原因|
|---|---|
|README.md|区分目标、官方范围、冻结基线、当前失败、入口/证据/环境导航|
|docs/repository_audit.md|记录范围、工作区保护、风险、验证与整理边界|
|docs/repository_audit_data.json|逐文件静态调用/测试关联、重复家族、大文件、抑制值的路径发现及166项main历史摘要|
|docs/script_inventory.md|235个重要脚本的可追溯分类与状态建议；不移动文件|
|docs/experiment_index.md|分列真实策略、专家CPU、几何/合成及计划，区分Commit和运行发布标识|
|docs/repository_maintenance.md|main/fix/experiment/chore及PR建议，不设置保护或授权资源|
|docs/repository_validation.json|记录实际验证结果与未执行范围|
|scripts/audit_repository.py|标准库只读AST/Git/Markdown审计工具；不运行项目或Slurm代码|
|scripts/README.md|标记早期部署说明为历史，链接新的阶段与脚本入口|
|scripts/pace_runtime_plan.md|仅修复5个链接的相对目录，不改执行参数或历史结果|
|.gitignore|防止未来误加入本地缓存、敏感材料、权重和原始轨迹|

## 实际验证

- 233份既有Python与1份新增审计脚本共234份AST检查通过；34份shell/Slurm文件经`bash -n`通过，只解析、不执行。
- G2轻量专项58项通过：`test_g2_remaining`、`test_g2e`、`test_g2e_host`、`test_g2_failure_repair`、`test_g2_adapter`、`test_g2_integration`。
- PACE/G1/驱动/几何/统计专项41项通过：`test_pace_g1`、`test_pace_runtime`、`test_robotwin_graphics_driver`、`test_robotwin_control_geometry`、`test_summarize_experiments`。
- 测试使用已有Python环境及固定OpenPI客户端源；未安装依赖、未恢复权重，设置CPU与单线程，进程有墙钟上限及超时清理。
- 13个合成episode单独运行通过，仅验证合成接口，不计入真实策略结果。
- 四份Stage 1公开摘要按baseline_manifest.json的Git blob及SHA-256全部一致；diff逐文件检查，不改原冻结结果。
- Markdown全库本地文件链接在5处修复后检查；最后结果及命令摘要见[验证记录](repository_validation.json)。远端URL、标题锚点与未入库运行资产不属于此检查。
- 完整188项数据依赖回归、完整原始物理回放、模型/CUDA、实际集群入口及GPU评测为NOT_EXECUTED：本轮维护不具备相关资源/原始轨迹，也明确禁止资源操作。不能拿99项替代这些检查。

## 删除、移动与合并边界

本轮没有删除或移动任何文件，也没有推荐可立即删除的具体文件。136组重复与vN家族是后续审查候选，不是删除清单；需要逐文件证明未被发布清单/调用者/报告依赖，再单独请求批准。

本PR只影响文档、忽略规则和新增只读维护工具。Stage 1冻结摘要、G2安全/控制/规划逻辑、GPU/Slurm脚本及历史结果字节保持不变。维护改动可供人工独立审查，Draft PR不自动合并；合并维护PR不表示G2具备资源提交条件。原生/跨本体CPU修复另在研究任务中继续，不混入该分支。
