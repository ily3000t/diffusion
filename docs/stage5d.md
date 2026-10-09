# 阶段5D：150→200轮原始速度监督续训

本阶段已完成代码、联合梯度校准、短GPU预演和现有200任务报告重放。正式50轮尚未启动，模型收敛和质量收益尚未验证。阶段6保持未启动。

## 1. 固定协议

| 项目 | 实际约定 |
|---|---|
| 父模型 | stage5c-v1/train/checkpoint_step_045000.pt，已完成150轮 |
| 父SHA-256 | 2a25293744e0f9ff8bec9a03b3a1c3e077275bc7c9cece4119cdda13f070900e |
| 数据 | 原1200训练、200独立验证窗口，三类道路；batch=4 |
| 新预算 | 真实151—200轮，15000更新，全局45000→60000步 |
| 初始化 | 继承模型、完整AdamW矩和噪声Generator/CPU/CUDA RNG；只重置日程 |
| 学习率 | 更新1为1e-6，更新600达2e-5，随后单段余弦至更新15000的1e-6 |
| 现有监督 | 模仿0.003、解码后时间0.001、道路/路线代理0.001，全强度保持 |
| 新原始速度项 | 内部5e-5，前四未来帧1e-5；两轮线性预热，仅预热新增项 |
| 验证 | 每轮全200噪声重构；160/170/180/190/200轮固定20任务自由生成 |
| 最终采样 | 全200验证任务，DDIM20、eta=0，与150轮基准逐任务初始噪声相同 |

正式配置是 configs/train/stage5d_raw_velocity.yaml，实验入口配置是 configs/experiments/stage5d_training.yaml。旧 stage5c.ps1 仍是100→150轮协议，不用于本次续训。

源AdamW在阶段5C曾重置，因此父模型的矩计数为15000，而不是全局45000。新checkpoint要求各参数计数为15000加本协议已执行更新数。源状态先复制再继承，避免CPU计数张量别名污染另一候选。严格resume恢复本次新日程，不重新开始预热；改变损失、LR、预算或数据身份会被拒绝。

## 2. 新损失与校准解释

在固定共同坐标系中，取最后两点观测速度与40点原始预测速度，计算二阶差分，再减去正常标签的相同差分。第k项按其最新未来帧索引；前4项覆盖0.1—0.4秒，其余36项覆盖0.5—4秒，内部项的差分仍可使用前面的帧。

残差单位用0.15m/s，逐车/帧/方向有效元素做场景内Huber平均。两个时间组独立平均，乘低噪声门限t<=300和alpha_bar，再按完整minibatch分母归一化。新系数rho_raw=min(追加更新/600,1)。该单位在dt=0.1秒时对应15m/s³的速度场jerk误差尺度，不是独立硬阈值或可行性保证。

监督匹配正常标签，允许正常加减速和转弯；不把jerk强制为零。原始速度项直接作用于原始速度字段；已有解码后目标经同一个解码器作用于位置/速度/朝向。未来标签仅作训练或诊断监督，不进入自由采样条件。

16个真实训练B4、自然随机t及同噪声固定t=200，均以新增rho=1校准。实际总辅助/噪声梯度比中位0.8477/1.2900，余弦中位0.1515/0.5348；总梯度最大范数0.0473/0.0548，clip阈值1.0均不触发。自然t仅10/16批符合辅助门限，固定t全部16批符合。详细范数、夹角及裁剪见 stage5d_preflight_summary.json；不能把分项范数比直接相加。

模仿仍保留0.003。减半模仿的冻结向量敏感性使组合比中位降到0.5750/0.7025，但未训练该候选，不能说明减半更好；范数大本身不说明有害。

## 3. 短验收与限制

正式短预演 stage5d-preflight-20261009-r1 的永久运行SHA是34a706738641549944c14ff3f8b182d28442ad50。它执行16更新；第8步恢复后8更新与连续训练的网络、完整AdamW、Generator、CPU/CUDA RNG、逐步损失、噪声、LR及两类预热记录逐位一致。父checkpoint未改。只完成0个新完整epoch，不能称50轮训练完成。

两个隔离副本分别用2e-5/5e-5峰值、rho_raw=1执行24更新；这是明确绕过预热的压力检查，不是正式日程前缀。均无裁剪且数值有限；验证epsilon为0.00056093/0.00109215，父模型为0.00049485。5e-5短期扰动更大，因此仍以2e-5保守起步，但不能据此宣布最优LR或长期收益。

实际16更新均时225.34ms，Torch allocated峰值182.03MiB、reserved626MiB；冻结联合审计allocated544.15MiB/reserved616MiB。缓存CPU张量约1.1975GB；RTX4070 Laptop、PyTorch2.5.1/CUDA12.4、Python3.10.16。预演墙时179.28秒。仅优化15000更新估计56.33分钟，验证、监测、保存、缓存和最终采样另计；短测速度及显存不是正式预算保证。

6任务自由生成1/6合格、0/22整段舒适性通过，无生成/指标失败。它处在16/600的新项预热阶段，仅验证输出链路，不用于质量或收敛判断。

现有200任务的报告审计 stage5d-report-audit-20261009-r1 在e5eccbe29ff7ab33f7364551620fb9650f6195fc完成：50/200质量、4/735整段舒适性，道路/路线各9、碰撞3，边界/内部混合jerk P99=278.366/94.304m/s³。全部自配对差值和区间为0，无新增训练或采样。失败车辆保留在计划舒适性分母，另报实际评估及未评估数；无适用碰撞对单列N/A。

首次训练预演因辅助能力标记与原尺度审计不兼容失败，原目录保留。修复仅把尺度审计中的非物理训练能力标记映射回原false，所有物理配置、数据及审计hash仍严格核对；实际checkpoint继续记录辅助监督开启。首次报告入口在RunRecorder初始化前因WindowsPath/string排序失败，日志保留；修复后新目录验收成功。上次自动审批额度问题已恢复，不再有待批准操作。

全量275测试通过16.91秒，CLI、PowerShell语法与diff检查通过。6个正式运行及7424个唯一输入身份核对通过。训练和审计SHA永久保留，后续文档/合并不回填。

## 4. 本地正式启动

在代码已提交、工作区干净时执行；RunId必须未使用。Run会续训50轮，随后自动采样全部200任务并写配对报告。不会再执行100→150轮，也不自动双倍训练候选。

~~~powershell
Set-Location E:\diffusion_new
.\scripts\stage5d.ps1 -Stage Run -RunId stage5d-v1
~~~

完整结果在 artifacts/runs/stage5d-v1/train、sample、report。可读摘要为 report/report.md，完整数值和配对区间为 report/paired_summary.json。训练进度为 train/progress.json、steps.jsonl，最近可恢复权重由 train/latest_checkpoint.json指向。

可选的600更新预热检查由用户另行启动，覆盖真实两轮及预热端点；本次未执行。这是独立短prefix，不能将它的checkpoint当作正式50轮实验resume。

~~~powershell
.\scripts\stage5d.ps1 -Stage WarmupCheck -RunId stage5d-warmup-check
~~~

重放现有200任务报告、无新增采样：

~~~powershell
.\scripts\stage5d.ps1 -Stage AuditBaseline -RunId stage5d-report-audit-repeat
~~~

## 5. 中断恢复及独立报告

保留原目录，读取它的最近已保存权重；下面两个变量使用本协议真实路径，不能用旧100轮或短预演checkpoint。

~~~powershell
$stage5dSaved = Get-Content .\artifacts\runs\stage5d-v1\train\latest_checkpoint.json -Raw | ConvertFrom-Json
$stage5dResumeCheckpoint = $stage5dSaved.checkpoint.location
.\scripts\stage5d.ps1 -Stage Train -RunId stage5d-v1-resumed -Resume $stage5dResumeCheckpoint
.\scripts\stage5d.ps1 -Stage Sample -RunId stage5d-v1-resumed-sample -TrainingRun .\artifacts\runs\stage5d-v1-resumed
.\scripts\stage5d.ps1 -Stage Report -RunId stage5d-v1-resumed-report -TrainingRun .\artifacts\runs\stage5d-v1-resumed -SamplingRun .\artifacts\runs\stage5d-v1-resumed-sample
~~~

Train只完成剩余预算；Sample只生成；Report检查从45001至60000的已提交训练链、每轮1200唯一窗口、AdamW继承和日程，再作200任务配对比较。仅training_recovery可恢复，monitor_assessment只用于采样。

正式报告独立列出质量、严格舒适性、交界及内部jerk、道路/路线/覆盖、碰撞、位置/速度协调修正和时间。连续量配对区间使用可用配对数，明确计划数与可用数；车辆/帧混合P99和场景P99均值区间分别解释。50轮终点同时改变训练预算、LR和监督，不单独归因新loss。

50轮结束后，只有交界与内部jerk共同改善、资格提高、其他质量没有明显恶化的联合证据，才支持讨论延长。没有自动延长、自动收敛、阶段6或tag。
