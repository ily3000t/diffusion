# Motion 全量控制训练协议

本协议只扩展阶段5的正式基础模型训练，不进入阶段6。沿用原 scripts/stage5a.ps1，不创建第二个PowerShell入口。旧六维训练、旧控制16步诊断配置及其checkpoint不变；新正式schema为 sumodiff.control.epoch.config.v1，随机初始化新模型与AdamW，不从诊断checkpoint继续训练。

## 冻结输入与目标

数据为 stage5a-motion-v1，data_id=sha256:d5377be30fa15745ec3ee9fcc2213ab355ab83973db337a3e1027dd14a5dfb29。6000训练、600验证、600测试，三道路及双参考配额不变；216源/215accepted/1rejected的账本保持。训练前校验选择v3的哈希、实际数量/运动覆盖/事件复用/回合上限、回合与几何隔离及来源隔离。只读test成员元数据用于确认固定划分，不读取test轨迹数组。所有实际训练/验证文件进入运行清单并按内容哈希核对，缓存未命中时仍核对内容。

模型为 control_directional.yaml 的directional v2，控制量[a,omega,r]尺度2m/s²、0.5rad/s、0.1m/s，使用既有全6000训练专用尺度审计。目标保持已校准的噪声MSE＋统一解码模仿0.003＋全40帧时间监督0.001＋道路/路线0.001＋物理jerk Huber尾部0.0003。辅助项每场景有效元素归一化、t<=300、alpha_bar加权；前10轮共同预热。不增加旧六维原始速度二阶差分项，控制模型没有该独立速度预测头。

训练配置 configs/train/motion_control_epochs.yaml 显式绑定既有 calibration/training_config.yaml 与完整组合梯度报告，核对模型、种子、batch、目标、尺度、学习率范围及实际选择权重；只借用其数值校准结论，不借用16步优化日程。校准不能证明这些权重最优或生成质量良好。

## 一次训练与监测

batch16，200epoch，6000/16=375更新/轮，共75000次。按epoch固定种子无放回排列，每轮恰好遍历全部训练窗口，最后不足批也保留。一次warmup＋cosine：第1步1e-6，第3750步1e-4，第75000步1e-6；辅助warmup3750步。前缀诊断也不压缩这些日程。

逐批读取、逐批车道padding；CPU窗口/道路缓存上限各64，不把6000张地图一次pad后装入GPU。float32、严格确定性、无AMP/TF32。统计同步文件核验、解压、批处理、道路几何和实际优化的每步耗时；验证/保存/自由采样另计。进度每25步与每次验证输出。

每轮：固定噪声下全部600验证窗口的场景平均MSE，以及固定60训练窗口学习探针。每5轮保存checkpoint，每10轮及最终轮对固定60验证任务做DDIM20/eta0自由生成；任务沿用已认证的普通/密集/转弯/换道分组，不将未来标签或运动标签传给网络。周期采样的随机状态在成功或失败后均恢复。

新的最低验证MSE也保存checkpoint，并维护 best_validation_checkpoint.json。此指针只是MSE候选，不是生成质量最优的结论，不构造综合场景分数。检查周期自由生成的场景资格、运动/jerk、道路路线和失败；完成后显式选择checkpoint并在全部600验证任务上评价。比较候选只用验证，测试不得用于调参。

latest_checkpoint.json 原子更新。检查点含原代码SHA、数据/尺度/梯度证据、完整协议、AdamW、噪声发生器、CPU/CUDA随机状态、全局更新位置、两个warmup及最佳候选。恢复拒绝更换预算、种子、batch、目标、监测设置、数据或尺度，不重启日程。周期采样前先保存含pending状态的checkpoint；成功后另存 _monitored.pt。若采样失败，恢复时先补做该监测，再继续优化。历史输出不覆盖，Resume需要新RunSuffix。best候选可能位于较早运行目录，应保留整个训练/恢复链。

checkpoint可能因MSE改善每轮保存，长训练前检查磁盘空间。未实现EMA、AMP、并行loader或训练中自动调权；无静默降级。

## 本地命令

在 E:\diffusion_new，工作区必须干净且已提交。无需重新Collect/Prepare/Audit/Calibrate。

~~~powershell
Set-Location E:\diffusion_new
.\scripts\stage5a.ps1 -Action Train -DataProtocol Motion -Name stage5a-motion-v1
~~~

输出 artifacts/runs/stage5a-motion-v1-train。Train始终从头开始，已有目录拒绝覆盖；新独立训练需明确不同RunSuffix。不要把旧16步训练配置传给此Train。

中断后从实际latest指针恢复，以下示例针对第一次训练；再次中断时改为最近Resume目录：

~~~powershell
Set-Location E:\diffusion_new
$taskLatest = Get-Content .\artifacts\runs\stage5a-motion-v1-train\latest_checkpoint.json -Raw | ConvertFrom-Json
.\scripts\stage5a.ps1 -Action Resume -DataProtocol Motion -Name stage5a-motion-v1 -RunSuffix resume1 -Checkpoint $taskLatest.checkpoint.location
~~~

完成训练后先检查 training_summary.json、validation.json、generation_monitor.json 与各周期采样metrics。可先取MSE候选进行全量验证，但不自动当作最终选择：

~~~powershell
$taskCandidate = Get-Content .\artifacts\runs\stage5a-motion-v1-train\best_validation_checkpoint.json -Raw | ConvertFrom-Json
.\scripts\stage5a.ps1 -Action Validate -DataProtocol Motion -Name stage5a-motion-v1 -Checkpoint $taskCandidate.identity.location
~~~

Validate必须600任务，失败保留分母。验证配置与测试共享噪声seed及采样/指标协议，只有split不同。明确选定模型后，Seal检查完整formal/clean的600验证运行、相同checkpoint/数据/采样配置，并要求干净训练的epoch边界checkpoint。Test必须匹配单独的control.test.seal.v1；旧诊断控制模型继续禁止读取test。

~~~powershell
.\scripts\stage5a.ps1 -Action Seal -DataProtocol Motion -Name stage5a-motion-v1 -Checkpoint $taskCandidate.identity.location
.\scripts\stage5a.ps1 -Action Test -DataProtocol Motion -Name stage5a-motion-v1 -Checkpoint $taskCandidate.identity.location
~~~

若Validate用RunSuffix，Seal需显式传对应 -ValidationRun；若Seal用RunSuffix，Test需显式传对应 -TestSeal。不要自动遍历多个checkpoint做test。

## 本次验收边界

实施验收做短前缀、实际恢复一致性及固定任务采样，不启动200轮。不保证200轮收敛，不根据几步样本评价最终模型或架构。短前缀命令需额外RunSuffix及PrefixUpdates，既有Motion Preflight仍是上一阶段的16步工程诊断。正式训练只有用户上述Train命令才会启动；不创建实验里程碑tag。
