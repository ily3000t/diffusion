# 阶段5A：固定目标、扩充数据、从头训练

现有 scripts/stage5a.ps1 已就地升级。默认 Name=stage5a-fresh；没有新增一套训练入口。历史1200/200、100至225轮配置与结果由Git及原运行清单保留；新模型随机初始化，不继承它们的参数或AdamW。

当前正式数据与200轮训练已经完成，完整600验证通过423/600（70.5%）；密集路口、转弯及严格舒适性仍不足。已核验训练恢复、来源和原始/解码两阶段；结果见[从头训练报告](fresh_training_results.md)及[轻量摘要](fresh_training_results.json)。此文件保留运行协议；测试尚未冻结和评价，不能把验证结果称为最终测试。

## 固定协议

| 项目 | 当前候选 |
| --- | --- |
| 训练/验证/测试 | 6000/600/600窗口；三类道路分别2000/200/200 |
| 候选回合 | 训练144，验证36，测试36；共216 |
| 每类几何×种子 | 训练8×6，验证3×4，测试3×4 |
| 正常交通需求 | 180秒，完整回合最多600秒并要求排空 |
| batch与预算 | 16，200个真实epoch，共75000次更新 |
| 学习率 | 1e-6起始，10轮升至1e-4，随后190轮余弦降至1e-6 |
| 全部辅助项 | 同一个前10轮预热；低噪声t<=300，alpha_bar加权 |
| 监测 | 每轮固定噪声验证MSE；每10轮60个固定自由生成验证任务 |
| 最终评价 | 选定checkpoint后完整600验证任务，再冻结配置评价600测试任务 |

按回合和实际几何先划分，再切窗。测试未来标签不进入训练缓存、尺度审计或权重校准。归一化尺度初始50m/20m/s；Prepare重新审计，不裁剪、不做逐场景标准化。窗口仍可能在同一回合重叠，划分隔离不等于窗口相互独立。

## 损失

噪声MSE仍为主目标。每个场景先按有效元素求均值，再乘alpha_bar、低t门控及共同预热，最后除以整个batch大小。

旧200轮辅助目标保留：模仿0.003、全40帧解码速度时间监督0.001、道路/路线0.001；原始速度二阶差分前4帧1e-5、内部36帧5e-5，单位0.15m/s。未沿用225轮边界10倍配权，未加独立末端目标。

新尾部项使用统一解码位置及固定最后3个历史位置，逐次后向差分除以dt得到全40帧物理jerk，和独立评价一致。对 max(0, norm(j)/15-1) 使用beta=1的Huber，再按场景计算有效元素均值加最差10%均值；K=ceil(0.1*M)，有效零值也计入M。覆盖交界、内部和末端，掩码在差分之前处理。

tail_weight默认未解析。Calibrate仅用当前训练信息挑选16个普通、16个密集路口窗口，检查随机初始及32次微小预热后的状态、自然t和固定t=0/50/200。计算完整辅助梯度与噪声梯度的夹角、范数和裁剪，直接autograd核对组合；不相加分项范数比例。扫描1e-9至1e-3的有界候选，要求尾部梯度<=噪声25%、总梯度相对基础辅助方案的余弦下降<=0.1、范数增加<=25%。这些是数值影响限制，不是最优权重或自由生成收益证明。

通过后再做4个B16完整辅助、峰值学习率的私有压力更新，写入资源记录；不保存成正式训练父模型。正式Train仍重新初始化。Calibrate冻结实际配置到artifacts，数据、模型或配方改变后不能复用原校准。开发旧数据校准明确标记development_only，不产生可供正式Train使用的配置。

## 标签质量与覆盖

核心完整性与质量资格分开。原始float64和存储float32均检查完整历史/未来、车身道路/路线/地图覆盖、运动及插值碰撞；拒绝原因保留，不平滑、不裁剪、不放宽阈值。

首次最小几何小试验的合格转弯数为0；34个实际转弯窗口均被运动规则拒绝，原始SUMO制动jerk说明这不是坐标转换问题。当前小试验在既定训练几何6、验证几何2检查路口，路口需求30秒，其余18秒；只复用了4个配置和种子完全一致的合格短回合，补采2个路口回合。审计697窗口，训练/验证合格转弯池9/1，坐标转换及审计异常0，小试验通过。小试验选出的12/6窗口仅供工程验证，不能作为正式数据。

正式准备要求每个划分选出的路口窗口至少5%实际发生明显转弯（历史+未来展开航向跨度>0.15rad），并报告数量、几何及回合覆盖；训练至少100，验证与测试各至少10。这个预先声明的覆盖下限不保证转弯充分，也不把标签标记放入推理条件。不足时保留报告并停止，不重复补齐或悄悄降低下限。

## 本地启动

当前代表性Pilot已在干净提交f1f5bf4运行并通过。以下按步骤在本地终端执行；Collect和Train可能耗时较长。进入后续步骤之前必须检查上一步成功状态。

```powershell
Set-Location E:\diffusion_new
$ErrorActionPreference='Stop'
.\scripts\stage5a.ps1 -Action Collect
.\scripts\stage5a.ps1 -Action Prepare
.\scripts\stage5a.ps1 -Action Calibrate
.\scripts\stage5a.ps1 -Action Preflight
.\scripts\stage5a.ps1 -Action Train
```

Prepare写入artifacts/processed/stage5a-fresh/dataset并重新审计尺度。Calibrate写入artifacts/runs/stage5a-fresh-calibration/training_config.yaml。Preflight使用完整正式数据、完整75000更新日程，只执行8次更新；正式Train另起随机模型，完整预算不因短预演缩短。

更换Name后先运行Pilot；输出目录不可覆盖。复用已有Pilot仅接受完成的干净正式来源，逐job核对真实配置、种子与划分，未匹配部分重新采集。Collect中断后用新的RunSuffix及ReuseCompleted重启控制入口，保持原采集计划；已完成回合不重采。Prepare失败保留全部候选和报告，修复配方后使用新数据输出目录，不能覆盖失败目录。

断点恢复必须使用本协议checkpoint和同一份冻结配置：

```powershell
.\scripts\stage5a.ps1 -Action Resume -RunSuffix recovery1 -Checkpoint artifacts/runs/stage5a-fresh-train/checkpoint_step_001875.pt
```

路径示例为第5轮checkpoint，实际使用latest_checkpoint.json指向的最后完整更新。恢复保持完整模型、AdamW、噪声及全局RNG和两条预热状态；更改目标、预算、数据或日程会拒绝恢复。Train不接受Checkpoint参数。

训练结束后选择checkpoint，先完整验证，再冻结测试。若验证用了RunSuffix，Seal必须用ValidationRun指定实际目录。

```powershell
$taskChosen='artifacts/runs/stage5a-fresh-train/checkpoint_step_075000.pt'
.\scripts\stage5a.ps1 -Action Validate -Checkpoint $taskChosen
.\scripts\stage5a.ps1 -Action Seal -Checkpoint $taskChosen
.\scripts\stage5a.ps1 -Action Test -Checkpoint $taskChosen
```

测试拒绝与冻结记录不同的模型、数据、种子或采样配置。失败与质量不合格保留在计划分母，无适用车辆对为N/A。并不要求必须选最后一轮；选择只能依据训练/验证。

## 短验收及限制

357项全集通过，并补验新增CUDA恢复用例（合计358用例）。实际小模型在CPU和CUDA上，同入口4+3与连续7更新完全相同，包括参数、AdamW、CPU/CUDA及噪声RNG和日程；PowerShell语法、旧Checkpoint误用拒绝与Pilot来源验收通过。

RTX4070 Laptop上B16完整压力更新峰值分配432.4MiB，保留954MiB；梯度探针峰值763.6MiB。普通与密集自然t微小预热平均0.766s/更新，完整辅助压力均值1.200s。按前者粗估75000更新约16小时，仅优化，不含采集、预处理、验证与保存，且新数据及学习状态可能改变吞吐。6600窗口缓存估计约5.65GB，实际值由新数据Preflight确认。

旧数据开发校准候选lambda_J=3e-4，完整辅助/噪声梯度范数比约0.0009至0.455，夹角与裁剪逐批保存；随机初始噪声自身已触发裁剪。它不授权新数据正式训练，也不证明模型质量。来源、哈希及精确资源见fresh_training_implementation_summary.json。

正式7200窗口、生产权重校准、200轮模型、完整验证与测试尚待用户本地执行。没有建立收敛或自由生成质量结论，没有启动阶段6或创建实验里程碑tag。
