# 阶段5E：200→225轮解码边界监督

本阶段从已完成的200轮模型继续，保留原1200训练/200验证窗口、batch4、网络、尺度、数据顺序、统一解码器和独立指标。只替换解码时间项的配权，不添加一个重复时间loss，不重新采集、不进行阶段6引导。25轮为第一段预算，不保证稳定或收敛。

## 来源和预算

父checkpoint：artifacts/runs/stage5d-v1/train/checkpoint_step_060000.pt，SHA-256 b7dcb40fcd9032b76d56ad8871e986067cd29873c8d84ec461feb55b8c96db54；原训练SHA c93a3b5保持。原200任务DDIM20/eta0输出作为配对基线，其自由生成质量82/200、严格舒适性10/735。诊断重构与标签替换不是基线生成结果。

正式协议固定全局60000→67500步、真实epoch201—225、7500更新。完整继承模型/190个AdamW参数矩/噪声Generator/Torch CPU-CUDA RNG；AdamW累积步30000→37500，仅重启调度器。前1轮lr从1e-6线性升到2e-5，其后24轮单段余弦降到1e-6。每epoch无放回遍历全部1200窗口，验证全200；每5轮20个固定任务，结束完整200任务配对报告。实际普通/密集分组只用于校准及审查，本版不改采样分布或密集权重。

## 时间损失

保持观测历史最后两个速度，拼接预测/标签的统一解码未来速度，计算匹配标签的离散一阶和二阶差分。差分索引由最新未来帧确定，前4帧作为边界，其余36帧作为内部；各组按每场景有效车辆、时间及二维元素独立平均。监督标签变化，不强制零加速度或零jerk。

J_boundary = Huber(Δv_pred−Δv_label) + 0.1 Huber(Δ²v_pred−Δ²v_label)，内部同定义。两种差分均除原速度差分单位1m/s；这是离散差分目标，不是已除dt的加速度/jerk值。物理指标仍由独立评估器按dt=0.1s定义。

完整标签下，原时间loss等于0.001×(0.1 J_boundary+0.9 J_interior)。因此旧目标的等效分组系数为0.0001/0.0009，不能把两组各设0.001后声称权重保持。

候选分组系数为边界0.001、内部0.0009；边界是原等效贡献的10倍，仍需校准，不能称最优。仅重配增量在300更新内预热：w_boundary=0.0001+rho×(0.001−0.0001)，w_interior=0.0009，rho=min(update/300,1)。起点等价旧时间项，最终替换成新两组权重。

保留模仿0.003、道路路线0.001、原始速度Δ²内部5e-5/边界1e-5（单位0.15m/s）；它们从第一更新全强度，原始速度项不二次预热。辅助项继续t<=300、alpha_bar、全batch分母。统一解码使边界目标的梯度可传到整个40帧，不能把它当作只作用原始首四帧的修复器。解码和评估阈值保持，未加入平滑或原始位置局部硬约束。

## 校准与短预演

Calibrate冻结父参数，按当前元数据选择普通/密集各16个训练窗口，几何/回合轮询、ID/seed哈希确定；不依据未来误差或验证失败选择。每组4个B4，分别自然t和固定t=0/50/200，同一批噪声，总32次模型前向。真实计算完整辅助/总loss参数梯度、与epsilon夹角、范数及理论clip；核对分项向量和，不把范数比例相加。

边界0.0001/0.0004/0.001/0.003为冻结向量敏感性，内部保持0.0009；不自动选权重，不把梯度校准当作学习收益。自然t统计对低噪声适用批次有条件，t=0仍有噪声。父训练steps.jsonl另记录普通/密集实际低噪声覆盖，计数不等于梯度贡献。

Preflight包含以上校准、12次隔离峰值/full-strength更新（无正式模型、跳过预热只查稳定）、16次实际协议前缀、第8步恢复后8步逐位比较及6任务自由采样/独立评估。调度总预算保持7500，短前缀不能算25轮完成；6任务质量不能证明新loss效果。

## 本地命令

要求代码已提交、工作区干净；选新RunId，已有失败/完成目录不能覆盖。实现和所有结果都在E:\diffusion_new，旧目录只读。

~~~powershell
Set-Location E:\diffusion_new
.\scripts\stage5e.ps1 -Stage Calibrate -RunId stage5e-calibration-local
.\scripts\stage5e.ps1 -Stage Preflight -RunId stage5e-preflight-local
~~~

完成短验收后，用户可在本地终端启动25轮、最终200任务采样及报告：

~~~powershell
Set-Location E:\diffusion_new
.\scripts\stage5e.ps1 -Stage Run -RunId stage5e-v1
~~~

只训练：-Stage Train。中断恢复要求同一不可变配置、新输出ID和training_recovery checkpoint；模型、优化器、RNG、日程及重配rho同时继承：

~~~powershell
.\scripts\stage5e.ps1 -Stage Train -RunId stage5e-resume-v1 -Resume E:\diffusion_new\artifacts\runs\stage5e-v1\train\checkpoint_step_061500.pt
.\scripts\stage5e.ps1 -Stage Sample -RunId stage5e-sample-v1 -TrainingRun E:\diffusion_new\artifacts\runs\stage5e-resume-v1
.\scripts\stage5e.ps1 -Stage Report -RunId stage5e-report-v1 -TrainingRun E:\diffusion_new\artifacts\runs\stage5e-resume-v1 -SamplingRun E:\diffusion_new\artifacts\runs\stage5e-sample-v1
~~~

恢复后只需剩余预算；已经完成的checkpoint不得重复训练。Sample/Report要求实际完整225轮。WarmupCheck只跑300更新，不能作完整模型。AuditBaseline只重放现有200轮200任务报告，零训练/新采样，检验报告链路。

## 产物与判断

每个子运行保存实际SHA、完整配置、命令、环境、种子、原数据及checkpoint身份、状态。新checkpoint使用sumodiff.decoded.temporal.train.config.v1，严格签名锁定分组权重/预算；原5C/5D仍按其旧合同加载。steps.jsonl保存两组一二阶差分、有效系数、temporal_reweight_delta/ramp及完整梯度clip前范数。

正式结果比较同200任务、初始噪声、DDIM与解码设置；报告资格、严格舒适性、交界/内部jerk、道路/路线/覆盖、碰撞、修正及成本，失败保留分母、无车辆对N/A。全部有益指标应一起判断，不能只按epsilon下降或边界改善延长；新增训练/日程与loss同时改变，不单独归因loss。验证集用于开发，测试集未使用。

当前工程测试通过，GPU校准/预演尚待执行。预演完成后补充实测来源摘要；正式25轮由用户本地启动，不自动启动长训练或阶段6。
