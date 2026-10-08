# 阶段5C：统一轨迹监督与余弦学习率

本阶段落实用户在四组10轮对照之后确认的新训练方案。数据仍为已通过质量资格检查的1200训练/200独立验证窗口，batch=4；从原100轮父模型参数继续学习50个真实epoch（全局101—150轮，30001—45000步）。这是一项初始预算，不保证收敛或生成轨迹合格。

默认只训练一个unified候选。保留epsilon_cosine配置供后续按需对照；两者共同使用参数起点、新AdamW、学习率和配对数据/采样。没有运行这个对照时，改善应归于整个训练方案，不能独立归因于新损失。

## 1. 起点、预算与恢复

原父checkpoint为artifacts/runs/stage5a-v1-train/checkpoint_step_030000.pt，SHA-256：

2315fa4d57b9d86a3254a0923590efbbd48f23666f3bf0d2c733ac04b4d1769b

新实验继承网络参数、噪声Generator以及Torch CPU/CUDA RNG，重新建立空AdamW和更新计数为0的调度器。AdamW维持weight_decay=1e-4、betas=(0.9,0.999)、eps=1e-8，clip=1；不沿用旧优化器矩状态。训练批次继续使用已有全局epoch无放回排列，每个完整epoch遍历1200窗口一次。数据、网络、50m/20m/s归一化尺度、float64协调解码器和独立指标保持原协议。

中断恢复是另一项操作：继承新实验网络、完整AdamW、Generator、CPU/CUDA RNG、学习率调度、损失预热计数和质量监测历史。训练签名包括完整损失、日程、预算、监测、数据与窗口顺序，不允许借resume更换目标或预算。只接受training_recovery快照；monitor_assessment快照用于评估，不用于恢复。

## 2. 更新级学习率和辅助预热

名义新增更新数15000；短预演仍用这个日程，不把完整余弦压缩到16步。

- 更新1：lr=1e-6；更新600（第2个新增epoch结束）：lr=1e-4，线性warmup。
- 更新601—15000：按单段余弦降至1e-6，无重启。
- 辅助系数rho(s)=min(s/1500,1)，新增第5个epoch结束时达到1。
- 调度在optimizer.step之前应用，成功更新后推进计数。checkpoint保存已完成更新数、最后实际lr、辅助系数及配置hash，恢复逐项核对。

降低并衰减学习率用于控制更新幅度，不保证跳出局部最优。

## 3. 训练目标

所有扩散时间步继续使用原每场景有效元素归一化的epsilon MSE。先用epsilon恢复干净状态、反归一化，再将预测和正常标签分别通过同一个固定起点协调解码器：

~~~text
L_scene = L_epsilon_scene
        + rho(s) * 1[t<=300] * alpha_bar[t]
          * (0.003 * L_imitation + 0.001 * L_temporal + 0.001 * L_road)
~~~

先计算有效车辆/未来元素的每场景均值，再在整个minibatch上平均。低噪声场景被筛出后，分母仍为完整batch，不能只平均符合门限的场景。Huber使用smooth_l1(beta=1)。附加目标只监督由对应真实标签加噪得到的配对重构；不会用一个真实未来强制对齐独立纯噪声生成样本。

### 模仿损失

位置误差投影到正常标签未来朝向定义的切向和法向，不固定在初始车道。标量位置项分别平均后相加，速度及朝向项按两个坐标平均：

~~~text
L_imitation = mean H(e_longitudinal / 2m)
            + mean H(e_lateral / 0.5m)
            + mean H((v_decoded - v_label_decoded) / 5m/s)
            + 0.1 * mean H(h_raw - h_label_unit)
~~~

朝向使用恢复后的原始sin/cos预测，避免归一化隐藏幅值错误；位置和速度使用协调后的状态。这里的模仿是低噪声配对轨迹监督，不是旧DDPO的行为克隆项或独立生成样本蒸馏。

初始候选模仿权重0.01。真实训练集16批次、父模型冻结、辅助ramp=1的参数梯度审计发现辅助/epsilon梯度范数比中位0.828、最大2.658。将模仿权重一次校准至0.003后，中位0.251、最大0.802，时间项保留0.001。此校准仅用训练批次，未用测试集；它不保证训练全过程始终保持相同比例。

### 时间差分监督

将最后两点观测速度与未来解码速度拼接，预测/标签共享相同历史。第一差分仅保留40个未来相关项；第二差分保留40项，包括历史—未来交界：

~~~text
L_temporal = mean H((Delta v_pred - Delta v_label) / 1m/s)
           + 0.1 * mean H((Delta2 v_pred - Delta2 v_label) / 1m/s)
~~~

这里不除dt或dt²，监督正常标签的离散变化，不要求加速度或jerk为零。正常转弯、加速和停车仍是监督目标。独立报告继续按dt=0.1计算物理加速度、jerk及交界，不修改阈值。本版不启用旧的raw一致性项或单独强交界/三阶物理损失；未知开关明确报错。

### 弱道路和路线项

每辆车每帧使用沿车身长度/宽度的5×5网格：25个角点、边缘和内部采样点。对精确向量可通行几何与该车t0已知计划路线走廊分别求越界距离，按1m尺度做Huber，再等权平均道路/路线和所有有效车身点。不把栅格外的点钳到边缘，不只约束车辆中心；合法换道仍由允许路线车道并集表达。

距离值来自Shapely精确最近点查询，几何投影点detach；外部点的距离梯度经Torch传到位置及朝向，再经统一解码器传到epsilon。静态集合距离的最近点包络导数在非退化点成立；转角或多重最近点处可能没有唯一导数。边界容差与现有道路评估一致为1e-6，沿用有界几何修复。

有限车身点是训练代理，可能漏掉孔洞或细窄缺口穿过点间车身区域；最终评估仍检查完整有向矩形、多边形孔洞、路线和地图覆盖。已知越界输入梯度下降不是正式生成修复或道路合规证明。

原16批次正常重构未触发道路项，不能据此说它已经改善生成。另在实际向量地图上，将正常标签构造为已知横向偏移16m的数学探针：道路代理4.30829降至4.29756，位置、速度、朝向梯度有限且非零，未更新网络。

## 4. 验证与模型候选

每个新增epoch在全部200验证窗上运行固定噪声MSE，保留best_epsilon.json所指checkpoint；如果父模型最好，它仍可作为最佳epsilon参考。每5个新增epoch对相同20个验证任务进行20步eta=0自由生成监测，保留原始/解码轨迹、修正幅度和独立质量指标。

quality_candidates.json记录质量通过数、道路/路线违规场景、运动失败数和内部jerk P99，保留多目标非支配候选，不引入旧综合场景分数。失败/未知不作为Pareto候选，仍保留原记录。候选只表示相对取舍，质量0/20的候选不是合格模型。

全50轮完成后，默认最终checkpoint在原200验证任务上与父模型进行相同DDIM、初始噪声和阈值的配对比较。报告保留失败分母、道路分层回合簇bootstrap差值区间。此结果是当前验证队列选择和诊断，不能称独立测试集结果。末轮报告不替代best_epsilon和质量候选记录。

## 5. 本地启动

使用项目已有.venv-model（Python/Torch以及data依赖包含Shapely），不修改全局环境。正式运行要求干净、已提交代码，RunId必须未使用。

~~~powershell
Set-Location E:\diffusion_new
.\scripts\stage5c.ps1 -Stage Run -RunId stage5c-v1
~~~

默认从原100轮父模型开始50轮，不接着短预演16步继续。Run自动执行训练、最终200任务采样和报告；输出为artifacts/runs/stage5c-v1/train、sample、report。它不会自动启动epsilon_cosine对照。

只训练：

~~~powershell
.\scripts\stage5c.ps1 -Stage Train -RunId stage5c-train-v1
~~~

中断后先从旧运行latest_checkpoint.json读取最新可恢复checkpoint，下面的路径须替换为真实存在的文件；选择新的RunId保留旧失败日志：

~~~powershell
.\scripts\stage5c.ps1 -Stage Train -RunId stage5c-resumed-v1 -Resume 'E:\diffusion_new\artifacts\runs\stage5c-v1\train\checkpoint_step_036000.pt'
~~~

训练完成后独立执行200任务采样及配对报告：

~~~powershell
.\scripts\stage5c.ps1 -Stage Sample -RunId stage5c-validation-v1 -TrainingRun 'E:\diffusion_new\artifacts\runs\stage5c-resumed-v1'
~~~

如果训练已完成、只有采样或报告失败，不再训练，给Sample指定已完成的训练目录；若只需重建报告，使用Stage Report以及真实TrainingRun和SamplingRun。

可选epsilon控制（单独运行才产生结果）：

~~~powershell
.\scripts\stage5c.ps1 -Stage Run -RunId stage5c-epsilon-control-v1 -Objective epsilon_cosine
~~~

新短预演：

~~~powershell
.\scripts\stage5c.ps1 -Stage Preflight -RunId stage5c-preflight-local-v1
~~~

它做16批冻结参数梯度审计、16步优化、从第8步恢复到第16步的实际一致性检查、6个自由生成任务；不计为50轮。定期监测的独立无标签重放接口：

~~~powershell
$env:PYTHONPATH='E:\diffusion_new\src'
.\.venv-model\Scripts\python.exe -m sumodiff.diffusion.stage5c monitor-probe --config configs/experiments/stage5c_training.yaml --training-run artifacts/runs/stage5c-preflight-calibrated-20261008/uninterrupted --output artifacts/runs/stage5c-monitor-probe-local-v1 --formal
~~~

## 6. 本次验收与实测

完整231项测试通过19.20s。实际校准后预演在9018625f5694c6fbdfc5cecf1a9063b69b7c9254上formal/clean完成，父val epsilon=0.001992598，16步后0.001777277；模型/AdamW/噪声Generator/CPU-CUDA RNG及逐步loss/lr/ramp恢复bitwise一致，原模型未在梯度审计中更新。

GPU为RTX4070 Laptop 8188MiB，Python3.10.16、Torch2.5.1/CUDA12.4。更新均时196.26ms，Torch allocated峰值181.01MiB、reserved284MiB；这些是Torch内存统计，不是整机GPU占用。CPU训练/验证张量缓存1,197,506,800字节，缓存准备21.26s、向量几何准备6.92s。仅15000次优化估计2943.83s，约49.1分钟；验证、监测、保存、环境采集与IO另计。预算估计不是收敛承诺。

预演主体121.67s；6任务质量0/6。随后在39c34e3d84ab09c54be645ecf71395014315b08c正式重放20任务监测，检查Torch CPU/CUDA RNG恢复和源checkpoint不变，未读取未来标签、未训练。质量0/20、道路/路线各4/20、运动失败20/20、内部jerk P99=1055.34m/s³。监测8.68s；这些短预算结果不证明新目标改善或失效。

完整输出见stage5c-preflight-20261008（原0.01权重）、stage5c-preflight-calibrated-20261008（当前权重）、stage5c-monitor-probe-20261008。小摘要见stage5c_preflight_summary.json。只提交协议、源码、测试和小摘要，数据/checkpoint/轨迹/日志在artifacts中忽略。本次完成实现与短验收；50轮、充分训练质量及收敛尚未执行或验证，阶段6保持未启动，没有创建研究里程碑tag。
