# 阶段5B：从100轮共同checkpoint继续10轮的四组对照

本轮保持阶段5A已审计的1200训练窗、200独立验证窗与网络/解码器/评估阈值；不重新采集，不进入阶段6。10轮是首轮诊断预算，不是收敛承诺。长训练仍由用户本地终端启动，助手先做每组12次更新的真实短预演。

## 共同起点与预算

共同父模型为 `artifacts/runs/stage5a-v1-train/checkpoint_step_030000.pt`，SHA-256：
`2315fa4d57b9d86a3254a0923590efbbd48f23666f3bf0d2c733ac04b4d1769b`。

A—D各自独立继承同一父模型的网络、AdamW完整状态、噪声Generator和Torch CPU/CUDA随机状态。batch=4、lr=0.0003、weight_decay=0.0001、梯度裁剪1保持父模型设置；无学习率调度、EMA或额外架构修改。四组均训练真正的101—110轮，每轮1200窗无放回遍历一次，共3000更新，最终step=33000。记录并核对每步窗口ID、扩散t及噪声哈希，避免额外随机调用混入对照。

新实验初始化与严格断点恢复分开。B—D有新的目标配置与checkpoint签名，不能作为原噪声MSE实验的原地resume；中断恢复只允许同组、同数据和同预算签名。父checkpoint和旧实验目录不覆盖。

## 四组训练目标

| 组 | 目标 |
|---|---|
| A | 原掩码噪声MSE；保持原算法计算图 |
| B | A + 原始预测的位置—速度一致性 |
| C | B + 预测速度的时间差分监督 |
| D | C + 统一解码后的历史—未来交界监督 |

原噪声MSE始终按每场景有效标签元素归一化。辅助项在扩散训练t≤300的场景上计算，由同一次网络epsilon输出恢复clean状态，无额外网络调用或随机抽样。各辅助项先取有效坐标的场景均值，乘对应alpha_bar(t)，再除整个batch大小（不是只除低噪声场景数），以限制高噪声重构放大的梯度。采用Huber(beta=1)，每个启用项固定权重0.01；不同组共有项的权重相同。权重与门限是第一轮候选，不预先声称最佳或有效。

记预测偏移q、预测区间均速v、正常标签v*、dt=0.1s。B约束 [(q_t-q_{t-1})/dt-v_t]/20，q_0=0为固定观测起点。此项在原始预测上计算，避免统一解码已经强制一致后损失恒为零。

C监督 [diff(v)-diff(v*)]/(dt×6)，只使用相邻有效未来区间。正常标签的加速、减速和转弯保留，不将真实运动强制变成零加速度/零jerk。

D使用现有float64位置—速度协调解码器，分别解码预测与正常标签，拼接最后3个历史位置，再比较最初两个未来区间的速度、加速度、jerk。分别除20m/s、6m/s²、15m/s³后取三项平均。t0位置固定，真实历史—未来变化作为监督，不强制预测速度等于历史末点或强制边界jerk=0。完整H+F要求显式校验；缺失标签不静默降级。当前未新增朝向损失，朝向问题仍可能限制最终质量。

future_mask只用于训练标签有效性，真实未来与未来退出时间不进入模型条件。A禁用的辅助项保持为零，B—D启用项必须在真实短预演中有有限、非零贡献。模型标签/噪声、解码梯度、有效槽位和无效NaN填充有独立必要测试。

## 验证与报告

每轮验证全部200窗的固定噪声MSE，保存checkpoint；每新增2轮监测相同6个自由生成任务。四组终点都按相同200个验证任务、同初始噪声、DDIM20步/eta=0采样，与100轮父模型已有200任务采样比较。无测试集参与参数选择。

短预演每组仅12次更新；另用真实GPU执行12次原训练算法，核对A的网络、AdamW、噪声Generator、CPU/CUDA RNG及逐步loss/裁剪norm是否bitwise一致。核对A—D起始状态、训练批次/扩散噪声和6任务初始采样噪声一致；记录实际辅助项、耗时、显存。预演不等同完成10轮，不用其质量通过率判断收敛。

正式四组全部结束后自动报告：父模型与A—D的质量通过数/200、运动、0.5—4s内部jerk、边界jerk、道路/路线/地图覆盖、碰撞、原始/解码修正、采样与训练成本。目标角色未定义，目标风险指标N/A。失败和质量未知保留在质量分母200中；连续指标缺失对数明确报告，不用零填补。

配对差值包括parent→A/B/C/D和A→B、B→C、C→D。95%区间按道路分层，以回合作为重采样簇，2000次bootstrap；这是当前验证队列条件下的区间，不能代表跨模型训练种子稳定性。单一父模型、单一配对采样seed的局限保留。训练恢复链只统计已提交到恢复checkpoint的更新，避免中断后未保存的日志重复计数。

## 本地命令

在PowerShell执行：

```powershell
Set-Location E:\diffusion_new
.\scripts\stage5b.ps1 -Stage Run -RunId stage5b-v1
```

此命令依次执行A—D各10轮、各200任务采样，并在 `artifacts/runs/stage5b-v1/report/report.md` 与 `paired_summary.json` 自动生成配对报告。入口设定项目虚拟环境及SUMO_HOME，无需修改全局环境。每次用新RunId，拒绝覆盖已有输出。正式运行要求代码已提交且工作区干净；所有子实验保存实际生效配置、父模型哈希、运行SHA、完整命令、种子、环境和状态。

只重新执行短预演：

```powershell
.\scripts\stage5b.ps1 -Stage Preflight -RunId stage5b-preflight-new
```

若B训练中断，使用其最后已保存的同组checkpoint，在新目录恢复剩余预算（不会再额外增加10轮）：

```powershell
.\scripts\stage5b.ps1 -Stage TrainArm -Arm B -RunId stage5b-B-recovered -Resume .\artifacts\runs\stage5b-v1\B\train\checkpoint_step_030600.pt
```

若训练已completed且终点33000，只有采样失败，保留已完成训练，单独在新目录采样：

```powershell
.\scripts\stage5b.ps1 -Stage SampleArm -Arm B -RunId stage5b-B-resampled -TrainingRun .\artifacts\runs\stage5b-v1\B\train
```

恢复后按A、B、C、D顺序指定四个最终组目录重新报告，例如B为已恢复的新目录：

```powershell
.\scripts\stage5b.ps1 -Stage Report -RunId stage5b-report-recovered -ArmRuns .\artifacts\runs\stage5b-v1\A,.\artifacts\runs\stage5b-B-recovered,.\artifacts\runs\stage5b-v1\C,.\artifacts\runs\stage5b-v1\D
```

完整训练不自动延长候选，不自动进入阶段6、不创建研究里程碑tag。应先看A的继续学习效果，再判断B/C/D的增量收益，最后才决定是否延长候选预算。
