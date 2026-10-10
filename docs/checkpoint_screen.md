# 阶段5E中间checkpoint短验证

先固定验证协议，再执行采样。使用原1200/200数据中的验证集；原200任务轮询序列前60个，三类道路各20个，包含4个8—12车路口。只按当前元数据和原任务顺序选择，不按未来质量挑任务。此任务集用于诊断与候选检查，不是独立测试集。

原seed20261015、DDIM20、eta0、全部尺度/解码/独立阈值保持。复用200/225轮保存输出，新增205/210/215/218/220/221轮各60次生成，总360次，无梯度或训练更新。覆盖所有正式保存的中间恢复checkpoint，不仅挑噪声MSE最好者。20任务训练监测不是本次任务/种子的替代。

generation_rows新增显式planned_window_ids，用于短诊断；默认仍严格要求完整200任务。显式任务必须与60任务采样目录或完整200任务目录的前缀逐个相同；缺失、乱序或重复拒绝，不能把短运行称作200任务结果。失败与质量不合格保留计划分母，碰撞无适用对N/A。

报告逐模型资格、两类jerk、前四帧/内部/末四帧超限、舒适性、道路/路线、修正和成本。按同任务及噪声核对来源哈希、独立运动重放。所有中间模型与200轮进行条件道路分层/回合聚类配对区间；多模型筛选未校正多重比较，不把某一最好值当作显著优于200轮。

如短检查提示值得进一步核实，完整200任务评估应另行说明，不能只以60任务替换当前候选。默认不新增loss、训练、数据或进入阶段6，不推送/tag。

运行：
~~~powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH='E:\diffusion_new\src'
$env:SUMO_HOME='E:\Program Files\sumo-1.22.0'
.\.venv-model\Scripts\python.exe scripts\validate_intermediate_checkpoints.py --output artifacts\runs\stage5e-checkpoint-screen-20261010 --formal
~~~

2026-10-10短验证已完成；结果见[checkpoint_screen_results.md](checkpoint_screen_results.md)，保留200轮候选。原主运行汇总保存失败，但六个采样子运行完成，已用--reuse-samples仅重算报告，旧失败目录保留。原RunId已使用，复现必须新目录。
