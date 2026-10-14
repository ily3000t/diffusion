# 既有训练数据的运动覆盖归因协议

本阶段只检查已冻结的训练数据，从原始回合到候选窗口、完整性、质量过滤及最终6000选集逐层追踪。读取现有全局索引和质量日志后只保留训练行；不加载验证／测试回合或窗口数组，不采集、不训练、不改阈值和选集。

固定范围为144个训练回合、44071个候选窗和6000个最终窗，数据ID及协议见configs/experiments/training_motion_coverage.yaml。三类道路分别报告，不能以道路家族齐全代替实际运动覆盖。

## 检查定义

原始位置从SUMO前保险杠与导航角独立恢复几何中心，不读取窗口targets。对每车参考时刻固定车身系，使用t0及未来40点计算展开航向范围>=5°、横向范围>=0.5m；不将弯曲道路的横向范围直接叫换道。未来不完整的车辆记未知，单独统计，不能当作静止。

真实换道由SUMO lanechanges.xml中的change标签提供，时间严格属于(t0,t0+4s]、车辆必须是t0选定对象。changeStarted/Ended不重复计数。原始事件、参考网格上对象已存在、80m半径内和实际选定对象四层分别统计，区分任务选择损失与过滤损失。窗口重复发生次数与唯一事件数同时报告。

汇入结束使用ramp路线对象从acceleration经过可选内部连接进入main_out的道路状态转换，不把普通main路线同一转换计为匝道事件；这只是已声明的路段转换标记。原始轨迹与窗口元数据按ID、回合、道路、t0及固定参考原点核对，完整历史所需的额外前一位置样本也参与生命周期检查。

质量过滤沿用既有raw64／stored32记录，报告各运动组拒绝原因，原因可重叠。确定性的有界案例用原始前保险杠／中心位置重算完整21历史与40未来、含三阶差分前缀的jerk，与原raw64审计最大值核对；不重算或放宽道路与运动资格。

最终6000窗的原始运动与之前存储标签审计逐车对照；航向数值容差1e-6rad、横向范围1e-4m，5°／0.5m分类必须完全一致。采样窗口和车辆不是独立观测，分位数与事件发生次数不得解读为真实交通总体概率。

## 运行与来源

~~~powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv-model\Scripts\python.exe' -m sumodiff.experiments.motion_coverage --config configs/experiments/training_motion_coverage.yaml --output artifacts/runs/training-motion-coverage-r1 --formal
~~~

在已提交干净代码运行，使用未占用目录。manifest记录完整实际配置、SHA、命令、环境、种子、原始输入和元数据身份。window_lineage.jsonl保留每个训练候选的阶段、已知／未知未来、事件与拒绝理由；episode_summary.json、source_cases.json及results.json保存回合、原始证据与汇总。遇到来源、分母、生命周期或原审计不一致就明确停止，保留失败记录，不跳过后宣称全量通过。

本阶段结论用于决定后续检查或新的采集／选窗方案，不自动执行补采、重选、平滑、阈值修改或正式训练。
