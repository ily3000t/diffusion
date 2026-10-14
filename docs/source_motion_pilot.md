# 正常交通源头与参考车辆的小规模配对试验

本阶段只做训练专用短回合对照，不替换既有6000/600/600数据，不优化模型，不进入阶段6。前置证据见[运动覆盖归因](training_motion_results.md)。一次试验固定14回合：每种配置2个相同种子，每回合正常需求30s，排空上限240s；预算越界明确拒绝。

## 候选与固定条件

| 道路 | 配置 | 变化 |
| --- | --- | --- |
| 直路 | baseline / sublane | 原LC2013持续3s换道；或SL2015、横向加速度0.3m/s²、横向速度0.8m/s、分辨率车道宽/4 |
| 汇入 | baseline / sublane / sublane_transition | 同上；第三种另用80m连接过渡及平滑匝道入口 |
| 路口 | baseline / eidm | IDM；或EIDM且jerkmax=3m/s³，其他共享输入保持 |

两种子20261105、20261106；几何来自既有stage5a计划的训练索引：直路／汇入0、路口6。对照固定需求、路线、种子、尺寸和原驾驶参数；模型变化会改变交互与随机数消耗，不声称轨迹逐车配对相同。EIDM比较是整个纵向模型候选，不是只改变jerkmax的因果消融。新几何只作为训练专用候选，不混入旧数据或沿用旧尺度。

本机SUMO1.22.0帮助和routeTypes.xsd实际包含lateral-resolution、SL2015参数及EIDM/jerkmax，另做七种配置编译与一步解析。文档依据为[SUMO SublaneModel](https://sumo.dlr.de/docs/Simulation/SublaneModel.html)、[EIDM](https://sumo.dlr.de/docs/Car-Following-Models/EIDM.html)及[PlainXML连接定义](https://sumo.dlr.de/docs/Networks/PlainXML.html)。在线文档不是版本固定源码；参数可接受不等于质量已改善。

配置显式拒绝SL2015与旧duration同时启用、缺横向限制、LC2013误用lcAccelLat及非EIDM误用jerkmax。旧默认仍为LC2013与原几何。保留speedMode31、laneChangeMode1621、碰撞／teleport记录与正常安全规则，不通过TraCI移动或强制换道制造事件。

汇入几何候选将加速段末端的横向位移分散到80m内部连接，理想曲线使用两端一二阶导为零的五次插值，按0.25m采样后交给netconvert。主路连接保持同一横向位置，匝道入口对齐加速车道。SUMO执行的是编译后的折线，不能声称其jerk已由解析曲线保证；须检查实际中心位置差分。新几何同时改变了道路／路线区域，跨几何资格变化不能全部归因于运动模型。

## 当前信息参考策略

first_current_id保留旧逻辑。interaction_current仅接收t0观测及地图／已知路线：直路最大化80m当前邻域数量；匝道优先选择尚未进入main_out且计划来自ramp的车辆，接近merge_end者优先；路口优先选择已知合法转弯、尚在入口段且接近center的车辆。没有此类当前对象时回退当前密集邻域，距离及ID确定性打破平局。

邻车仍按t0距离排序，最多12车，槽位固定。未来位置、future_mask及真实退出时刻不参与策略。interaction窗口ID增加-interaction后缀，避免同回合、同t0与旧策略的不同样本混名。这是为交互覆盖设计的任务分布，不能当作自然交通总体比例。

## 独立检查和分母

每个回合对全部候选参考时刻做轻量当前选车、实际换道change和匝道退出覆盖统计。完整未来才能计算5°航向／0.5m横向范围，真实事件以原始日志和道路转换为准。

详细质量检查取当前全局至少2辆车的时刻，按时间等间隔选最多32个参考时刻，两参考策略使用完全相同的时间清单。此选择不查看未来或质量结果。逐窗构造21历史／40未来及最多12车，复用原audit_window的raw64、stored32、完整车身道路／路线、帧间碰撞和运动门槛；不降低jerk15等阈值。所有不完整、单选车、失败均保留。

只有详细检查子集报告合格率和“合格已评估窗覆盖的唯一事件”。全部候选的未评估质量保持unknown，不能将子集计数当作全回合合格事件总数，也不能把未见事件写成质量通过。原始事件上下文单独报告中心jerk及完整性；上下文不完整时通过状态为N/A。路口60m内另报中心与SUMO报告纵向加速度差分，帮助区分纵横向变化，不视为实车动力学。

原始与子运行配置、清单、种子和输入SHA均校验；正常交通不合格的回合保留为失败案例，不跳过后缩小14回合分母。输出results.json、逐窗window_quality.jsonl与每回合原始记录，不生成新正式dataset。

## 运行

在已提交干净代码、未占用输出目录运行：

~~~powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
$env:SUMO_HOME = 'E:\Program Files\sumo-1.22.0'
& '.\.venv-model\Scripts\python.exe' -m sumodiff.experiments.source_motion_pilot collect --config configs/experiments/source_motion_pilot.yaml --output artifacts/runs/source-motion-collect-r1 --formal
& '.\.venv-model\Scripts\python.exe' -m sumodiff.experiments.source_motion_pilot audit --config configs/experiments/source_motion_pilot.yaml --collection artifacts/runs/source-motion-collect-r1 --output artifacts/runs/source-motion-audit-r1 --formal
~~~

已有目录不能覆盖。采集与审计分开，便于复核原始记录而不重新采集。资源与收益以实际结果为准；两个种子、一种训练几何及最多32个详细参考时刻只是定位机制的预算，不足以冻结最终数据分布或控制尺度。若收益不足，保留负结果后再决定下一轮有限修改，不自动扩大到正式采集／训练。
