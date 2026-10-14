# 汇入出口权限与入口对齐短对照

本阶段只使用训练几何，不替换既有6000/600/600数据，不训练模型，不进入阶段6。前置问题见[source_motion_results.md](source_motion_results.md)：80m过渡已减少连接jerk，但仍有出口横向偏移及入口细小车身越界。

## 固定预算与因素

复用source_motion_pilot的collect/audit入口，配置为configs/experiments/ramp_alignment_pilot.yaml。固定12回合、种子20261105/20261106、30s正常需求、240s排空上限。全部沿用SL2015、横向加速度0.3m/s²、横向速度0.8m/s、分辨率车道宽/4、80m过渡及正常安全规则。

| 训练几何索引 | 配置 | 每配置回合 |
| --- | --- | --- |
| 0 | control / permissions / alignment / combined | 2 |
| 1 | control / combined | 2 |

permissions只设置三项方向规则：acceleration_1禁止乘用车向右进入加速车道；其通向main_out_0的内部连接禁止向右；加速车道通向同一出口的内部连接禁止向左。使用changeLeft/changeRight=emergency表示乘用车限制，保留应急类别。外部加速车道向左汇入及其他正常换道不受这三项规则限制。必须同时认证netconvert编译结果和TraCI实际getChangePermissions，失败立即保留并报错。

alignment把匝道入口五次曲线的末端移到x=-10m，随后以10m直线接入；不加宽道路、不改质量容差、不平滑采集标签。combined同时应用两因素。几何0的control应与上一阶段sublane_transition的原始轨迹复现；几何1用于小范围泛化检查，不据两几何宣称广泛有效。

SUMO官方说明直行内部连接允许换道，方向规则语法见[Road Networks](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html)、[PlainXML](https://sumo.dlr.de/docs/Networks/PlainXML.html)及[Motorways](https://sumo.dlr.de/docs/Simulation/Motorways.html)。在线文档不是固定1.22源码；本机还需实测解析与运行。

## 窗口与独立检查

每回合固定32个t0：tick=30,60,...,960（3–96s），两参考策略共用相同时间。计划质量分母12×32×2=768；回合失败、无当前车辆或超出回合的时间点记为unavailable，不移到其他时刻，不从计划分母删除。既有全部候选轻量运动覆盖照常统计，未详细评估质量保持unknown。不能将768重叠窗口当作独立回合。

质量使用原raw64/stored32及统一道路、路线、完整车身、帧间碰撞、速度/加速度/jerk门槛；额外独立核对原始世界坐标车身、出口横向偏移、运动分布、事件及来源哈希。分别报告已评估和计划窗口分母、唯一事件；不保证本次修复有效，失败与退化如实保留。

## 版本与模型边界

scenario.v2仅支持本轮匝道接口：geometry.alignment_length和network.protect_acceleration_lane。旧v1不增加默认键，原14配置快照必须相同。精确地图保留方向权限。现有神经网络车道特征未编码该权限，因此模型适配器遇到乘用车限制明确拒绝；不能直接把此候选用于新正式训练。后续若采用权限方案，必须先扩展规则编码及checkpoint合同。

正式运行先提交并保持工作区干净：

~~~powershell
$env:PYTHONPATH='E:\\diffusion_new\\src'
$env:SUMO_HOME='E:\\Program Files\\sumo-1.22.0'
& '.\\.venv-model\\Scripts\\python.exe' -m sumodiff.experiments.source_motion_pilot collect --config configs/experiments/ramp_alignment_pilot.yaml --output artifacts/runs/ramp-alignment-collect-20261014-r1 --formal
& '.\\.venv-model\\Scripts\\python.exe' -m sumodiff.experiments.source_motion_pilot audit --config configs/experiments/ramp_alignment_pilot.yaml --collection artifacts/runs/ramp-alignment-collect-20261014-r1 --output artifacts/runs/ramp-alignment-audit-20261014-r1 --formal
~~~

结果保留运行SHA及完整manifest/config/command/environment/status。无长训练、测试数组读取、旧数据替换、push或tag。
