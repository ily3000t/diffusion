# 阶段1场景与采集协议

## 运行入口和环境

工作目录为E:\diffusion_new。使用项目 `.venv-sumo` 或具备PyYAML的Python，显式指定SUMO_HOME；入口只接受经过安装路径、bundle revision、二进制版本和TraCI协议核对的SUMO_HOME/tools客户端。客户端源码树哈希进入resolved_config和episode_manifest，不复制外部源码。不修改旧工程或全局包。

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv-sumo\Scripts\python.exe' -m sumodiff.simulation collect --config configs/scenarios/ramp_a.yaml --output artifacts/raw/my-ramp-episode --seed 21 --formal
& '.\.venv-sumo\Scripts\python.exe' scripts/validate_collection.py --output artifacts/runs/my-stage1-validation --formal
```

每次用新的输出目录，正式模式要求已提交、干净的工作区和Git忽略的仓库内输出。第二条命令固定为6个正常短场景回合、1个换道/停车诊断、1个teleport诊断、1个预期截断失败，不是批量数据集采集。预期截断子运行status为failed，验收父运行在确认失败保留正确后可为completed。

## 三类几何
- 三车道单向直路：入口缓冲、核心道路、出口缓冲，保持三车道合法连接；参数含长度、宽度、限速和整体旋转。
- 匝道：三车道主线和单车道匝道接入四车道加速段，再合并为三车道出口；显式合法车道连接，入口优先、末端zipper正常合并规则。变化匝道偏移、加速段、入口/出口、宽度、限速和旋转。
- 无信号路口：四臂、每向1或2车道，priority或right_before_left；直行、左转、右转路线及正确车道连接，不设置U-turn或traffic_light。变化臂长、南北偏角、宽度、转弯半径、限速、通行规则和整体旋转。

正常车辆类型为passenger，Krauss跟驰及LC2013换道；速度模式31、换道模式1621在原始测量中逐车核验。通过lanechange.duration=3s启用非瞬时换道，不设置关闭安全规则的模式。流量、计划路线、驾驶参数和SUMO种子均可变化；需求是有限时段，采集等待全部车辆排空。

geometry_id对编译后地图的拓扑、车道几何/宽度/速度、连接和优先关系做规范化哈希，排除生成注释中的时间和绝对路径；不含流量、种子或回合名。网络原始文件另外保存内容哈希。验收重编译同一地图到不同路径/种子，核对geometry_id不变。

## 原始数据与完整性
每个回合除阶段0六文件清单外还包含：
- `scene/`：nodes/edges/connections、编译net、已知路线与需求、sumocfg、netconvert命令/日志和scene.json。
- `frames.jsonl`：一个JSON对象每tick，dt=0.1；tick0为空初始状态，其后记录全部当前车辆，不按12车模型上限截断。
- `events.jsonl`：启动时loaded及各步loaded/entered/arrived、由到达或teleport证据解释的exited、teleport始末、停止/停车始末、紧急停车、SUMO碰撞信息及诊断控制请求。
- `episode_manifest.json`：实际运行SHA/分支、种子、运行状态、原始字段语义、几何ID、客户端来源、场景和输出哈希、完整数据内容标识、正常训练资格。
- `quality.json`：时间网格、采样/车辆数量、生命周期与未到达ID、连续换道/停止/转弯证据、原始前保险杠单步位移摘要及资源实测。这里的SUMO碰撞事件不是阶段3的独立几何碰撞指标。
- `process.json`、`collection_partial.json`、SUMO命令/日志、lanechanges/collisions/tripinfo XML；截断时另有`truncation.json`。

帧结构：`schema_version=sumodiff.raw.frame.v1, tick, time_seconds, vehicles`。逐车字段为vehicle_id及sumo_raw：前保险杠x/y、导航角degree、SUMO速度/加速度、尺寸/类型/车辆类、当前edge/lane/index/沿车道位置/横向偏移、当前已知route_id/完整edges/index、stop_state、安全模式、实际出发时刻。

阶段1只保存明确命名的SUMO原始字段，没有混入二维差分速度或固定局部坐标；中心、数学角度、后向差分和窗口在阶段2实现。getRoute在当前采样时刻读取，未来退出时刻未用作条件。离开当前ID列表但无到达/teleport证据时记unexplained_disappearance，不冒充到达。

dt由实际服务器核对；每步tick/time核对，不以浮点相等判断。到达流量结束时刻且getMinExpectedNumber=0后才结束；max_seconds达到但未排空是失败，保留原始帧/事件、质量摘要、进程与回合清单，不转成completed。进程异常也保留failed，不吞掉错误。

## 诊断与训练资格
straight_probe含一次遵守默认安全模式的TraCI换道请求和正常计划stop，用于验证，不进入正常训练数据。teleport_fixture仅将拥堵teleport阈值缩短至0.5s，实际采集teleport始末和位移跳变，明确diagnostic_only且不进入正常训练。truncation_fixture在13s终止未排空回合，用于验证失败记录，明确失败并保留。

正常配置不包含诊断控制。eligible_for_normal_training要求完整回合、正常质量通过且非diagnostic_only；资格只是采集质量标记，不能证明真实驾驶行为或动力学可行。任何被排除回合仍保留及计数，阶段2再决定窗口规则。

资源字段是逐tick采样的进程RSS峰值估计，不是OS连续监测的精确峰值；字段名明确sampled。阶段1不使用GPU，不给出模型显存或训练吞吐量。

独立文件审计可复现：

```powershell
& '.\.venv-sumo\Scripts\python.exe' scripts/audit_collection.py --run artifacts/runs/my-stage1-validation --output artifacts/cache/my-stage1-audit.json
```

审计直接读取原始JSONL和SUMO tripinfo，对照采样网格、逐帧ID/路线/安全模式、生命周期数量、输入输出哈希以及运行时SHA，不只读取采集器的质量标记。输出必须是新文件。
