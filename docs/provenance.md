# 来源与实质变化

## 来源记录
1. 用户在本次会话给出的完整 SUMO 分阶段方案：设计、张量接口、数据边界、评估、Git和实验合同的主要来源，已整理到 design/data_schema/AGENTS。
2. 旧目录 `E:\diffusion_fix\guide`：只读参考。阶段0仅检查相关文件名称，包括 `danger/diffscene_style_guidance_highd.py`、`diffusion/GaussianDiffusion.py`、两份旧评估脚本；没有复制算法源代码、数据、权重或运行旧代码。
3. 新清单工具为本工程重新实现，未从旧工程复制。阶段0不实现引导算法，也不将 `diffscene_style` 名称解释为已复现原论文。

## 与旧工程的计划差别
从HighD高速公路纵横向积分表示切换为SUMO共同固定局部系中的多车二维状态；从旧checkpoint/旧评分切换为重新训练、统一解码和独立指标；基础模型不含攻击角色；引导直接调干净状态并分离风险与约束修复；后续PPO控制器只优化自身离散动作概率。

这些是设计计划，是否有效须由后续实验验证。层次融合、PPO或几何引导不能仅凭实现就称为新贡献。

## 后续借鉴要求
每次实际引用或改写代码，新增准确来源URL/文件/版本或commit、许可证、使用范围和实质修改。未核实许可证的源代码不直接复制。模型、数据不提交Git，生成方法、标识、哈希、获取位置另行记录。不得让新工程运行依赖旧路径。


## 阶段1来源
- [SUMO PlainXML](https://sumo.dlr.de/docs/Networks/PlainXML.html)：节点、edge、连接及priority/zipper规则。自行生成PlainXML并调用netconvert，未复制示例路网。
- [连续换道](https://sumo.dlr.de/docs/Simulation/SublaneModel.html)：采用lanechange.duration，而非关闭安全模式。
- [仿真事件接口](https://sumo.dlr.de/docs/TraCI/Simulation_Value_Retrieval.html)、[车辆状态接口](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html)：原始状态订阅和生命周期语义。
- 实际核对本机SUMO1.22.0随包tools/traci及sumolib，bundle revision为v1_22_0+0002-63e50f52594、TraCI协议21。运行时再核对服务器，源码树SHA-256写清单。随包客户端使用其原有EPL-2.0或GPL-2.0-or-later授权，源代码未复制进仓库；本工程场景/采集/生命周期代码自行实现。

## 阶段2来源与实现

- [SUMO路网定义](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html)：车道中心线、宽度、连接链、junction shape及右向读取的request bitset。代码自行实现，没有复制SUMO解析器。
- [SUMO车辆状态定义](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html)：前保险杠位置、导航角；本工程自行转换几何中心与固定局部系。
- 本机SUMO1.22.0的sumolib.net.Connection.getJunctionIndex和Node.areFoes用于独立交叉验证，不复制源码；来源树哈希进入审计清单。内部junction的incLanes是冲突/禁止通过信息，不能覆盖普通junction请求索引。
- Shapely2.1.2用于明确的车道半宽buffer和向量几何并集，Pillow用于栅格。新窗口、掩码、划分、独立审计和数据读取均新写，不读取/改写旧工程实现。
