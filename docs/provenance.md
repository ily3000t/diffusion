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


## 阶段3来源与实质变化

- 位置—速度协调目标和解算由本工程自行推导实现；使用[PyTorch线性求解API](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve.html)及本机2.5.1，关闭autocast，另用NumPy增广最小二乘独立核对，不复制PyTorch源码。
- 有向矩形、SAT、恒定朝向扫掠及旋转界递归自行实现，未沿用旧高速公路积分器、平滑器或综合评分。
- [Shapely covers](https://shapely.readthedocs.io/en/stable/reference/shapely.covers.html)用于完整车身覆盖；[make_valid](https://shapely.readthedocs.io/en/stable/reference/shapely.make_valid.html)的linework可返回GeometryCollection，数值修复保留全部组成、限制变化并记录。实际Shapely2.1.2；不复制第三方源码。
- 独立运动、边界、布尔资格、失败/未知分母、资源探针和审计均新写。未读取或改写旧评估脚本，不将标签质量称为生成模型性能。


## 阶段4来源

- 新模型、适配、Q/K/V注意力、FiLM时间块、路线/规则图消息与验收工具为本工程自行实现，未复制或运行旧工程网络。CNN、U-Net、缩放点积注意力及FiLM作为常见组件使用，不能据此称为原创贡献。
- [PyTorch缩放点积注意力说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)用于核对标准形式；本工程显式处理全mask与float32 softmax，不复制官方示例或绑定flash backend。
- [GroupNorm接口](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)用于样本内归一化，实际执行库仍为torch2.5.1。在线stable目前指向更高版本，API与确定性行为以本机短验证为准。
- 路线次序、via movement代表及yield/foe来自阶段2已审计的SUMO1.22.0数据，仍保留完整原字段；没有读取未来退出信息。原始与处理数据未改写，合成12车仅用于负载测试。


## 阶段5基础扩散

noise objective及DDIM方差来自Ho等DDPM（arxiv:2006.11239）和Song等DDIM（arxiv:2010.02502）公开数学定义；在线核对作者ermongroup/ddim的functions/denoising.py，独立实现公式，没有复制该项目文件/代码或依赖其目录。论文大PDF网页读取受大小限制，公式使用作者仓库核对。固定状态尺度、每场景mask loss、元数据、安全tensor-only checkpoint、恢复签名、无标签任务及独立评估均为本工程实现。旧工程不读取训练数据/权重，不修改。

项目torch2.5.1在本机实际验证weights_only=True、float32 AdamW、严格确定性及CUDA RNG恢复。库版本变化不代表验收可复现；恢复明确检查版本，环境记录保留实际CUDA/GPU驱动。
