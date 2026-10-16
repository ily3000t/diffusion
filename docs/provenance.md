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


## 2026-10-07 有界训练诊断

新增diagnostics.py和render_training_diagnosis.py均在新工程独立实现，复用本仓库已验收的训练、扩散、解码、评估与RunRecorder接口。未复制或运行会写回旧工程的脚本，未使用旧权重；标签只进入重构/oracle与采样结束后的绘图参考，不进入自由采样条件。第三方绘图依赖Matplotlib，版本与实际运行来源进入environment及manifest。


## 2026-10-07 阶段5A

采集计划、完整窗口独立资格核对、流式处理/多样性选择、真正epoch遍历和定期监测均在E:\diffusion_new自行实现，复用本仓库已验收接口。未复制旧工程，不修改旧目录。显式IDM接口及原始前保险杠/导航角定义来自本机SUMO1.22.0及官方TraCI/Intersections文档；没有复制第三方源码。正常源标签保留，连续直路用单edge去除人为分段，跟驰参数变化不等于真实行为校准或动力学保证。硬链接只在新项目内材料化已选窗口，记录方式与哈希。第三方依赖及实际环境均由RunRecorder保存。


合法路线连接修复自行实现：端点相接且连接合法时，用相同半宽的round join消除分开flat cap缓冲产生的细缝，裁剪到drivable；不按测得轨迹膨胀路线或增加全junction。新SUMO路网细分/精度和转弯限速参数依据[netconvert官方文档](https://sumo.dlr.de/docs/netconvert.html)及本机1.22.0 --help核对，自行写入调用配置。改变源几何后重新采集短预检，不事后修改正常标签。


有效的裁剪GeometryCollection现在沿用有面积/有限/合法检查并保留全部组成，与现有bounded linework修复的点集语义一致；无面积集合仍拒绝。未移除几何组成、改变1e-6道路容差或平滑源轨迹。第二轮检查失败和原始记录保留，第三轮严格复用相同新回合，运行SHA分开记录。新N/A分类与selected turn统计只澄清审计，不改变正常标签或推理条件。


## 2026-10-08 阶段5A训练复核

有界重构复核、全200验证配置和通用报告范围在新项目内实现，复用现有独立采样/评估/追溯接口。未引入旧目录依赖或复制旧权重。原始实验不修改；诊断先验证全部训练身份，再以固定前缀均衡选择6/6标签探针。自由生成只使用当前状态条件，事后标签叠加用于图表。模型、训练目标、解码数学和质量阈值保持原样；完整来源及结果见stage5a_training_summary.json。


## 2026-10-08 基础轨迹质量归因

quality_audit.py、必要合成测试及render_quality_audit.py均在新工程自行实现，复用当前冻结采样/解码/独立评估和RunRecorder。NumPy增广最小二乘作为正常方程的独立数值核对；通道oracle仅在已生成结果上使用真实未来，重放使用WindowDataset.inference与源初始噪声，不加入标签条件。恒加速度控制使用后向区间均速约定、仅作指标数学对照，不复用旧积分器或引入新的正式生成方法。未复制外部/旧目录代码、模型或数据；Matplotlib实际版本进入环境及图表manifest。生产模型、训练目标、解码器和阈值未修改。


## 2026-10-08 阶段5B四组损失对照

physical_losses、controlled_training、stage5b入口及配对报告在新项目内自行实现，复用本仓库已验证的噪声目标、float64协调解码器、数据资格和独立评估。辅助项遵循现有后向区间均速约定，用正常标签监督差分与交界；未复制旧DDPO/DiffScene或依赖旧目录。真实原训练步骤用于A的bitwise对照，不引入新采样器或平滑器。共享完整AdamW/RNG继承和逐步噪声哈希用于控制实验混杂；来源、运行SHA、资源和结果进入独立manifest。


## 2026-10-08 阶段5C统一轨迹监督

unified_losses、schedule、unified_training、unified_diagnostics、stage5c入口及报告均在E:\diffusion_new自行实现，复用本工程已验证的epsilon、固定尺度、协调解码、数据资格、独立评估和RunRecorder；未复制外部或旧工程，不依赖旧目录。损失设计依据本地阶段5B横向位置/jerk诊断及用户确认方案，候选权重仅在训练批次作参数梯度校准。

Huber计算使用已有Torch smooth_l1接口；位置/速度投影、差分、门限及闭式warmup/余弦更新函数自行实现，无SGDR重启，调度状态随checkpoint保存。道路距离通过已安装Shapely covers/shortest_line/get_point接口计算精确最近点，Torch对点到detach投影的距离求导；不复制第三方代码，不用低分辨率道路栅格替代独立全车身评估。几何修复沿用本工程既有有界点集语义及1e-6容差，地图和标签未改变。

实际梯度审计未更新模型；人工位移探针仅测试道路数学，不作为生成结果。20任务监测重放只读当前输入与地图，无未来标签。原0.01权重预演eff9dda、校准0.003预演9018625、监测39c34e3的运行SHA和来源清单永久保留，后续摘要提交不回填。新初始化明确重置AdamW，新实验恢复保留全部新状态；原阶段5B继承AdamW协议不修改。


## 2026-10-09 阶段5D原始速度监督与继承续训

raw_velocity_losses/training/diagnostics、stage5d入口与报告在新工程实现，复用本仓库已验证的噪声、统一解码、正常标签、精确道路代理与追溯；不依赖或复制旧工程。原速度二阶差分数学从本工程velocity_validation提升为共享训练函数，保留原诊断接口。UnifiedProtocol复用训练循环但保留Stage5C默认100→150及新AdamW语义；Stage5D另设150→200、完整矩继承、只重启日程和独立签名。

使用已有Torch差分/Huber/autograd，校准实际总辅助及总loss向量并核对分项向量和，不相加范数比。半模仿只作冻结敏感性，没有自动权重选择。两峰值各24更新为隔离压力检查，正式预热和总预算没有被替换；源模型/优化器哈希保持。正常标签差分监督不是零jerk平滑器，也不修改独立质量阈值。

首次formal预演392e749因原尺度审计能力标记不兼容failed；原文件保留。34a7067修复仅非物理aux开关的审计映射后，以新目录r1完成，SHA永久34a706738641549944c14ff3f8b182d28442ad50。第一次报告在Recorder前路径排序失败，原日志保留；统一路径后e5eccbe新目录完成原200任务重放，不更新模型或重采样。正式运行SHA与原父训练/采样SHA不因后续结果文档和合并回填。

失败车辆纳入计划舒适性分母，实际评估率另列；无车辆对N/A与未知分开。自配对200任务零差值、6个formal/clean/completed运行及7424唯一输入hash核对通过。全部产物留artifacts，Git只包含代码/配置/文档及小型摘要。当前50轮尚未运行，没有阶段6、推送或实验tag。


## 2026-10-10 交界与密集路口诊断

boundary_validation及冻结入口在新工程独立实现，复用本仓库WindowDataset、prepare_conditioning、NoiseSchedule、统一decode_future、独立evaluate_motion及RunRecorder；不复制或写入旧工程。首帧差分模板和局部替换新交界有合成数学测试。800次前向仅用于同标签/噪声重构，不创建优化器，不修改源模型。

CPU后处理以Torch自动微分和独立NumPy法方程核对固定解码器的线性敏感性，真实未来仅用于oracle分解，既有200任务保持。科学图用已安装Matplotlib绘制并目视检查；脚本/源/图身份记录在formal=false清单和轻量摘要。正式诊断a1812b6、原训练c93a3b5及源父训练8ceb573永久分别保留，不回填结果提交SHA。

独立来源审计核对2012个唯一输入身份、3190个重构行、140组汇总、1600个oracle行与200个敏感性行；先前核对脚本将场景RMS误当车辆混合RMS而触发断言，修正为有效车辆数加权后与源输出一致，原实验和结果未改。训练探针刻意集中密集路口，不能用其总体统计宣称过拟合。候选loss配权尚未训练，本轮未推送、tag或进入阶段6。


## 2026-10-10 阶段5E解码边界监督

新decoded_temporal_losses/training/diagnostics/stage5e模块在本工程实现，复用unified/raw_velocity的预测解码和正常标签差分，未复制旧目录。共享loss仅增加内部可选分组接口，旧5C/5D目标默认保持；0增量与原total/梯度等价有测试。新目标以分组加权替换旧时间项，保留原速度/模仿/道路，不叠加重复项或用标签回填推理。

续训复用本仓库严格训练循环、AdamW复制继承、UpdateSchedule与RunRecorder；新的预算和loss签名单独定义，checkpoint加载显式支持5E。共用stage5d_report参数化套件及epoch来源，默认5D输出合同保留；5E结果parent200/candidate225，不称已存在完整225轮模型。梯度校准只用训练当前元数据/几何/回合选择，计数与实际向量贡献分开，保留自然适用批次和固定t0裁剪的解释边界。

校准第一次formal 08a4aae因索引类型failed，无更新；72051ac修复后独立r1完成。缓存独立审计脚本及输入身份进入轻量摘要，六个formal完成目录SHA永久72051ac6e6071d755f3d5a6bbd132fa971ba3e0f。源正式训练c93a3b5/200轮checkpoint b7dcb40f…不变，文档合并不回填。

12次私有full-strength压力、16次前缀及8次恢复重放共36次实际更新，只作短验收；6任务同噪声资格与父同为1/6，未宣称收益。完整200基线重放不训练或采样，自配对为0。正式25轮尚未启动，候选权重没有自动择优，不推送/tag或进入阶段6。


## 2026-10-10 阶段5E正式结果分析

用户本地正式stage5e-v1运行SHA858ad6bcd205f1966c3b251b6e048b106da5eda1；9个formal/clean/completed目录及7833唯一清单输入身份核对。新CPU脚本stage5e_result_analysis_r1.py复用本仓库已审计detail函数、独立运动差分和SVD最小二乘，重算保存结果，无模型更新/新采样或标签回填。stage5e_result_figures_r1.py用Matplotlib导出两张科学图并保存轻量摘要，脚本/源文件/图SHA在stage5e_results.json及后处理清单中。

资格82→64/200、两类jerk尾部退化；严格舒适性改善不替代整场景资格。保留200轮候选和225负结果，不宣称新loss有效、因果确定、稳定或收敛。结果文档提交不回填源SHA。全部实现/分析在新工程，旧目录未写入，无新训练/推送/tag/阶段6。

后处理首份清单误标上一阶段bootstrap种子，实际重算使用正式20261022；原脚本/目录保留，r1新增源模型/RNG哈希检查且正确标记种子，最终文档引用r1。没有覆写原清单或正式结果，全部数值一致。


## 2026-10-10 中间checkpoint短验证

新scripts/validate_intermediate_checkpoints.py复用本仓库基础DDIM采样、独立指标、配对报告及RunRecorder；没有旧工程依赖、模型/loss变更或训练更新。报告公共函数增加显式任务前缀，默认200任务合同保留。采样源17643cb，报告修复/复用源76eab2c，模型训练858ad6b分别保留。首次报告JSON整数失败的主目录保留为failed，六个采样子目录completed；独立复用报告不重复360采样。

CPU脚本audit_checkpoint_screen_r1.py核对7个完成正式运行、1197输入身份及区间并导出科学图，r1修正终端字符编码，旧图保留。轻量摘要保留脚本/图/源报告文件身份。本验证60任务只用于诊断，保留200轮候选，不锁定损失独立因果或证明收敛。阶段6/push/tag未启动。


## 2026-10-11 冻结时间目标结果整理

新temporal_diagnosis在本工程复用boundary_validation的当前元数据选择/标签缓存，严格调用既有decoded_temporal_objective及统一解码、位置差分运动评估。没有修改训练目标或复制旧目录。源formal/clean运行SHA b832aaedc00fe45ee1b0167bcfbdc1f880987316，640前向零优化，两个参数/源checkpoint哈希保持。实际运行日期2026-10-10，11日整理，不重命名原目录或回填SHA。

CPU audit_temporal_diagnosis.py核对1602唯一输入、112组均值/计数并独立重放400条已有自由轨迹，保存完整摘要和Matplotlib图。准备时旧core_only推理保护及缺少清单种子被明确拒绝；修正为按源任务ID读取无标签推理输入和显式继承真实diagnostic_noise=20261020，随后completed。没有覆写正式输出或静默降级。重构混合分位数使用源输出，仅核对计数/均值，未重复GPU张量重放。

平均监督改善与自由生成退化分开报告。整段物理jerk超限尾部只是后续可审阅候选，未实现/加训；保留200模型，不作loss或学习率独立因果结论。阶段6/push/tag未启动。

## 固定从头训练与物理尾部（2026-10-11）

新jerk_tail模块由当前方案实现，物理差分和独立quality_audit.physical_derivatives交叉核对；复用本仓库RawVelocityLossConfig的200轮目标、统一解码器、道路代理和UpdateSchedule。直接修改epoch_training/training/cli及stage5a.ps1，没有复制旧工程或创建重复训练入口；不依赖E:\diffusion_fix或E:\diffusion运行。

数据复用仅引用本仓库已完成的4个正常短回合，并核对原清单、实际配置、种子和划分；路口补采2个代表性回合。原0转弯小试验及拒绝原因保留，没有平滑原始轨迹、放宽运动阈值或覆写失败结果。固定目标和Huber尾部不自行构成创新或生成质量证据。

验收的开发资源校准真实运行SHA为cb557d6843f63ee63d0a3e60de2f687d487d7073；代表性正式Pilot真实SHA为f1f5bf4，完整哈希见其manifest及轻量摘要。后续文档或merge的SHA不回填成这两个运行SHA。开发校准明确仅用旧训练标签，不能据此冻结新6000数据权重。


## 2026-10-13 控制反解与回放

controls.py按本项目SUMO几何中心/前保险杠关系和后向区间速度独立实现，未复制CTG/CTG++源码。控制加状态rollout的思路参考原始论文 https://arxiv.org/html/2210.17366 和 https://arxiv.org/html/2306.06344 ，这里不是论文动力学或训练方法复现。显式横向速度仅作失配诊断，不称无侧滑车辆动力学。

control_replay复用本仓库WindowDataset、RunRecorder和既有独立运动/道路/插值碰撞指标。任务ID来自已保存的56窗冻结归因，来源文件SHA写入轻量任务清单；运行只依赖本工程的现有窗口数据，不依赖旧工程目录或原诊断脚本。核心状态、速度监督及评估阈值不改变。未涉及模型训练、旧checkpoint转换、未来推理条件或测试集调参。


控制回放源SHA30975255601bfbb047b4faa609b7533004cf2007；独立NumPy审计源SHA404d1bfe2de543be7b672d06dc3ad41d9b14039a。两者formal/clean/completed，数据身份与56固定任务保留。汇总只聚合原NPZ和JSON，物理jerk P99按有效点np.quantile(.99)，末4帧明确属于内部36帧；label未执行回放门槛记N/A，源文件的计数0保留不覆写。没有重新生成轨迹、训练或修改原数据。


## 2026-10-13 控制回放边界与追溯复核

98906f1修复NaN验收、保留名称、差分前填充屏蔽、历史角度别名和失败车辆分母；107c210补充独立NumPy反解与门槛/摘要核对。61077a9新增历史Git输入精确快照恢复，未复制旧工程或外部实现。

原YAML/任务JSON因Git切换换行变为LF，来源检查正确拒绝；从3097525历史blob恢复CRLF候选，只有SHA和大小同时匹配原记录才使用。两快照与强化审计实际源61077a947312538a8286324b1423e51161b268ce，formal/clean/completed。280反解、280门槛、336指标和396身份通过，原42/56及56/56结论保持；原运行未改，文档提交不回填运行SHA。


## 2026-10-14 完整训练控制尺度与覆盖交付

control_scale_audit与control_scale_review由本项目当前控制反解／回放、WindowDataset及RunRecorder实现；独立NumPy计算和分组网格测试在本仓库新增，未复制旧工程或外部模型。仅消费stage5a-fresh既有6000训练窗口，未改原始回合、选集、标签、尺度源文件或checkpoint。

正式扫描运行SHA为24e67696e9fd037237336df99c82402cd00bb424，缓存复核为8934fb777c2d3422cb8fe675dc2ed4c8fece8a0b，均formal/clean/completed。扫描记录18004输入身份；复核绑定8个扫描文件，重算7组分位数／候选包络并检查任务和车辆分区，未再次逐个读取18004原输入。交付进一步核对14项配置、环境、命令或来源缓存身份，不回填来源SHA。

候选2/0.5/0.1来自运行前固定规则。覆盖诊断5°／0.5m并未变为标签资格阈值；直路／汇入未来运动稀少的事实保留，原因未隔离。control_scale_results.md/json仅提交轻量结果与来源，大型轨迹／缓存仍在artifacts且排除Git；未新训练、采集、测试评价、阶段6、push或tag。


## 2026-10-14 三维控制接入及有界验证

表示、目标变换与共同回放基于本仓库controls、统一loss、物理jerk及RunRecorder实现，未复制旧工程或外部模型。95a865e接入明确三维表示，f42a11f接入原训练／采样／恢复，82a5d19增加组合梯度与有界协议；35dade3修复实测CUDA cumsum严格确定性不支持，采用相同有限和的下三角线性映射；29e066a扩展完整loss／日程恢复审计，9a2cc30拒绝无效旧尺度，72a7ff3补齐失败采样的实际调用计数。

初始化校准r2、完整16与8+8训练实际源35dade30f087c8234cdc7da3fc792dc5c48d25d7；后校准和恢复审计实际源29e066a5bec427eb3b42941a1a4b2838747981e8；采样r2与独立CPU复核实际源72a7ff32ed6b9a7d0edc62db9d71120d81cc2a36。全部从formal／clean代码启动，状态如实保存：初始化r1因CUDA失败，两次采样均3/3失败，未删除或改写。

独立CPU复核脚本保存在artifacts/cache/control_target_review_20261014.py，并以哈希进入365输入身份清单；NumPy独立积分、56窗恢复及采样重复逐元素核对通过。3个任务重新运行仅验证调用记录修复，没有产生额外独立任务或改善输出。未修改原始数据、选集、阈值或旧200模型；测试轨迹、长训练、阶段6、push／tag未启动。运行SHA不回填为交付SHA。


## 2026-10-14 训练运动覆盖来源追踪

eef1e77新增data.motion_coverage与experiments.motion_coverage，8f554b4冻结训练专用配置、来源认证和有界测试。原始位置从SUMO前保险杠／导航角独立计算，案例三阶差分与本仓库原raw64质量记录交叉核对；未复制旧工程或外部模型代码，未修改原采集、窗口、筛选或训练算法。

主扫描、独立复核与最终图表的实际源均为8f554b4bcee60f9f6332396edae2cf5236942535，formal/clean/completed。主扫描44804输入，独立复核44227输入、165汇总值及4回合17案例；最终图表7输入，仅修正注释遮挡，初版图和复核仍保留。独立复核及渲染脚本位于artifacts/cache，其字节哈希进入各运行manifest，不依赖旧工程目录。复核没有再次扫描所有原始回合。

解释换道端点机制参考SUMO官方SublaneModel文档，事件时间语义参考官方Lanechange输出文档；在线文档不冒称1.22.0固定源码。SL2015／横向加速度约束仅为待本机小试验的候选，尚未实施或证明有效。交付training_motion_results.md/json，来源SHA不回填为结果或合并提交；数据、checkpoint及原始结果仍排除Git。


## 2026-10-14 源头小对照与参考选择

fd66ec2扩展本仓库仿真配置与编译、5ddb1a1加入当前路线／地图参考选择、277d8b9冻结14短回合协议／896上限质量记录及18新增测试。未复制旧工程或外部模型代码；模型与参数依据本机1.22.0帮助／routeTypes.xsd和SUMO官方SublaneModel、EIDM、PlainXML文档。平滑几何为本项目五次插值采样，不冒称SUMO提供的解析动力学。

正式采集、审计、原始独立复核和图表实际源277d8b9583f02c33adfdac1e8d2e1ce5030d0677，formal/clean/completed；原始复核重建14回合、3532次选车、84分布与896缓存记录。数值摘要先提交38fb133后，补充质量复核实际源38fb1332c8554b635d33945188a9704353bdc281，重放14个代表窗口并独立核对8道路拒绝窗。功能代码未变，源运行SHA不回填交付SHA。

开发预检formal=false，包括七几何编译及七配置一步运行。初始单via片段距离统计对分段路口不适用，最终沿完整via链200端点对均为0，原预检结果保留。独立复核／质量定位／绘图脚本保存于artifacts/cache，其哈希进入相应manifest。两张科学图已目视核对。EIDM退化、SL2015在原汇入几何的极端尖峰、参考策略取舍及残余越界全部保留，未放宽资格或替换正式数据／旧模型。


## 2026-10-14 汇入出口与入口对齐短对照

2f61c2b在本仓库场景/采集中实现三项方向权限及10m入口尾段；ca62a26保留精确地图规则，并显式拒绝尚未编码的模型输入；f44da9c冻结12回合与768固定时刻协议/17新增测试。使用SUMO官方Road_Networks、PlainXML、Motorways和本机1.22.0 xsd及TraCI实测，未复制外部或旧工程代码；在线文档不冒称版本固定源码。

采集、详细审计、独立原始复核和图表实际源均f44da9ca0dd3427423ac3a8860c80791e681d27c，formal/clean/completed。独立复核脚本artifacts/cache/review_ramp_alignment_20261014.py的字节身份进入247输入清单，认证234个主审计输入；复算99380原始车身帧、626窗口世界几何与12完整质量重放，两份旧control认证历史哈希并逐字节一致。几何0/1组合分别仍有全回合最大jerk55.118/41.324m/s³，负结果保留。图脚本figure_ramp_alignment_20261014.py及4输入进入图清单，PNG/PDF目视通过；对数图0值在绘图下界标为0，不代表正越界面积。

结果ramp_alignment_results.md/json仅轻量交付，运行SHA不回填交付SHA。开发预检formal=false，旧14配置快照保持；487全量测试通过。未写旧目录、放宽质量阈值、替换正式数据、训练、读取验证/测试轨迹、进入阶段6、push或tag。


## 2026-10-14 方向换道权限条件编码

063d405在本仓库模型适配器与共享车道编码器中加入类别权限MLP，97323d7在原控制训练/校准/采样及checkpoint中绑定架构，bf74057增加现有模型CLI rules-smoke及固定协议/19专项测试。未复制旧目录或外部模型代码，不把共享MLP或版本接口作为创新证据。类别语义依据前轮已验证的SUMO XML和TraCI权限；独立复核从原编译XML直接读取，不借用同一适配函数生成期望值。

主探针与独立复核实际源均bf74057839d7891986cf52bfb371ea5ee302b8b1，formal/clean/completed；原始训练回合来源f44da9ca0dd3427423ac3a8860c80791e681d27c。主探针106输入，复核115输入含artifacts/cache/review_directional_conditioning_20261014.py字节身份；认证旧200轮及控制16步checkpoint与历史摘要哈希，5输入XML左右方向完全匹配，CPU/GPU重放最大差2.50e-6。参数哈希不变、正式优化0，生成质量未评估。

506全量测试通过。合成checkpoint测试只在临时fixture做少量CPU更新，核对真实AdamW/噪声Generator/日程续接；未执行新架构CUDA优化恢复或正式训练。方向条件不是动态可行性/规则遵从保证；5原始前缀、合成B4负载不代表新数据分布覆盖。结果与轻量来源见directional_conditioning_results.md/json，源SHA不回填为交付SHA；无数据替换、采集、验证/测试标签读取、阶段6、push/tag。


## 2026-10-15 运动平衡数据协议及短预演

a325103冻结新stage5a v3源头、划分/预算与双参考目标，60bdfb4实现离线MotionIndex/事件标识及有界选择和仅物化入选数组，f57bab4接入原PowerShell数据入口并加入15专项测试/协议。未复制旧工程或外部代码，保持原质量引擎及阈值；模型/尺度/loss未修改。源头继承前轮已实测候选，不将数据均衡/共享编码等工程设计自动称为研究创新。

正式6回合Pilot、preparation及两版独立复核实际源均f57bab438b979656f8abf43ef9ded60db4bfddda。Pilot/preparation完成，首版复核只检查首段内部via而失败，脚本及status保留；r2新脚本artifacts/cache/review_motion_data_20261015_r2.py沿完整合法via链复核通过，旧脚本未覆盖。两脚本身份均进入各自manifest，r2还绑定r1的失败元数据。未修改原始回合或生产标签/配额以使复核通过。

288计划记录、220合格池、24入选，独立核对7次运动证据及5246原始车辆时刻，位置/速度/航向误差均在原容差内；模型适配不包含未来标签。521全量测试及PowerShell解析/实际Pilot启动通过。入选24窗数据ID388aa78c044dd18c45881792cbaae1450ac725eaf53a4f1ec3af6d2e4236529e，仅短预演，不替换旧正式数据。结果/轻量来源见motion_data_results.md/json，源运行SHA不回填交付SHA；216回合正式采集、正式训练/验证/测试、阶段6、push/tag未启动。


## 2026-10-15 Motion采集恢复

新增source_quarantine.py并接入原stage5a.ps1、stage5a Collect及Motion Prepare，复用本仓库RunRecorder、原始身份/语义几何和load_episode验证；无旧工程依赖、文件复制或新第三方代码。异常源依据实际失败案例保留证据，独立quality_policy.v1/sources.v2显式变更源人口合同；原窗口配方和质量不放宽。合成元数据测试仅验证恢复/拒绝/账本接线，不能冒充SUMO采集或窗口质量结果。真实检查只读既有96回合，不自动继续批量采集。


## Motion控制训练前检查（2026-10-16）

新增内部任务生成与有限检查编排，复用本仓库控制反解、NumPy回放、联合目标、原训练及checkpoint接口；无旧工程或外部代码复制。Motion v3标签需要显式版本适配，不静默映射缺失旧字段。独立复核含梯度几何关系、实际裁剪、AdamW及checkpoint身份；原始数组/模型和运行日志全部排除Git。 详见[motion_control_preflight_results.md](motion_control_preflight_results.md)，协议[motion_control_preflight.md](motion_control_preflight.md)。
