# 数据与追溯接口 v1

本文件规定后续阶段的数据合同。阶段0已实现运行清单，阶段1已实现原始回合采集；坐标、地图提取和窗口合同在阶段2实现。具体原始字段与文件见sumo_collection.md，不代表模型训练已实现。

## 1. 标识、单位与时间
- `schema_version` 使用独立字符串版本，例如 `sumodiff.raw.v1`、`sumodiff.window.v1`；修改语义须更新版本。
- 每个 episode 保存 `episode_id`、`scenario_family`、`geometry_id`、地图/配置 SHA-256、SUMO 版本、种子、dt、完整生命周期事件、实际采样时间。
- `geometry_id` 表示实际拓扑和几何，不只使用文件名或场景类别。数据版本以生成配置、原始内容哈希和划分清单共同确定。
- 位置/尺寸：m；速度：m/s；加速度：m/s²；jerk：m/s³；时间：s；数学角度：rad。SUMO 原始导航角保存 degree。
- 时间使用整数 tick 配合 `dt=0.1`，避免浮点相等判断。原始采集不裁成定长回合。

## 2. 原始记录和生命周期
每车每 tick 保存：`vehicle_id`、tick/time、SUMO 原始前保险杠 x/y、导航角、SUMO 报告速度、length/width/type、road_id/lane_id、当前已知计划 route edge 列表及其记录时刻。转换后中心坐标和数学角度使用单独字段，不能覆盖原始字段。

生命周期记录进入、到达、退出、teleport 开始/结束等实际可观察事件。事件来源及时间明确标注；不把查不到 ID 自动解释成到达。路口内部 edge/连接保留。驾驶参数、流量、路线、地图构建输入和运行退出码进入回合 manifest。

连续换道使用 SUMO 正常安全规则及连续换道配置，不关闭所有安全检查制造变化。短回合验证包括换道、转弯、停车、进入/退出与 teleport 记录能力。

## 3. 坐标与后向差分
设 SUMO 导航角 α，数学角 θ = (90° − α) × π/180；航向单位向量 h = (cosθ, sinθ)。几何中心 c = p_front − (length/2) h。每次转换保留输入以便核对。

参考车在 t0 的中心 o 和朝向 θ0 固定整个窗口的参考系：

```
R = [[cosθ0, sinθ0], [-sinθ0, cosθ0]]
p_local = R @ (p_world - o)
v_local = R @ v_world
θ_local = θ_world - θ0
p_world = R.T @ p_local + o
```

地图、路线、其他车辆共享同一变换。各车未来位置学习标签为 `p_local(t) − p_local(t0)`，并非每一步位移，朝向不转到逐车自身坐标系。

`v(t) = (c(t) − c(t−dt)) / dt`。历史21点为 t0−2.0…t0，需额外 t0−2.1 的中心位置。未来40点为 t0+0.1…t0+4.0。历史第一速度缺乏前点则明确标记缺失或过滤理由；禁止读取 t0 后的位置补历史。SUMO 报告速度仅用于核对，不混用为训练二维速度。

## 4. 窗口字段
| 字段 | 单样本形状/语义 |
|---|---|
| history | [N,21,6]，x,y,vx,vy,sinψ,cosψ |
| future | [N,40,6]，Δx,Δy,vx,vy,sinψ,cosψ；训练标签 |
| initial_position | [N,2]，各车 t0 在共同局部系的位置 |
| vehicle_attributes | 长宽、类型编码、参考车标记；具体编码须版本化 |
| agent_ids | [N]，有效 ID 固定，不通过未来信息重排 |
| agent_mask | [N] bool；真实被选定车辆 |
| history_mask | [N,21] bool；六维状态完整有效 |
| future_mask | [N,40] bool；只用于标签、损失和离线统计 |
| reference_slot | 一个有效槽位；参考车必须纳入 |
| frame_origin/frame_yaw | 固定世界原点与 θ0，支持反变换 |
| t0_tick/episode_id/geometry_id | 时刻、回合和地图追溯 |
| route_conditions | 各车 t0 已知路线/合法走廊；独立有效掩码 |
| map_conditions | 栅格、折线、连接及规则；独立有效掩码 |
| selection_summary | t0 总车辆数、选定数、排除数、确定性选择规则版本 |

N 的初始上限为12，包含参考车。batch 前加 B 维。填充状态置零并通过掩码隔离；零数值不代表停车或缺失。无效槽位不能进入注意力归一化、损失、风险和统计。输入包含非有限数应明确拒绝或记录过滤，不静默修补。

选择只用 t0 可用状态与已知路线。槽位和 ID 在整个窗口稳定；未来缺失不会替换为另一车辆。推理可使用 agent/history/route/map 掩码，不传 future_mask 或真实未来退出时间。

先按 episode 和 geometry 分组隔离划分，再切 t0 间隔1s的窗口。训练、验证、测试间 episode_id 与 geometry_id 均不可交叉；同几何的多种种子和流量仍属于同一几何组。不得先切重叠窗再随机分配。划分清单固化且哈希，类别覆盖及不足如实报告。

核心训练优先完整未来窗，但所有不完整窗保留其标识、mask、过滤/退出原因和数量。缺失数据、过滤、车辆排除数及地图覆盖统计均写数据质量摘要。

## 5. 地图与计划路线
- 初始栅格 `[C,256,256]`：可通行区域、边界、中心线；C的精确通道定义与栅格映射版本化，不按车辆槽位增通道。
- 保存栅格物理 extent、像素尺度、坐标变换。物理覆盖须按速度、4s时域和车辆范围检查后冻结，不能只写像素尺寸。
- 车道折线含采样点、方向、宽度、限速、edge/lane 标识；点与折线各有掩码。
- 连接图保留合法后继、路口内部连接、优先通行关系及其来源。
- 每车允许路线走廊按 t0 计划构建，纳入合法换道；精确几何另存，不用低分辨率栅格代替完整车身越界检查。
- `out_of_map` 与 `off_road` 独立输出，不能将地图外未知区域默认视作道路。

## 6. 输出轨迹和 checkpoint（后续实现）
每次采样分别保存原始预测、统一解码轨迹、引导轨迹（base 标记不适用）、位置/速度/朝向修正幅度、朝向回退次数及原始一致性误差。元数据记录单位、固定局部系、车辆 ID、窗口 ID、任务车对、初始噪声标识、DDIM设置、引导调用和失败。

解码固定观测起点，用合适精度求解位置—速度协调，最终速度由位置后向差分定义。正常标签输入只应发生数值误差级修正。原始和各解码阶段运动指标分别报告；边界变化计算应包含最后历史点。

checkpoint 保存格式版本、模型/能力配置、数据版本与划分哈希、固定归一化尺度、运行 SHA、训练/恢复状态和种子。旧 checkpoint 明确不兼容；CFG/部分扩散/滚动/控制器能力关闭，启用未支持项报错。

## 7. 已实现的实验清单 v1
每次运行独立目录，不覆盖已有目录：

| 文件 | 内容 |
|---|---|
| manifest.json | `sumodiff.run.v1`；运行 ID/用途；运行时 Git SHA/分支/工作区状态；配置哈希；种子；数据清单（内容标识、位置与可选文件哈希）；checkpoint路径和SHA-256；命令、实际启动工作目录与环境文件SHA-256 |
| resolved_config.yaml | 调用方提供的完整生效配置及清单命令自身的生效参数；不只保存配置路径 |
| command.txt | 完整命令，使用操作系统的参数引用规则；另外在manifest保存 argv |
| environment.json | Python及解释器、平台、全部已安装发行包版本、PyTorch/CUDA/GPU查询、SUMO/驱动查询；失败保留原因 |
| metrics.json | 真正产生的指标，运行初始为空对象；不得填入示例科学结果 |
| status.json | running/completed/failed、UTC时间、错误类型及信息 |

`RunRecorder` 的上下文正常结束标记 completed，异常标记 failed 并重新抛出，初始化失败不得生成看似可用的 completed 运行。正式模式启动前拒绝无提交或脏工作区；开发 smoke 明确 `formal=false`。环境查询失败字段显式带 error，CUDA build 与驱动版本分开记录。

数据清单应包括所有实际输入及划分/任务清单，不能只记录目录名。若提供文件路径则流式计算 SHA-256；尚无数据/模型的阶段0metadata smoke使用空清单和空checkpoint，不伪造身份。实际训练/采样入口在后续阶段必须验证必需数据和checkpoint已提供。

## 8. 阶段2已实现的窗口/地图合同

具体命令、算法约定和限制见[窗口协议](window_dataset.md)。数据状态仍为物理单位，归一化候选未冻结。raw只读，未删除或平滑原始轨迹。

| 文件或字段 | 实际形状/语义 |
|---|---|
| conditioning.history | float32 [N,21,6]，共同固定局部系，后向速度 |
| conditioning.agent_mask / history_mask | bool [N] / [N,21] |
| conditioning.attributes | float32 [N,4]：[长m,宽m,乘用车类型码1,参考标记]；原始type_id另存 |
| conditioning.initial_positions | float32 [N,2]，各车t0在共同局部系的位置 |
| conditioning.map_raster | uint8 [3,256,256]，可通行区域/外边界/中心线；后续转float |
| conditioning.lane_polylines | float32 [L,64,8]，xy/方向xy/宽/限速/internal/priority |
| conditioning.lane_mask / lane_point_mask | bool [L] / [L,64]；batch补齐时false |
| conditioning.lane_adjacency | bool [L,L]，合法连接source→via→后续via→target |
| conditioning.route_lane_mask | bool [N,L]，已知计划路线全部合法车道及内部连接 |
| targets.future / future_mask | float32 [N,40,6] / bool [N,40]，仅标签 |
| input.json | sumodiff.window.input.v1，t0、ID、固定frame、已知完整路线及当前route_index |
| labels.json | sumodiff.window.labels.v1，历史/未来完整性和QA；不送入inference |
| map.json | sumodiff.local.map.v1，局部向量几何、原始折线及通行规则 |
| dataset_manifest.json | sumodiff.dataset.v1，预处理SHA、源清单、实际配置、窗口索引及输出哈希 |
| sources JSON | sumodiff.sources.v1，每条显式manifest路径和train/validation/test归属 |

默认N=12；不同L由collate补齐，不把车辆槽位绑定到地图通道。真实回合唯一键采用data_id内容哈希，source_episode_id保留原始目录名；不再依赖basename全局唯一。窗口ID为完整内容哈希+参考tick。

当前reference_policy为first_current_id，槽位0；半径80m按t0距离/ID选车。从3s开始每1s取参考点；历史第一点需t0−2.1s原始采样。新生车辆允许历史mask不完整，未来消失不换槽。全部不完整窗保存，core_training_eligible只用于训练过滤，不能筛推理任务。

未来Δxy必须加各槽位initial_positions才能得到共同局部坐标，再使用固定frame反变换到世界系。不存在逐车旋转参考系或移动参考系。

精确向量几何在[-192,192]m栅格外仍可有已知路网；out_of_map指CNN范围不足，off_road依完整向量道路单独判断。8m余量用于覆盖检查，不是道路误差容忍阈值。实际车身道路、路线及动力学指标留阶段3。


## 9. 阶段3解码与评估输出

解码API不接收标签future_mask，完整生成输出仍为40点；显式初始heading为当前观测[N,2] sin/cos，history末点有效时可读取，否则报错。states为绝对局部xy、后向速度、单位heading；输出精度float64默认。详见decoding_evaluation.md。

labels验收的stages.npz保存原始future delta及float64绝对状态、标签mask、初始位置、agent mask；完整标签窗口还保存解码绝对状态、初始heading、position/velocity/heading correction、heading fallback mask、raw velocity residual。未实现引导字段不填零假轨迹，metrics明确guided_stage=not_applicable。

独立指标schema为sumodiff.scene.metrics.v1，含completed/failed、窗口ID、适用性、未选车数、collision/road/motion及quality/effective event。无适用或无法判定使用JSON null，未知与失败数量另报；quality是资格布尔，非综合场景分。几何数值修复逐项包含reason、方法、类型、组成与变化，不改map原文件。

trajectory_index.json为逐窗NPZ/指标文件SHA-256及split/status；validation_summary.json与metrics.json保存完整汇总、运行SHA、资源和限制，标准manifest记录所有实际输入的哈希。audit_summary.json保存独立核对结果及来源运行SHA和审计SHA。


## 10. 阶段4模型条件与验证输出

prepare_conditioning仅消费collated inference条件及当前metadata/map，新增map_extent_m[B,4]、lane_rule_relations bool[B,L,L,2]（定向yield/foe）、route_lane_features float[B,N,L,2]（t0相对计划次序/剩余标志）。原窗口文件不变，future_mask及labels不得进入model条件字典。阶段2栅格是0/1，不是0/255。

ConditionalDenoiser输入noisy_future[B,N,40,6]、整数timestep[B]（候选0..999）、上述条件；输出pred_noise[B,N,40,6]，可附condition[B,N,128]/gates[B,N,3]。基础模型无攻击角色embedding，两种融合权重布局相同。未来状态归一化、扩散日程和checkpoint留阶段5；初始方向另遵守阶段3合同。

模型短验收schema为sumodiff.stage4.validation.v1，标准清单包含初始化seed/参数哈希、实际配置及全部当前输入身份；checkpoint=null、training_started=false。untrained_outputs.npz只保存随机latent和两种模式的epsilon/condition/gates，不能作为物理生成轨迹。resource_cases.json记录逐负载的实际batch/选车数、时间/torch显存。input_audit.json记录train/validation窗口、t0任务选择，future_labels_accessed=false。audit_summary.json记录CPU重放误差及来源运行SHA，保留运行时标识。


## 基础扩散checkpoint与阶段输出（阶段5）

sumodiff.base.checkpoint.v1保存epsilon网络/AdamW、CPU及CUDA随机状态、完整配置、固定50m/20m/s尺度、数据/窗口ID/审计hash和运行SHA。旧checkpoint明确拒绝。未来标签mask只进入训练损失；推理阶段只按agent_mask生成全部40点。

采样stages.npz独立保存initial_noise、raw_normalized_future、raw_future_delta（前2维相对各车t0）、raw_absolute_states、decoded_absolute_states（前2维共同局部绝对位置）、observed initial heading、agent_mask及解码修正/残差/回退。没有future_mask或真实退出时刻作为推理输入，没有伪造guided轨迹；guided阶段在JSON中N/A。原始输出未clip或平滑。详见base_diffusion.md。


## 阶段5A数据资格与epoch合同

原window.v1与dataset.v1维持不变。候选目录保留全部窗口；window_quality.jsonl逐窗分开记录core_complete、eligible、reasons、raw64/stored32运动/道路/碰撞、坐标精度误差、实际t0边界及源jerk归因。quality_report.v1记录配额、保留比例和几何/回合覆盖。最终dataset附selection.json（sumodiff.stage5a.selection.v1）；manifest.curation保存selection哈希和候选manifest身份。标签资格不进入模型条件，future_mask只作标签。

训练配置sumodiff.base.epoch.config.v1验证最终数量和资格。epoch无放回遍历，保留尾批；checkpoint.v1保存完整epoch配置、严格数据/尺度签名及原随机/优化器状态，恢复由总step重建批次，旧step模型不能跨数据签名续训。progress.json、latest_checkpoint.json、generation_monitor.json持久化真实步数/保存点和定期自由生成结果。convergence_status=not_assessed。


阶段5A路网配置新增network几何细分、输出精度和正常转弯横向加速度限值，进入完整resolved_config及源scene身份。route_corridors保留合法相接端点的round join并裁剪drivable，正面积GeometryCollection保留全部线/点组成；没有允许整个junction替代路线。window_quality和selection进一步区分无车辆对N/A与真实碰撞/未解析，并分别报告eligible与selected实际方向变化窗口数量。


## 阶段5B受控继续学习

checkpoint.v1保留现有状态布局，training_config新增sumodiff.controlled.train.config.v1，记录arm、共同父SHA-256、source_step/source_epochs、继承初始化、名义/实际追加预算与physical_loss完整配置。A additional_losses_enabled=false，B—D为true；true要求可验证的受控训练目标元数据，旧训练入口拒绝隐式启用。严格恢复签名包含目标、数据及预算；不能把不同组当作同实验resume。原窗口/模型条件/采样stages结构不变，未来标签仍仅作监督。

steps.jsonl包含全局step、真实epoch/batch、窗口ID、timestep、noise_sha256、各目标/梯度norm与时间。initial_state.json包含model/AdamW/Generator/CPU-CUDA RNG哈希。完整实验最终step33000、每组3000更新；preflight_prefix另记录真实12更新，不计为10轮。arm_outputs.json允许恢复训练或独立重采样的显式路径；报告追溯原始运行SHA，失败/未知始终在计划200任务质量分母内。


## 阶段5C统一训练与调度状态

原dataset/window/checkpoint.v1外层布局和采样stages.npz保持。新增training_config.schema_version=sumodiff.unified.train.config.v1；signature.training包含完整实际配置，不沿用旧训练签名的可变预算例外。loss为sumodiff.unified.loss.config.v1，schedule为sumodiff.warmup.cosine.config.v1。源模型30000步/100轮，新增15000步/50轮，短prefix单独记录execution_budget_kind及实际预算。

checkpoint.runtime保存scheduler_state（配置hash、completed_updates、last_lr、last_auxiliary_ramp）、best_epsilon、quality_history和snapshot_kind。scheduler_state的已完成更新数必须等于checkpoint.step-source_step，实际AdamW lr须一致。best_epsilon.checkpoint=null只表示本checkpoint本身为该保存点；恢复时解析为当前文件身份，其余身份检查hash。training_recovery可恢复，monitor_assessment仅用于采样。

initial_state.json记录网络/优化器/RNG哈希、实际AdamW state数量和调度起点。steps.jsonl增加lr、rho、模仿/时间/道路分项、加权贡献、道路代理越界比例、朝向回退计数及clip前范数。best_epsilon.json指向实际最佳epsilon参考；quality_candidates.json为各监测点与非支配checkpoint列表，无综合评分。full_budget_complete=true仅在实际满45000步且非preflight时成立，不能据此推断收敛或质量。

gradient_audit.json保存冻结父参数、完整候选rho=1的逐批参数梯度范数/与epsilon夹角、训练窗口与时间步。road_activation_probe.json是明确标注的标签衍生位移数学探针，optimizer_updates=0，不是生成事件。monitor_probe.json使用当前条件和地图，future_labels_accessed=false、training_updates=0；检查Torch RNG恢复与源文件不变。正式200配对报告继承既有独立指标和失败分母协议。


## 阶段5D原始速度续训与报告合同

外层checkpoint/window/dataset/stages布局保持。training_config.schema_version新增sumodiff.raw.velocity.train.config.v1，loss为sumodiff.raw.velocity.loss.config.v1，内含原sumodiff.unified.loss.config.v1；签名锁定所有新损失/原损失/日程/预算和数据身份。source_step=45000/source_epochs=150，完整终点60000/200轮；preflight_prefix与正式预算分开。

checkpoint仍保存完整AdamW/RNG及调度状态。源参数矩step为15000，新增模型矩step必须为15000+checkpoint.step−45000；日程completed_updates只计新阶段，不继承旧15000日程或重置矩。源状态复制避免计数别名；仅training_recovery可恢复，monitor_assessment不能作训练恢复。尺度审计显式保存原additional_losses_enabled=false，其余尺度/扩散参数、数据与原审计hash必须相同；实际新训练/采样能力标记仍true。

steps.jsonl新增inherited_auxiliary_total、raw_second_boundary/interior、weighted_raw_boundary/interior、raw_auxiliary_total、inherited_auxiliary_ramp=1、raw_auxiliary_ramp。通用调度last_auxiliary_ramp在本协议只指新项预热；继承项从首步全强度。initial_state记录190个真实AdamW矩哈希，正式日志真实epoch151—200逐轮1200窗口无放回。

joint_gradient_audit.json记录自然t/固定t200及同噪声、实际组合向量、范数、余弦、角度和理论clip。peak_stress.json明确隔离副本各24更新、峰值/full-strength、无正式checkpoint；不能算正式50轮完成。恢复检查含所有状态和两种ramp逐位比对。

paired_summary.json/scene_rows.json报告计划200任务及可用配对数、交界/内部两时段、舒适性计划/已评估/未评估车辆数和两种率、道路/路线/覆盖、碰撞、位置/速度修正及耗时。无适用车辆对单列not_applicable而不是unknown或零事件概率。baseline_report_audit.json仅重放现有150轮输出，自配对区间为0；optimizer_updates=0/new_sampling_tasks=0，不宣称200轮模型存在。


## 阶段5D交界与密集诊断合同

sumodiff.boundary.density.config.v1锁定两个checkpoint、训练119/验证200、五个低噪声t与同一窗口Gaussian。sumodiff.boundary.density.metrics.v1保存cohorts、模型不变性、同噪声输入配对、标签控制、全部20组重构、固定历史差分恒等式、原始/解码误差和prefix_oracle。重构行包含窗口/几何/回合、有效车辆数、各运动违规及物理量摘要；label_conditioned_reconstruction=true/free_generation=false，不作为独立自由生成指标。

训练密集选择及普通控制仅按当前元数据，不按未来误差或验证失败挑选；train/validation回合和几何隔离。prefix_rows保留每模式全部200任务，真值替换只能作oracle。后处理敏感性相对于同一解码器恢复的标签作线性向量分解，车辆混合RMS与每场景RMS区分，分量可相消。

sumodiff.boundary.density.summary.v1只提交必要数值和来源，保存原运行SHA、源文件/脚本身份、配置环境哈希及验证范围。重构逐场景摘要不能还原逐帧分位数；轻量摘要明确标记混合P99来自正式源输出，计数、均值、极值和每场景均值分位数独立重算。原训练/推理数据接口保持。


## 阶段5E解码时间监督续训合同

checkpoint外层保持，新增training_config.schema_version=sumodiff.decoded.temporal.train.config.v1；loss为sumodiff.decoded.temporal.loss.config.v1，完整保留inherited的5D目标。source_step=60000/source_epochs=200，225轮终点67500/7500新更新；inherited_optimizer_updates=30000，总矩步37500。signature锁定新loss、真实epoch、数据、尺度、监测及日程，不沿用5D签名或重复150→200入口。

通用scheduler auxiliary_ramp在本协议只表示temporal_reweight增量rho；inherited_auxiliary_ramp/raw_auxiliary_ramp始终1。steps.jsonl增加分组一二阶差分、decoded_temporal_boundary/interior、weighted_decoded_boundary/interior、effective_decoded_*_weight、inherited_weighted_temporal及temporal_reweight_delta。旧weighted_temporal被替换，不在total中重复叠加。

sumodiff.decoded.temporal.gradient.audit.v1记录训练元数据选择、自然/fixed t=0/50/200同噪声输入、实际完整向量及clip、冻结候选敏感性和N/A；不访问验证失败来选训练组。父训练timestep_coverage只统计自然低t实际访问，不称梯度占比。formal短preflight保存私有压力更新、真实前缀恢复、参数/RNG/AdamW/调度和完整loss逐位核对，full_budget_complete=false；正式报告parent200/candidate225，原报告parent150/candidate200合同保留。


## 阶段5E正式结果摘要

sumodiff.stage5e.result.summary.v1为保存输出的CPU审计摘要：source_sha保持真实858ad6b，checkpoint和source_parent_checkpoint为文件身份；verification包括完整epoch/AdamW累计步/同噪声/独立SVD重放。counts、strict_comfort、pooled和paired_intervals沿用报告口径，混合帧分位数与场景配对均值区间分开。transitions/道路子集/dense_intersections/temporal_failures保留全部任务，新增失败时段不互相替代。label_control仅核对正常标签质量，不进入生成条件或修复。

analysis/figure运行均formal=false，仅CPU后处理、输入与脚本身份可追溯。正式训练和轨迹接口未改；结果SHA不会被摘要提交SHA替换。225轮结果不自动替代200轮候选，不触发tag或阶段6。


## 显式诊断任务前缀

generation_rows默认保持完整200任务要求；planned_window_ids显式指定诊断计划时，目录必须为相同有序短计划或完整200目录的相同前缀。重复、缺失、错序拒绝，不通过删除失败样本重建分母。sumodiff.checkpoint.screen.metrics.v1记录planned_tasks/new_sampling_tasks/source_sampling_tasks/reused_sampling_tasks及optimizer_updates，将复用报告与实际新采样成本分开。对应summary.v1保存来源身份、条件配对区间、子集资格、分时段超限及成本；混合分位数不能当作整场景资格。


## 冻结时间目标诊断

sumodiff.temporal.diagnosis.config.v1固定200/225来源，可选至多四个t<=300；sumodiff.temporal.diagnosis.run.v1保存119训练控制、200验证的选定ID、共用225目标与full-ramp、标签/地图/模型身份，明确future_labels_used=true、optimizer_updates=0、new_free_generation=0。各逐场景文件记录实际目标分项、motion及分时段摘要，metrics.v1另保存分组混合和逐时刻分位数。组件含alpha_bar、不含lambda；按父/新权重组合的total是固定目标探针，不是历史训练日志。

sumodiff.temporal.objective.result.summary.v1为Git轻量结果，保留真实运行SHA、原诊断/CPU脚本/完整摘要/图身份及验证范围。jerk误差来自解码速度监督差分，独立物理jerk来自固定历史与解码位置；前4/中间32/末4互斥，内部36包含末4。CPU自由输出重放使用输入任务ID，不用未来完整性选择推理数据；没有新增标签修复或推理条件。

## 固定从头训练协议（v2）

sumodiff.stage5a.config.v2新增test_geometries/test_seeds_per_geometry/test_quotas，三划分几何索引使用3*index+split_index，种子相隔100000；v1仍支持历史复现。新增代表性路口Pilot几何/30秒需求和minimum_turning_fraction；均进入Pilot协议哈希，旧0转弯报告无法授权新批量计划。复用原始Pilot逐job验证真实配置、种子、划分与完成状态，只重采未匹配部分。

sumodiff.stage5a.selection.v2记录三个划分配额、完整标签与质量资格、实际转弯标记、覆盖下限及turning_coverage；训练校验选定转弯计数与证书一致。actual_turning_agents是离线标签审计字段，不进入conditioning。测试标签不参与训练缓存、尺度统计或lambda选择；准备测试标签时仅执行已经固定的质量规则，不调用覆盖阈值校准。窗口掩码、固定t0坐标、未来位移及地图格式不变。

sumodiff.base.epoch.config.v2记录随机初始化、200轮/75000更新的完整不可变配方，raw_loss、jerk_tail、tail_weight、calibration_report与序列化schedule。核心MSE与所有辅助项按场景归一化；全辅助共用低t门控、alpha_bar和预热。尾部输入是统一解码绝对位置+固定最后3个历史点；K=ceil(q*M)，含有效零项，掩码先于差分；dt=0.1。旧checkpoint外层schema保留，新训练schema单独检查AdamW步数、完整日程及恢复状态，拒绝旧模型续接。

sumodiff.fresh.calibration.v1保留普通/密集训练窗口ID、自然/固定低t、初始及32更新状态、分项和实际组合梯度、范数比/夹角/裁剪、候选筛选、4个全强度B16压力更新与资源。development_only禁止生成正式训练配置；生产报告、实际配置、数据/model/recipe哈希锁定。校准checkpoint不继承给Train。

sumodiff.fresh.test.seal.v1绑定选定checkpoint、数据哈希、采样配置及已完成完整验证来源。测试先验证冻结证书，再读取任务输入；模型、数据、随机种子或采样设置漂移会拒绝。现有manifest/config/command/environment/status格式保持，完整训练日志新增实际LR、统一辅助ramp、loss分项与裁剪。短prefix不缩短总日程，并明确complete_budget=false。


## 控制回放诊断合同 v1（2026-10-13）

控制模块读取absolute local六维标签（future前两维先加回各车initial_positions），不修改现有targets.npz。ControlConfig明确dt=0.1、center/front参考点、midpoint/endpoint积分航向、是否包含lateral_velocity、float64计算。InitialControlState只读最后两个有效历史姿态和t0已知length，不接收future_mask；rollout_controls也不接收未来标签掩码。inverse_label_controls是离线标签变换，要求完整active未来，输出a/omega及显式可选r、舍弃横向残差、原速度/位置差分误差。未支持的离散间隔及角度别名报错，不裁剪标签。

诊断任务schema为sumodiff.control_replay.tasks.v1；结果为sumodiff.control_replay.results.v1。逐窗NPZ保存label_states、agent/history/future masks、initial_positions、lengths、各方法states/controls/lateral_velocity/jerk；JSON独立保存质量、逐车舒适性、恢复误差、门槛和失败。future_mask仅用于标签审计，不能进入生产采样条件。新控制checkpoint、训练归一化和噪声目标尚未定义；不得把现有六维checkpoint用于两/三维控制推理。


## 控制回放继续检查合同（2026-10-13）

回放results.v1兼容新增planned_active_agents、unknown_planned_agent_tasks、strict_comfort_confirmed_pass_agents、strict_comfort_assessed_agents、strict_comfort_confirmed_pass_fraction_all_planned。完整严格舒适性不使用缺失时域推断通过；失败车辆保留计划分母，无法知道的数量N/A。

sumodiff.control_replay.audit.v2增加independent_control_inversions、independent_error_and_gate_checks、verified_group_summaries、maximum_control_disagreement及实际复算的gate_pass_all_tasks。原始身份和字节哈希不放宽；旧label门槛计数0按未执行明确归为N/A。

sumodiff.historical_git_inputs.v1保存原manifest身份、原Git状态、original/current/snapshot文件身份。snapshot必须同时匹配original的SHA-256和字节大小，并属于同一source manifest。只恢复Git可追溯的小型文本输入，不能掩盖数据或模型漂移；通过显式CLI参数使用，不静默回退。
