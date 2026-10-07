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
