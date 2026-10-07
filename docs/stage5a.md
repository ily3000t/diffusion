# 阶段 5A：扩大有效数据，再进行基础模型训练

本阶段先核查新正常 SUMO 标签，再由本地终端采集与训练。目标是 1200 个训练窗口（各道路 400）和 200 个独立几何/回合验证窗口（直路 67、汇入 67、无信号路口 66），batch=4，100 个真实 epoch。这个预算不是收敛标准。本轮只运行少量新回合和有界入口预演，不启动全量采集、100 epoch 或阶段 6。

## 标签与诊断

“核心窗口”仍只表示选定车辆的 21 点历史和 40 点未来完整。新增质量资格单独记录：完整 H+F 的原始双精度重建及存储 float32 标签，都必须通过已声明的硬运动阈值、车身道路/已知路线/地图覆盖、插值碰撞检查；至少两辆车。检查增加最早历史点之前的导数上下文，无此上下文明确拒绝。comfort 指标保留，但不等同于硬质量资格。

独立核对从 SUMO 原始前保险杠位置、导航角及车长直接重建几何中心、后向速度和固定局部状态，不复用生产坐标转换代码；同时比较世界和局部车身道路/路线结果。记录车道归属、实际转弯覆盖、原始前保险杠/几何中心位置差分 jerk 与 SUMO 报告的标量加速度变化。原始标签不平滑、不裁剪、不做事后运动修复；窗口拒绝和原因保存。

旧 train/validation 原始标签的只读排查发现：分段直路 core 入口/出口有声明 0.10m、形状长度 0 的内部车道，车辆跨连接时位置增量比 speed×dt 少约 0.10m，引起约 208m/s³ 的位置差分 jerk。原始前保险杠序列也存在尖峰，不能归因于局部坐标旋转。匝道与路口转弯存在内部车道几何/声明长度差异和车身超出狭窄路线车道并集的问题，需要逐窗保留。

本阶段新直路显式使用一条连续三车道 edge，避免人为分段内部连接；旧场景配置默认保持 segmented_straight=true。正常跟驰改为显式 IDM、sigma=0、保留安全规则；变化几何、驾驶参数和种子。这个选择不保证 jerk 合格。汇入/路口仍保留合法 SUMO 连接，质量检查不通过的转弯窗口被拒绝，报告中独立记录保留的实际转弯数量。不能将全部 junction 加入路线走廊来掩盖车身问题，也不能宣称数据完全覆盖转弯。

原始位置定义见 [SUMO TraCI 文档](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html)；道路连接语义见 [SUMO Intersections](https://sumo.dlr.de/docs/Simulation/Intersections.html)。位置差分向量 jerk 与标量加速度 jerk 并非同一指标。

## 采集、划分和资格选择

配置为 configs/experiments/stage5a.yaml。Pilot 是 6 个新短回合：三类道路各一个 train 几何、一个 validation 几何，需求时段 18s，随后排空，最大回合 600s。Pilot 保存候选窗口，选出 12/6 个质量合格窗口供入口 smoke；每类至少 3 个合格候选、合格比例至少 10%，坐标或检查失败会阻止后续 bulk。

批量计划预先固定 78 个回合：每类 train 5 个几何×4 个种子，validation 2 个不同几何×3 个种子。需求时段 120s，然后排空。先指定回合与几何划分，后切窗；实际编译后的 geometry_id 也检查隔离。没有访问测试集。流量、路线、限速、几何、驾驶参数及随机种子进入实际场景配置。

Prepare 按回合流式处理，避免同时加载全部原始回合。所有完整与不完整窗口存入 candidates/windows；逐窗质量写 window_quality.jsonl。按 geometry/episode 轮转、窗口 hash 顺序选择合格样本，每类每划分至少 2 组几何、4 个回合，单回合上限为该类配额的 25%。不足 1200/200 或多样性不足即失败，不重复、补零或无提示减量。

最终 dataset 为独立目录，窗口用本卷硬链接或显式记录的复制方式保留；运行不依赖旧项目。manifest 和 selection.json 记录资格、配额、哈希和来源。验证几何及回合与训练隔离，回合内部相邻窗口仍重叠，不能把 200 窗当作 200 个统计独立回合。

质量过滤后的数据/验证分布更简单：这是合格正常标签子集。保留候选池拒绝比例、转弯数量及回合/几何覆盖，不能把本子集验证直接推广为任意交通、真实驾驶行为或未过滤的无标签测试质量。

## 尺度与真实 epoch

Prepare 后必须对新的 dataset 全部 train/validation 重新运行 audit-scales。50m/20m/s 是固定物理单位，允许归一化数值大于 1，不是裁剪范围。检查新的位移/速度 P99、max、比例和历史速度，若决定改变尺度，先改受控配置并重新审计、从新模型开始，不能继承旧尺度 checkpoint。

configs/train/stage5a_epochs.yaml 使用 epoch schema，严格验证质量选择、配额、划分及实际数量。每 epoch 对全部 1200 窗以确定种子洗牌，不放回，保留最后不足 batch 的批次。1200/4=300 更新/epoch，100 epoch=30000 更新。旧 step 训练接口仍兼容，但不会把随机抽 30000 步标称 100 epoch。

每 epoch 验证全部 200 窗及固定 60 窗训练噪声探针；每 5 epoch 保存 checkpoint；每 10 epoch 和结束时，使用固定验证任务、初始噪声及 20 步 eta=0 DDIM 检查自由生成。验证任务只从已知当前条件选择，推理不读取 future_mask。报告 raw 位移、解码修正、质量资格和完整独立指标；监测器保存并恢复 Python/NumPy/CPU/CUDA RNG，避免采样改变后续训练顺序。

检查 progress.json、validation.json、generation_monitor.json、training_summary.json 和 latest_checkpoint.json。convergence_status 默认为 not_assessed；当前没有自动宣称收敛或早停。判断模型稳定应一起看验证曲线、固定任务原始轨迹、解码修正和运动/道路指标。100 epoch 用完仍不合格，先分析结果，不自动增加预算。

## 本地终端入口

在 E:\diffusion_new 的 PowerShell 终端执行：

~~~powershell
Set-Location E:\diffusion_new
.\scripts\stage5a.ps1 -Action Pilot -Name stage5a-v1
.\scripts\stage5a.ps1 -Action Collect -Name stage5a-v1
.\scripts\stage5a.ps1 -Action Prepare -Name stage5a-v1
.\scripts\stage5a.ps1 -Action Train -Name stage5a-v1
~~~

逐条执行，前一步通过后才执行下一步。Collect 是长采集，Prepare 包含质量检查和新的尺度审计，Train 才启动 100 epoch。脚本使用已有项目 .venv-model，设置本进程环境，保留完整生效配置，不修改全局环境。正式入口拒绝脏 Git 工作区，改配置后先提交。

可以复用本轮验收 Pilot：Collect 加 -PilotReport 指向已完成 Pilot 的 preparation/quality_report.json，配置 protocol_id 必须相符。不要拿旧数据或直接 prepare --profile pilot 代替新回合预检。

受本地脚本执行策略阻挡时，对当前进程使用：
~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stage5a.ps1 -Action Pilot -Name stage5a-v1
~~~
这里的 Bypass 仅作用于该进程，不改变机器执行策略。

采集被中断时，原尝试全部保留；在新 controller 目录继续：
~~~powershell
.\scripts\stage5a.ps1 -Action Collect -Name stage5a-v1 -RunSuffix retry1 -ReuseCompleted
~~~

训练被中断，读取 latest_checkpoint.json 的真实路径，在新目录恢复：
~~~powershell
.\scripts\stage5a.ps1 -Action Resume -Name stage5a-v1 -RunSuffix resume1 -Checkpoint '真实 checkpoint 完整路径'
~~~

恢复必须使用同一数据、尺度、模型配置、训练种子/优化器/精度；每步的 epoch/permutation 由完整步数确定，旧 12/6 数据 checkpoint 不允许续到新 1200/200 数据。没有覆盖旧目录、强制跳过质量检查或自动重试认证。

若 Prepare 的数据成功但随后尺度审计失败，保持已完成 dataset，用单独 Audit 入口和新后缀重做审计；训练通过 -ScaleAudit 指定该新 scale_audit.json。若配额不足，先看 quality_report，不启动训练。按拒绝来源调整新采集计划并使用新 Name；不混用原 campaign 的不同配置。

## 成本边界和交付状态

旧 12/6 窗口测得 B4 单步 173–181ms，只能给相同缓存/车数/地图负载下约 86–91 分钟 optimizer-only 的粗略 30000 步估算。新数据可能多达 12 车且地图更多车道，采集/质量检查/缓存、每 epoch 全验证和 checkpoint/自由生成另计，不能承诺总时间。短预演的实际结果见 progress.md 和 stage5a_validation.json；正式采集后 training_summary 给新负载吞吐量及显存。

1200/200 是计划资格数量，78 个回合只是首批候选预算，不保证保留足够窗口。此次不会自动采集第二批或启动长训练。没有阶段 6 引导、100 epoch 收敛证据或实验里程碑 tag。
