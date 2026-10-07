# 实施进度

## 阶段 0：已验收

所有实现位于 `E:\diffusion_new`。旧目录仅只读检查，没有复制工程、权重或数据，没有修改参考目录。

### 交付
- 设计、数据/坐标合同、阶段验收、Git规则、实验协议、环境和来源记录落盘。
- 独立src包、候选配置分类、保留模块目录及README、pyproject、artifacts忽略规则。
- 清单工具：Git SHA/分支/准确工作区状态，完整实际配置，种子，完整argv/PowerShell命令/工作目录，输入与checkpoint哈希，真实环境及其哈希，指标与运行状态。
- 正式模式拒绝脏工作区、无提交仓库、未忽略的仓库内输出；拒绝覆盖已有运行、无效种子、未实现配置组合和科学流程。异常保留failed状态并抛出，初始metrics为空。

### 实际验证
验证解释器：`E:\Programs\EnvAnaconda3\envs\pytorch\python.exe`。

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe' -m pytest -q --basetemp artifacts/cache/pytest-stage0-final
& 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe' -m pip --disable-pip-version-check wheel --no-deps --no-build-isolation --no-index --wheel-dir artifacts/cache/wheels-final .
& 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe' -m sumodiff.experiments record --config configs/experiments/manifest_smoke.yaml --output artifacts/runs/stage0-accepted-20261004 --seed 20261004 --formal
```

- 最终19项测试通过，10.00s；配置YAML/TOML与UTF-8检查通过，git diff --check通过。
- 本地wheel构建成功，直接从wheel独立导入通过，没有安装/升级全局环境，没有下载依赖。
- 真实环境清单六文件完整，状态completed，配置与环境SHA-256核对通过；运行代码已提交、工作区干净。
- 已有运行目录不能重用；重跑请选新目录。
- 验收清单是metadata smoke，不是模型实验。metrics为空，没有科学结果或实验里程碑tag。

### 资源实测与待测
RTX 4070 Laptop GPU：8585216000 bytes / 8188 MiB；驱动610.47；PyTorch2.5.1/CUDA build12.4，CUDA可用。最终元数据清单记录耗时 `3.124` 秒，包括环境查询，不是生成或训练耗时。SUMO二进制1.22.0，现有TraCI/sumolib1.25.0。

没有模型单批峰值显存、batch size、吞吐量或正式训练时长测量，阶段4/5再测。总显存不能代替模型峰值。

### Git 与追溯
远端成功只读查询无引用，才创建新历史。main初始无功能基线 `e297483`，开发分支 `chore/project-foundation`。

| 提交 | 职责 |
|---|---|
| e5a3107 | 完整设计与实验合同 |
| 85451c2 | 独立包布局与依赖分组 |
| 45f25a5 | 环境与清单实现 |
| 6bdfd3f | 追溯、失败和保护测试 |
| 84ebd81 | 工作目录、准确状态和环境哈希 |

最终清单运行SHA为 `84ebd8192289a696de5419b3e52bcaefa5849106`，运行分支为chore/project-foundation，输出位置为artifacts/runs/stage0-accepted-20261004。本进度与summary随后提交，不回填该SHA。可提交的小型验收summary在 `stage0_validation.json`，原始环境和结果不进Git。

验收分支已通过 `--no-ff` 合并main，验收合并提交 `4aaf5b0`。随后在独立文档分支 `docs/foundation-delivery-status` 记录远端交付状态，再合并main；最终HEAD以Git记录和交付报告为准。

远端推送未能成功确认：普通原子推送和一次HTTP/1.1重试均返回 `Recv failure: Connection was reset`。这是实际网络错误，没有返回认证或权限错误。已停止推送，全部本地提交和工作保留。网络恢复后可执行：

```powershell
git -C E:\diffusion_new push --atomic -u origin main chore/project-foundation docs/foundation-delivery-status
```

运行SHA仍为84ebd81；文档提交和交付状态不会改写原实验清单。

### 阶段0交付时的下一步（历史记录）
- 阶段1—8均未实现；没有采集数据、模型、采样、评估或RL控制器。
- 阶段1先建立隔离的SUMO客户端环境，将客户端与二进制统一版本并短回合验证；不升级旧训练环境。
- 下一阶段分支建议 `feat/sumo-collection`：参数化三类场景、正常安全规则/连续换道、0.1s完整回合与生命周期记录，只运行少量短回合，验证换道、转弯、停车及进入退出。
- 不默认开展批量采集、长训练或阶段8研究，不创建版本tag。


## 阶段1：已验收

### 交付与边界
- 独立功能分支feat/sumo-collection；沿用已有main历史。开始时远端与本地main均为14d4352，未创建独立历史或改写共享提交。
- 参数化三车道直路、匝道汇入、无信号路口；每类两种正常几何配置，变化流量、路线、驾驶参数和种子。
- 固定0.1s原始采集，完整回合可变长，直到需求结束且车辆排空。记录原始前保险杠位置/导航角/速度/加速度、尺寸类型、ID、道路车道、当时已知路线、安全模式与实际出发时刻。
- 连续换道使用lanechange.duration，保留默认速度模式31及换道模式1621；未关闭安全规则。采用SUMO1.22.0随包客户端，bundle revision及协议21已核对，路径和源码哈希进清单；项目环境不修改旧训练环境。
- loaded包含启动时已加载车辆，记录进入、到达、由证据解释的退出、teleport始末、停止/停车始末、紧急停车和SUMO碰撞事件。未知消失不冒充到达。
- 回合清单/完整配置/命令/环境/原始与质量摘要均保留；限时未排空为failed，进程清理及截断原因可追溯，未丢弃失败样本。
- 六个主配置均无外部控制，全部正常质量通过、可供后续正常数据处理；换道停车、teleport、截断诊断单独配置且不进入正常训练。
- 不进行正式批量采集，未进入阶段2坐标/窗口、模型或训练，不创建实验里程碑tag。

### 最终验证

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv-sumo\Scripts\python.exe' -m pytest -q --basetemp artifacts/cache/pytest-stage1-final
& '.\.venv-sumo\Scripts\python.exe' scripts/validate_collection.py --output artifacts/runs/stage1-accepted-20261004 --formal
& '.\.venv-sumo\Scripts\python.exe' scripts/audit_collection.py --run artifacts/runs/stage1-accepted-20261004 --output artifacts/cache/stage1-final-audit.json
```

重复执行需新的输出目录/审计文件名。39项测试通过（8.18s），24项真实仿真验收检查通过。最终九个回合的输入输出哈希、逐帧ID/时间/路线/安全模式、tripinfo到达与生命周期数量由独立脚本核对通过。阶段1wheel构建和独立导入通过，git diff --check通过。

| 正常配置 | 车辆数 | 车辆记录数 | 仿真时长(s) | 采集阶段实耗(s) |
|---|---:|---:|---:|---:|
| straight_a | 3 | 1109 | 45.3 | 1.024 |
| straight_b | 4 | 1735 | 51.2 | 1.070 |
| ramp_a | 4 | 1302 | 40.3 | 0.978 |
| ramp_b | 4 | 1376 | 39.5 | 1.003 |
| intersection_a | 4 | 1413 | 37.9 | 0.963 |
| intersection_b | 4 | 1448 | 42.3 | 1.001 |

六个正常回合共8383条车辆记录，全部正常到达且无SUMO碰撞或teleport事件。每类正常几何ID均不同；重编译同一几何到不同路径/种子时ID不变，为阶段2按几何分组提供稳定标识。

诊断straight_probe实际连续横向偏移29帧、低速停止20帧，stop始末均记录；teleport_fixture实际6次开始/结束，原始位置跳变保留，质量不合格且明确排除训练；truncation_fixture在13s未排空，270条记录及failed状态/原因均保留。九个回合总15642条记录，原始及追溯输出约9.46 MiB，未提交Git。

### 资源与结论限制
已完成回合的Python逐tick采样RSS峰值约54.00–54.84 MiB，SUMO约21.31–21.47 MiB，未使用GPU。这是采样RSS估计，非OS精确峰值。表中耗时是采集阶段（启动/连接/采样/保存），不含全部配置检查、环境记录及netconvert，不能称为端到端训练或生成耗时。

这些是小规模工程验收数据，不证明真实驾驶真实性、独立几何碰撞指标或动力学执行。只有每类两个正常几何、共六个正常回合，不能据此宣称完整正式训练数据或训练/验证/测试三份均具全类别覆盖。

### Git与原始运行SHA
最终正式验收运行SHA：`36eadaed424bc46a1b6019beb9353650f21806e4`；运行分支feat/sumo-collection；原始位置artifacts/runs/stage1-accepted-20261004。summary保存在stage1_validation.json，不回填成summary的新提交。

| 提交 | 职责 |
|---|---|
| d2d875e | 项目内客户端隔离记录与忽略规则 |
| 0d26159 | 客户端/二进制来源与版本核对 |
| 6c52287 | 三类场景及合法需求 |
| 5d6e235 | 完整回合采集和失败保留 |
| b80b1a2 | 拓扑、生命周期及短回合验收 |
| 5b589a2 | 实际协议、来源及环境文档 |
| c765028 | 将正常配置与诊断控制拆开 |
| 36eadae | 可复现的独立原始文件审计 |

开发分支经上述验收后以--no-ff合并main。最终交付SHA和远端同步结果以Git记录及交付报告为准。

### 阶段2前置条件
- 在feat/window-dataset分支实现中心/数学角度、固定参考系与后向差分，额外历史前点不可省略。
- 只读取明确的回合清单，诊断、失败、过滤理由和数量均统计；不得误把历史开发验证目录混入正常数据。
- 先按episode及geometry隔离划分，再切窗；当前六正常回合仅适于小验证，类别/几何不足须显式报告。如需三划分全类别覆盖，先补充少量独立几何验证数据，不默认正式大采集。
- 保留ID/槽位、掩码、逐车计划路线及完整net的连接/优先规则；未来退出时刻不能进条件。
- 阶段2不启动模型训练；CPU采集成本不能用于推断8GB显存上的模型batch size或训练时长。

## 阶段2：已完成验收（2026-10-04）

### 交付与范围

开发分支feat/window-dataset。实现中心/数学角转换、t0固定坐标正逆变换、后向速度、额外历史前点、21/40点窗口、t0选车及12车稳定槽位。实现显式回合清单/内容ID、输入完整性校验、按几何与回合先隔离再切窗、三类mask；不完整窗口保留。

地图包含三通道栅格、64点车道折线、合法内部连接链、优先/冲突规则及逐车路线走廊，向量几何独立于栅格。补充零几何长度内部车道的明确方向处理；独立SUMO检查发现并修复内部等待路口覆盖普通请求索引的问题。未来标签单独存储，inference不读targets/labels，训练完整窗过滤不可用于推理筛选。

新增可追溯预处理入口、NumPy reader/collate、逐窗独立审计，使用文档见window_dataset.md；实际schema见data_schema。安装仅在项目虚拟环境增加Shapely，未修改全局或旧目录。

为三划分各有三类别，只补采三个C几何短回合：straight_c seed41（1224记录）、ramp_c seed42（960记录）、intersection_c seed43（1402记录）。原始位置artifacts/raw/stage2-supplement-20261004，运行SHA为0f8ce09126afa67d148cd8a5e5301f6f9c745a2a。其余输入仅来自明确的阶段1验收清单，未扫描或混入历史开发采集。

### 最终验收命令

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
$env:SUMO_HOME = 'E:\Program Files\sumo-1.22.0'
& '.\.venv-sumo\Scripts\python.exe' -m pytest -q --basetemp artifacts/cache/pytest-stage2-final
& '.\.venv-sumo\Scripts\python.exe' -m sumodiff.data process --config configs/data/window.yaml --sources configs/data/sources_stage2.json --output artifacts/processed/stage2-accepted-20261004 --formal
& '.\.venv-sumo\Scripts\python.exe' scripts/audit_window_dataset.py --dataset artifacts/processed/stage2-accepted-20261004 --sources configs/data/sources_stage2.json --output artifacts/runs/stage2-accepted-audit-20261004 --formal
```

输出不可覆盖，重跑需新目录。55项测试通过（8.77s）；wheel构建及无src路径的独立wheel导入/数据推理通过。git diff --check通过。此前preview/map-check是开发验收过程记录；后续只使用stage2-accepted的最终数据，不改写既有实验文件。

| 划分 | 回合/几何 | 全部窗口 | 核心完整窗口 | 未来不完整 | 历史不完整 |
|---|---:|---:|---:|---:|---:|
| train | 3/3 | 116 | 84 | 30 | 2 |
| validation | 3/3 | 126 | 95 | 27 | 4 |
| test | 3/3 | 110 | 87 | 21 | 2 |

共9个正常回合、11969条原始车辆记录、352窗口、266核心完整窗；78未来不完整与8历史不完整均保留。3诊断/失败源回合明确排除并记录多重原因。全部三个集合均覆盖三道路类型，几何与回合无交叉；窗口重叠不跨集合。选定车辆累计764车窗，当前未选定车辆累计377车窗（不是唯一车辆数）。

独立审计核对15988个有效历史点、28319个有效未来点及全部352份实际保存的局部地图。状态最大绝对误差7.4033e-6（float32存储误差；各维单位不同）；82条通行连接、694项foes关系与SUMO随包解析器一致；18条零几何长度内部车道保留。此处通行核对不是车辆动力学验证。

train/validation实际区间速度最大21.7520m/s，车身半对角2.5164m，通过预定25m/s与8m覆盖设计包络；test不参与包络选择。固定±192m栅格中，有效标签中心/车身四角覆盖问题均为0，向量道路中心越界为0。完整车身道路/路线与加速度/jerk仍未评估，不以这些0宣称动力学/行为质量通过。

### SHA、资源与限制

最终预处理及独立审计运行SHA均为eb29838025350d12537dad2b92282c201b9923c2，分支feat/window-dataset；summary为stage2_validation.json。数据/审计清单保存了所有实际输入、配置、状态与源SHA；后续summary提交不回填运行SHA。原始数据和处理窗口不提交Git。

最终预处理实耗7.256s（含清单准备、切窗及写文件，不是模型生成时间），逐窗采样Python RSS峰值85.13MiB；审计末次RSS约94.01MiB，非OS峰值。未用GPU，未测模型batch显存。实际版本Python3.12.7、NumPy1.26.4、Shapely2.1.2、Pillow10.4.0、psutil5.9.0。

数据仅供工程验收，每集合每类只有一个小回合，实际最多同时选定4辆；12车容量已用合成样例验证，但真实12车场景/模型负载未测。不是足够的正式训练/论文基准；没有新checkpoint、训练、碰撞成功率或创新性结果，不创建实验里程碑tag。归一化候选尚未冻结。

### 原子提交

72fff7f 几何与固定frame；4e32953 零长度连接；90c1c41 隔离窗口；f70fcfb 条件/标签持久化；0f8ce09 三几何验收清单；5a50bae 内部等待点规则修复；07a2567 车身覆盖QA；5d80f92 独立地图/轨迹审计；7a55b54 容量与可配维验证；4c0e230 数据命令与依赖；eb29838 接口文档。

验收通过后以--no-ff合并main并同步远端；实际summary/merge交付SHA以Git记录和最终报告为准，不更改运行SHA。

### 阶段3前置条件

- 新建feat/trajectory-decoding分支，先读本进度及window_dataset/data_schema；使用最终已验收数据清单，不混入preview。
- 在项目内配置/记录兼容torch的数值验证环境，不升级全局pytorch。数据环境本身无torch。
- 实现可微位置—速度最小二乘协调，固定t0锚点，适当求解精度与朝向回退；未来偏移先加initial_positions恢复共同局部位置。
- 实现独立有向矩形及帧间插值碰撞、完整车身道路/路线/覆盖、原始运动与历史—未来边界指标；无适用车辆对输出N/A，不合成旧综合分。
- 以已知匀速、转弯、停车、碰撞、越界样例和正常标签检查数值误差/梯度。只做短验证，不进入模型训练或正式大采集。


## 阶段3：已完成验收（2026-10-05）

### 交付

分支feat/trajectory-decoding，从已验收main 4fc73de开始。实现可微全时域位置—速度最小二乘协调、固定t0锚点、显式初始朝向、sin/cos规范化及近零回退计数。支持批维度和padding，拒绝半精度线性求解，默认float64。API不接收未来标签mask，避免按真实退出截断生成。

实现独立有向矩形、恒定朝向精确扫掠SAT和带保守界的旋转帧间检查；预算不足标未知，初始接触不计新目标事件。完整车身向量道路/路线/地图范围检查互相独立。运动包含原始速度一致性、纵横加速度、jerk、朝向和历史—未来交界，不合成旧综合评分。场景质量为可审计的布尔资格；无车辆对或目标任务N/A，失败与未知留在适用分母。

新增统一评估配置/入口、标准运行清单、逐窗raw/decoded轨迹与修正、CPU梯度和GPU资源探针，以及独立保存结果审计。配置和接口见decoding_evaluation.md/data_schema.md；小型最终summary见stage3_validation.json，数据和原始输出不进Git。旧目录、旧checkpoint均未修改或引用。

### 正式短验收

命令见decoding_evaluation.md，使用项目.venv-model，PYTHONPATH指向本工程src；pytest设置SUMO_HOME为1.22.0安装。82项测试通过（14.49s，含此前55项及新增27项），git diff --check通过。wheel构建通过，移除src路径后从wheel独立导入和解码通过。

正式标签运行artifacts/runs/stage3-accepted-20261005、独立审计artifacts/runs/stage3-accepted-audit-20261005，两者运行时SHA均为8befcda2cf0928b9ba147b12e41928b53d1f3ea2、工作区干净、status=completed。summary提交与合并SHA不回填为运行SHA。输入只来自最终阶段2数据集及全部实际窗口文件哈希。

| 核对项目 | 结果 |
|---|---:|
| 原始标签窗口 | 352，失败0 |
| 全部所选车辆未来完整、用于数值解码 | 274 |
| 未来不完整、保留原始指标并跳过标签解码 | 78 |
| 原始有效未来车帧 | 28319 |
| 选定/未纳入当前车窗 | 764 / 377（非唯一车辆数） |
| 最大位置修正 | 6.6653e-6m |
| 最大速度修正 | 3.9248e-6m/s |
| 标签方向回退 | 0 |
| NumPy增广最小二乘与torch最大位置差 | 1.9895e-13m |
| 保存后速度与位置差分最大误差 | 1.4211e-13m/s |
| CPU gradcheck与CPU/CUDA有限梯度 | 均通过 |
| 旋转碰撞未解决区间 | 0（当前标签） |

合成样例验证匀速、停车、转弯数值恢复，梯度、NaN padding隔离，帧间高速穿越，旋转未知预算、角度绕回，车身跨洞/道路/路线、合法换道、边界跃迁、失败分母、不完整标签及配置错误。独立审计另逐文件核对SHA，用NumPy SVD、独立车身多边形、直接三阶差分核对保存数据。

### 标签质量的实际负结果

27个局部drivable几何出现浮点变换后的自交；严格预演保留为failed。最终明确采用有上限的make_valid linework，保留所有组成，逐项记录，不修改阶段2原图。面积变化≤4.55e-12m²、完整几何Hausdorff变化0，低于公开的1e-6m²/1e-6m上限。默认API仍strict；不是扩大通行区域或以测试调质量阈值。

原始完整车身道路越界314车帧、路线越界1206车帧（分母28319有效车帧），栅格物理范围越界0。逐车多边形审计确认；此前阶段2只查中心点的0不能代表车身通过。

固定候选阈值下，纵向加速度违规190、横向536、jerk1947车帧，最大原始jerk390.4245m/s³。速度、yaw rate、heading-motion及预测速度一致性违规均0。尚未查清并修正全部源头，不能把正常SUMO标签称为舒适/动力学合格，也不能用协调器把这些负结果抹掉。

原始资格通过104/352=29.55%，未知64；完整标签解码子集通过104/274=37.96%。子集与全集不同，不能称为解码提升质量。无确认车辆碰撞，262个多车窗口中221明确无事件、41因部分标签仍未知；不能把0下界称为全时域证明。无目标/攻击角色，目标及角色事件率均N/A。

### 资源与边界

正式标签评估/保存用时9.899s（不含RunRecorder构造/环境采集；不是模型推理时间），逐窗采样Python RSS峰值1117175808bytes≈1065.42MiB，含PyTorch和CUDA上下文，非OS精确峰值。

RTX4070 Laptop上预热后decoder-only B2×N12×T40 float64前后向均耗9.684ms，5次，torch峰值allocated17824768bytes=17MiB，reserved23068672bytes=22MiB。这不含网络、DDIM、生成、驱动显存，不能据此决定模型batch size。torch2.5.1/CUDA12.4及其他实际依赖见environment/运行清单，未修改全局包。

道路指标仅在未来帧检查，不保证帧间道路合规；碰撞连续检查使用明确的离线线性位置/最短角度插值，不证明真实动力学或闭环攻击。当前数据仅九个工程短回合、真实最多4选车，不是正式研究数据或12车网络负载。未训练模型、未输出引导结果、未创建任何实验里程碑tag。

### 原子提交与交付

d0325cf 全时域可微解码；cf04b20 独立扫掠碰撞；9013fbe 车身道路/运动边界；c844144 失败/未知分母；cc026cc 有界几何修复与舒适性分母；78f1304 可追溯标签评估/GPU探针；cd70a10 同精度物理恢复与严格mask；741555e 独立结果审计；8befcda 接口/环境/指标文档。

验收后提交本summary，以--no-ff合并main并同步main与功能分支。最终交付SHA/远端状态以Git记录和交付报告为准；上述运行SHA不改变。

### 阶段4前置条件

- 新建feat/conditional-denoiser；读取最新协议与进度，使用本工程独立环境及已验收数据接口。
- 实现共享逐车历史CNN、地图CNN、路线/折线、层次与平行融合、共享时间U-Net及瓶颈按时间车辆注意力；掩码贯穿，不绑定车辆槽位到CNN通道。
- 先小候选128维/4头，验证车辆重排一致性、无效槽位隔离及单批前后向，实测整个网络显存；不拿解码17MiB推断网络资源。
- 解码与生成仍需显式当前heading；history末点不可用时给当前观测而非未来。未来mask仅训练标签。
- 现有SUMO标签足以查工程接口，但正式数据/阶段5长训练前应只在train/validation分析道路、路线和运动异常源头，决定采集/校准方案，保留旧结果，不以放宽测试阈值替代。
- 阶段4不训练基础扩散、不启动大采集或后续引导/RL。


## 阶段4：已完成验收（2026-10-05）

### 交付

分支feat/conditional-denoiser，基于已验收本地main 5207240。实现共享逐车历史时间CNN、空间地图CNN、折线/合法连接及yield/foe图消息、逐车已知计划路线顺序编码。层次与平行融合共用全部权重和U-Net，区别只在车辆交互是否预先接收道路条件；每车门控输出128维条件。基础模型无攻击角色或槽位embedding。

时间U-Net为40→20→10、64/128/256通道，FiLM接受逐车条件及扩散整数步embedding；瓶颈在每个下采样未来时间位置做车辆注意力，再两倍重复上采样和skip，输出[B,N,40,6] epsilon。所有有效状态和mask检查，padding输入/输出/注意力隔离。cfg/部分扩散/滚动/攻击角色及未支持的结构开关明确报错。完整结构、条件单位及路线代表语义见conditional_model.md，代码源目录models，配置initial.yaml已成为实际可运行规格。

适配器只读当前条件/metadata/map，不读标签；正确保留0/1栅格，支持已知内部连接路线。测试还验证整个处理数据目录搬迁并移除targets/labels后仍可运行当前条件审计，哈希来自实际读取路径，不能依赖原目录。

### 最终验收

97项测试通过（16.99s，已有82项+阶段4新增15项），包含两种融合的车辆/车道重排、padding增减、NaN payload隔离与零梯度、全部有效参数梯度、history/map/route/attributes/current position/timestep输入路径、跨车未来交互、每瓶颈时间位置独立注意力、等参数模式、配置拒绝及资源探针。git diff --check通过。wheel构建及删除src路径后的独立模型导入/参数数目核对通过。

正式GPU短验收位置artifacts/runs/stage4-accepted-20261005；CPU重放artifacts/runs/stage4-accepted-audit-20261005。两者运行时SHA为34f324a3313632787a3585a2ab3cd42e9809711a，分支feat/conditional-denoiser，干净/已提交、status=completed。summary见stage4_validation.json；后续summary/merge提交不回填运行SHA。

核对train116、validation126，共242个inference窗口，test不用于架构/资源选择；未读取future标签。当前最多4选车、最多32个地图车道token，有6个车辆历史不完整。只按t0选定车数为三类各选一个负载窗口，实际选车为3/3/4，槽位仍12。

默认模型3414233参数（13656932bytes），两种模式参数布局相同。初始化参数SHA-256为28c500f86c340465eb740e5182b726d80fdba6ffef0f4ffdaaea768a5cb40a72，运行前后相同，没有optimizer、权重更新或checkpoint。noise种子/配置及全部实际输入哈希均保存。

GPU车辆重排epsilon最大绝对误差hierarchical1.2033e-6、parallel1.0394e-6；condition最大1.4305e-6。屏蔽位置改为NaN后有效输出差0、padding梯度0、参数梯度均有限。CPU重放重新初始化同参数、核对所有输入/输出/config/environment身份：epsilon最大差hierarchical1.1884e-6、parallel1.4305e-6，condition2.8610e-6，均在公开容差内。两种未训练输出有所差异不表示任何质量收益。

### 实际资源

环境继续使用项目.venv-model，Python3.10.16、torch2.5.1/CUDA12.4、RTX4070 Laptop 8188MiB、驱动610.47；未升级全局包。float32、严格确定性、TF32/AMP关闭，warmup1、计时3次，CUDA同步。表中为完整网络和接口验证开销，不包含数据加载、optimizer状态、DDIM、解码或正式训练。

| 输入负载 | 模式 | forward ms/批 | forward+backward ms/批 | allocated MiB | reserved MiB |
|---|---|---:|---:|---:|---:|
| 三类真实条件，B3、12槽位、实际3/3/4车 | hierarchical | 121.50 | 142.51 | 134.65 | 158 |
| 同上 | parallel | 128.32 | 147.13 | 134.65 | 158 |
| 合成12有效车，B1，32车道 | hierarchical | 55.58 | 76.70 | 105.71 | 122 |
| 同上 | parallel | 54.89 | 93.54 | 105.71 | 122 |
| 合成12有效车，B2，32车道 | hierarchical | 109.71 | 110.41 | 115.78 | 126 |
| 同上 | parallel | 88.60 | 117.33 | 115.78 | 126 |
| 合成12有效车，B4，32车道 | hierarchical | 148.51 | 189.24 | 154.82 | 188 |
| 同上 | parallel | 167.71 | 189.29 | 154.82 | 188 |

合成12车仅为完整负载测试，重复已知地图/路线并改变当前位置，不能作为真实SUMO密集场景或研究指标。batch4已执行但没测最大batch，正式训练仍须包含optimizer再测；建议阶段5先从B2或B4短验证。三次计时的微小模式差异不构成速度优势证明。

网络验证和逐case资源部分8.420s，不含此前242窗adapter审计、RunRecorder构造/环境查询；不是端到端生成/训练时间。torch allocated/reserved不等于驱动总显存。CPU或新库版本不保证bitwise一致，重放采用明确容差而非静默降级。

### 失败与当前限制

最初linear1d上采样反向在本机严格CUDA确定性下失败，保留stage4-preview的failed记录；最终改为nearest_repeat并继续严格确定性。另一个资源入口调用遗漏参数的开发失败已修复并加实际profile测试；更新配置后旧preview源哈希不匹配也被拒绝，后续配置一致的预演/重放及最终正式运行均通过。原失败不覆盖。

全部是未训练噪声网络和工程验收，没有生成场景、基础DDIM、训练checkpoint、攻击事件率或层次融合优越性结果。不创建里程碑tag。阶段3检出的正常SUMO车身越界、路线越界和较大jerk仍未解决；新网络不能自动修复源数据质量，也不能证明真实行为。movement以via/source车道作代表的规则摘要不是完整连接级优先决策；循环路线未实现并明确报错。

### 原子提交与同步

561b4f0 当前路线/规则输入；78d0e13 共享编码/同参数融合；d3a7ada 联合时间U-Net及语义测试；df1ac8c CUDA确定性上采样；5a70e4d 可追溯资源/CPU重放；2a93b3a 模型/协议文档；34f324a 搬迁后实际输入身份与无标签独立性。

验收后提交本summary，以--no-ff合并main。结束前远端只读核对main仍为4fc73de，阶段3的5207240因上次网络失败尚待同步；本次同步main及两个功能分支，不强推。最终交付SHA与同步状态以Git记录/交付报告为准，验收运行SHA不改变。

### 阶段5前置条件

- 创建feat/base-diffusion，读取最新模型/解码/指标接口；只完成基础扩散阶段，不默认启动长训练或批量采集。
- 用train/validation检查并冻结位置/速度尺度，与固定条件特征单位区分；checkpoint保存完整ModelConfig和归一化，旧checkpoint拒绝。
- 实现前向加噪、逐场景按有效元素归一化的future_mask噪声MSE；未来mask仅标签/损失，不能进入条件编码或采样退出规则。
- 完成小数据短训练、验证、恢复和元数据，再20步eta0 DDIM及统一解码/独立评估；附加物理loss默认关闭。CFG/部分扩散等未支持能力继续明确报错。
- 先实测含optimizer的显存/吞吐量，从B2或B4候选开始；不拿未训练forward时间估计正式训练时长。
- 正常标签质量问题只在train/validation分析/校准，保留原始结果；当前九回合仅供查错，不能当作足够的正式研究数据。
- 只有完整采集—切窗—训练—base—评估且可复现才考虑v0.1.0，目前条件尚不满足。


## 阶段5：工程验收通过；短训练模型的收敛及充分训练质量未验证（2026-10-05，结论于2026-10-07澄清）

### 交付

分支feat/base-diffusion，基于main a078c75。完成固定50m/20m/s尺度审计、1000步线性DDPM、每场景masked epsilon MSE、AdamW/固定验证探针、checkpoint/恢复、20步eta0 DDIM、raw/decoded轨迹和既有独立评估。条件缓存一次。future_mask仅训练损失，推理按t0当前ID/family选车，无标签/core过滤、目标角色或旧综合评分。CFG、部分扩散、滚动、额外物理loss和guided/diffscene_style等未实现功能明确报错，不静默降级。

checkpoint保存配置/数据/尺度/窗口/代码SHA、权重、optimizer、CPU/CUDA RNG、训练Generator及父checkpoint。旧checkpoint拒绝；weights_only=True；恢复检查torch版本及训练签名，可增加总步数，超过1000更新需显式allow-long-run。没有启动长训练、采集或阶段6/RL。

接口/命令见base_diffusion.md，机器可读结果stage5_validation.json。

### 验收与运行身份

最终120项测试通过（18.16s，已有97+新增23），git diff --check、wheel构建和python -I从wheel独立导入通过。覆盖单位/不clip、mask/NaN梯度、逐场景权重、DDIM oracle/随机方差/更新后epsilon、恢复状态/签名、能力拒绝、任务种子、失败分母、当前ID选任务和无标签搬迁投影。

六个正式追溯短验收均completed/formal=true/dirty=false，运行SHA为1bacc0a44e2f2676c07bfa38b45b1841cb2cfd5d，分支feat/base-diffusion。位置为artifacts/runs/stage5-{scales,train,resume,continuation,sample,inference}-accepted-20261005；花括号表示六个独立目录，不是PowerShell路径写法。每项的manifest/config/command/environment/metrics/status与hash均保存，小型summary记录真实运行SHA，后续提交不回填。数据、权重、轨迹不提交Git。

### 尺度与学习

只使用train116/validation126共242窗、19364有效未来车帧。位移P99=71.258m、max=79.850m，21.256%大于50m；速度P99=19.951m/s、max=19.999m/s。50m是单位而非上界，保留大于1的值；未clip/标准化sin-cos。test不参与尺度/参数选择。

短训练core12窗、验证6窗，B4，240更新，lr3e-4/weight decay1e-4/clip1，float32严格确定性，AMP/TF32关闭。训练固定epsilon探针1.087919→0.037835（下降96.522%），验证1.111236→0.047379；权重确实改变。最终参数hash为1a0f4e11a8c9557c5477d14a88420ea2e8246a1b1b4e399989249219a447ad6f。

从120步恢复到240，与连续240步的model、AdamW、Generator、CPU/CUDA RNG以及后120步窗口ID/t/损失/裁剪前梯度范数全部bitwise一致。搬迁投影缺少全部targets/labels文件，6任务初始noise、normalized/raw physical/raw absolute/decoded absolute重放全部bitwise相同，最大误差0。

### 资源实测

项目.venv-model，Python3.10.16、torch2.5.1/CUDA12.4、RTX4070 Laptop 8188MiB、driver610.47；SUMO1.22.0自带客户端入清单。含optimizer峰值allocated186094592bytes（177.474MiB），AdamW状态tensor27314624bytes；reserved及实际环境见summary。B4更新mean203.09ms/median195.91ms，19.696窗/s，排除首步warmup，含索引/传GPU/前反向/裁剪/AdamW，不含cache0.512s、validation和checkpoint IO。训练/validation/checkpoint循环54.018s，不含环境查询。

同缓存负载/精度假设10000更新，optimizer-only估计2030.892s（33.848min），不代表收敛或正式训练规模；更多车/车道、数据IO、验证/保存另计。未探测最大batch，未采集真实12車密集新数据。

6场景采样+解码mean0.25585s/P99=0.50940s（含首例warmup），20去噪调用/场景，梯度调用0，allocated51568640bytes（49.18MiB）。独立评估/循环时间见summary，不是研究效率收益证据。

### 必须保留的负结果

生成/解码/评估无运行失败；raw、decoded质量通过率均0/6，全部保留。任意/非目标碰撞2/6；目标/有效目标N/A。三类道路各2窗，实际选车2/2/3/2/2/3，共14 vehicle-windows，未纳入当前车辆2 vehicle-windows。最大位置协调修正4843.106m、速度修正701.792m/s。解码后位置—速度关系一致，但道路/路线/覆盖、速度/加速度/jerk/朝向大量不合格。

当前仅完成240步短训练，checkpoint自由生成严重失真；收敛及充分训练后的生成质量未验证，失真原因尚未确定。epsilon loss下降不能替代生成质量，也不能据此判断架构失败。高噪声误差经x0恢复放大是机制解释，不表示已排除其他学习问题。没有clip、平滑、放宽阈值或追加未授权长训练来掩盖结果。九回合规模及阶段3源正常数据道路/路线/大jerk问题仍需在train/validation治理；不创建v0.1.0，可用基础研究模型及正式训练尚未完成。

开发期间修正jsonl换行生成语法、系统tmp权限、非空seed记录接口及UTF-8读取问题；预演结果保留，其中采样preview为dirty，最终验收使用上述干净SHA。旧目录始终只读。

### Git与下一阶段

a0d8f2a 噪声数学/尺度；9833bab 训练/checkpoint/恢复；ffd4332 base采样/独立评估；72406c0 续训与标签隔离；a494841 协议；1bacc0a 搬迁无标签重放。验收后summary提交，再--no-ff合并main并同步main/feat/base-diffusion；最终交付SHA与推送状态以Git记录/报告为准。

阶段6留待后续授权。接口可用于固定目标引导及合成动作梯度验证，当前短checkpoint质量不是阶段6模块实现的硬性阻断条件。真实风险—质量实验须独立报告基础模型训练程度和生成质量。正式训练仍需进行，先分析train/validation数据质量和高噪声误差，确定数据规模与收敛/生成质量标准；不沿用旧三天估计，不默认扩大采集/长训练，也不把guidance当前0质量模型当成研究成功。未来引导更新干净状态后重算对应epsilon，使用同一解码轨迹/独立评估，主干冻结。


## 阶段5补充：有界训练诊断已完成（2026-10-07）

用户授权在原12/6窗口上进行1000—3000总步数诊断。本轮新分支feat/bounded-training-diagnosis从main 3a11718建立，保留原模型/尺度/学习率/AdamW/精度/种子，从240依次续训到1000、2000、3000，总新增2760更新，达到预算后停止。

新增独立diagnose入口：9个固定t、每窗3个固定noise的有标签重构，精确noise/标签解码oracle，以及current-only任务的train/validation纯噪声自由生成；明确区分这三类证据。增加20/50/100步DDIM对照、原始/解码修正统计、seen身份和源文件哈希。评估主干不更新模型，所有失败/不合格保留；不增加综合分数或指导模式。新增可复现绘图脚本与reports可选依赖。

### 结果与实际成本

完整表格、解释和源清单见[training_diagnosis_summary.md](training_diagnosis_summary.md)，机器摘要见[training_diagnosis_summary.json](training_diagnosis_summary.json)。验证ε MSE在240/1000/2000/3000步为0.047379/0.015260/0.012439/0.006607；相同6验证任务的20步DDIM原始最大位移5351.17/1612.97/962.23/725.86m，最大位置协调修正4843.11/1500.61/935.90/719.03m。解码质量均0/6；运行失败0。学习与自由输出失真明显改善，仍不能宣称收敛或充分训练质量已验证。

3000步20/50/100DDIM原始最大位移725.86/667.42/635.93m，质量均0/6，均时0.192/0.390/0.784s（采样与解码，不含IO/独立指标）。增加采样步数不能单独解决本轮固定任务的失真。全部oracle通过，精确噪声恢复误差≤1.72e-12，标签位置修正≤5.40e-6m；检查范围不等于排除全部实现或学习问题。train-split自由任务仅2/6在实际训练12窗中，不能误称全是训练已见样本。

三段训练/验证/保存循环合计505.214s（8.42min），独立诊断/环境查询另计；batch4更新均时173–181ms，峰值allocated177.474MiB，未测最大batch。仍为RTX4070 Laptop 8188MiB、torch2.5.1/CUDA12.4，未修改全局环境。

### 验证、追溯和提交

必要短验证25项通过；最终完整128项测试通过（14.55s）。wheel构建、python -I从wheel独立导入诊断和元数据检查通过，git diff --check通过。损失/重构/修正趋势及三类场景轨迹图已目视核对，不遮盖异常raw输出。原240步固定验证采样与阶段5旧输出逐数组bitwise相同，保证比较起点一致。

训练及诊断7个顶层正式运行的SHA为d51400e7ac79c225c29bc558dc1c9bea4936e390；报告SHA为41ea18f4535b8ad4c6b70e6908ecf5d8d2bff053，均clean/formal/completed，子采样另有完整RunRecorder。运行目录、环境、实际命令、配置/数据/checkpoint哈希及资源定义进入summary；最终checkpoint位于artifacts/runs/diagnostic-train-3000-20261007/checkpoint_step_003000.pt。轨迹/数据/权重/图表均不提交Git。

本轮原子提交cdcaca5诊断；d51400e验收语义及预定预算；41ea18f报告绘图；cb2d012可选图表依赖；之后小型summary提交并--no-ff合并main，最终SHA/远端同步以Git及交付报告为准。实验SHA不改写，不创建tag。

### 当前结论与下一步边界

阶段5工程验收通过，模型继续显示学习能力；收敛及充分训练后的生成质量未验证。数百米异常仍需诊断，不能唯一归因欠训练，当前证据也不足以要求架构重构。本轮不默认追加训练、采集新数据或启动阶段6。

后续若使用现有完整core84/95窗口，需新建实验而非当前严格resume，先检查源数据质量并另定训练预算及收敛/生成质量标准。阶段6模块与合成梯度验证不受当前短模型质量硬阻断；真实引导效果须独立报告基础质量。测试集本轮未访问，阶段3正常数据道路/路线/jerk问题继续保留。
