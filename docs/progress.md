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


## 阶段5A：数据资格、扩充计划与真实epoch入口（开发验收中，2026-10-07）

用户授权目标1200训练/200验证核心窗（3类均衡）、batch4、100 epoch，并要求先核查少量新回合标签；批量采集和长训练由本地终端启动。本阶段从main d1b3155建立feat/stage5a-data-training，不进入阶段6。新增质量与完整性分离、原始坐标独立重建、完整H+F导数/车身/碰撞检查、拒绝保留、多几何/回合配额和新尺度审计；6回合Pilot及78回合预分配批量入口均拒绝覆盖。

新epoch配置每轮无放回遍历全部资格窗，每轮完整验证、5epoch保存、10epoch固定纯噪声自由生成，监测恢复全部RNG。保持旧step接口与checkpoint兼容，对新数据训练不复用旧权重。PowerShell入口及失败恢复见stage5a.md。已完成必要50项短测试及此前141项全测试，进一步新回合与短训练预演尚未运行，不能宣称1200/200数据已存在、100epoch已训练或收敛。正式预演将在干净提交上执行，实际结果另补。


### 第一轮新回合发现与修正

65f83de干净代码完成stage5a-accepted-20261007-pilot：6个新回合、597候选窗，3类多车核心窗口保留率80%/54.26%/66.67%，248个合格候选中选择12/6。坐标核对及审计0失败；但明显转弯的路口窗口32个全部拒绝、合格转弯0，不作为转弯质量验收。原运行完整保留。

定位到合法车道flat-cap buffer连接裂缝，并补全仅相接合法中心线的round join；同时提高新源路网几何精度和细分，显式降低正常转弯速度。指标阈值未放宽，原始标签不平滑。新版protocol要求新的短Pilot，第一轮不会被当作当前批量协议的质量凭据。后续预检及资源结果另补。


### 阶段5A最终交付：入口工程验收完成，正式数据与100epoch待用户本地执行

当前feature为feat/stage5a-data-training，基于main d1b3155。原子提交包括2c09315连续直路/IDM、ce7a2c8原始资格、6ac6ea9预分配采集、e9ed596真实epoch/监测、2841009语义测试、65f83de终端文档、c02213c合法相接路线、53d6c08源精度/正常转弯、3f50ddf首轮负结果、2fb9a58保持有面积集合、7c4a256新原始回合复查、a36aa58资格/N/A/覆盖、cd9f600正式来源恢复保护。全部修改在E:\diffusion_new；旧项目始终只读。

原第二轮stage5a-sourcefix-20261007-pilot在27窗口检查中因几何集合输入限制失败，保留原始输出；修复后用严格相同的6个新源回合生成stage5a-ready-20261007-pilot。最终602候选、216合格候选，选择12/6用于短验收。core多车保留率80%/39.08%/68.25%，坐标/审计0失败。合格池的路口方向变化窗口6个，预演选定队列0个；不宣称转弯模型学习或整个源路网物理合格。merge/换道/转弯仍存在真实原始差分尖峰，保留拒绝；无适用车辆对N/A已与碰撞区别命名。

Stage5A正式preflight、尺度、2epoch、4epoch连续、2→4恢复、continuation audit及子采样均completed/formal/clean，运行SHA=7c4a256073fc2fdc751966232d1dc2f262b0f3ba，源回合采集SHA=3f50ddf。机器读结果与路径见stage5a_validation.json；后续summary不回填运行SHA。旧第一轮/第二轮均不覆盖。

新尺度审计仅覆盖预演18窗：位移P99/max58.9306/63.0720m、速度P99/max15.7680m/s；固定50m/20m/s、未clip。训练B4仅最大12更新，不是100epoch：valepsilon从1.073069→0.601917，生成监测仍0/6、raw/修正数十公里，所有失败及不合格保留。每epoch真实遍历12窗一次；连续/恢复model、AdamW、噪声Generator、CPU/CUDA RNG及后6步trace全部bitwise一致；固定监测312数组重放一致。

RTX4070 Laptop8188MiB、torch2.5.1/CUDA12.4，短训练allocated177.474MiB、更新均时178—184ms，CPU缓存15,396,516bytes，三训练循环含监测合计约45.37s，cache/环境查询另计。预演真实选车2—6，不能保证完整新负载同成本；30000步按相同缓存负载仅optimizer估计89—92分钟，不是收敛或端到端时间。

最终155项测试通过（14.27s），git diff --check、PowerShell语法、离线wheel与python -I从wheel独立导入通过。最终代码再次读取实际已完成Pilot/资格队列检查，接受原协议身份。没有默认批量采集、正式1200/200数据、100epoch、阶段6/RL或tag。

接下来由本地终端按stage5a.md执行Collect（78回合）、Prepare（资格/多样性/重新全量尺度审计）、Train（新模型B4/100真实epoch=30000更新）；每epoch验证全部200、5epoch保存、10epoch固定任务监测。配额或质量不足就保留报告并停止，不重复补足或静默缩减；查看最终转弯覆盖及过滤分布后再判断数据代表性。100epoch只是初始预算，不自动标记收敛。

本轮远端只读核验Connection reset，未推送；保留本地原子提交，验收后--no-ff合并main。完整手动同步命令已写stage5a.md，包含尚未同步的bounded-training-diagnosis分支；不强推。最终交付分支/SHA以Git与交付报告为准。

## 阶段5A：正式数据与100epoch训练完成后的复核（2026-10-08）

用户本地已完成stage5a-v1：78回合采集，17397候选窗，独立原始资格检查0审计/转换失败，选中1200/200完整且质量合格窗口；三类训练各400，验证67/67/66，训练每类5几何20回合，验证每类2几何6回合。实际转弯窗口训练23、验证5，覆盖限制保留。新位移P99/最大56.87/64.00m、速度P99/最大16.00m/s，50/20尺度保持、不裁剪。

在main d07676f干净正式代码上训练100epoch、30000更新、batch4已completed。逐条日志核对每轮1200窗一次，无遗漏重复，loss/梯度/时间均有限，最终checkpoint哈希核对通过。最终验证ε MSE=0.001993，最低第88轮0.001528（该轮未保存权重）；末20轮均值0.002749仍有波动，未执行收敛判定。训练/验证/保存/监测循环6954.71s，缓存准备26.48s；更新均时195.82ms、吞吐20.43窗/s、CUDA allocated/reserved峰值177.47/196.00MiB，CPU缓存约1.20GB。不能将CUDA统计等同整机占用。

本轮从main d07676f建立feat/stage5a-training-review。443e7ed加入有界重构探针并检查完整训练队列身份，162项全测试通过14.59s（临时目录指定新项目artifacts；默认系统pytest临时目录权限错误已避开）。2ec2096移除绘图/采样说明中写死的旧12/6与3000步，正确记录当前源预算和探针范围。a220c90加入固定20步的完整200验证窗口配置。全部在新项目实现，旧项目始终只读，未改训练目标、模型、解码或物理阈值。

正式诊断6训练/6验证、9个t每窗3噪声，48.56s完成；精确噪声恢复最大1.66e-12、标签位置修正最大3.54e-6m，全部oracle通过。训练已见自由任务6/6但质量0/6；验证相同6任务20/50/100步仍0/6，修正最大19.90/17.15/16.92m，均时0.208/0.433/0.806s。验证t0/t999单步标签位置重构RMSE0.308/271.15m，这不是自由生成误差。

完整200验证任务、735车辆窗已completed，失败/未知0，raw/decoded质量均0/200。decoded碰撞4/200、道路越界50/200、路线越界51/200、地图覆盖超出0；全200场景有运动/jerk超限，排除边界后0.5—4s内部jerk仍全200超限。位置修正帧合并P50/P99/最大0.453/8.064/28.119m。生成与解码均时0.200s，循环61.04s（独立指标17.81s）。6个原监测任务的78个数组逐一bitwise重放。目标角色未定义，相关指标N/A；评估仅选定车辆，未纳入3005个当前车辆实例。

结果、来源哈希、命令、环境、资源定义见stage5a_training_summary.md/json；曲线和轨迹图已目视核对，保存于artifacts/runs/stage5a-v1-review-report-20261008。原始采集/准备/尺度/训练SHA d07676f、有界诊断443e7ed、绘图2ec2096、完整验证a220c90均formal/clean/completed，摘要与合并不改写这些SHA。

本轮完成数据与训练预算的结果验收；基础生成尚未达到声明质量标准，不能宣称稳定或收敛。建议保持当前数据，下一步短对照定位位置—速度一致性、历史未来边界及内部jerk，并评估学习率衰减/EMA/验证最优保存，改变算法须新实验身份。未追加训练、重采数据、开始阶段6或创建tag。最终本地合并和远端同步以Git状态及交付报告为准。

## Git交付状态（2026-10-08）

本轮功能分支为`feat/stage5a-training-review`。代码与配置提交443e7ed、2ec2096、a220c90，结果文档提交163e6a1。远端引用查询实际main为`3a11718192632513a41dbd5f84a43a2efd5b6cf2`，与现有历史兼容；随后fetch因`Recv failure: Connection was reset`失败，未执行推送。保留本地验收合并，不重写共享历史或反复尝试网络连接。精确main合并SHA以Git交付报告为准；原实验SHA保持不变。

网络恢复后手动执行，每条成功后再执行下一条：

```powershell
git -C E:\diffusion_new fetch origin
git -C E:\diffusion_new switch main
git -C E:\diffusion_new merge --ff-only origin/main
git -C E:\diffusion_new push --atomic origin main feat/stage5a-training-review feat/stage5a-data-training feat/bounded-training-diagnosis
```

若无法快进或出现认证/权限错误，保留本地提交，先检查实际远端和分支状态，不能强推。

## 阶段5A：基础轨迹质量短诊断完成（2026-10-08）

用户授权检查当前基础生成，保持阶段6未启动。feature `feat/trajectory-quality-audit` 从main30a8464建立，新增4311a75冻结质量归因及6项必要测试、fd9fc37可复现图表；生产模型/训练/解码/物理阈值未改，参考工程只读，未采集或训练。

200个验证窗口、735车辆窗、9种对照全部完成，失败/未知0。真实标签raw/decoded均200/200质量合格；生成decoded仍0/200。内部jerk P99原decoded719.225、替换真实位置709.826、替换真实速度107.575；真实速度使内部超限场景200→98，但交界仍200超限。位置+速度替换后176/200合格，剩余24场景的朝向质量不合格、19道路/路线越界。恒加速度算术控制190/200合格，不称行为真实/生成模型成功。

独立NumPy增广LS/Torch最大位置差1.492e-13m，源GPU/CPU解码状态差7.105e-14，200真实标签最大位置修正6.914e-6m。6当前条件纯噪声去噪轨迹重放全部bitwise一致，模型参数不变，120次网络调用、0梯度。最后t53→0使5/6内部jerk改善，所有中间/最终运动仍失败，不支持跳过最后去噪来修复。仅epsilon目标、恒定lr3e-4、无EMA/自动最优保存核实；症状归因不等于已证明学习策略根因。

诊断59.697s（环境查询/输入hash另计），SHA4311a755dcc76421d5097d0b7edc69ffa54911e6；图表SHAfd9fc375719aa2803f69062c9ea36173a9a389a9；均formal/clean/completed。源200采样a220c90、源训练d07676f、checkpoint SHA-256 2315fa4d57b9d86a3254a0923590efbbd48f23666f3bf0d2c733ac04b4d1769b保持。完整环境与1408来源hash进入manifest，本次未另测峰值显存。168项全测试通过15.67s，绘图编译/CLI/正式运行与两幅图目视核对通过。git diff --check通过。

报告与小摘要见[trajectory_quality_audit.md](trajectory_quality_audit.md)/json，artifact位于stage5a-quality-audit-20261008、stage5a-quality-report-20261008，不提交权重/轨迹/图表。建议下一授权阶段5B保持1200/200及网络，显式新实验初始化与严格resume分开；短对照一致性/速度差分/边界/朝向损失，学习率/EMA另作验证。高噪声clean梯度需有界，不把正常未来强制零jerk；不改阈值或平滑。长训练默认不启动，下一阶段6需后续授权，无tag。

本轮远端查询及fetch成功，main3a11718与现有历史兼容；诊断feature已存在远端。验收后按--no-ff合并main，最终提交与同步以Git交付为准，实验运行SHA不回填。同步受阻时完整手动命令见本轮报告。

本轮远端多分支推送命令被自动审批拒绝：当前授权为检查，未明确授权更新共享main。被拒命令未执行；仅继续本地提交和验收合并，不继续尝试远端推送。等后续明确授权再同步main和本轮feature，实验SHA不变。


## 远端同步补充（2026-10-08）

用户明确授权后，已普通推送并核对main 4e69ec9与feat/trajectory-quality-audit 5a09806，前节自动审批拒绝的同步事项已解决；未改写远端历史。

## 阶段5B四组受控继续训练：实现与短验收准备（2026-10-08）

从已同步干净main 4e69ec9建立feat/physical-loss-ablation。d14a76c实现低噪声有界的原始位置—速度一致性、正常标签速度差分与解码后交界目标及12项必要测试；c53fda2实现共同父模型/完整AdamW/Generator/CPU-CUDA RNG继承、真实101—110轮预算、严格同组resume、四组配置及16项训练测试。父SHA-256固定2315fa4d57b9d86a3254a0923590efbbd48f23666f3bf0d2c733ac04b4d1769b，batch4/lr3e-4保持。A保留原训练图，B/C/D共享启用项权重0.01。

提供scripts/stage5b.ps1：完整Run依次四组各3000更新、统一200验证任务采样与自动配对报告；Preflight每组12更新，核对真实原算法A连续学习bitwise一致和训练/采样噪声配对。恢复到新目录、已完成训练后的独立重采样及完整报告均有明确入口。当前记录是实现状态，实际正式短预演结果将在后续补充；未执行四组完整10轮、不宣称辅助项改善模型质量。长训练按前约定由用户本地启动。未修改旧项目、重采数据、进入阶段6、扩大候选预算或创建tag。协议与命令见stage5b.md。

## 阶段5B实际短预演验收（2026-10-08）

入口/配对报告提交9d92312、协议提交b40f5ad。202项全测试通过33.68s，CLI/PowerShell语法/git diff --check通过。干净正式短预演stage5b-preflight-20261008在SHA b40f5ade48db4d5ea3fb993ad4faa3c3516642c1完成，外层和8个子运行均formal/clean/completed。每组12更新、6采样，新增真实完整epoch=0，不是10轮完成。A与真实GPU原训练算法逐步loss/norm、网络、AdamW、Generator及CPU/CUDA RNG全部bitwise一致；四组共同起点和逐步训练/初始采样噪声配对一致。B—D启用目标贡献有限且非零，14/48场景符合低噪声门限。

相同起点valepsilon=0.001992598；A/B/C/D短终点分别0.001970300/0.001966814/0.002072765/0.004504386。质量均0/6，失败0，不能作为候选收益或收敛结论。D平均辅助目标0.0103378，比噪声项0.0045363影响较大，待完整物理指标对照，首轮不临时换权重。

更新均时185.51/191.57/203.93/242.62ms；CUDA allocated峰值177.54/178.81/178.84/179.60MiB、reserved194MiB，CPU缓存1.1975GB。缓存33.24s、预演主体162.82s，环境/哈希另计；四组各3000更新仅优化预测2470.92s≈41.2min，不包含验证、保存、监测与最终采样，不是收敛承诺。只读真实200任务报告源核对通过，内部jerk P99=719.2254527及道路/碰撞统计重放既有基准，自配对CI为0。

结果小摘要见stage5b_preflight_summary.json，完整定义与运行/恢复/重采样命令见stage5b.md。正式训练已准备，由用户本地终端运行scripts/stage5b.ps1 -Stage Run -RunId stage5b-v1；各组独立100→110轮，完整200验证任务及配对报告自动执行。不自动追加候选训练，不进入阶段6或创建tag。验收后本地--no-ff合并main，新分支不自动推送；最终本地SHA以Git与交付报告为准，实际预演SHA永久保留。


## 阶段5B完整10轮结果分析（2026-10-08）

用户本地stage5b-v1已完成四组各3000更新/真实101—110轮，原SHA15f6dfe5511382b2f3e41c69cb7302ca43b2ce77。逐条日志核对每轮1200唯一窗口，30个运行全部formal/clean/completed。主体3892.93s≈64.9min、缓存25.52s另计。父/A/B/C/D质量均0/200，内部jerk P99为719.23/889.98/236.99/261.52/1311.10，道路50/51/183/181/75，路线51/66/183/184/82，碰撞4/4/5/5/11。

只读归因支持B/C横向偏移为车身违规主要来源：中心越界6/9但车身183/181；只替换真实朝向仍180/181，事后消除真实横向误差后6/8。oracle非生成修复。D辅助/噪声项比2.095、边界占辅助82.74%、clip270次，A—C4/4/5次，待分项梯度校准。图与独立几何重放通过；后验诊断和绘图formal=false，有完整来源和脚本。结果与待审修改点见stage5b_results.md/json。

建议先已有检查点小规模质量筛选，再新协议的统一低lr和较小一致性短对照；D先梯度诊断。没有实施建议、追加训练/采样、修改生产代码或指标、进入阶段6或tag。本轮无生产代码变动，不重复原202项测试；完成源hash、日志断言、几何重放与图表核对。文档分支docs/stage5b-result-analysis，验收后本地--no-ff合并，不推送；正式运行SHA不改写。


## 阶段5C统一训练目标与余弦日程：实现及短验收（2026-10-08）

用户确认在原100轮模型表现基础上实施统一损失、warmup/余弦和本地50轮入口。feature feat/unified-trajectory-training从干净main8205d29建立。53ce3fe实现低噪声配对模仿/速度一二阶差分/车身向量道路代理与11项测试，6c795d7实现更新级双预热/余弦状态与3项测试，8a0f236实现参数/RNG暖启动、新AdamW及严格同实验恢复，eff9dda实现独立协议/分项梯度/启动与配对报告，9018625按真实训练批次将模仿权重0.01校准至0.003并验证实际道路梯度，39c34e3补充无标签20任务定期监测重放。原阶段5B协议、已有数据、模型架构、尺度、解码器、阈值和旧工程均未改。

预算固定原1200训练/200独立验证、batch4，101—150真实epoch共15000更新；新AdamW保持wd1e-4/betas0.9,0.999/eps1e-8，继承父参数与Generator/CPU-CUDA RNG。lr更新1为1e-6、600为1e-4、15000余弦至1e-6，无重启；辅助rho五轮升至1，t<=300/alpha_bar/全batch分母。lambda imitation=0.003、temporal=0.001、road=0.001。配对监督经同一解码器，原sin/cos监督幅值，不强制正常加速度/jerk为零；完整车身独立指标保留。

两个短预演及监测共12个RunRecorder运行，全部formal/clean/completed。初始0.01权重eff9dda训练梯度比中位0.8278/最大2.6577；校准0.003后9018625中位0.2505/最大0.8024，父模型冻结16批审计无优化器更新。人工16m横向偏移的真实地图数学探针road4.30829→4.29756，位置/速度/朝向梯度有限且非零；它不是生成修复。正常重构16批道路项为0，不能宣称道路损失已有质量收益。

实际新AdamW为空；连续16更新与第8步checkpoint恢复后8更新，网络、完整AdamW、Generator、CPU-CUDA RNG、逐步noise/loss/norm/lr/rho全部bitwise一致。全200val epsilon0.001992598→0.001777277，短自由生成0/6。39c34e3重放固定20任务监测，未读取未来标签/未训练、源文件和Torch RNG保持；0/20质量、road/route各4、motion20不合格、内部jerk P99=1055.34m/s³。这些是工程诊断，不是50轮、收益或收敛验收。

231项完整测试通过19.20s；CLI、PowerShell语法和git diff --check通过。实际RTX4070 Laptop8188MiB/Torch2.5.1 CUDA12.4/Python3.10.16；更新均时196.26ms，Torch allocated峰值181.01MiB/reserved284MiB，CPU张量缓存1.1975GB，缓存21.26s、几何6.92s。优化15000步估计49.1min，验证/20任务监测/保存/环境/IO另计；预演主体121.67s。实测不会改写旧运行SHA。

完整定义、启动与恢复见stage5c.md，小摘要stage5c_preflight_summary.json保留来源manifest/hash。默认本地scripts/stage5c.ps1 -Stage Run -RunId stage5c-v1：从原父模型训50轮，自动200任务采样/配对报告。每epoch验证epsilon并保存最佳参考，每5轮20任务保留质量多目标候选；epsilon_cosine只提供可选配置，不自动双倍训练。失效/不合格保留，不用综合场景评分、平滑或阈值放宽完成验收。

本轮只完成实现与短验收，正式50轮由用户本地执行，基础模型充分训练质量仍未验证。未进入阶段6、重采数据或创建tag；新提交不自动推送远端。按规则验收后本地--no-ff合并main，精确文档/合并SHA以Git交付为准；工程完成不等于模型合格。


## 阶段5C完整50轮结果与事后诊断（2026-10-09）

用户本地stage5c-v1已完成原100轮模型的真实101—150轮、15000更新与全200配对验证任务，运行SHA永久为8ceb573cbe76a6d224869ef52c82533891f7869d。14个原运行全部formal/clean/completed，7836个不重复清单源文件身份、配置/环境哈希通过；逐条日志核对每轮1200唯一窗口、日程/ramp及29次clip。训练/验证回合与几何隔离，400份配对解码运动重放及200初始噪声一致。

验证epsilon0.001992598→0.000494853（下降75.17%），场景质量0/200→50/200，道路50→9、路线51→9，内部0.5—4s混合jerk P99=719.225→94.304m/s³；碰撞4→3，差值区间含0，不宣称改善。失败/未知均0，全部预定分母保留。全部150个质量失败均有jerk，138个无其他运动违规；27个只在前0.4s失败、123个内部仍失败。直路/匝道/路口通过29/67、13/67、8/66；4/735整段车辆窗口舒适性通过，原始输出仍0/200，不能称已收敛或物理质量稳定。

相同指标正常标签200/200运动通过、jerk P99=0.496、整段舒适性721/735。800份CPU字段替换解码验证：模型位置/真实速度内部P99=13.810，真实位置/模型速度=95.455，原输出=94.304；线性残差分解RMS位置3.245/速度23.290m/s³、重放误差1.42e−13。真实未来只用于oracle归因，不作为生成修复或baseline。证据指向原始速度的高频变化是当前内部jerk主源，交界及位置漂移仍需检查。

完整流程78.20min、训练/验证/保存75.77min，更新256.37ms；Torch allocated185.38MiB/reserved198MiB，非整机总占用。采样/解码226.78ms，200任务循环70.39s。末5轮epsilon再降1.65%，20任务质量4→6，不能以日程已达min_lr宣告收敛。45000步同时最佳epsilon/当前最高监测质量；最终checkpoint SHA-256为2a25293744e0f9ff8bec9a03b3a1c3e077275bc7c9cece4119cdda13f070900e。

报告stage5c_results.md/json记录条件区间、失败/道路分层、损失值占比与梯度解释边界、下一步建议和完整路径；图已目视检查。posthoc formal=false的stage5c-analysis-20261009-r1与stage5c-field-diagnostic-20261009 completed。第一次后验脚本因NumPy int64 JSON序列化failed，修正后以新目录重放，失败记录保留；原正式运行不受影响。缓存诊断脚本哈希在各自manifest及轻量摘要中。

建议先固定候选进行20/50步DDIM配对及低噪声速度/分项梯度诊断，必要时校准原始速度二阶差分与独立交界分组，保留模仿/道路目标；不盲目放大时间系数或再训50轮。未经epsilon_cosine同日程/预算对照，不能单独归因loss。阶段6可在后续授权下做工程，但当前不作充分模型质量验收。

本轮仅结果分析与文档，无生产代码变动、不重复已通过的231项工程测试，未启动新训练/采样、重采数据、推送、tag或阶段6。文档分支docs/stage5c-result-analysis，通过源哈希/日志/运动与oracle重放/JSON及文档检查后本地--no-ff合并main；新文档SHA不回填实验SHA，精确文档/合并提交以Git为准。


## 阶段5C采样步数与速度诊断验证（2026-10-09）

用户授权执行验证后，从干净main cb058eb建立feat/sampling-velocity-validation。4a7a46a提交仅steps=50的采样配置；0cac507实现冻结参数重构、原始速度二阶差分诊断、配对核对与15项数学测试，8a523d9记录协议及CLI。与原采样/扩散/统一损失共39项相关测试通过1.90s，CLI/diff检查通过。未改生产训练目标、模型、解码器或评价阈值。

四个正式新运行全部formal/clean/completed：stage5c-steps20-validation-20261009（SHA4a7a46a）、stage5c-steps50-validation-20261009（0cac507）、stage5c-velocity-validation-20261009及其report（8a523d9）。固定45000步checkpoint SHA-256 2a25293744e0f9ff8bec9a03b3a1c3e077275bc7c9cece4119cdda13f070900e，训练源8ceb573不回填。200个20步输出/初始噪声/解码状态全部bitwise重现旧结果，50步只有采样步数变化；两组200任务噪声配对。

20/50步质量50/200→43/200，运动失败150→156，道路/路线9→11，碰撞均3，生成/指标失败/未知均0。内部混合jerk P99=94.304→82.581，但场景交界最大jerk平均变化+2.596，条件95%CI[1.798,3.553]；质量变化−3.5百分点CI[−8.11,+1.00]含0，不宣称50步显著更差。失去资格14、改善7、共同通过36；失去的13个jerk、1个道路，其中9个首未来帧jerk>15。严格整段舒适性4/735→0/735。

48训练/200验证窗口在t=0/50/100/200/300真实标签加噪重构。训练运动通过24/48、之后四时刻均0/48；验证111/200、之后均0/200。训练内部jerk P99=28.10/96.77/121.99/158.50/219.58，验证34.48/108.32/145.61/180.86/236.53。所有标签运动控制通过。重构访问真实未来，非自由生成资格；t是噪声索引，不是DDIM步数。训练集也存在时间误差，支持目标校准，不能只归于采样或验证泛化。

16个训练B4、64唯一窗口无更新梯度诊断，自然t适用10批及同噪声强制200适用16批。当前加权时间梯度/epsilon中位0.074/0.072，模仿0.706/1.271，总辅助0.725/1.285；不能用损失值占比代替梯度，也不能沿用父模型校准。0.15m/s归一化的未加权原始速度二阶差分内部梯度比中位3185.9/2743.6、前4帧5045.6/6778.7。待审尺度内部5e-5、前4帧1e-5仅为单项线性梯度换算，未施加到训练；总组合梯度及学习收益仍待检验。

采样+解码均时195.99→404.01ms，200循环59.50→100.61s；Torch采样allocated48.49MiB。重构/梯度主体71.77s，allocated307.85MiB/reserved342MiB，非整机占用。模型哈希一致，parameter.grad未填入，optimizer_updates=0，checkpoint未改。图已目视核对；来源与详细结果见stage5c_validation_results.md/json、重放命令见stage5c_validation.md。

建议保留45000步候选与20步采样，下一版校准原始速度内部/交界二阶差分和当前总辅助预算，保留模仿/道路目标。先有界学习再定正式预算，不自动加训、重采、改架构或进入阶段6。本轮未新训练、推送、tag；按规则验证后本地--no-ff合并main，结果运行SHA永久保留，实际最终提交以Git为准。

最终来源验收：四个新正式运行配置/环境哈希与2776个不重复输入文件身份一致；20步200个bitwise重放、摘要数字及UTF-8检查通过。


## 阶段5D：150→200轮原始速度续训实现及验收（2026-10-09）

用户确认当前方案并授权实施；feature feat/raw-velocity-supervision从干净main5c7a13c建立。127febe实现原始速度二阶差分及掩码/梯度测试，927cec0实现独立150→200真实epoch/AdamW矩与RNG继承/严格恢复，392e749实现联合校准、隔离峰值压力和实验入口。首次预演因辅助标记与原尺度审计不兼容failed，34a7067显式映射审计中的非物理开关并保留全部物理配置、数据及原audit hash；首次失败目录保留。

正式协议固定1200/200、B4、45000→60000/15000更新；新lr两轮1e-6→2e-5、48轮余弦至1e-6。旧模仿0.003/解码时间0.001/道路路线0.001从首更新全强度。新增原始速度标签二阶差分内部5e-5、前四帧1e-5，0.15m/s单位；t<=300/alpha_bar/全batch分母，仅新增项两轮预热。不是零jerk平滑，架构、尺度、解码及独立阈值保持。

stage5d-preflight-20261009-r1在34a706738641549944c14ff3f8b182d28442ad50完成。16训练B4冻结审计自然t适用10批、固定t200适用16批；总辅助/噪声梯度比中位0.8477/1.2900、cos中位0.1515/0.5348，实际总梯度max0.0473/0.0548，均未触发clip1。新增内部/边界单项不能相加；直接组合向量及分项向量和核对通过。模仿减半仅冻结敏感性，比中位0.5750/0.7025，未训练或自动下调。

两隔离副本2e-5/5e-5峰值/full-strength各24更新，均有限/无clip；验证epsilon0.00056093/0.00109215，短期不支持扩大峰值。它们没有正式checkpoint，不替代预热或15000预算。正式16更新初期val epsilon0.000494853→0.000494678；第8步恢复后8步的网络、AdamW/Generator/CPU/CUDA RNG、noise/全部loss/LR/两类ramp逐位一致。真实190个AdamW矩继承，不重置其15000累计更新，只重启日程；父checkpoint不变。

更新均时225.34ms，allocated182.03MiB/reserved626MiB；冻结梯度审计allocated544.15MiB/reserved616MiB。缓存1.1975GB，RTX4070 Laptop8188MiB/Torch2.5.1/CUDA12.4/Python3.10.16。预演主体墙时179.28s；仅优化15000更新估计56.33min，验证/监测/保存/缓存/最终采样另计。6任务1/6质量、0/22整段舒适性、失败/未知0，仅链路探针，新增完整epoch=0，不能解释为质量收益或收敛。

4766a54修正失败车辆保留计划舒适性分母；ac7fb95提供不采样的200任务报告审计；首次排序WindowsPath/string错误发生在RunRecorder前，日志保留，e5eccbe统一路径后stage5d-report-audit-20261009-r1完成。50/200质量、4/735严格舒适性，道路/路线各9、碰撞3、边界/内部混合jerk P99=278.366/94.304；全部自配对差值和CI=0，没有新训练/采样。15f6e81完善碰撞N/A与未知区分，单车场景不填0概率，失败适用任务保留分母。

275项完整测试通过16.91s，后续新增碰撞适用性等6项报告测试通过1.80s；CLI、PowerShell语法、diff与JSON检查通过。6个正式formal/clean/completed运行的完整配置、环境及7424个唯一输入身份核对通过。验收摘要见stage5d_preflight_summary.json，协议及本地Run/Train/Resume/Sample/Report/WarmupCheck/AuditBaseline命令见stage5d.md。训练/报告运行SHA永久保留，文档和合并不回填；原训练8ceb573和父checkpoint 2a252937…保持。

上次因自动审批额度无法执行的收尾已在本轮恢复，保留代码与失败日志；没有待审批操作。已补齐设计、schema、来源、README和运行说明，按规则验收后本地--no-ff合并main，实际文档/合并SHA以Git交付为准。正式50轮由用户本地启动，本次未启动长训练、重采数据、Stage6、推送或tag。接下来按固定任务检查交界/内部jerk、资格、舒适性、道路路线与修正成本，再决定是否延长；不只按噪声MSE或日程结束宣布稳定。

## 阶段5D正式运行结果（2026-10-09）

用户本地已完成stage5d-v1，运行SHA c93a3b55532d4b943401cd84fc9c8d1c7feb5b61。真实150→200轮、15000更新、1200/200窗口、B4，完整AdamW继承/日程/损失记录均一致；200任务及初始噪声配对。基础质量50→82/200（25%→41%），95%回合聚类配对区间改善+10.56至+21.87百分点；33改善、1退化。严格舒适性4→10/735，前四帧/内部混合jerk P99 278.366→266.431 / 94.304→88.667，道路9→8、路线9→9、碰撞3→3，失败/未知0。阈值和解码器未改，未平滑。

9个正式formal/clean/completed运行及7830个唯一输入身份核对，15000条日志和每轮1200唯一窗口、190个AdamW矩与最终30000优化器累计步、源checkpoint/配置/环境哈希一致。原60000步checkpoint SHA-256 b7dcb40fcd9032b76d56ad8871e986067cd29873c8d84ec461feb55b8c96db54。结果原SHA不回填成分析/文档提交SHA。

剩余118失败全含jerk，108仅jerk运动违规；28仅前四帧、90内部超限。超限帧3590→2365（−34.12%），整段严格舒适性仍1.36%。直路/汇入/路口质量40/67、25/67、17/66；16个8—12车路口任务仍0通过。一个11任务路口回合贡献43.93%超限帧，未删除。训练已有47个高车数窗口，不能仅凭验证失败断言数据不足或架构失败。

CPU字段oracle诊断重放误差<1e-9、线性分解<1e-8；内部残差位置/速度RMS 2.82/20.03，前四帧46.17/25.01m/s³。仅支持该解码器的字段误差定位；标签不用于生成、修复或基线，不把分量范数相加为占比。下一步可小规模检查原始位置交界和训练/验证密集子集，再校准新增loss；不自动加大速度权重。总体支持讨论有界延长，稳定与收敛尚未证明。

正式运行墙时78.92min，训练/验证/保存76.66min；更新262.78ms，训练Torch allocated185.38MiB/reserved196MiB，缓存1.1975GB，采样+解码224.43ms/任务。组合总梯度max1.877、33/15000次clip；这不能替代每步辅助夹角审计。末10轮噪声验证均值仅比前10轮改善2.43%；20监测资格6/20→5/20、边界尾部恶化，复杂子集尚未稳定。

本轮分支docs/stage5d-result-analysis；交付stage5d_results.md/json、README及协议结果状态。独立审计/绘图目录stage5d-analysis-20261009、stage5d-figures-20261009，formal=false；它们只读已有结果，脚本/源/图哈希可追溯。绘图首次因种子字典为空在RunRecorder前失败，日志保留，补确定性seed=0后完成。两张图已目视检查。未改生产代码，按结果分析范围验证来源、日志、重放、分解、图和文档；没有重复全量模型测试。

200轮保留当前候选；不自动启动训练、采集、阶段6、push或tag。现有入口固定150→200，后续200→225/250必须明示新协议及预算。本轮验收后本地--no-ff合并main，实际结果提交/合并SHA以Git交付为准。


## 阶段5D交界位置与密集路口诊断（2026-10-10）

用户授权开始检查分析；从干净main d935d14建立feat/boundary-density-diagnosis，a1812b6提交冻结诊断、配置/PowerShell入口和数学测试。仅访问新工程数据/模型，未修改旧目录或生产loss/解码器/阈值。22项交界与解码测试通过10.86s，CLI、PowerShell语法和diff检查通过。

正式stage5d-boundary-density-20261010在a1812b60cef144626183e4fd0014408b2d90a472上formal/clean/completed：两模型、119训练/200验证、t=0/50/100/200/300、同窗口同Gaussian、800前向和3190重构。主体138.75s，Torch allocated56.87MiB/reserved74MiB。模型参数哈希/grad为空、源checkpoint文件不变，optimizer_updates=0/new_free_generation=0。CPU后处理stage5d-boundary-analysis-20261010为formal=false/completed，主体4.56s，无新增网络调用；图已目视检查。

标签训练119/119、验证200/200运动通过。200轮t=0普通路口17/24训练、30/50验证运动通过，密集8—12车路口0/47训练、0/16验证；其他t两个模型两划分均0运动通过。密集问题在训练侧也出现，不能只归于验证泛化，也未证明唯一原因是loss或容量。t=0仍有约0.5m/.2mps输入扰动，重构访问标签，不能替代自由生成82/200。

原始首四帧位置oracle替换只让全未来jerk通过82→88/200；位置＋速度替换使边界101→180通过，但内部110→89、P99 88.67→102.55。全时域协调首帧误差分解：首四帧位置RMS1.094cm、后36帧5.846cm、全速度1.978cm，分量可相消而不能相加。线性分解1.37e-14、原输出解码重放7.11e-14；未用标签修复生成或截断后续时域。

来源验收核对2012个唯一输入身份、3190重构行/140组计数均值极值及场景均值分位数、1600 oracle行、200敏感性行；混合P99保存正式源输出，不从场景分位数反推。轻量摘要boundary_density_results.json保留原SHA、输入/脚本/图哈希和限制；报告boundary_density_results.md说明下一版联合校准解码边界/内部监督、全段原速度及密集训练贡献的候选。没有选择或训练新权重，没有重采数据。

保留200轮候选。下一步前置为普通与密集训练批次上的完整组合梯度、低噪声覆盖及短预演；200→225/250需另设明确协议，不能重跑旧150→200入口或保证自动收敛。阶段6未启动，本轮不训练/push/tag。结果文档验收后按规则本地--no-ff合并main，实际文档/合并提交以Git为准，文档SHA不回填运行SHA。


## 阶段5E：200→225轮解码边界监督实现及短验收（2026-10-10）

用户授权开始实现；从干净main fa5c06c建立feat/decoded-boundary-supervision。3a8020b实现共享解码时间组及替换式配权，3c0d68b实现200→225真实epoch、30000步AdamW/RNG继承/严格恢复；03e0cd4补标签非零运动与全batch门限测试，868ff15实现元数据普通/密集联合校准、独立入口、参数化配对报告与配置，08a4aae落盘协议。初次formal预演因列表索引不符合select_batch数组接口failed，原目录保留；72051ac作最小索引修复后新目录r1完成。

正式配置1200/200、B4、60000→67500/7500更新，1轮lr1e-6→2e-5及24轮余弦至1e-6。将旧解码时间项替换成前4帧/内部36帧分别平均的匹配标签一二阶差分；旧等效系数0.0001/0.0009，新边界候选0.001/内部0.0009，仅增量300更新预热。模仿0.003/道路0.001/原速度内部5e-5边界1e-5从首更新全强度；数据/网络/解码/指标保持。每5轮20任务，终点200配对；不改变密集抽样或epoch无放回定义。

stage5e-preflight-20261010-r1、uninterrupted/resumed/recovery_audit/sample及stage5e-report-audit-20261010六个正式运行均formal/clean/completed，SHA 72051ac6e6071d755f3d5a6bbd132fa971ba3e0f永久保留，结果文档SHA不回填。7427唯一输入身份/配置/环境/父模型哈希核对。冻结32调用：自然t普通2/4、密集3/4批适用，总辅助/epsilon中位0.214/0.539、余弦0.393/0.265，无clip。固定t200比2.457/1.142、余弦0.378/0.315，总范数未超1；模仿仍主导部分梯度但不自动下调。强制整批t0八次clip源于epsilon自身1.48—2.25，辅助约0.002—0.007，不视为新增loss不稳定或自然clip率。

源50轮密集窗口2350次访问，低t适用696（29.62%），普通29.78%；未见密集被门限系统性排除，计数不等于梯度贡献。12次隔离峰值/full-strength均有限无clip，max0.4146，但val epsilon0.000521高于父0.000439，只作数值压力；不认定收益或最优lr。实际16更新正确预热，val epsilon0.000438751→0.000436205；第8步恢复后8步模型/190个AdamW矩/Generator/CPU-CUDA RNG/日程及全部loss与实际系数逐位一致。累积AdamW30000→30016，父checkpoint b7dcb40f…文件保持。

6任务与父输出同噪声配对，两者1/6资格，新模型0/22整段舒适性、失败/未知0。完整200报告重放82/200、10/735，道路8/路线9/碰撞3，全部自配对差值和CI为0，零训练/新采样。未将标签重构或冻结权重敏感性当作新方法收益。

322项全量回归16.76s通过，新增入口/覆盖14项及索引修复8项通过，CLI/PowerShell/diff检查通过。最初测试默认Temp权限错误，改用新工程fresh basetemp后通过，不改全局环境。预演主体174.87s，更新226.51ms，allocated180.74MiB/reserved702MiB；冻结梯度allocated623.85MiB，缓存1.1975GB/准备23.21s。7500纯优化估计28.31min，验证/保存/监测/最终采样报告另计，不是收敛估计。

命令、公式、恢复和限制见stage5e.md，轻量来源及数值见stage5e_preflight_summary.json。默认边界0.001/内部0.0009及峰值2e-5保持候选。正式25轮追加epoch尚为0；用户本地运行，终点质量与密集改善待验证，不启动阶段6/重采/push/tag。验收后本地--no-ff合并main，实际结果及合并SHA以Git交付为准。


## 阶段5E正式运行结果与CPU审计（2026-10-10）

用户本地已完成stage5e-v1，正式运行SHA858ad6bcd205f1966c3b251b6e048b106da5eda1。真实201—225轮、60000→67500/7500更新，1200/200窗口、B4；逐轮1200唯一训练窗口、1轮warmup/24轮余弦及新增增量预热一致。继承190个AdamW状态，累计30000→37500，未重置矩。9个正式formal/clean/completed运行、7833唯一输入身份、配置/环境哈希核对通过。

200任务初始噪声配对，场景行、混合jerk、运动违规及全部12组区间重算一致；5次20任务同噪声监测也核对。独立SVD位置协调重放400条轨迹，最大误差1.56e-13m。标签运动200/200、严格舒适性721/735；训练/验证回合和几何隔离。没有新训练、采样、平滑、标签修复或改阈值。

资格82→64/200，41%→32%，条件95%道路分层/回合聚类配对区间[−12.04,−6.08]百分点；61维持、21退化、3改善、115仍失败。前四帧/内部jerk P99=266.431→274.555 / 88.667→94.267，超15帧2365→2714/29400。严格舒适性10→18/735，道路/路线8/9→7/8，碰撞均3、失败/未知均0。道路违规帧418→450，不能仅按场景少1个称整体改善。

直路40→30/67、汇入25→20/67、路口17→14/66，8—12车路口仍0/16。21新增失败均jerk单项：7只在前四帧，14内部，其中9涉及末四帧。内部超限场景90→108、末四帧60→82。新增349超限帧中329来自内部，密集仅贡献90，不能仅归于密集样本问题。

终点验证epsilon0.000438751→0.000428515（−2.33%）；221轮最低0.000416562尚无完整生成评估。20任务监测4/4/4/1/5，父为5，没有持续质量上升证据。29/7500次clip、max1.858，有限无训练失败；标量日志和clip不替代逐步辅助梯度夹角，也不锁定损失独立因果。

全流程36.32min，训练/验证/保存34.26min、更新232.69ms，Torch训练allocated185.26MiB/reserved196MiB、CPU缓存1.1975GB；采样+解码195.76ms/任务，循环60.37s。225checkpoint SHA-256 3625b5c3028e93fde90e527161f988bcba5f97cf4dd918c70daf809140eac937，200父b7dcb40f…保持。

本轮分支docs/stage5e-result-analysis，交付stage5e_results.md/json及两张已目视核对的科学图；CPU审计与绘图在artifacts/runs/stage5e-analysis-20261010-r1、stage5e-figures-20261010-r1，均记录脚本/输入身份、非正式后处理状态。来源运行SHA不回填。只改结果文档和当前状态索引，不改模型/loss/采样/评价代码；验证后本地--no-ff合并main，实际原子提交/合并以Git为准。

保留200轮候选，将225轮记录为此联合协议的负结果；不只按MSE或训练轮数继续加训，稳定与收敛尚未证明。后续若授权，优先已有中间模型配对采样或检查全时域/末端物理尾部目标，再决定新预算；本轮不自动实施。阶段5工程验收保持，阶段6未启动，无push/tag。

最终CPU审计另独立核对模型/Generator/CPU-CUDA RNG哈希。首份后处理清单误标bootstrap20261017，实际区间重算一直使用20261022；保留原目录，最终r1清单正确记录20261022，数值不变。无正式实验更改。

文档验收：validate_stage5e_results.py核对9个交付文件身份、41个本地Markdown链接、UTF-8、关键数字及实际bootstrap20261022；r1图与已目视核对的图逐字节一致。只改文档，未重复全量模型测试。结果原子提交8213b12a3837e28b2a3be39c59d0be139ca01692；状态索引另提交。实际分支docs/stage5e-result-analysis，验收后本地main --no-ff合并；最终合并SHA以Git记录为准，未推送。


## 阶段5E中间checkpoint短验证（2026-10-10）

用户“继续”授权执行已有中间模型短验证；从干净main0065b4a建立feat/intermediate-checkpoint-validation。bddbef1增加显式任务前缀目录校验，默认完整200合同保持；17643cb提交固定协议/采样配置/入口。26项相关回归3.62s通过，原200模型60任务CPU重放通过。固定原任务序列前60，三类各20，214车辆窗口、4个密集路口，seed20261015/DDIM20/eta0。

205/210/215/218/220/221六中间恢复模型各60采样，共360次，6子目录formal/clean/completed，源SHA17643cb。首次主流程在汇总JSON时因NumPy int64失败，failed原目录保留；76eab2c修复整数写入和严格--reuse-samples，独立formal报告仅复用全部360输出，零重采样。7完成正式运行、1197唯一清单输入身份、配置/环境哈希及全部配对区间核对，8模型各60初始噪声一致，480轨迹运动重放。训练SHA858ad6b、父模型及原输出不改。

同60任务资格依次200/205/210/215/218/220/221/225=26/6/13/22/19/15/20/19。215最接近，差−6.67pp区间[−14.46,+1.36]含0，不称显著更差；221虽最低epsilon仍20/60。4密集均0通过。全部失败/未知0，本队列碰撞0不代表完整200碰撞消失。部分混合P99改善不替代资格；内部与末端超限依然突出。没有发现值得替换200的证据，未自动扩大全部模型的200评估。

新360采样+解码均值202.53ms，生成循环合计107.98s、主流程至汇总失败135.87s；CPU复用报告主体9.88s，准备/哈希另计。Torch采样allocated48.49MiB/reserved50MiB。没有训练或引导梯度更新。来源CPU核验与科学图在stage5e-checkpoint-screen-audit-20261010-r1，原字符编码图保留，最终图目视核对。

交付checkpoint_screen.md、checkpoint_screen_results.md/json、新诊断入口和显式计划校验；后续只作结果文档及当前状态索引。源训练/采样/报告SHA分开保留，不回填结果提交SHA。保留200候选，当前日程/目标联合负结果未锁定原因；建议先审查全时域/末端尾部或可区分的短控制，暂不新loss、训练、数据、阶段6/push/tag。验收后本地main --no-ff合并，实际分支/原子提交/合并以Git为准。

短验证交付验收：26相关测试通过、60任务CPU指标及JSON预演通过，7完成正式来源/1197唯一输入、4交付身份及39本地链接核对；最终科学图已目视检查。代码原子提交bddbef1、17643cb、76eab2c；结果及状态分别提交。实际分支feat/intermediate-checkpoint-validation，验收后本地main --no-ff合并，最终SHA以Git记录为准，未推送。


## 阶段5E冻结目标诊断（2026-10-11整理）

用户“继续”授权短冻结检查；从干净main f6fc42d建立feat/temporal-objective-diagnosis。43eb494实现固定200/225配对目标诊断及39相关测试，b832aae提交协议；真实窗口CPU前向/目标/JSON预演通过。正式运行于2026-10-10，SHA b832aaedc00fe45ee1b0167bcfbdc1f880987316，119训练控制（47密集+三类各24普通）及完整200验证，seed20261020/t0,50,200,300/B4。模型参数、梯度缓存及源checkpoint不变，同加噪输入配对，零优化/新自由生成。

640前向主体218.79s，Torch峰值allocated60.55MiB/reserved76MiB，准备/地图/哈希另计。CPU准备的无标签推理接口和非空种子要求曾拒绝，修正后审计completed；1602唯一输入身份、3 formal/clean/completed来源、112组组件均值/计数及400保存输出独立运动重放通过。CPU绘图/写出0.53s不含身份校验/重放准备，未重跑GPU分位数。

所检查固定t的平均监督总体下降：验证t200边界/内部−9.48%/−6.31%、模仿−1.49%、原始速度内部−4.46%；内部时间200/200任务改善。最低噪声重构motion训练58→57/119、验证131→126/200；密集训练0/47、验证0→1/16只作标签重构，不是自由生成收益。平均目标确实优化，不意味着极值或整场景通过改善，也不能锁定loss/日程的独立因果。

已有完整自由输出仍82→64/200。超限帧前4=509→529，中间32=1570→1860，末4=286→325；新增349中20/290/39分别来自三段，不能只增加边界或末端权重。末4超限场景60→82。保留200候选，整段物理jerk超限尾部与原监督组合写为后续方案，未实现/加训。

交付temporal_objective_results.md/json、诊断协议/配置/模块、科学图已目视核对；源诊断/完整CPU摘要/脚本/图身份保留。更新设计/schema/README/provenance，验证后本地main --no-ff合并。实际代码提交43eb494/b832aae，结果/状态及最终合并以Git交付为准。无新训练、数据、损失/指标修改、阶段6、push或tag；后续前置是用户选择新单一监督改动及预算，不能把本轮标签重构作为自由生成验收。

## 固定从头训练工程验收（2026-10-11）

已完成：就地修改stage5a.ps1及主训练路径，6000/600/600三划分计划、75000更新、随机初始化、新AdamW、统一10轮辅助预热及10轮LR预热/190轮余弦；恢复200轮全40帧监督，新增与独立评价一致的物理jerk Huber全时域尾部。新增有界联合梯度校准、严格配置/数据身份锁、8更新正式数据Preflight，以及完整600验证后冻结600测试入口。阶段6未进入。

验证：357项全集通过，新增CUDA恢复用例通过（合计358用例）；真实小模型在CPU及CUDA上，同入口连续7与4+3更新参数、AdamW、CPU/CUDA/噪声RNG及日程完全一致，掩码、场景分母、topK、三划分泄漏和校准/测试身份漂移检查通过。PowerShell语法、Checkpoint误用拒绝和正式Pilot来源校验完成。分项范数比例没有相加作为完整梯度。

实测：已提交代码cb557d6上旧训练标签开发校准，共32私有预热+4压力更新；B16压力分配453418496 bytes、保留1000341504 bytes，梯度探针800714240 bytes；预热均值0.7655s、全辅助压力均值1.1999s。候选lambda_J=0.0003，development_only=true；不生成正式训练配置。缓存6600窗口粗估约5.65GB，新数据资源和优化耗时由Preflight复核；75000优化更新粗估约16小时，验证/保存/采集另计，非收敛预测。

标签排查：最小几何Pilot虽通过旧基础资格，但合格转弯0；34个转弯核心窗均被原运动阈值拒绝，制动jerk来自原始SUMO，非坐标错误。保留来源与阈值，新的代表性正式Pilot引用4个完全匹配短回合，补采2个路口回合，审计697候选，坐标/审计异常0，合格转弯候选训练9、验证1；精选12/6仅供小试验。正式选集各路口家族至少5%转弯（训练100、验证与测试各10）不足即停止。默认运行名stage5a-fresh，当前Pilot已可供Collect验证。

Git：功能分支feat/fixed-fresh-training；按职责提交dcf4528、e67ccee、9c20474、cb557d6、f1f5bf4、bc169fa、2f06646及CPU/CUDA恢复测试提交；验收后按规范以--no-ff合并本地main，功能分支保留。没有自动推送或创建实验里程碑tag；运行SHA保持真实来源，不改成摘要提交SHA。命令及轻量实测见stage5a.md和fresh_training_implementation_summary.json。

待完成：用户在终端按Collect→Prepare→Calibrate→Preflight→Train顺序启动。尚无新正式7200窗口、生产校准、200轮checkpoint或600/600自由生成结果。配额/覆盖失败应先审查数据报告，不能改loss掩盖缺陷；充分训练后再依据生成资格、两类jerk、舒适性、道路/路线和修正判断方案收益。


## 固定从头训练结果与完整验证（2026-10-12）

用户本地已完成新stage5a-fresh采集、准备、生产联合梯度校准、B16预演及200轮训练。源SHA dd7591418773e1576dbbe7f715f069dd504ebf3d。新数据6000/600/600，三类各2000/200/200，回合144/36/36、实际几何24/9/9互斥；64472候选，审计/转换异常0。训练转弯116、密集路口273；验证21/46，两组重叠19。测试仅核对索引/选集及来源文件哈希，未加载测试轨迹或用其指标调参。

生产校准lambda_J=0.0003，16组自然/固定低t、普通/密集及两状态完整梯度探针通过，最高尾部/噪声范数15.96%，总梯度余弦最低0.9178；不是最优性或生成质量证明。

原训练日志到70703，69375 checkpoint恢复后完成75000/200轮。1328重叠更新任务、t、损失/梯度/日程核验；每轮6000唯一训练窗口。190组AdamW75000，LR1e-6、统一预热1；原status遗留running，原训练进程已不在，原文件未改写。最终验证epsilon0.0002767385、最低197轮0.0002752209；收敛未判定。

本轮从干净main创建feat/fresh-training-results，现有stage5a Validate入口在SHA dd75914下正式执行完整600验证，无训练/引导更新。通过423/600=70.5%，条件道路分层/回合聚类95%区间[67.58,73.44]%；直路180/200、汇入152/200、路口91/200。失败/未知0；严格舒适190/2287=8.31%，整场景5/600。177不合格均有jerk超15，137为单项；物理jerk前4/内部36/末4 P99=59.737/34.118/49.406。道路14、路线21、碰撞15场景均路口，地图覆盖0；目标指标N/A。密集0/46、转弯0/21，不能称复杂场景质量稳定。

原始网络输出资格0/600；统一解码后423/600，位置修正均值0.06095m/P99 0.50030m/max3.20405m；固定60最大0.98337m不可代替完整600。独立QR/SVD重放840轨迹最大差1.71e-13m，运动、jerk、修正及汇总核对通过。600验证标签原始/解码硬运动均通过、舒适2259/2287=98.78%，正常标签解码修正max6.95e-6m；差距来自生成输出质量，未发现解码数值问题。

170/180/190/200同60任务资格40/38/39/39，舒适7/16/17/18（213车辆窗口）；170→200配对1改善/2退化，差−1.67pp区间[−10.00,+1.67]含0。保留200候选，未扩大全部中间checkpoint评价，未因MSE最低自动换模型。

RTX4070 Laptop约8GiB；75000唯一优化日志18.79h、均值0.902s，恢复重算0.314h，启动至完成21.22h含重启/验证等。恢复段Torch allocated419.6MiB/reserved692MiB、缓存5.645GB；600采样解码均值0.406s、评价85.57s、循环343.46s、Torch allocated48.49MiB/reserved50MiB。

12个主要formal/clean来源及36259唯一输入身份核验。CPU审计/标签对照/科学绘图均保存脚本、命令、配置、环境、种子、输入身份与completed状态，formal=false；图已目视检查。首份审计NumPy标量比较及首份绘图字段错误的failed来源保留，r2修正后全部完成，正式来源未改变。交付fresh_training_results.md/json，更新README和stage5a协议当前状态。本轮只改结果文档；必要的交付链接/数字/身份验证后按职责提交、--no-ff合并本地main；实际SHA以Git为准，无push/tag。

阶段5A执行已完成，模型质量局限保留。建议固定协议与200模型，后续若授权先做小规模冻结复杂场景归因，再决定阶段6或训练变更，不回到逐项续训循环。阶段6、Seal及600测试均未启动；旧41%与本次70.5%不是同数据消融，不称tail项独立收益。


## 200轮模型下一步决策与冻结归因（2026-10-13）

用户要求检查是否接受结果或继续修改训练、交通扩散论文loss，以及框架/数据问题与收敛证据。从干净main 3348338创建feat/fresh-model-diagnosis；仅保存诊断脚本、结果与文档，不改变loss/模型/正式数据。

冻结200 checkpoint e6f20667…，训练/验证各28诊断窗：三类普通各4，密集无转弯8，转弯8；标签分类仅作归因，纯噪声采样条件未加入未来。336前向重构t0/50/200/300/700/999，86新采样（含2个精确重放），复用28既有20步输出，64标签替换控制。GPU主体149.47s，allocated51017728/reserved52428800 bytes；参数哈希51190020…及grad缓存不变，optimizer_updates=0，无测试访问。

训练自由20步普通直/汇/路口4/3/3（各4），密集/转弯均0/8；验证20/50/100复杂两组均0/8。最低噪声t0训练密集0/8、转弯1/8，验证1/8、0/8，标签运动均通过。不能只归为新地图泛化或采样20步太少，也不能凭此宣布整个架构无效。

验证位置+速度未来替换控制使密集8/8、转弯3/8资格，单换位置/速度/朝向均0；提示位置—速度联合误差及转弯朝向关系。控制使用真未来，不是可部署改进。原数据正常标签仍有效，困难样本训练占比较低；密集46中24触及容量，但余22也失败，不能以截断解释全部。

末段191—195/196—200：训练总loss0.000321753/0.000316452（−1.65%）；验证epsilon均值0.000277982/0.000277700（−0.101%）；加权tail0.000082346/0.000082655（+0.375%）。末20轮clip0，LR至1e-6。近期平台不等于数学收敛，历史没有完整验证辅助曲线；原样延长不能承诺复杂质量改善。

已查原始论文CTG/CTG++、MotionDiffuser、SceneDiffuser、DiffScene，基础干净动作/状态L2、PCA与预条件、v-prediction、推理引导的区别明确；不是把风险loss都加进基础训练。保留当前200模型作基线，下一方向优先核对物理一致状态表示及SUMO标签控制反解/回放，再固定一次修改预算；未实施新loss、动力学或训练。阶段6/测试未启动。

CPU核验289来源输入，重放336重构、224原始/解码自由轨迹、64oracle控制通过；两条20步轨迹与原正式600逐元素相同。诊断/审计/图均formal=false，实际SHA保持3348338，不回填文档提交；科学图目视检查。交付fresh_model_diagnosis_results.md/json，论文原始链接、脚本和完整来源均保留，更新README索引。本轮只做结果文档验收与本地原子commit/--no-ff合并；实际提交以Git记录为准，无push/tag。


## 控制反解与状态回放：实施第一步（2026-10-13）

用户“开始逐步实现”授权先验证控制表示与现有SUMO标签，不自动进入新训练或阶段6。干净main df285eb上建立feat/action-state-replay；实现历史初始化、两维区间a/omega反解与可微回放，center/front及midpoint/endpoint选择显式化；横向速度第三通道仅为重构参照。原六维训练入口、loss、权重及数据保持原版本。

51项相关短测试通过，含CPU float64 gradcheck、车辆重排/共同坐标变换、前保险杠/中心关系、航向跨pi、填充NaN隔离、失败分母和完整清单，CUDA/autocast梯度测试实际运行。一次零值断言误把约3.6e-14浮点误差当作失败，改为数值容差后通过，未修改算法以制造零误差。

固定56个训练/验证任务（各28），不访问测试轨迹；比较四种两维配置和显式横向参照。兼容性门槛与原独立指标已落盘，不做动态放宽。提交实现和协议后再执行可追溯短回放，结果及下一步依据将在本节追加。没有优化更新、长采集/训练、push或tag。


第一步结果：主回放源3097525，独立审计源404d1bf，均formal/clean/completed。56任务/342车辆窗口、280次控制回放、336份场景指标；396个输入/输出身份核验，NumPy与Torch状态最大差1.42109e-13。56项相关不同测试通过（首批51，再新增5独立审计用例）；CUDA/autocast梯度实际检查。CPU回放/评价35.10s、审计30.70s，准备/哈希/环境另计；无新模型训练资源估计。

原标签56/56硬场景资格、严格舒适329/342。中心中点两维保留56资格但兼容42/56、转弯2/16，最大位置0.137779m/速度0.119440m/s，新增2个严格舒适失败（同时1个改善，净328/342）。中心终点两维兼容41/56、位置0.446377m；前保险杠两配置只43/56资格、兼容41/56，最大位置2.904738/2.571610m。显式横向参照兼容56/56、转弯16/16，位置1.78e-15m、舒适329/342保持；速度舍入残差7.39098e-5m/s，不是新训练生成结果。所有非转弯40任务两维兼容；密集原模型失败不能全归为这种转弯表示失配。

交付action_state_replay_results.md/json、可运行模块/配置/56任务、独立审计和协议。label回放门槛基线N/A，原汇总的0仅是未执行门槛的计数，交付明确语义，未覆写来源。两维候选研究验收为负结果，第一步工程及诊断完成，不自动换模型或训练。下一步前置为训练集控制尺度审计并冻结输出表示，再在原入口实现切换及短校准；不新采集、长训练、阶段6、测试、push或tag。

原子提交dd48d3e控制模块、b30e27f回放/协议配置、3097525协议文档、404d1bf独立审计；结果文档及状态另提交。验收后本地main按--no-ff合并，功能分支feat/action-state-replay保留，实际最终SHA以Git记录为准。运行SHA不回填为结果提交或合并SHA。

基线门槛N/A显示在7d0a9d0单独修正，10项相关短测试通过；实际源3097525/404d1bf及完成结果仍保持，未重算或回填。


## 控制回放继续检查与边界修复（2026-10-13）

用户“继续检查”授权复核第一步，不自动进入新表示训练或阶段6。从干净main d3a940d建立fix/control-replay-validation。复现NaN恢复误差误判通过、label方法覆盖基线、physical_jerk填充NaN未隔离，补充初始历史pi别名拒绝、任务几何/回合/类别绑定、车辆槽位/碰撞对一致性及失败车辆计划分母。98906f1按职责提交修复，107c210扩充独立NumPy反解、误差、逐车舒适性、门槛、jerk及分组核验。

84项相关测试通过，另2历史快照测试通过（不同用例共86），包含真实batch梯度及CUDA/autocast反向。首次强化审计在计算前拒绝两份tracked配置的字节漂移：Git切换CRLF→LF，语义相同仍不能忽略SHA。61077a9从原3097525 Git blob恢复两份原SHA/大小精确匹配的独立快照，显式参数引用，原运行目录不改；数据/非Git文件漂移继续拒绝。审计v2目前只支持float64。

快照与强化审计均formal/clean/completed，真实源61077a947312538a8286324b1423e51161b268ce。396身份、280独立反解/积分/误差及门槛、336独立场景指标、10分组摘要通过，控制/状态最大分歧1.599e-13/1.421e-13；主体35.41s，准备/哈希/环境另计。原中心中点两维资格56/56、兼容42/56、严格舒适328/342；显式横向参照兼容56/56、舒适329/342；原标签及原运行SHA保持。没有新采样、优化、采集或测试轨迹访问。

交付control_replay_review.md/json，更新协议/schema/provenance和README。来源读取工具一次自动批准超时，改用受限只读调用取得统计，无需放宽权限或用户审批。待完成仍是全训练集控制尺度审计和输出表示定稿，当前未扫描6000窗、不启动长训练/阶段6/push/tag。按职责提交文档，验收后本地main --no-ff合并并保留功能分支，最终SHA以Git记录为准。


## 完整训练控制尺度与运动覆盖审计（2026-10-14）

用户“进行下一步／继续执行／继续完成”授权完成训练专用控制尺度审计；从干净main d88172f建立feat/training-control-scale-audit。c337bfb实现完整训练反解／分布与候选网格，24e6769冻结协议，8934fb7新增尺度和未来运动覆盖测试／缓存复核。未修改原六维训练入口、网络、loss、正式选集、质量阈值或200轮checkpoint。

正式扫描实际源SHA24e67696e9fd037237336df99c82402cd00bb424，缓存复核实际源8934fb777c2d3422cb8fe675dc2ed4c8fece8a0b；两者均在2026-10-13运行，formal/clean/completed。6000训练窗（三类各2000）、144回合／24几何、21960有效车辆窗口／878400车辆时间元素，失败0；验证与测试轨迹数组读取0、优化／去噪／自由生成0。扫描保存18004实际输入身份，复核绑定8个扫描输出并重算7组分位数及全部任务／车辆分区；未再次逐个读取18004原输入。

中心／中点三维控制[a,omega,r]位置往返最大2.66e-14m、航向2.02e-15rad、NumPy与Torch状态差异7.11e-14；标签速度舍入1.34e-4m/s在0.001容差内。固定训练组绝对P99.9包络1.907716／0.472434／0.093055，向上网格尺度为2m/s²／0.5rad/s／0.1m/s；保留零点，不减均值、不裁剪。统计认证通过，训练接入未实现，旧六维checkpoint不兼容。r是横向区间速度，其时间差分不是世界坐标横向加速度。

两维无r参照仅状态容差5896/6000，104失败，最大位置0.163198m；本轮未执行完整道路／碰撞／严格舒适性门槛，不能称98.27%场景质量或与42/56兼容门槛比较。原标签物理jerk前4／内部36／末4的P99为0.433460／0.423431／0.418313m/s³，最大14.786654m/s³；负纵向区间速度点0。

覆盖复核以每车t0观测车身固定坐标系、未来含t0的范围计算。航向>=5°的车辆窗口170/21960（约0.774%），含此车辆的场景114；均来自路口。直路6328车辆窗口未来最大航向0.001576°、横向0.001553m，>=0.5m横向范围为0；汇入7496车辆窗口航向>=5°为0、横向>=0.5m为1；路口8136相应170／144。横向范围在弯道不能识别实际换道／汇入数量。既有转弯标记116采用历史+未来及>0.15rad，与本复核未来5°定义不同。覆盖偏少已确认，采集／选窗／资格过滤的原因尚未隔离。

短验证：python -m pytest tests/test_control_scale_audit.py tests/test_control_review_audit.py tests/test_control_rollout.py -q首批50项通过（3.61s）；新增缓存运动测试后python -m pytest tests/test_control_scale_review.py tests/test_control_scale_audit.py -q共10项通过（2.43s）。不同相关用例合计53，当前阶段新增10；未重新跑仓库全量测试。报告7个文档链接及17个关键来源文件身份通过，另核对两运行配置／环境／命令和8份缓存输入，共14项匹配。命令路径、UTF-8、JSON有限性、范围标记与git diff --check通过。初始默认沙箱Git工作树读取受限，按已有授权使用实现目录的受审查调用完成；未写旧目录。

资源：CPU扫描与摘要主体82.50s，采样观察RSS最高615743488 bytes（587.22MiB）；缓存复核主体0.40s，文件准备／哈希／环境另计。非连续观察的绝对进程峰值，无GPU训练显存／吞吐估计。

交付control_scale_results.md/json，更新协议、设计、schema、provenance和README；结果提交df0a84154036f0837fb73f6b233b48afb60fc335。进度提交和本地main --no-ff合并SHA见Git交付，功能分支保留；源运行SHA不回填为交付提交。没有push或tag，artifacts仍排除Git。

下一步前置：在原入口显式接入中心／中点三维控制目标、固定尺度和统一可微回放，重新定义状态辅助loss并做短联合梯度／采样链路校准。正式新训练前先沿既有回合与选集检查未来运动覆盖，决定重选、采样分布或必要补采，保持划分隔离；当前不决定补采规模、不自动启动长训练。阶段6与600测试评价继续冻结。


## 三维控制目标接入与有界校准（2026-10-14）

用户在尺度审计交付后“继续”授权接入原训练／采样入口并短验证。从干净main b3cef8a建立feat/control-target-integration。95a865e明确三维扩散／模型schema及共同回放，f42a11f接入原train、sample和checkpoint恢复，82a5d19增加组合梯度校准与16更新协议；没有另写独立训练入口。新模型随机初始化，不继承旧200模型；历史和数据仍六维，扩散未来为[a,omega,r]三维，尺度2／0.5／0.1。状态模仿／时间／道路／物理jerk尾部在共同回放状态上计算，去掉不适用的原始速度头监督。

第一次formal GPU校准因Torch2.5.1严格确定性CUDA cumsum不支持而失败，原运行保留。35dade3以固定下三角线性映射实现相同前缀和，不关闭确定性，补真实CUDA反向与CPU数值比较。29e066a扩展完整loss／日程恢复审计，9a2cc30拒绝控制表示无效的旧状态尺度选项，72a7ff3补齐失败采样实际去噪调用计数。

测试：初始438项全量通过；CUDA修复后61项相关通过；最终python -m pytest -q --basetemp artifacts/cache/pytest-control-integration-full-20261014-r2，440项全量通过（50.19s），当前阶段新增12项。包含标签往返／掩码、两融合车辆重排、三维DDIM、完整目标场景分母、组合梯度抵消、真实模型AdamW／RNG／日程恢复和控制测试数据读取前拒绝。重复用例不累计为新测试数。

初始化r2和16更新后各11探针，普通三类／密集／转弯自然t及t200，含B16密集转弯；每次67真实梯度调用。组合辅助／噪声范数最高14.62%／10.19%，总／噪声余弦最低0.992205／0.995749，组合梯度相对残差最大6.93e-7，无非有限值；未将分项比例相加。两次参数哈希不变、优化0、验证和测试标签读取0。11/11探针将触发1.0裁剪，探针未实际裁剪；真实16更新实际16/16裁剪，阈值未改。

正式短训练28训练／28验证、B4、16更新，固定验证噪声MSE1.059090→0.730960→0.670197（0／8／16更新），训练探针下降37.0%；不判断收敛。另8+8恢复，三个训练运行实际32优化更新，其中16为重复验收。原CLI正式audit-continuation比较后8更新的完整loss、LR、辅助预热、裁剪、权重、AdamW、噪声Generator及CPU/CUDA RNG、更新日程，全部逐位一致。新训练配方最大64更新，正式长训练预算和控制模型测试封存尚未实现，超范围明确报错。

3个验证任务20步DDIM，3/3均因单步角增量未严格小于pi而失败，真实60次去噪／梯度0。原始预测有限但严重失真：a最大约714–842m/s²，omega139–193rad/s，r31–36m/s。没有可评价的六维道路／碰撞／舒适性轨迹，未知指标不写成0违规。失败分母、原始控制和初始噪声保存；补调用计数后的r2与r1逐元素相同，是同3任务复核而非6独立任务。未放宽保护、裁剪或重采制造成功。

资源实测RTX4070 Laptop约8GiB，Torch2.5.1／CUDA12.4；B4优化均值0.323s、约12.39窗/s，allocated268956160 bytes／reserved316669952 bytes，缓存47900272 bytes；训练验证保存循环11.48s，准备／哈希另计。B16梯度探针allocated700700160 bytes，未运行完整B16优化器训练，不据此承诺正式资源。校准主体8.38／6.77s。

独立CPU复核56标签窗／342车辆窗口，位置恢复max1.23e-7m、速度7.39e-5m/s、sin/cos分量3.15e-8，NumPy与Torch状态差1.42e-13，均通过。脚本及365实际输入身份进入formal清单；交付再次逐个重哈希365输入、7关键摘要来源及11文档链接通过。所有实际来源保留配置／命令／环境／种子／状态，包含初始CUDA失败与两份采样失败，不覆盖来源。

真实运行SHA：初始化r2、完整16和8+8训练35dade30f087c8234cdc7da3fc792dc5c48d25d7；恢复审计／更新后校准29e066a5bec427eb3b42941a1a4b2838747981e8；采样r2和独立CPU复核72a7ff32ed6b9a7d0edc62db9d71120d81cc2a36。均从formal／clean代码启动，状态如实保留。交付control_target_results.md/json及control_target_integration.md，更新README、设计、schema和provenance；结果提交8a754236509f1141d2a38163b06a51ead866ff0b，进度提交／本地main --no-ff合并SHA见Git，功能分支保留，运行SHA不回填。

当前结论：表示与有界训练／恢复检查完成，尚无合格新控制生成模型。下一步优先沿既有回合、候选窗和过滤理由追查未来运动覆盖；覆盖及采样分布冻结后，再制定一次固定新控制正式训练协议与校准／预演。当前不扩大训练、采集或修改质量阈值，不以16更新自由采样失败否定充分训练后的方法。阶段6、600测试、push及tag未启动。
