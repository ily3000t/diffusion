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
