# 阶段2：固定坐标、地图与窗口数据

本阶段只处理正常SUMO回合，不训练模型。入口为 `python -m sumodiff.data process`；使用项目 `.venv-sumo`，不依赖旧目录或旧checkpoint。

## 快速复现

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
$env:SUMO_HOME = 'E:\Program Files\sumo-1.22.0'
# 以下三个短回合只需生成一次；重跑改输出目录并同步清单。
& '.\.venv-sumo\Scripts\python.exe' -m sumodiff.simulation collect --config configs/scenarios/straight_c.yaml --output artifacts/raw/stage2-supplement-20261004/straight_c --seed 41 --formal
& '.\.venv-sumo\Scripts\python.exe' -m sumodiff.simulation collect --config configs/scenarios/ramp_c.yaml --output artifacts/raw/stage2-supplement-20261004/ramp_c --seed 42 --formal
& '.\.venv-sumo\Scripts\python.exe' -m sumodiff.simulation collect --config configs/scenarios/intersection_c.yaml --output artifacts/raw/stage2-supplement-20261004/intersection_c --seed 43 --formal
& '.\.venv-sumo\Scripts\python.exe' -m sumodiff.data process --config configs/data/window.yaml --sources configs/data/sources_stage2.json --output artifacts/processed/my-windows --formal
& '.\.venv-sumo\Scripts\python.exe' scripts/audit_window_dataset.py --dataset artifacts/processed/my-windows --sources configs/data/sources_stage2.json --output artifacts/runs/my-window-audit --formal
```

上述命令还需阶段1的六个正常回合与三个诊断回合，生成方式见progress和sumo_collection。切窗前必须提供完整清单，不扫描所有历史开发目录。每个命令检查退出码，失败输出保留，不能继续当作验收成功。正式运行需已提交、干净工作区；输出拒绝覆盖。

新机器安装可在项目虚拟环境中使用 `pip install -e ".[dev,sumo,data]"`；SUMO及其匹配客户端来自已核对的安装目录。当前实际版本见environment。独立切窗不调用SUMO服务；独立通行规则审计需要SUMO_HOME自带sumolib。

## 来源、划分和完整性

`sources_stage2.json`为显式清单：A几何归train、B归validation、C归test，每类各一个回合。三个诊断/失败回合保留在源清单并报告排除理由。清单相对路径以清单文件目录为基准。

先验证所有回合及几何的分组归属，再构造窗口。同几何的不同种子、流量不能跨集合；重复内容拒绝。回合唯一键采用原始data_id内容哈希，原始目录名作为source_episode_id保留，防止两个不同目录同名产生ID冲突。重叠窗口继承回合划分，不能随机拆分。

核对回合声明的原始输出和场景文件SHA-256/大小，重新计算语义geometry_id和data_id；轨迹时间必须在0.1s整数tick网格，重复ID、无效路线、非有限位置等报错。失败/诊断回合明确记录原因，不静默混入正常数据。

这是工程验收清单，每集合仅3个回合、每类1种几何，不能代表正式数据规模或统计充分性。正式研究需要在后续阶段建立足量、固定的训练/验证/测试清单；不能将本次已查看的工程数据宣传为未经检查的盲测基准。

## 坐标与窗口

SUMO原始位置为前保险杠，导航角从正北顺时针计。转换为几何中心和数学角度，再在参考车t0的固定原点、固定朝向中表达全部车辆、地图和路线。raw不改写。21点历史为t0−2.0s…t0，40点未来为t0+0.1s…t0+4.0s。

速度统一为中心位置后向差分，需要额外t0−2.1s的原始点。即便当前有位置，若前一采样缺失，该六维状态点整体置零且mask=false；不使用未来插值补历史。原始SUMO标量速度不直接用作二维速度。

默认t0在1s网格，从3s开始（首个满足22原始采样要求的整秒），到回合末尾。即使未来40帧不足仍保存。缺失原因区分回合范围之外、车辆缺失和速度前点缺失；它们只进入标签质量信息。

默认参考车为t0当前ID中字典序首车，槽位0；其他车辆按t0中心距离和ID稳定排序，半径80m、最多12辆。只使用t0可用状态和已知计划路线。记录当前总数、半径排除及容量排除；不按未来轨迹完整度替换车辆。该简单参考选择不是交通任务均衡采样策略，后续正式任务可以采用经过版本化的其他策略。

未来学习位置是各车相对于自己t0位置的局部偏移，不是逐步位移，也不是各车自己的旋转坐标。恢复共同局部位置需加`initial_positions[slot]`，恢复世界位置再用该窗口frame的逆变换。

## 地图和路线

固定栅格范围为局部[-192,192]m×[-192,192]m，256×256，像素1.5m，通道依次为可通行区域、道路外边界、车道中心线。行从+y到−y，列从−x到+x，连续像素坐标以像素中心为整数（栅格绘制是近似表达，不能代替车身约束）。

范围事先根据80m选择半径+25m/s×4s+8m车身余量=188m检查。实际速度与尺寸的包络核对仅使用train/validation；test只处理和报告，不据其结果重调范围。25m/s是覆盖设计包络，不是动力学可行阈值。当前检查未使用或拟合50m/20m/s归一化候选，状态仍为物理单位。

车道折线保留全路网的已知向量地图，每条64点，8维为[x,y,方向x,方向y,宽度,限速,内部车道标记,edge priority]；精确原始折线另存。已知全路网向量信息可以在CNN裁剪范围之外，out_of_map指固定栅格覆盖不足，不能等同off_road。

车道通行区域采用原始分段线半宽buffer（平端、圆角，quad_segs=16），道路几何再与合法junction shape并集；这是一套明确的向量几何约定，不宣称与SUMO内部车身执行完全等价。路线走廊包含t0完整已知计划路线上的全部乘用车车道，允许同edge内换道，并追踪合法内部连接链（包括等待点后的第二段内部车道）；不加入其他出口。当前路线不截去已行驶的部分，route_index_at_t0另存供后续方向和通行规则判断。

直路零几何长度内部车道仍保留连接、SUMO声明length和明确zero_geometry_length标记；方向从合法后继车道获取，不虚造道路长度。普通junction的request索引与内部等待junction分开，foes/response从右读bit0。精确地图保留连接、优先/冲突关系和内部禁止占用信息。独立审计以SUMO自带解析器交叉核对索引和冲突关系。

质量摘要分别记录未来中心越过栅格、车身四角越过栅格、中心越过向量道路。后者只是数据QA中心检查，完整车身道路/路线违规与运动质量将在阶段3实现。

## 存储与读取

运行目录保存六个标准追溯文件，另有dataset_manifest、split_audit、coverage_calibration、quality、skipped_reference_ticks和windows索引。每窗有：

| 文件 | 内容 |
|---|---|
| conditioning.npz | 历史、agent/history mask、属性、t0锚点、栅格、折线、折线mask、连接邻接、逐车路线mask |
| input.json | t0、ID/槽位、固定frame、t0已知路线与类型、范围及排除数 |
| map.json | 同一局部系的精确道路、路线走廊、原始折线和通行规则 |
| targets.npz | future和future_mask，仅训练/标签使用 |
| labels.json | 完整度、缺失原因、核心训练资格、未来数据QA，仅标签使用 |

```python
from sumodiff.data.dataset import WindowDataset, collate_numpy
all_tasks = WindowDataset('artifacts/processed/my-windows', split='test')
request = all_tasks.inference(0)  # 不打开targets或labels；不含future_mask
training = WindowDataset('artifacts/processed/my-windows', split='train', core_only=True)
batch = collate_numpy([training[0], training[1]])
```

默认保存/读取全部窗口；core_only只能用于训练，使用core_only调用inference会报错。核心资格要求选定有效车辆的21历史与40未来均完整；不完整数据仍在索引中。不能依据标签完整度挑选推理任务或把真实退出时刻传给生成器。

读取默认核对窗口文件哈希，拒绝pickle；支持重定位数据目录。NumPy collate给不同路网的车道维补零、mask=false，车辆维不绑定栅格通道。模型及torch数据加载在后续阶段接入。

## 验收范围

合成测试覆盖坐标往返、时间边界、未来更改不影响条件、第一历史前点、缺失、退出、12车容量、栅格覆盖与offroad区别、同几何隔离、同名回合、哈希损坏和不读取标签的推理。

独立真实数据审计逐状态从原始前保险杠/导航角重算坐标和后向速度，核对mask、padding、t0属性及ID、路线、map局部变换、向量几何往返、完整度和数量，再与SUMO SDK核对通行规则。审计自身也记录真实SHA和failed/completed状态。

本阶段不进行碰撞事件评估、轨迹一致性解码、模型显存实测或训练，不创建实验里程碑tag。下一阶段只实现统一解码器和独立指标。
