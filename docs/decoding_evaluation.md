# 阶段3：统一解码与独立指标

本阶段只实现物理状态恢复、指标和短数值验收。输入仍为物理单位；不接受旧checkpoint，不实现归一化、扩散、引导或模型训练。阈值是预先给定的工程候选，不代表已完成真实行为校准。

## 解码合同

`decode_future(raw_future, initial_positions, initial_heading, agent_mask, DecoderConfig)` 支持 `[...,N,T,6]`，本任务T=40。未来前两维为各车相对t0的位移，方向均在共同固定局部坐标系。输出 `states` 是共同局部系中的绝对位置、位置后向差分速度、单位sin/cos向量。`positions_with_start` 单独含固定观测起点。

解码器不接收future_mask，完整生成时域不允许依据真实退出标签截断。标签数值验收只对全部选车未来完整的窗口重构，其余窗口保留原始指标并标注跳过，不作为推理任务筛选规则。

令D为单位阵减下一条副对角，q0=0固定，alpha=w_pos/unit_pos²、beta=w_vel/unit_vel²，求解

    min_q alpha ||q-q_raw||² + beta ||Dq/dt-v_raw||²
    (alpha I + beta DᵀD/dt²)q = alpha q_raw + beta Dᵀv_raw/dt

初始配置dt=0.1、两权重及两单位尺度均为1。单位尺度只是协调目标的无量纲化单位，不是阶段5的模型归一化尺度。求解关闭autocast并使用float64，支持明确指定float32，拒绝半精度求解；输出保留求解精度。未来位置相加前先提升精度，原始标签delta保持原存储值。

近零方向向量继承上一个有效方向；第一次继承观测t0方向，每个回退计数。非零方向归一化，近零原始方向的角度修正N/A。有效车辆非有限值报错；padding先屏蔽再计算，不能传播到解算或梯度。

初始方向须来自当前观测。优先接受显式initial_heading；如采用history末点，history_mask末点必须有效。缺失时明确报错，不从未来推断。此要求比位置可用更严格：刚进入、缺前一采样的车辆仍可能拥有当前方向，后续适配器应显式提供该观测。

解码只保证所定义的位置与区间平均速度关系，不保证运动约束、道路或行为质量。保存原始、解码状态及位置/速度/方向修正、回退与原始一致性误差；本阶段没有引导轨迹，以not_applicable注明。

## 碰撞与未知结果

独立有向矩形SAT检验四条车身轴，接触也计事件；签名分离裕度不是碰撞概率或欧氏距离。恒定朝向区间采用精确扫掠SAT，能检出端点未碰撞但帧间穿越的情况。转向区间采用最短角度插值、递归SAT和保守平移/旋转运动界。时间分辨率、递归深度或调用预算不足时输出未知，不能记为无碰撞；旋转情况的检测时刻只是已证实接触的采样时刻，非精确首次接触。

评估含t0观测，初始已接触单独报告且不能成为新目标事件。缺失标签不跨空洞连接；未来仅部分观察且未发现接触时，整时域结果未知。没有车辆对/攻击任务时输出null（显示N/A）。默认标签验收不设攻击角色，目标事件率N/A。

非目标安全覆盖指定攻击—目标对以外全部车辆对。场景汇总采用预定适用任务分母，失败、未知、不合格结果均保留；confirmed_event_rate是确认事件的下界，并伴随confirmed_negative和unknown_or_failed数量，不将未知补成已知负例。

## 道路、路线与运动

道路、逐车合法路线走廊、地图栅格物理范围分别检查完整有向车身多边形，包括凹区和洞；允许合法换道，不能只检四角。向量几何不依赖CNN分辨率。道路检查只覆盖提供的未来帧，未实现连续道路穿越证明；帧间碰撞另行处理。

默认RoadConfig.geometry_repair=strict，无效几何报错。显式stage3配置使用bounded_make_valid：仅允许linework保留所有组成，面积变化≤1e-6m²、Hausdorff距离≤1e-6m、结果有效且正面积；逐项记录原始原因、类型、组成、变化量。未删除或写回原图，超限报错。保留的塌缩线只会按公开的1e-6m覆盖容差形成极细区域；不能覆盖正常大小车身。此规则处理坐标浮点变换的拓扑问题，不作为地图补全或质量调参。

速度、加速度、jerk用几何中心位置连续后向差分；纵横分量依当前朝向，航向速度矛盾只对速度≥0.5m/s适用。原始预测速度与几何速度误差单报。历史—未来第一步单报位移、速度变化、加速度、加速度变化、jerk和方向变化；缺失必要历史输出N/A。舒适性报告实际评估点/车数及完整分量点数，观测通过率不表示完整缺失时域通过。

候选硬阈值：速度35m/s，纵横加速度各6m/s²，jerk15m/s³，yaw rate3rad/s，移动方向偏差pi/4，heading norm误差0.1，速度一致性误差0.5m/s。另报舒适性候选：纵横加速度各3m/s²、jerk5m/s³。均在配置显式保存，测试标签不用于放宽阈值。

有效目标事件是新目标接触且同时满足候选硬运动/一致性、车身道路/路线/范围及非目标安全条件。这是布尔资格，不是加权综合评分；所有组成仍独立报告。已知任一不合格为false，其余缺失为null。

## 短验收与追溯

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
$env:SUMO_HOME = 'E:\Program Files\sumo-1.22.0'
& '.\.venv-model\Scripts\python.exe' -m pytest -q --basetemp artifacts/cache/pytest-stage3-final
& '.\.venv-model\Scripts\python.exe' -m sumodiff.evaluation labels --dataset artifacts/processed/stage2-accepted-20261004 --config configs/evaluation/stage3.yaml --output artifacts/runs/stage3-accepted-20261005 --formal
& '.\.venv-model\Scripts\python.exe' scripts/audit_stage3_evaluation.py --dataset artifacts/processed/stage2-accepted-20261004 --run artifacts/runs/stage3-accepted-20261005 --output artifacts/runs/stage3-accepted-audit-20261005 --formal
```

只在已提交且干净时正式启动；重跑用新输出目录，禁止覆盖。labels保存标准六文件、numeric_probe.json、trajectory_index.json、validation_summary.json及逐窗stages.npz/metrics.json；全部输出位于忽略目录。保存全部实际窗口输入哈希、配置、种子、环境与运行SHA，无checkpoint时为null。

独立审计逐文件核对身份，使用NumPy增广最小二乘SVD校验torch正常方程解，用独立车身顶点和Shapely验证道路/路线/覆盖，用直接三阶位置差分验证jerk；标签无目标任务时核对N/A。合成测试另外验证匀速、停车、转弯、扫掠碰撞、未知预算、洞、边界跳跃、掩码和梯度。

GPU probe是预热后的decoder-only B=2,N=12,T=40前后向；不含网络、DDIM、采样或驱动显存。CPU RSS含PyTorch/CUDA上下文。不能据此推断阶段4网络batch size、吞吐量或正式训练时间。
