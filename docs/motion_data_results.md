# 运动平衡数据协议与预演验收

新数据配方和本地终端入口已完成，小规模预演通过。正式216回合采集、6000/600/600选集与新模型训练尚未开始；原正式数据及checkpoint保持。

## 已冻结的完整协议

同一道路/划分内旧参考与交互参考各50%；直路换道窗口至少20%，汇入退出20%及主路换道10%，路口已观测转弯10%。每唯一标识最多3窗，同回合同t0最多一个视图。运动唯一数量、几何/回合覆盖、原质量筛选和缺额规则见[协议](motion_data_protocol.md)。这些是预声明的首批目标，不保证有足够合格候选。

复用原stage5a collect/prepare及PowerShell脚本，使用DataProtocol Motion显式分流；旧配方保持。直路SL2015、汇入80m连接/10m入口对齐与方向权限、路口IDM。完整计划预分配144/36/36回合和8/3/3几何/道路，先划分再切窗；test只按冻结配方构造，不参与参数选择。

## 实际预演

6/6训练专用短回合正常排空，30s正常需求、240s上限。固定24个t0×两参考/回合，共288计划且可构造记录；253标签完整并进行raw64质量检查，35不完整保留。另13个完整单车窗无适用车辆对，20个路口窗有原始运动超限；重叠拒绝理由均保存，不作为0风险/通过。最终合格池220，选取24个，无平滑、裁剪或门槛放宽。

| 道路 | 合格池/计划96 | 最终窗口 | 旧/交互参考 | 入选唯一运动标识 |
| --- | --- | --- | --- | --- |
| 直路 | 77/96 | 8 | 4/4 | 换道1 |
| 汇入 | 73/96 | 8 | 4/4 | 主路换道1、退出3 |
| 路口 | 70/96 | 8 | 4/4 | 转弯车辆动作2 |

三类最终各覆盖两个几何/两个回合；本次各事件复用最大1。路口入选转弯来自1个几何，不能替代完整协议要求的每运动3几何/4回合。小预演的运动下限是每类别至少1窗/1唯一标识，与完整配额不同，不能将24窗当作已完成正式数据扩容。

独立复核24窗口/7次入选事件证据，核对原始change日志、道路转换、转弯航向及完整合法内部连接；首版复核只看首段via导致失败，原脚本/失败状态保留。r2沿完整via链通过，未修改原始数据、生产规则或配额。

重新计算5246个有效车辆时刻的原始前保险杠/航向→中心/固定局部坐标/后向速度，最大存储误差为位置3.81e-06m、速度9.45e-07m/s、朝向2.97e-08；均在原容差内。新模型适配器可消费全部24窗口，未来掩码/运动标签未进入条件。正式collect的pilot准入检查也实际通过。

## 资源及来源

总仿真808.2s，车辆记录70971条，raw frames共41424986bytes。质量审计主体107.89s，准备含24窗物化记录114.95s，整个Pilot记录163.58s；复核r2主体2.14s，准备/环境另计。
质量审计回合间观察RSS最高451.55MiB，非连续绝对峰值。仅保存24窗口数组，数据集15198106bytes；288条质量记录与原始回合保留，可重建候选。未训练或测GPU吞吐，稀疏预演不能线性保证完整数据处理时间。

521项全量测试通过（57.18s），本阶段新增15项；59项相关子集不重复累计。PowerShell语法解析通过，实际Pilot经原PowerShell脚本启动。测试覆盖划分/种子、旧接口、运动证据、唯一事件/重复视图、缺额保留、冻结协议、内容摘要和来源预检。

实际运行SHA均为f57bab438b979656f8abf43ef9ded60db4bfddda，formal/clean。Pilot/preparation/review_r2 completed，review_r1 failed按实保留；所有配置/命令/种子/环境/身份入清单，交付不回填运行SHA。预演数据ID为sha256:388aa78c044dd18c45881792cbaae1450ac725eaf53a4f1ec3af6d2e4236529e，仅作为短验证集，不替换正式数据。

- pilot：[stage5a-motion-v1-pilot](E:/diffusion_new/artifacts/runs/stage5a-motion-v1-pilot/manifest.json)，completed。

- preparation：[preparation](E:/diffusion_new/artifacts/runs/stage5a-motion-v1-pilot/preparation/manifest.json)，completed。

- review_r1：[motion-data-review-20261015-r1](E:/diffusion_new/artifacts/runs/motion-data-review-20261015-r1/manifest.json)，failed。

- review_r2：[motion-data-review-20261015-r2](E:/diffusion_new/artifacts/runs/motion-data-review-20261015-r2/manifest.json)，completed。

## 现在由本地终端启动

已有预演通过，可直接运行以下长任务；本轮未自动执行：

~~~powershell
Set-Location -LiteralPath 'E:\diffusion_new'
.\scripts\stage5a.ps1 -Action Collect -DataProtocol Motion -Name stage5a-motion-v1
.\scripts\stage5a.ps1 -Action Prepare -DataProtocol Motion -Name stage5a-motion-v1
~~~

采集中断可用-ReuseCompleted及新-RunSuffix复用已完成回合。Prepare也属于长任务，目前没有增量质量缓存恢复；若中断，用新输出目录从原sources重新准备，不需要重新采集：

~~~powershell
$env:PYTHONPATH='E:\diffusion_new\src'
& '.\.venv-model\Scripts\python.exe' -m sumodiff.experiments.stage5a prepare --config configs/experiments/motion_data.yaml --sources artifacts/raw/stage5a-motion-v1/sources.json --output artifacts/processed/stage5a-motion-v1-retry1 --formal
~~~

完整准备必须通过全部质量/运动/唯一事件/几何/配额检查，缺额时停止并保留报告，不补重复窗或静默降级。随后再审计新训练集控制尺度及完整组合梯度，扩展一次固定正式训练协议；原控制入口仍限64更新，不能直接运行旧六维200轮命令。阶段6、模型验证/测试、push及tag未启动。
