# 完整训练集控制尺度审计

本阶段只扫描已冻结的6000训练核心窗口。原200轮模型、训练入口、loss及数据保持；不读验证/测试轨迹，不新采集或训练。三类道路各2000，几何与回合来自既有隔离划分。

## 审计量与表示

采用几何中心、区间中点航向、float64的三维区间控制[a,omega,r]。a是纵向区间速度差分(m/s²)，omega是航向差分(rad/s)，r是横向区间速度(m/s)。r保留标签的横向残差；不是无侧滑车辆模型。

初始化只使用最后两个历史姿态。训练未来标签反解用于构建候选目标，不进入模型推理条件。二维速度仍是中心位置的后向差分，原标签速度误差单独记录。

逐窗执行完整21历史/40未来和有效mask检查；Torch和独立NumPy控制反解/回放交叉核对。数值门槛固定为位置/航向/独立状态/控制差异1e-8，原float32速度舍入容差0.001m/s。失败保留6000分母，有失败就不认证全部训练统计。

保存每个通道的signed mean/std/range、绝对值P50/P90/P95/P99/P99.9/max。附加观察r的时间差分，第一帧包含已观测的历史横向区间速度。该差分是旋转基底分量的变化，不直接称为世界坐标横向加速度。记录初始及未来纵向区间速度、负速度点、标签三阶位置jerk。

两维去横向参照仅检查位置/速度恢复0.05m、0.05m/s候选容差，不调用全部道路/碰撞/舒适性门槛。因此不能把本审计的state_tolerance_pass写成整场景资格或与此前42/56直接比较。

## 固定尺度候选规则

每个通道使用训练组绝对值P99.9的最大值，再向上取预先声明的候选网格。参与组：全体、三类道路、转弯窗口、密集路口。每组至少8窗，不足或超过网格就明确不可选，不扩大网格、删除坏点或裁剪标签制造通过。

网格：
- a：1、2、3、5、8、10m/s²。
- omega：0.05、0.1、0.2、0.3、0.5、1rad/s。
- r：0.02、0.05、0.1、0.2、0.3、0.5、1m/s。

规则保留控制零点，仅除固定物理尺度，不减均值、不裁剪。报告归一化后绝对值超1/3/5的比例。分位数是有效车辆—时间元素加权；重叠窗口存在相关性，不能解释成独立真实交通总体分位数或概率。分组尾部规则避免大量普通直行淹没转弯运动。

转弯来自既有选集离线标签标记，密集定义为路口当前选定车辆>=8，另报告密集无转弯子组。这些分层字段只供数据审计；组有重叠，不能相加得到总数。

候选尺度保存data_id、控制定义和来源。统计认证不等于新训练目标已实现或模型质量通过，旧六维checkpoint明确不兼容。最终目标、辅助loss和采样质量要在后续工程/实验中验证。

## 可复现入口

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv-model\Scripts\python.exe' -m sumodiff.decoding.control_scale_audit --config configs/experiments/control_scale_audit.yaml --output artifacts/runs/training-control-scales-r1 --formal
```

先提交代码、保持干净、使用新输出目录。入口验证dataset ID、窗口数、选集哈希、回合/几何/道路声明、三类配额。只加载conditioning.npz中的history/history_mask/initial_positions/attributes/agent_mask及targets.npz标签；不解压地图栅格或读map.json。

完整配置、命令、代码SHA、种子、环境、选集与所有实际文件身份进入运行清单。tasks.json固定6000计划，window_results.json保留逐窗结果/失败及有效车辆偏移；active_controls.npz保存压缩有效控制/横向分量变化/速度/标签jerk。results.json、metrics.json及candidate_normalization.json保存汇总。候选归一化没有训练消费入口，不能直接启动旧模型训练。

资源只测CPU审计的主体时间与采样观察的进程RSS，未测新网络训练显存或吞吐量。


## 已完成结果与缓存复核

正式6000窗口扫描与缓存复核已完成，见[结果](control_scale_results.md)和[轻量摘要](control_scale_results.json)。候选尺度2/0.5/0.1来自训练分组包络；统计认证不等于生成模型验收。未来运动覆盖偏少，正式新训练前需查明来源。

复核必须引用已完成、无失败、formal且clean的扫描；重新执行使用新输出目录：

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv-model\Scripts\python.exe' -m sumodiff.decoding.control_scale_review --source artifacts/runs/training-control-scales-20261013-r1 --output artifacts/runs/training-control-scale-review-reproduce-r1 --formal
```

复核检查全部任务顺序与有效车辆数组分区、独立重算分组P99.9与候选网格，并量化每车t0固定车身系中的未来航向／横向范围。5°及0.5m用于覆盖描述，不新增训练资格条件，不检测真实换道事件。来源由8个扫描文件身份绑定；未再次逐个读取全部原始训练输入。保持已提交干净代码，保存实际运行SHA，不改写旧运行目录。
