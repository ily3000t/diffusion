# 控制反解与状态回放：第一步诊断

本步骤承接200轮冻结归因，只验证新的控制表示与现有SUMO标签的兼容性。原六维状态模型、loss、训练入口、checkpoint和数据文件保持原版本。未开始控制扩散训练、阶段6或测试集评价。

## 物理和数据语义

所有位置和航向在共同固定局部坐标系。标签二维速度是几何中心的后向位置差分，属于采样区间平均速度；SUMO报告的瞬时标量速度不作为初始控制状态。

候选参考点为几何中心或前保险杠。若使用前保险杠，F=p+(L/2)d(theta)，L是t0已知车辆长度。初始姿态与初始参考点速度只从最后两个已观测历史姿态得到，不使用未来标签或真实退出时刻。dt=0.1秒。

每步控制为a、omega：

```
theta[k] = theta[k-1] + dt*omega[k]
u[k] = u[k-1] + dt*a[k]
basis[k] = theta[k-1] + dt*omega[k]/2   # midpoint
         = theta[k]                    # endpoint
R[k] = R[k-1] + dt*u[k]*d(basis[k])
p[k] = R[k] - offset*d(theta[k])        # offset=0或L/2
v[k] = (p[k]-p[k-1])/dt
```

a是纵向参考点区间速度的差分，omega是采样航向差分，并非真实连续控制输入。该离散模型保证自身定义的状态一致性，不保证道路、动力学或舒适性。速度允许带符号，全部控制不裁剪、不平滑。单步航向增量达到pi拒绝，避免采样角度别名。计算显式使用float64；半精度输入不会导致半精度积分。

标签反解先将未来相对位置加回各车p0，再计算参考点位移。航向用最短圆弧差分；u为区间位移在basis方向的投影，横向投影r独立记录。两维候选舍弃r，不隐藏误差。

第五个对照显式使用三维[a,omega,r]，将dt*r*n(basis)加回参考点位移。它用于检查舍弃横向运动是否造成回放误差，是已知未来的代数重构参照，不是无侧滑车辆模型、生成结果或下一版生产方案。现有标签保留，原速度与位置差分的浮点残差另外记录。

## 固定短协议

configs/experiments/action_state_replay_tasks.json固定此前诊断的56窗：训练/验证各28，普通直路、汇入、路口各4，密集无转弯与转弯各8。困难组根据离线标签分层，仅作归因，不能作为推理输入。不是随机总体估计；不同组车辆数不同，不能直接作因果消融。

只读取这56窗。训练和验证按已冻结的独立几何/回合划分；不加载测试轨迹。四个两维候选为center_midpoint、center_endpoint、front_midpoint、front_endpoint，显式三维参照为center_midpoint_lateral。各方法共用同一标签、初始状态与原独立指标。

在运行前固定兼容性检查：
- 最大中心位置恢复误差<=0.05m；
- 最大中心速度恢复误差<=0.05m/s；
- 最大航向角误差<=1e-5rad；
- 不丢失原场景资格；不增加运动超限计数、道路/路线/地图越界计数；
- 原来无碰撞的同一车辆对不能产生新碰撞或未知结果；
- 同一车辆原严格舒适性通过时，回放不能新增失败或未知。

这是一组工程兼容性候选阈值，不是行为真实性校准。运动硬阈值、舒适性和几何容差沿用已有600验证设置。失败、未知和不兼容任务保留分母，不放宽阈值制造通过。每个车辆的舒适性单独对照，不能只比较通过车辆总数。

位置、速度、朝向恢复误差、横向残差、a/omega分布、交界前4帧/内部36帧/末4帧物理jerk、道路、碰撞、资格、时间分别保存。末4帧属于内部36帧，不能把两者当作互斥分组相加。jerk直接来自三阶位置差分，交界使用最后三帧已观测位置。

## 运行与追溯

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv-model\Scripts\python.exe' -m sumodiff.decoding.control_replay --config configs/experiments/action_state_replay.yaml --output artifacts/runs/action-state-replay-r1 --formal
```

运行前提交代码并确认干净。重复时更换输出目录，禁止覆盖。默认CPU；--device cuda是同一算子的GPU诊断，不能将其成本当作完整网络训练显存或吞吐量。

输出包含manifest、完整生效配置、命令、环境、数据与任务文件身份、状态、metrics及results.json。逐窗NPZ分别保存原标签、控制量、横向残差、回放轨迹和物理jerk；逐窗JSON保存独立指标和兼容性原因。无checkpoint，没有模型前向或优化更新。

当前方法受交通扩散论文控制—状态rollout设计启发，公式按本项目区间速度和SUMO参考点独立实现，不称CTG/CTG++复现。若两维回放不兼容，先记录失配原因与范围；三维参照通过也不足以直接决定训练目标。新的表示、归一化、预测目标与训练预算需作为下一步单独固定。


## 已完成的短回放

实际结果及资源见[action_state_replay_results.md](action_state_replay_results.md)。独立审计复现命令：

```powershell
& '.\.venv-model\Scripts\python.exe' -m sumodiff.decoding.replay_audit --source artifacts/runs/action-state-replay-r1 --output artifacts/runs/action-state-replay-audit-r1 --formal
```

审计逐项核对输入/输出身份、源标签、NumPy逐时刻积分及原场景指标，不调用Torch控制回放；原结果不覆写。


## 继续检查与审计v2

完整检查见[control_replay_review.md](control_replay_review.md)。审计v2新增独立控制反解、误差/门槛、逐车舒适性、jerk及任务/分组核验，目前只支持float64控制计算。遇到tracked历史输入字节变化时，可用input_snapshots从原Git提交恢复精确匹配SHA和大小的独立快照，并显式传入--input-snapshots；没有快照时继续拒绝漂移，原输入不改。
