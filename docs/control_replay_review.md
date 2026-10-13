# 控制回放继续检查（2026-10-13）

继续检查完成。发现并修复了验收、填充及追溯的边界问题；强化审计重新核对原56窗，原数值与结论保持。当前检查没有改善或重新评价生成模型，不启动新训练或阶段6。

## 发现与修复

1. compatibility_gate接收NaN恢复误差时，旧比较会因NaN>阈值为False而误判通过。现在拒绝非有限、负数误差及无效阈值；未知场景资格不能通过比较。
2. label被允许作为方法名，会覆盖原标签基线。现在明确拒绝这个保留名称。
3. 独立physical_jerk辅助函数没有在差分前隔离无效槽位的NaN。现在先屏蔽填充，再计算物理差分，活跃位置异常明确报错。生产控制回放原有掩码处理不受该缺陷影响。
4. 初始两个历史航向相差接近pi时，旧初始化采用最短弧但未明确拒绝角度别名。现在与未来反解一致地拒绝。
5. 固定任务的回合、几何和道路类别声明现在与真实索引/input.json绑定；比较不能遗漏车辆槽位、车辆对或重复车辆对。
6. 汇总增加planned_active_agents和strict_comfort_confirmed_pass_fraction_all_planned。失败方法的已计划车辆仍在分母，无法确定数量时报告N/A及unknown_planned_agent_tasks。不完整未来的“已观察舒适性”与完整时域严格舒适性明确分开，原字段保留兼容。

这些问题在边界回归中可复现。原56窗的恢复误差有限、方法名无冲突、填充原为零、任务身份和历史航向合法，因此没有证据说明上述边界问题改变了原报告。

## 强化审计

原审计主要核对状态积分与场景指标。本次增加独立NumPy逐时刻控制反解，并独立重算误差、逐车舒适性、门槛、jerk、分组通过数、完整任务顺序和JSON/NPZ一致性。反解不调用Torch生产反解；验收不调用生产compatibility_gate。

| 核验 | 实际结果 |
|---|---:|
| 源输入/输出身份 | 396 |
| 独立控制反解 | 280 |
| 独立状态积分 | 280 |
| 独立误差及门槛判断 | 280 |
| 独立场景指标 | 336 |
| 分组摘要 | 10 |
| 控制量最大分歧 | 1.599e-13 |
| 状态最大分歧 | 1.421e-13 |

86项不同相关短测试通过：84项几何/运动/控制/审计测试，另2项历史字节快照测试。涵盖真实batch初始化与控制梯度、CUDA/autocast反向、NaN/角度别名拒绝、身份篡改、车辆分母及独立闭式控制反解。

正式强化审计主体35.41秒，准备/文件哈希/环境查询另计。没有优化更新、自由生成或测试轨迹访问；不把本次算子验证成本解释为新模型训练成本。

## 原结论保留

| 方法 | 硬场景资格/56 | 兼容门槛/56 | 严格舒适车辆窗口/342 |
|---|---:|---:|---:|
| 原标签 | 56 | N/A | 329 |
| 中心点中点两维 | 56 | 42 | 328 |
| 中心点终点两维 | 56 | 41 | 327 |
| 前保险杠中点两维 | 43 | 41 | 322 |
| 前保险杠终点两维 | 43 | 41 | 322 |
| 中心点中点显式横向参照 | 56 | 56 | 329 |

中心点中点两维仍有2个新增严格舒适失败、1个原失败改善。三维参照使用已知未来标签，仍只证明可恢复，不证明动力学可行或生成收益。原最大位置恢复误差和分时段jerk保持，见[第一步报告](action_state_replay_results.md)。

## 历史配置字节与快照

第一次强化审计在实际计算前被来源检查拒绝：Git切换分支后，tracked YAML和任务JSON由原CRLF变成LF，内容语义相同但字节哈希不同。所有其余原输入身份核对一致，未据此归因为轨迹或模型变化。旧输入的“工作区路径永久不变”假设需要修正。

新增input_snapshots从原运行记录的Git SHA 30975255601bfbb047b4faa609b7533004cf2007读取两份历史文件。只接受原blob或LF/CRLF候选中SHA-256及字节大小同时与原记录一致的版本，保存到新的独立目录。源文件和源运行目录未改，不能用语义相同作为放宽哈希的理由。

恢复的原字节哈希：
- YAML：112f1152cff0d9556fd02d3853c0bae51eb6f71b7b0de0f390d34ffbb1267ba6。
- 任务JSON：b52b7697df39dfe4fb1fb88f62b30ee52bf4fa159b3483da118d16bdcfcde6e2。

审计通过显式--input-snapshots引用冻结副本，快照必须属于同一原运行，且仍与原输入身份精确匹配；无快照时继续拒绝字节漂移，不静默替代。数据或非Git文件变化不能用此功能忽略。新审计版本v2目前明确只支持float64控制计算。

## 来源和复现

快照恢复与强化审计均formal/clean/completed，运行代码SHA61077a947312538a8286324b1423e51161b268ce。原3097525主回放与404d1bf首审计保持原文件和运行SHA。首次来源拒绝发生在运行上下文建立前，没有生成伪completed目录；恢复后另行执行正式审计。

- [强化audit.json](../artifacts/runs/action-state-replay-review-audit-20261013-r1/audit.json)
- [历史输入快照清单](../artifacts/runs/action-state-replay-input-snapshots-20261013-r1/snapshots.json)
- [轻量结果及关键来源哈希](control_replay_review.json)

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv-model\Scripts\python.exe' -m sumodiff.decoding.input_snapshots --source artifacts/runs/action-state-replay-20261013-r1 --output artifacts/runs/my-historical-inputs --formal
& '.\.venv-model\Scripts\python.exe' -m sumodiff.decoding.replay_audit --source artifacts/runs/action-state-replay-20261013-r1 --input-snapshots artifacts/runs/my-historical-inputs/snapshots.json --output artifacts/runs/my-strengthened-audit --formal
```

先提交并保持干净，使用新输出名。代码修复原子提交98906f1、107c210、61077a9，结果文档单独提交；本地合并main，实际SHA以Git为准，无push或tag。

下一步仍是训练集控制量/横向残差分布和尺度审计，再固定输出表示。尚未扫描全部6000训练窗的控制尺度，不能据56窗拍定正式归一化或直接开始新训练。
