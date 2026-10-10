# SUMODiff

独立的 SUMO 多车轨迹扩散生成工程。已实现阶段0至5及阶段5A：1200/200窗口、batch4、100个真实epoch（30000更新）已完成。基础生成仍0/200质量合格；冻结质量检查已定位速度抖动、位置—速度不一致、边界及朝向问题，收敛与稳定生成尚未达成。旧checkpoint不兼容。阶段6引导尚未实现，下一步短训练修改建议见[质量诊断报告](docs/trajectory_quality_audit.md)。实施范围见 [设计](docs/design.md)、[数据合同](docs/data_schema.md)、[进度](docs/progress.md)。

## 目录
`configs/` 分类保存候选配置，`src/sumodiff/` 为新代码，`scripts/` 为入口辅助，`tests/` 为短验证，`docs/` 为合同和协议。`artifacts/{raw,processed,cache,checkpoints,runs}` 全部忽略，不提交数据或权重。保留模块目录标明后续阶段，不包含假实现。

## 阶段0短验证（PowerShell）
以下使用本机已核对的环境，不要求安装工程或修改全局环境：

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe' -m pytest -q
& 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe' -m sumodiff.experiments record --config configs/experiments/manifest_smoke.yaml --output artifacts/runs/manifest-smoke --seed 20261004
```

重复执行时改用新的输出目录；工具拒绝覆盖。命令只记录真实环境和追溯信息，不执行科学实验，metrics初始为空。正式实验入口在后续阶段实现；清单工具 `--formal` 会拒绝未提交或脏工作区。运行清单包含实际生效配置，不只保存文件路径。

## 环境与安装方案
已核对：Python 3.10.16、PyTorch 2.5.1/CUDA build 12.4、RTX 4070 Laptop GPU（8188 MiB）、驱动610.47、SUMO二进制1.22.0。已有TraCI/sumolib为1.25.0，与二进制不一致，阶段1先统一版本。默认Anaconda Python 3.12.7没有PyTorch。

基础追溯依赖PyYAML，测试依赖pytest；数据阶段额外依赖NumPy、Shapely、Pillow和psutil。阶段1使用项目 `.venv-sumo` 和1.22.0安装目录自带客户端，严格检查来源与版本，不修改全局包。`sumo`依赖组仅提供进程内存测量的psutil，不重复安装客户端。阶段3/4在项目内复用兼容PyTorch/CUDA环境，不在阶段0下载大型依赖。具体方案见 [环境记录](docs/environment.md)。

## 继续实施
每次先读AGENTS、design、data_schema、progress，检查Git和已完成内容；只完成指定阶段。正式大采集和训练默认不启动。目标远端为 https://github.com/ily3000t/diffusion.git，阶段验收后再合并main，不根据代码完成自动打实验tag。

## 阶段1短回合采集

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv-sumo\Scripts\python.exe' -m sumodiff.simulation collect --config configs/scenarios/ramp_a.yaml --output artifacts/raw/my-ramp-episode --seed 21 --formal
```

详见 [采集协议](docs/sumo_collection.md)。六个正常场景验收配置覆盖每类两个几何；诊断回合单独标记，不进入正常训练数据。达到限时但未排空会失败并保留输出，不能当作完整回合。

## 阶段2窗口数据

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv-sumo\Scripts\python.exe' -m sumodiff.data process --config configs/data/window.yaml --sources configs/data/sources_stage2.json --output artifacts/processed/my-windows --formal
```

须先提供清单中的原始回合，所有输出使用新目录。详见[切窗协议](docs/window_dataset.md)：按几何及回合先划分，t0选车，历史21点/未来40点，未来标签与推理条件分文件；独立审计核对原始状态及SUMO规则。不完整窗口保留，工程验收数据不能代替正式训练数据。


## 阶段3解码与标签评估

详见[解码与指标协议](docs/decoding_evaluation.md)。使用项目`.venv-model`的兼容PyTorch环境，安装组为`.[model,data,dev]`。`python -m sumodiff.evaluation labels`与`sumodiff-evaluate labels`等价，原始和解码轨迹分开保存，无引导阶段明确N/A。完整标签数值重构不代表生成模型结果或动力学验证。阶段4单批网络显存见后述模型协议及progress。


## 阶段4条件模型

共享历史CNN、栅格CNN、路线/规则图编码，hierarchical/parallel共用同参数，时间U-Net输出[B,N,40,6]噪声。完整接口及短验证命令见[条件模型协议](docs/conditional_model.md)。当前是未训练网络，GPU资源probe不会更新权重，不构成模型生成性能结果；不自动开始长训练。


## 阶段5基础扩散

已提供固定尺度审计、240步小数据学习验证、严格checkpoint/续训、20步DDIM、raw/decoded轨迹及独立指标。入口`python -m sumodiff.diffusion`，使用项目`.venv-model`；详见[基础扩散协议](docs/base_diffusion.md)。工程链路验收与基础模型质量分开；短checkpoint不作为可用研究模型，不自动长训练或创建tag。


## 补充训练诊断

已完成原12/6窗口、总3000步的有界诊断，未改变架构或追加采集。固定噪声标签重构、纯噪声自由生成与oracle数值检查分别报告；20/50/100步DDIM对照和图表均可追溯。结果见[诊断摘要](docs/training_diagnosis_summary.md)，复现命令与依赖见[诊断协议](docs/training_diagnosis.md)。不把当前质量失败视为阶段5工程失败，也不将loss下降当作收敛证明。


## 阶段5A数据扩充与正式基础训练入口

已完成阶段5A正式数据扩充及100个真实epoch训练：1200/200窗口、batch4、30000更新。完整200验证窗口的基础生成质量仍0/200，基础模型稳定与收敛尚未达成。见[训练结果报告](docs/stage5a_training_summary.md)与[阶段5A协议](docs/stage5a.md)。采集/训练复现入口为[scripts/stage5a.ps1](scripts/stage5a.ps1)；当前不自动追加训练或进入阶段6。正式运行要求干净已提交代码，核心标签完整性与物理质量分开检查。

## 基础轨迹质量检查

已用现有checkpoint与200个验证样本完成冻结归因、独立解码数值核对和6个逐步去噪重放；真实标签200/200合格，生成仍0/200。报告区分通道oracle和自由生成，不改模型/指标或启动训练。结果、下一轮一致性/边界/速度差分短对照建议及复现命令见[trajectory_quality_audit.md](docs/trajectory_quality_audit.md)。诊断入口为`python -m sumodiff.diffusion.quality_audit`，图表工具为`scripts/render_quality_audit.py`。

## 阶段5B四组继续训练

已实现并验收A原噪声MSE、B加位置—速度一致性、C再加速度差分监督、D再加历史—未来交界监督。共同100轮checkpoint/完整AdamW/RNG、batch4/lr3e-4和配对噪声保持一致；各组固定继续10轮，3000更新。202项测试及每组12更新真实GPU预演通过；这不是完整10轮结果，短预演质量仍各0/6。完整协议、损失定义、资源与恢复命令见[stage5b.md](docs/stage5b.md)。

以下为四组训练与自动200任务报告的复现入口；本次已完成，重复运行须改用新RunId：

```powershell
Set-Location E:\diffusion_new
.\scripts\stage5b.ps1 -Stage Run -RunId stage5b-v1
```

仅优化预计四组合计约41分钟，验证/保存/监测/最终采样另计。正式四组已完成，新增损失尚未带来合格模型；下一步校准建议见结果报告。不自动延长预算或开始阶段6。


用户本地已完成阶段5B四组各10轮，质量仍各0/200。完整结果、横向偏移与损失归因及待审修改点见 [stage5b_results.md](docs/stage5b_results.md).


## 阶段5C统一训练入口

已实现统一解码轨迹模仿、正常速度差分和弱车身道路/路线损失，加两轮warmup与单段余弦衰减。原100轮模型参数/RNG暖启动，新AdamW；默认追加50轮、batch4、原1200/200数据。实际GPU短预演、恢复、道路梯度和20任务监测已验收，231项测试通过；充分训练质量与收敛仍待验证。

~~~powershell
Set-Location E:\diffusion_new
.\scripts\stage5c.ps1 -Stage Run -RunId stage5c-v1
~~~

这个入口由本地终端执行50轮、最终200任务采样和配对报告；预计仅优化约49分钟，其他步骤另计。损失定义、一次梯度校准、结果边界、恢复/单独采样及可选epsilon对照见[stage5c.md](docs/stage5c.md)。阶段5C正式50轮已完成，场景质量50/200，收敛尚未证明；新续训使用下方阶段5D入口。


## 阶段5D：150→200轮续训

已实现并短验收原始速度二阶差分监督：内部5e-5、前四帧1e-5；保留现有模仿/时间/道路目标，继承150轮模型的完整AdamW/RNG，仅重启低幅warmup及余弦。默认峰值2e-5，原1200/200、batch4，新增50真实epoch。用户已完成正式50轮，场景质量82/200（41%），严格舒适性10/735（1.36%）。完整诊断见[stage5d_results.md](docs/stage5d_results.md)及[轻量结果摘要](docs/stage5d_results.json)。

联合梯度、真实GPU恢复和完整200任务报告重放均通过。275项全量回归及新增报告检查通过；估计仅优化约56分钟，验证/监测/保存/采样另计，不保证收敛。完整配置、资源、结果及恢复命令见[stage5d.md](docs/stage5d.md)，轻量来源摘要见[stage5d_preflight_summary.json](docs/stage5d_preflight_summary.json)。

~~~powershell
Set-Location E:\diffusion_new
.\scripts\stage5d.ps1 -Stage Run -RunId stage5d-v1
~~~

Run从150轮checkpoint续训到200轮，每10轮固定20任务，最后自动评估200配对任务并写报告。要求已提交干净代码和未使用RunId。本入口是独立新协议；旧stage5c入口仍保留100→150轮的复现用途。该RunId已完成；复现须使用新RunId。200轮保留为当前候选，稳定与收敛仍待验证；后续延长不能重复此固定150→200入口。阶段6未启动。


## 阶段5D交界与密集路口诊断

已完成冻结150/200轮模型的低噪声重构及现有200任务oracle检查，未继续训练。t=0时密集路口训练0/47、验证0/16运动通过；普通路口分别17/24、30/50。协调后的首帧受到整段预测影响，仅新增原始首四帧位置loss的证据不足。当前自由生成质量仍为82/200；重构与标签替换不替代生成评价。

结果、限制及下一版监督建议见[boundary_density_results.md](docs/boundary_density_results.md)，数值与来源见[轻量摘要](docs/boundary_density_results.json)，冻结入口见[诊断协议](docs/boundary_density_diagnosis.md)。200轮保持当前候选，后续先校准解码边界与整段速度监督，阶段6未启动。


## 阶段5E：200→225轮解码边界监督

已实现并完成短GPU验收：解码时间监督按前4帧/内部36帧独立归一化并替换旧时间项；候选系数0.001/0.0009，仅增量预热1轮。保留原始速度、模仿和道路目标，继承200轮完整AdamW/RNG，逐轮无放回数据保持。322项全量回归及后续入口/修复检查通过，实际恢复逐位一致。

~~~powershell
Set-Location E:\diffusion_new
.\scripts\stage5e.ps1 -Stage Run -RunId stage5e-v1
~~~

用户本地已完成stage5e-v1的25真实epoch及完整200任务配对评估。资格82→64/200（41%→32%），前四帧/内部jerk P99均变差；严格舒适性10→18/735，道路/路线各少1个违规场景，密集路口仍0/16。保留200轮候选，不自动按现配方延长训练。全流程36.32分钟，收敛尚未证明。结果见[stage5e_results.md](docs/stage5e_results.md)和[轻量摘要](docs/stage5e_results.json)，公式、恢复和复现入口见[stage5e.md](docs/stage5e.md)。上述RunId已使用，复现必须使用新RunId、已提交干净代码。阶段6未启动。


已检查全部六个已有中间checkpoint，同60任务均未超过200轮的26/60资格；无需再训练即可复核。协议和报告入口见[checkpoint_screen.md](docs/checkpoint_screen.md)，结果、成本及限制见[checkpoint_screen_results.md](docs/checkpoint_screen_results.md)。保留200轮候选，阶段6未启动。
