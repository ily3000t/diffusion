# 阶段5A训练结果复核（2026-10-08）

1200/200合格训练与验证窗口已经生成，100个真实epoch、30000次更新已经完成。模型显示持续学习，当前基础生成尚未达到声明的道路与运动质量标准，不能宣称稳定或收敛。本轮复核保持原模型、权重、尺度、扩散和解码算法、质量阈值；没有启动阶段6或追加训练。

## 数据与实际覆盖

全量采集78个回合，先分配几何和回合，再切窗口。17397个候选窗口保留，审计和坐标转换失败均为0。最终选择如下：

| 道路 | 训练窗口 | 验证窗口 | 训练几何/回合 | 验证几何/回合 | 选中实际转弯训练/验证窗口 |
|---|---:|---:|---:|---:|---:|
| 三车道直路 | 400 | 67 | 5/20 | 2/6 | 0/0 |
| 匝道汇入 | 400 | 67 | 5/20 | 2/6 | 0/0 |
| 无信号路口 | 400 | 66 | 5/20 | 2/6 | 23/5 |

“实际转弯”沿用资格报告中H+F航向变化范围超过0.15rad的统计，不能等同于完成指定转弯动作。路口转弯覆盖有限；正常源数据中的jerk异常仍通过资格过滤处理，过滤后的数据分布存在选择偏向。几何、回合隔离不代表同一回合内相邻窗口相互独立。全部200个验证窗口来自18个回合、6组几何；本轮没有使用测试集。

新尺度审计使用完整训练和验证队列：未来位移P99=56.87m、最大64.00m；速度P99和最大均16.00m/s。位置50m、速度20m/s是物理归一化单位，保持不裁剪；约4.09%的位移超过50m不构成归一化错误。SUMO合格标签不能独立证明真实驾驶行为真实性。

## 训练完成情况与资源

逐条核对30000条更新日志，100轮均各使用1200个训练窗口一次，无重复或遗漏；记录的loss、梯度范数和耗时均有限。最终checkpoint的SHA-256已重新核对。每轮验证覆盖200个窗口，固定训练探针为均衡的60个窗口。

| 项目 | 实际结果 |
|---|---:|
| batch / epoch / 更新 | 4 / 100 / 30000 |
| 初始固定验证ε MSE | 1.133042 |
| 最终固定验证ε MSE | 0.001993 |
| 最低固定验证ε MSE | 0.001528，第88轮 |
| 第61—80轮验证均值 | 0.003133 |
| 第81—100轮验证均值 | 0.002749 |
| 训练、验证、保存及定期生成循环 | 6954.71s，约1小时56分钟 |
| CPU张量缓存准备 | 26.48s；缓存约1.20GB |
| optimizer更新均时 / 吞吐 | 195.82ms / 20.43窗口每秒 |
| CUDA峰值allocated / reserved | 177.47 / 196.00MiB |

CUDA数值只包括PyTorch管理的内存，不能解释为整机GPU占用；未测最大可用batch。环境为RTX4070 Laptop 8188MiB、PyTorch2.5.1/CUDA12.4，详细版本保存在原始environment.json。第96轮验证MSE曾升到0.007879，末段仍有波动；平均下降和最低点均不等于收敛判定。第88轮未保存checkpoint，不能拿第90轮文件冒充该轮最优权重。

最终checkpoint：
`artifacts/runs/stage5a-v1-train/checkpoint_step_030000.pt`
SHA-256：
`2315fa4d57b9d86a3254a0923590efbbd48f23666f3bf0d2c733ac04b4d1769b`。

## 完整200窗口的独立生成评估

固定最终checkpoint，DDIM20步、eta=0、seed=20261015。评估735个被选车辆窗口，29400个有效未来车辆帧；另外3005个当前车辆实例未纳入生成，不能将其安全性纳入本结果。运行失败0、质量未知0，失败和不合格样本均保留在200分母中。

| 指标 | 原始输出 | 一致性解码后 |
|---|---:|---:|
| 全部质量条件通过 | 0/200 | 0/200 |
| 任意车辆碰撞场景 | 15/200，7.5% | 4/200，2.0% |
| 道路越界场景 | 107/200 | 50/200 |
| 路线越界场景 | 107/200 | 51/200 |
| 地图覆盖超出场景 | 0/200 | 0/200 |
| 运动质量不合格场景 | 200/200 | 200/200 |
| 历史—未来边界jerk超限场景 | 200/200 | 200/200 |
| 未来0.5—4.0秒内jerk超限场景 | 200/200 | 200/200 |

独立碰撞检测包含帧间插值。本轮没有目标和攻击角色，目标事件及相关角色分类指标为N/A；基础模型产生的碰撞不能描述为有效攻击。

解码后道路越界按直路/汇入/路口分别为4/67、15/67、31/66；路线越界分别4/67、15/67、32/66。路口问题尤其突出。位置协调修正的车辆帧合并统计为中位数0.453m、P99=8.064m、最大28.119m；这三项与场景等权比例具有不同分母。排除交界及前0.4s后，解码轨迹jerk的车辆帧合并P99仍为719.23m/s³，高于声明硬阈值15m/s³。当前问题还存在于未来内部，修复第一帧不能单独保证合格。

生成与解码均时0.200s/场景，P99=0.261s；此计时不含输入输出和独立评估。200场景循环共61.04s，其中独立评估17.81s；实际引导梯度调用为0。固定6个原监测任务在这次完整评估中78个数组逐一bitwise一致（相应NaN视为一致），验证复核没有改变模型采样行为。

## 有界重构与采样步数诊断

重构仅使用均衡的6个训练和6个验证窗口、9个固定扩散时刻、每窗3份噪声；先核对完整1200/200训练身份，再缩小探针。训练自由生成6个任务全部是已见训练窗口，仍0/6合格；这说明当前问题也存在于训练已见任务。本项不代表全部训练集的自由生成质量。

精确epsilon的CPU float64加噪/恢复误差最大1.66e-12，真实标签解码位置修正最大3.54e-6m，数值核对通过。其范围是扩散数学、尺度和标签位置—速度协调，不能据此排除所有实现问题或认定动力学可行。

验证标签重构探针在t=0时位置RMSE=0.308m，在t=999时为271.15m；后者对应epsilon误差放大约157倍。这是已知加噪标签的单步重构，并非最终自由生成误差。低epsilon MSE不能直接替代物理空间及运动质量指标。dt=0.1s下三阶差分包含dt的三次方分母，帧间位置抖动需要独立控制。

| 固定6个验证任务，DDIM步数 | 原始最大位移m | 最大位置协调修正m | 解码合格 | 生成与解码均时s |
|---:|---:|---:|---:|---:|
| 20 | 58.09 | 19.90 | 0/6 | 0.208 |
| 50 | 57.89 | 17.15 | 0/6 | 0.433 |
| 100 | 57.50 | 16.92 | 0/6 | 0.806 |

步数比较保持相同任务、checkpoint、初始噪声、eta和指标设置。更多采样步数小幅减小修正，但本组任务的合格率未改善。全200窗口只执行了20步评估；50/100步结果不能扩展成全部验证集结果。

## 当前判断与下一步建议

阶段5A的数据扩充、尺度审计和预定训练预算已经完成；“基础模型稳定”仍未达成。现有证据不足以判定架构或研究方法失败，也不足以把剩余异常唯一归因为欠训练。

建议下一步维持这1200/200数据，先设小预算做位置—速度一致性、历史末速度/加速度与未来边界连续性的训练和解码对照，再检查未来内部jerk与道路质量。验证最优checkpoint保存、降低后期学习率/学习率衰减、EMA可作为稳定性候选，需同任务和预算对照支持，不能保证自动修好质量。改变损失、优化器或EMA规则应建立新实验身份；现有严格Resume不允许伪装成算法不变的续训，warm-start能力仍需明确实现。

本轮保持训练目标与物理阈值，不启动新100epoch训练、扩大采集、引导或RL，不创建里程碑tag。后续是否进入阶段6遵循用户选择的先稳定基础模型路径；本结果不构成阶段6研究效果证据。

## 追溯与复现

数据采集、准备、尺度审计及训练运行SHA为`d07676f6ad47cb4758ccf2592831fb0e18918ac9`；
有界诊断SHA为`443e7edc7cf7485f4e282cf95cc3ff560882e8f6`；
绘图SHA为`2ec2096bf314fdcc106a134c8f3145d39c99a09c`；
完整验证SHA为`a220c9016046cc14a858d184672db1a28ded3737`。
所有来源均formal、clean、completed。摘要提交和main合并SHA不回填到原实验身份。

完整数据、checkpoint、轨迹和图表保留在artifacts；Git仅提交配置、代码与[小型机器摘要](stage5a_training_summary.json)。已完成162项测试（14.59s）、图表目视检查和Git差异空白检查。以下仅复现评估，使用新的输出目录，不覆盖本次结果：

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = 'E:\diffusion_new\src'
$env:SUMO_HOME = 'E:\Program Files\sumo-1.22.0'
$dataset = 'artifacts/processed/stage5a-v1/dataset'
$checkpoint = 'artifacts/runs/stage5a-v1-train/checkpoint_step_030000.pt'
& '.\.venv-model\Scripts\python.exe' -m sumodiff.diffusion diagnose --dataset $dataset --checkpoint $checkpoint --config configs/experiments/stage5a_training_review.yaml --output artifacts/runs/stage5a-v1-review-replay --formal
& '.\.venv-model\Scripts\python.exe' -m sumodiff.diffusion sample --dataset $dataset --checkpoint $checkpoint --config configs/experiments/stage5a_validation_sampling.yaml --output artifacts/runs/stage5a-v1-validation-replay --formal
& '.\.venv-model\Scripts\python.exe' scripts/render_training_diagnosis.py --diagnoses artifacts/runs/stage5a-v1-review-replay --training-runs artifacts/runs/stage5a-v1-train --dataset $dataset --output artifacts/runs/stage5a-v1-review-report-replay --formal
```

每条命令成功后再运行依赖步骤；正式复现要求干净已提交代码。不同运行的SHA按实记录。

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
