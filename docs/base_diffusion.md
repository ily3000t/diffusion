# 基础扩散、短训练与独立采样协议

本阶段实现基础链路，不实现引导、RL、CFG、部分扩散或滚动生成。旧HighD/DDPO checkpoint不兼容。`python -m sumodiff.diffusion`与安装后的`sumodiff-diffusion`等价。依赖使用项目`.venv-model`，不修改全局环境。

## 表示与尺度冻结

标签为[B,N,40,6]：各车相对t0位置的delta x/y、共同局部坐标系中的vx/vy、sin/cos。固定除数[50m,50m,20m/s,20m/s,1,1]，逆变换后交给既有统一解码器。条件编码器自己的特征单位在ModelConfig中单独记录；不由场景统计量改变。

`audit-scales`只读取train/validation的有效标签和历史，保存范围、分位数、超出物理单位的比例、数据身份和完整DiffusionConfig。50m是单位而非边界：大于50m的位移不截断，sin/cos不标准化。训练必须传入与数据及配置匹配的审计文件。更换数据或尺度先重新审计，不能复用旧checkpoint或从测试集选参数。

## 训练目标与采样

1000步线性beta，1e-4到0.02，alpha_bar为(1-beta)的累计乘积。0..999索引，最后一步alpha_bar约4.04e-5。x_t=sqrt(alpha_bar_t)*x0+sqrt(1-alpha_bar_t)*epsilon，预测epsilon。每个场景的平方误差除以其有效agent_mask AND future_mask对应的六维元素数，再对场景取均值；空监督场景明确报错。附加物理/一致性loss关闭。

当前训练及固定验证探针优先使用H+F完整的core窗口；过滤是训练/标签操作，采样不允许依赖core或future_mask。噪声函数只屏蔽无效车辆，不根据未来退出mask改变噪声。标签与条件结构分离。训练随机抽取窗口（有放回），epsilon和t从独立CPU Generator取样，随后传到模型设备。全部条件编码器和去噪主干共同训练，AdamW、梯度裁剪、float32、严格确定性，AMP/TF32关闭。

默认12个训练窗、6个验证窗，按输入道路family轮询选取；固定validation_seed令噪声/时间步可重复。每40步保存训练/验证epsilon探针，每120步保存checkpoint，总240更新。固定探针仅为小数据学习诊断，不是轨迹分布指标或泛化保证。总步数>1000须显式--allow-long-run；不会自动扩大训练规模。

DDIM默认20个从999到0的均匀整数子序列，末转移到索引-1（alpha_bar=1），eta=0。完整Eq.12方差实现支持0<=eta<=1，随机采样显式记录种子。无x0 clipping。干净状态及噪声逆关系独立暴露，为阶段6更新后重算epsilon预留接口；本阶段没有引导开关。条件编码器每场景计算一次并缓存，20次U-Net调用。目标参考：

- [DDPM论文](https://arxiv.org/abs/2006.11239)，噪声预测简化目标。
- [DDIM论文](https://arxiv.org/abs/2010.02502)，Eq.12；公式核对[作者实现](https://github.com/ermongroup/ddim/blob/main/functions/denoising.py)。代码在本工程独立实现，没有复制作者实现。

## checkpoint与恢复

schema sumodiff.base.checkpoint.v1；只保存tensor及基本类型，torch.load(weights_only=True)，不加载任意类。包括权重、AdamW状态、更新次数、训练Generator、CPU/CUDA RNG、完整模型/训练/扩散配置、数据manifest/index哈希、尺度审计哈希、训练/验证窗口ID、运行SHA/分支/dirty、父checkpoint和实际torch/device。临时文件写后原子替换，已有checkpoint不覆盖。

恢复检查训练签名：模型、数据、尺度、样本ID、batch、优化器、训练及验证种子、设备等必须一致。允许增加总步数及改变记录频率/诊断门槛，不允许改变学习算法；无需假装跨版本bitwise一致。恢复要求同torch版本、CUDA RNG设备数。同一运行时验收实际续训权重、optimizer、RNG及后半段窗口/t/损失/裁剪前梯度范数完全相同，计时不比较。每次续训为新独立目录，并记录父checkpoint。

## 采样与独立评估

任务从t0的agent_ids和family选择（默认每窗至少2辆、validation6窗、三family轮询），不使用真实未来标签、退出时刻或core过滤。可配置minimum_agents=1使单车任务进入，无车辆对的碰撞率N/A。base没有攻击角色，target/effective target为N/A；所有车辆对计入non-target。跨新数据推理暂需另行定义明确协议，当前入口要求checkpoint训练数据manifest身份一致，搬迁保持内容哈希可接受。

每个任务保存预定ID、噪声种子、initial_noise、未解码normalized prediction、物理raw_future_delta、raw_absolute_states、decoded_absolute_states及位置/速度/朝向修正、回退mask、原始速度残差。无guidance输出，metadata明确N/A。解码不增加道路或动力学保证。全时域推理，无future_mask。

raw与decoded分别运行既有旋转矩形及帧间碰撞、车身道路/路线/覆盖、原始运动/边界指标。保持既有阈值和有界数值几何修复策略。失败、未知、质量不合格保持在各自适用分母；错误保留后整次执行报告failed。综合分数不实现。成本分别记录采样+解码、独立评估、循环总时间、显存及实际梯度调用0；计时不含RunRecorder环境查询。

## 可复现短验收命令

从仓库根执行；新输出目录不可重复使用。正式运行前工作区应干净且代码已提交，`--formal`指追溯规范，不表示论文正式实验或已充分训练。

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = 'E:\diffusion_new\src'
$dataset = 'artifacts/processed/stage2-accepted-20261004'
& '.\.venv-model\Scripts\python.exe' -m sumodiff.diffusion audit-scales --dataset $dataset --config configs/train/base.yaml --output artifacts/runs/stage5-scales-accepted-20261005 --formal
& '.\.venv-model\Scripts\python.exe' -m sumodiff.diffusion train --dataset $dataset --model-config configs/model/initial.yaml --config configs/train/base.yaml --scale-audit artifacts/runs/stage5-scales-accepted-20261005/scale_audit.json --output artifacts/runs/stage5-train-accepted-20261005 --formal
& '.\.venv-model\Scripts\python.exe' -m sumodiff.diffusion train --dataset $dataset --model-config configs/model/initial.yaml --config configs/train/base.yaml --scale-audit artifacts/runs/stage5-scales-accepted-20261005/scale_audit.json --resume artifacts/runs/stage5-train-accepted-20261005/checkpoint_step_000120.pt --output artifacts/runs/stage5-resume-accepted-20261005 --formal
& '.\.venv-model\Scripts\python.exe' -m sumodiff.diffusion audit-continuation --uninterrupted artifacts/runs/stage5-train-accepted-20261005/checkpoint_step_000240.pt --resumed artifacts/runs/stage5-resume-accepted-20261005/checkpoint_step_000240.pt --output artifacts/runs/stage5-continuation-accepted-20261005 --formal
& '.\.venv-model\Scripts\python.exe' -m sumodiff.diffusion sample --dataset $dataset --checkpoint artifacts/runs/stage5-train-accepted-20261005/checkpoint_step_000240.pt --config configs/experiments/stage5_base_sample.yaml --output artifacts/runs/stage5-sample-accepted-20261005 --formal
```

上述原始数据不在Git，来源及生成过程见sources_stage2.json、progress阶段1/2和window_dataset.md。完整清单、配置、命令、依赖、hash和状态在各目录；小型summary另存docs，不回填运行SHA。测试tmp统一指定新artifacts/cache目录，避免用户系统临时目录权限冲突。

## 质量限制与后续条件

九个工程验收回合及少量窗口不是足够的正式研究数据。短训练epsilon MSE下降不能证明正确自由采样；高噪声x0恢复会放大epsilon误差，20步生成还需要独立质量检查。出现公里级原始/解码修正时必须公开，不截断、平滑或降低阈值来宣称通过。源正常数据已有道路/路线/jerk问题，先在train/validation定位数据/路线几何与运动质量，再确定正式训练规模和收敛标准。

本阶段实测吞吐只估计相同缓存batch/车道/精度下给定更新次数的优化器时间，排除数据准备/validation/checkpoint，不能估计收敛，也不能假定最大batch。正式基础训练仍需单独授权及执行。v0.1.0需要有可用的完整可复现基础生成实验，本阶段不自动tag。阶段6可做引导数学及合成动作验证，但真实风险—质量比较以可用基础模型为前提；不得用guidance掩盖当前输出失真；不能仅凭240步结果判定欠训练为唯一原因或判定架构失败。当前短checkpoint质量不阻断阶段6模块实现/合成验证，研究结论仍须独立质量证据。


补充无标签搬迁验收：`audit-inference --dataset ... --checkpoint ... --source-run artifacts/runs/stage5-sample-accepted-20261005 --output artifacts/runs/stage5-inference-accepted-20261005 --formal`。核对源config/checkpoint/输出哈希，然后在新目录只复制该split的输入metadata及指定任务的conditioning/map（所有targets/labels文件缺失），重选任务并重放初始噪声、raw和decoded数组，要求本机bitwise相同。投影保存原始认证catalog，用于证明标签文件无需存在，不是新的完整训练数据集。
