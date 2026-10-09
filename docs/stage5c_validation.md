# 阶段5C采样步数与速度诊断验证

本轮范围是45000步基础模型的验证，不实施新loss、不继续训练或进入阶段6。固定模型SHA-256：2a25293744e0f9ff8bec9a03b3a1c3e077275bc7c9cece4119cdda13f070900e；训练运行SHA永久为8ceb573cbe76a6d224869ef52c82533891f7869d。

## 验证协议

20步重放与50步采样使用同一checkpoint、现有200个验证任务、seed 20261015和逐任务初始噪声、eta=0、模型精度、解码器和独立阈值。50步配置仅更改steps。比较器要求20步原始归一化预测、解码状态和初始噪声逐场景bitwise重现既有结果，保留全部失败分母。比较质量、道路路线、碰撞、交界/内部jerk、协调修正及耗时，使用2000次道路分层回合聚类配对bootstrap。不同采样步数是本次唯一生成干预，生成不读取未来标签。

低噪声诊断采用48个训练窗口（三类各16）、全部200验证窗口，在t=0/50/100/200/300分别向真实标签加噪，再恢复干净状态；同一窗口各噪声时刻共享Gaussian噪声。这是读取真实未来的条件重构，不是自由生成质量。只检查噪声分项、位置速度误差、原始速度二阶差分误差、解码运动、交界/内部jerk和修正；重构的motion_pass不能替代包含道路/碰撞的场景quality。

梯度诊断只用训练集16批B4。车辆窗口ID由原训练规则的假设下一轮前16批产生，噪声Generator继承checkpoint状态。每批对比自然随机t与同噪声固定t=200，计算当前加权模仿/时间/道路/总辅助梯度，以及未加权原始速度二阶差分诊断项。后者以0.15m/s归一化（dt=0.1对应15m/s³），包含前4帧与内部独立分组。低t<=300、alpha_bar和全batch分母保持；这些probe不是注册的训练loss，没有应用梯度更新、创建优化器或声称该项能改善生成。梯度范数/夹角来自eval模式、局部补齐的B4，不冒充原完整缓存的bitwise训练重放。无适用批次明确N/A。

所有新正式运行须从干净提交启动，保存生效配置、命令、环境、源checkpoint/数据身份和状态；采样与诊断不并发占用GPU，以便成本比较。代码与checkpoint前后哈希检查，参数梯度缓冲区不填入。验证集用于开发判断，不用测试集调参。

## 重放入口

在E:\diffusion_new本地PowerShell中设置环境：

~~~powershell
$env:PYTHONPATH='E:\diffusion_new\src'
$env:SUMO_HOME='E:\Program Files\sumo-1.22.0'
$env:PYTHONUNBUFFERED='1'
~~~

已有200任务的20步重放与50步采样：

~~~powershell
.\.venv-model\Scripts\python.exe -m sumodiff.diffusion sample --dataset artifacts/processed/stage5a-v1/dataset --checkpoint artifacts/runs/stage5c-v1/train/checkpoint_step_045000.pt --config configs/experiments/stage5a_validation_sampling.yaml --output artifacts/runs/stage5c-steps20-validation-20261009 --formal

.\.venv-model\Scripts\python.exe -m sumodiff.diffusion sample --dataset artifacts/processed/stage5a-v1/dataset --checkpoint artifacts/runs/stage5c-v1/train/checkpoint_step_045000.pt --config configs/experiments/stage5c_sampling_steps50.yaml --output artifacts/runs/stage5c-steps50-validation-20261009 --formal
~~~

无更新的速度重构及分项梯度验证：

~~~powershell
.\.venv-model\Scripts\python.exe -m sumodiff.diffusion.velocity_validation diagnose --dataset artifacts/processed/stage5a-v1/dataset --checkpoint artifacts/runs/stage5c-v1/train/checkpoint_step_045000.pt --config configs/experiments/stage5c_velocity_validation.yaml --output artifacts/runs/stage5c-velocity-validation-20261009 --formal
~~~

完整来源核对、配对报告与诊断图：

~~~powershell
.\.venv-model\Scripts\python.exe -m sumodiff.diffusion.velocity_validation compare --dataset artifacts/processed/stage5a-v1/dataset --checkpoint artifacts/runs/stage5c-v1/train/checkpoint_step_045000.pt --original20 artifacts/runs/stage5c-v1/sample --fresh20 artifacts/runs/stage5c-steps20-validation-20261009 --fresh50 artifacts/runs/stage5c-steps50-validation-20261009 --diagnostic artifacts/runs/stage5c-velocity-validation-20261009 --output artifacts/runs/stage5c-velocity-validation-report-20261009 --formal
~~~

原目录已存在时入口拒绝覆盖；再次运行须换输出ID。完整实际命令也在各运行command.txt，不把文档提交SHA回填到既有实验。

## 验收与解释

新增15项数学测试覆盖非零真实加速度、交界定位、常数速度偏置、NaN无效槽位、缺失数据、全batch分母、导数、梯度隔离与预算拒绝；与原采样、扩散、统一损失的相关测试共39项通过。随后实际运行结果和资源见stage5c_validation_results.md/json及progress.md；本节不预判结果。无模型质量或新loss收益时保留负结果，不扩预算。

扩散网络、原训练目标、解码器和评价阈值未变。新增模块仅用于有限诊断与验证来源比较；任何训练目标调整应基于本轮证据另行实施。
