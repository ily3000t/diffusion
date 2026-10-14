# 三维控制目标接入与短校准

本阶段将已审计的中心／区间中点三维控制接到现有 `python -m sumodiff.diffusion train/calibrate/sample`。旧六维路径继续由原schema识别；不复制旧工程，也不另写训练入口。6000训练窗口的运动覆盖局限见[尺度报告](control_scale_results.md)。

## 表示与兼容性

历史仍为21点六维状态；模型未来输出为40点 `[a, omega, r]`，固定尺度2 m/s²、0.5 rad/s、0.1 m/s。中心位置、固定共同局部坐标系及逐车条件编码保持。新模型schema为sumodiff.model.control.config.v1，state_dimension=3表示扩散输出通道；历史始终六维。扩散schema为sumodiff.control.diffusion.config.v1，旧六维checkpoint不得作为新控制模型父模型。

训练标签先加回各车初始位置，再用统一观测初始化反解，最后除固定尺度。初始化仅读取最后两个有效观测姿态；不使用未来出生、退出或future_mask决定推理。标签完整性掩码只参与离线训练变换与噪声监督，核心窗口不完整会明确拒绝。

训练辅助监督与采样都调用同一float64可微控制回放：积分纵向区间速度、航向及横向区间速度，再由中心位置后向差分得到最终速度。没有额外最小二乘协调、平滑、裁剪、道路投影或动力学执行保证。回放保留原有单步角度别名检查；不支持的巨大角增量明确失败并保留采样分母。

采样保存raw_normalized_controls、raw_controls及六维回放状态。raw_future_delta只表示回放状态的相对位置／速度／朝向，不能将其前三维当作网络控制输出。raw_absolute_states与decoded_absolute_states在该表示下为同一回放状态，零修正是没有后处理阶段的定义结果，不能作为模型质量改善证据。失败时仍保留已经计算的原始控制预测。

## 候选目标与短预算

噪声MSE按每场景有效元素归一化、最后按场景平均。辅助项是统一回放状态与配对标签的模仿、全40帧速度时间差分、车身道路／路线代理与全时域物理jerk尾部；沿用alpha_bar、t<=300及共同预热。控制头没有原六维“原始速度预测”，因此不保留旧raw_velocity头监督。

起始候选：模仿0.003、时间0.001、道路0.001、jerk尾部0.0003。它们只是新表示的数值校准候选；旧权重不能直接视为新目标的最优值。物理jerk直接使用统一回放位置与固定最后3个历史点，阈值15 m/s³、Huber、最差10%加均值。

短配置固定28训练／28验证诊断窗口，来自原显式任务清单，含三类普通、密集无转弯和转弯。分组仅供离线诊断，不进入模型条件。batch4、16次优化更新；2更新LR预热、统一4更新辅助预热，再单段余弦。新模型与新AdamW随机初始化，不继承旧200轮策略；保持旧模型作基线。配置只允许3至64更新，正式长训练协议尚未实现，超预算明确拒绝。

checkpoint保留模型、完整训练配置、数据与尺度身份、任务清单哈希、AdamW、CPU/CUDA与噪声RNG及更新日程。总预算不可在恢复时修改；prefix仅截短执行，不改变日程。新控制模型的测试封存尚未实现，sample split=test会在读取数据之前拒绝。

## 命令

在干净已提交的功能分支运行。以下目录名仅为示例，已有目录不得覆盖：

~~~powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
$python = '.\.venv-model\Scripts\python.exe'
$data = 'artifacts/processed/stage5a-fresh/dataset'
$model = 'configs/model/control.yaml'
$config = 'configs/train/control_integration.yaml'
$scales = 'artifacts/runs/training-control-scales-20261013-r1/candidate_normalization.json'

& $python -m sumodiff.diffusion calibrate --dataset $data --model-config $model --config $config --scale-audit $scales --output artifacts/runs/control-gradient-initial-r1 --formal
& $python -m sumodiff.diffusion train --dataset $data --model-config $model --config $config --scale-audit $scales --output artifacts/runs/control-short-r1 --formal
& $python -m sumodiff.diffusion calibrate --dataset $data --model-config $model --config $config --scale-audit $scales --checkpoint artifacts/runs/control-short-r1/checkpoint_step_000016.pt --output artifacts/runs/control-gradient-after-r1 --formal
& $python -m sumodiff.diffusion sample --dataset $data --checkpoint artifacts/runs/control-short-r1/checkpoint_step_000016.pt --config configs/experiments/control_integration_sampling.yaml --output artifacts/runs/control-sample-r1 --formal
~~~

恢复验收可在同一个16更新配方下先加 `--prefix-updates 8`，再用 `--resume <prefix checkpoint>` 在新目录完成16更新。完整运行与8+8必须使用相同配置、尺度、数据、任务及随机种子。不默认执行正式训练、600验证／测试、重新采集或阶段6。

## 验收与解释

短校准只加载训练标签；每个诊断组分别检查自然随机t及固定t=200，另检查B16密集／转弯实际组合梯度与显存。记录分项范数、实际组合辅助梯度、总梯度、与噪声梯度夹角、组合残差及是否会触发裁剪。不能把分项范数比例相加代替组合梯度。梯度探针不更新参数，也不实际裁剪；真实训练日志另记实际裁剪。

工程验收包括标签恢复、掩码、车辆重排、公共回放梯度、DDIM、实际模型／AdamW／RNG／日程恢复及短训练／采样的追溯。生成失败与质量不合格保留，短训练不要求收敛或合格自由生成。正式训练前仍需查明未来运动覆盖不足，并冻结正式配方与预算。


## 本次执行结果

已完成固定短协议，见[结果](control_target_results.md)和[摘要](control_target_results.json)。最终440项全量测试通过，56标签窗口独立CPU恢复与真实GPU16／8+8逐位续接通过。当前PyTorch严格确定性CUDA不支持cumsum的问题已使用固定下三角前缀和修复；保留原失败来源。

16更新能够降低固定噪声探针，但3个自由采样均触发角增量保护，原始预测及实际60次去噪调用保存；没有放宽保护或裁剪成合格轨迹。本阶段不认为已获得合格新模型，不自动扩预算。后续先追查既有数据的运动覆盖，再制定正式训练协议。
