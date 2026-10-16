# Motion 新数据控制训练前检查

范围固定为新数据ID sha256:d5377be30fa15745ec3ee9fcc2213ab355ab83973db337a3e1027dd14a5dfb29，model.control.config.v2。沿用原stage5a.ps1的Audit/Calibrate/Preflight动作；Train/Resume/Validate/Seal/Test仍明确拒绝。本阶段不做正式长训练、自由采样或阶段6，也不声称收敛。

## 数据和尺度

motion_control_scales.yaml使用显式control_scale_audit.protocol.v2，从selection.v3的实际运动标记/唯一事件构造转弯、换道、匝道退出分组。旧v1仍要求旧actual_turning_agents，字段缺失不能默认为零。全部6000训练标签参与中心坐标/区间中点航向三控制[a,omega,r]反解、NumPy独立回放和尺度统计；验证和测试数组均不参与尺度估计。

各训练组绝对控制值p999的最大包络向上取到预声明网格，不中心化、不裁剪。分组含三道路、转弯、密集路口、密集无转弯、换道和匝道退出，最低每组8窗；网格不足、数值回放或组覆盖失败时不能认证。原独立control_scale_review复算组分位数与候选尺度。窗口重叠，不把分位数当作独立自然交通总体估计。

## 固定诊断任务

control_replay.tasks.v2绑定dataset_manifest/index/selection哈希和任务种子，按明确算法重新生成验证，身份或任务改变立即拒绝。每个train/validation各60窗：普通12（各道路4）、密集无转弯16、转弯16、换道16（直路/汇入各8）。各组双参考各半、至少3个几何；优先分散几何/回合，密集与转弯组再偏向较多车辆，然后按固定种子哈希排序。任务不重复、不缺额回退，测试不入任务；运动信息仅用于离线分组，不加入模型条件。

## 联合梯度校准

保留现控制目标：epsilon MSE + 解码轨迹模仿/时间/道路代理 + 物理jerk Huber均值与top10%尾部。控制表示没有独立的六维预测速度头，不把旧六维raw速度loss配置伪装成已启用。

基线权重imitation=.003、temporal=.001、road=.001、tail=.0003，辅助项仅t<=300并按alpha_bar和有效场景分母计权。随机初始化条件模型使用固定任务、种子、自然随机t及固定200/50；记录完整辅助梯度、总梯度、各项范数/夹角、组合残差、预计裁剪、梯度调用和显存。除普通组12窗外，其他组和硬批次为真实B16；不把单项比例相加。

预声明数值预算为完整辅助梯度范数/噪声梯度范数<=.5，遍历预声明共同缩放因子网格，为全部辅助项选择满足最坏训练探针的最大因子。需要缩放时用相同模型/噪声重测，校验完整组合，网格不足失败。共同缩放保持各项相对权重，但该上限和权重只是随机初始化的数值候选，不是生成质量、最优权重或收敛认证。全辅助强度测试与后续预热分开记录。

## 真实优化预演与命令

固定B16、60训练/60验证诊断窗、16 AdamW更新，新模型从头初始化。诊断日程16步，LR 1e-6→1e-4→1e-6、前2步LR预热/前4步辅助预热；不能冒称未来200轮正式日程。随后克隆已保存模型/AdamW做3个训练专用压力更新（密集、转弯、换道各16窗），固定t200/全辅助/峰值LR，原checkpoint不改写，不产出可续训的压力checkpoint。该检查覆盖实际反传/裁剪/优化器状态，报告真实时间/显存，不从小缓存吞吐承诺全量长训练时长。

~~~powershell
Set-Location -LiteralPath 'E:\diffusion_new'
.\scripts\stage5a.ps1 -Action Audit -DataProtocol Motion -Name stage5a-motion-v1
.\scripts\stage5a.ps1 -Action Calibrate -DataProtocol Motion -Name stage5a-motion-v1
.\scripts\stage5a.ps1 -Action Preflight -DataProtocol Motion -Name stage5a-motion-v1
~~~

每一步要求上一步完成、正式干净来源和固定输入身份。默认输出依次为runs/NAME-scales、NAME-calibration、NAME-preflight。输出目录存在时拒绝覆盖；重跑使用新RunSuffix，并在下一步显式传-ScaleAudit指向新scale/candidate_normalization.json或-TrainingConfig指向新calibration/training_config.yaml，避免引用旧结果。完整生效配置、命令、环境、输入/模型哈希、种子和status保留。

正式6000窗训练缓存、全部600验证/冻结测试、单段完整warmup/余弦预算、周期采样/选择checkpoint、失败恢复和完整200轮入口是后续独立交付。本阶段继续保留控制<=64更新限制。
