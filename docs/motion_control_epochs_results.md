# Motion 全量控制训练入口验收（2026-10-16）

本阶段已完成正式入口和短验证；尚未启动200轮长训练，不代表模型收敛或轨迹合格。继续使用原 scripts/stage5a.ps1，完整本地命令见[motion_control_epochs.md](motion_control_epochs.md)，机器可读证据见[motion_control_epochs_results.json](motion_control_epochs_results.json)。

## 交付与固定范围

- control.epoch.config.v1：6000训练、600验证、600封存测试，batch16、200轮/75000更新；375更新/轮，每轮无放回全覆盖，保留末批。
- 随机初始化新的directional control v2模型和AdamW；旧六维/16步模型不兼容此恢复协议。
- 已认证尺度[a,omega,r]=[2,0.5,0.1]及校准目标保持。MSE主目标＋模仿0.003、解码时间监督0.001、道路路线0.001、物理jerk尾部0.0003；无独立原始速度头loss。
- 一次3750步LR和辅助预热，1e-6→1e-4→1e-6；恢复不重启。CPU窗口与道路各64的LRU，按批padding，不全量占GPU。
- 每轮600验证MSE与60训练探针；每5轮存checkpoint，每10轮固定60验证任务自由采样；MSE最低候选单独保存，最终显式全600验证选模型，再Seal/Test。
- checkpoint/签名保存数据、尺度、梯度证据、完整配方、模型、AdamW、随机数、两个预热及pending监测位置；新RunSuffix恢复，原结果不覆盖。

数据仍为 d5377be30fa15745ec3ee9fcc2213ab355ab83973db337a3e1027dd14a5dfb29，216原任务/215合格源/1隔离。独立核对每个训练运行33017输入身份，完整train/validation的33000文件均入清单，未读取test数值数组；没有重采或重做质量准备。

## 实际验收

最终588项全量测试通过（52.55秒），PowerShell解析通过。首轮兼容测试发现旧封存identity验证不应强制反序列化fixture，及旧bounded test guard的缺字段测试问题，均修复后通过；不据首轮失败报告通过。

实际执行正式原日程的连续8步、从头4步和恢复第5—8步，总16次真实优化。三个运行均正式/clean/completed；每次起止均完整检查600验证窗口。固定验证MSE：初始1.0862613、第4步1.0749714、第8步1.0634465。8步只到第1轮的8/375批、aux ramp=8/3750，不能与旧16步压缩预热训练比较学习速度，不能据此判断收敛。

实际恢复审计逐位一致：模型参数、AdamW、噪声发生器、CPU/CUDA随机状态、LR/辅助日程、窗口/时间步/损失/梯度日志；16/16真实更新触发clip1，未称无裁剪。独立review再次验证signature、状态、分母、完整验证范围与测试隔离。

## 预演发现并修复的监测边界

8步模型在固定60任务全部产生大于或等于pi/0.1s的航向变化，违反既有控制rollout定义域；r1监测及其60任务、异常和原始控制保留为failed。没有修改控制积分、定义域、损失、采样噪声或任何质量阈值。

新增明确的InvalidGeneratedControl类型，仅把这一已知定义域错误视为模型输出失败。正式epoch监测/评价可完成记录，状态为completed_with_invalid_generated_controls；generation_failures仍60/60，质量通过率仍0%，事件未知/失败计数60。输入、代码、形状、数值与评价执行异常仍failed_execution并中断。旧bounded采样继续原报错合同。不能把“监测程序完成”写成“生成成功”。

r2以相同8步checkpoint、60任务及初始噪声复测：实际1200次去噪网络调用、0梯度调用、全部Python/NumPy/CPU/CUDA随机状态恢复。独立逐任务对比r1/r2的initial_noise、raw_normalized_controls、raw_controls均逐位相同，证明分类修复没有修正或裁剪预测。无有效解码轨迹，位移与修正最大值为N/A；r2监测17.31秒不能作为合格轨迹的完整几何评价耗时估计。

## 实测资源与成本边界

RTX4070 Laptop 8GB、Python3.10.16、Torch2.5.1/CUDA12.4，float32、无AMP/TF32、严格确定性。B16每步含同步文件核验/读取、解压、padding、条件、道路几何、前后向、clip与AdamW；连续8步去掉首步后均值1.7875秒，中位1.6735秒。另两个4步段均值1.8263/1.5008秒，样本很少且密度/磁盘缓存不同。

峰值GPU allocated约407.36MiB、reserved684MiB；三个进程RSS采样峰值最高约2.10GiB，非连续绝对峰值。初始化60训练＋600验证探针约37.77—43.00秒；首次完整身份核验与清单写入有额外启动成本，见JSON的近似launch-to-manifest范围，不能与每步速度混算。每份checkpoint约42MB。

按8步均值机械外推75000优化步约37.24小时，还不含每轮验证、checkpoint、自由采样及启动；不是完整总时长承诺、训练密度的全面测量或收敛预测。不据此自动扩大batch或训练预算。

## 来源与后续

训练/恢复、第一次监测失败的actual SHA是2bf4c9b39055ad6a42d098890fe252d42b1056cb；分类修复后的r2监测与独立复核actual SHA是5d39eb8a89d5225bc91bd6fa113171a5c4f58e1b，均在feat/motion-control-epochs、formal/clean。后续报告提交/合并SHA不回填上述实际运行SHA。源代码数值训练路径未更改，分类修复只涉及异常类型和监测输出语义。

下一步由用户从头启动正式Train。运行目录stage5a-motion-v1-train尚未创建；不要把epochs-check8等诊断checkpoint当作正式父模型。阶段6、长期训练、test评价、push及实验tag均未执行。200轮仅初始预算；后续按验证生成质量和曲线共同判断是否延长。
