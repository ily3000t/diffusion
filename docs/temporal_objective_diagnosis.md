# 阶段5E冻结目标诊断

在完整200/225模型及中间模型短验证之后，检查实际监督目标是否改善、误差是否集中在特定未来时段。只冻结前向，不创建优化器、不训练、不自由生成、不改损失和评价阈值。

固定协议：原1200/200数据，训练侧所有47个密集窗口，加每类24个按几何/回合轮转的普通控制，共119窗口；验证侧完整200窗口。选择只依赖元数据，seed=20261020，与既有交界诊断的训练选择一致。两模型同一窗口、同一Gaussian、固定t=0/50/200/300，B4。验证/训练回合和几何必须隔离，不访问测试集。

实际调用现有 decoded_temporal_objective，两个模型都按同一225轮配置、full ramp=1计算。同时按200轮权重组合旧总目标；不能把两模型各自不同权重的历史总loss直接比较。各分项含alpha_bar，不含lambda；各t比较固定输入，四个时刻均值不代替自然随机t下的期望。模仿、道路/路线代理及原始速度项全部保留；不将道路训练代理称为完整车身合规。

时域分组为前4帧、内部中段32帧、末4帧；另报告内部36帧，后者包含末4帧。实际解码速度的一阶/二阶差分误差分别除dt/dt²得到加速度/速度场jerk误差；独立物理jerk仍由共同固定历史和解码位置差分得到。两者分开记录。逐时刻、逐场景及混合帧分位数均保存。

运行：

```powershell
$env:PYTHONPATH='E:\diffusion_new\src'
$env:SUMO_HOME='E:\Program Files\sumo-1.22.0'
& '.\.venv-model\Scripts\python.exe' -m sumodiff.diffusion.temporal_diagnosis --config configs/experiments/stage5e_temporal_diagnosis.yaml --output artifacts/runs/stage5e-temporal-diagnosis-20261010 --formal
```

在干净已提交代码运行。输出清单锁定代码SHA、两个checkpoint哈希、实际配置、命令、种子、选定数据和标签、环境及状态；验证模型哈希未变、梯度缓存未建立、输入逐位配对及源checkpoint未改。轨迹与原始诊断结果进入忽略目录。

限制：已知未来标签的加噪重构不是自由生成；t=0仍有噪声(alpha_bar=0.9999)。训练控制过采样密集窗口，不据此计算无偏泛化差距。一组Gaussian及四个固定t不构成多种子研究。冻结诊断可以核对目标和现象，不能区分学习率重启与时间权重修改的独立因果，不触发下一段训练或阶段6。
