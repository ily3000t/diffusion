# 阶段5补充有界训练诊断（2026-10-07）

240步工程验收通过，噪声预测目标显示学习能力；收敛及充分训练后的生成质量未验证。原始公里级失真不能直接归因为欠训练，亦不构成架构失败证据。本轮保持12个训练/6个验证core窗口、模型/归一化/学习率/optimizer/种子/精度，从240恢复至总1000步，检查后最多总3000步。训练max_steps指总更新数，不是新增更新数。不新增数据，不改架构，不加入引导或物理loss。

## 预先固定的诊断

- 各checkpoint用现有固定train/validation epsilon探针（训练每100步记录），以及9个扩散t=[0,50,100,250,500,750,900,950,999]、每窗3个可复现noise的独立重构。报告每场景等权、六通道epsilon/物理重构MSE、位置/速度/方向向量RMSE；所有checkpoint使用相同窗口与noise。
- 在真实标签上以CPU float64核对精确epsilon的前向加噪/逆变换误差及label解码修正，检查数学、尺度与协调链路。oracle是数值检查，不是模型生成结果；若失败，先停止训练并定位实现。
- 同样的current-only选择器在train/validation各选6个多车任务，noise seed沿用20261008；不依赖future_mask或core。记录训练seen/unseen身份（仅用于解释，不筛选任务）。保留raw/decoded轨迹、全体失败/未知/不合格分母、非目标碰撞/道路/运动/历史边界和修正幅度。
- 先20步eta0，对照240/1000/2000/3000；若epsilon改善而自由采样仍异常，再同一checkpoint同任务/noise比较20/50/100步。步数对照是诊断，不自动选择测试最优参数。
- 达到3000步即停止本轮；数值发散或oracle/实现失败则提前定位。不设置“必须收敛/轨迹必须合格”的诊断验收门槛，不用loss下降代替收敛，不自动追加训练。源数据质量和小样本泛化限制保留。

## 入口与记录

使用项目.venv-model，所有数据/checkpoint/轨迹置于artifacts，每个独立运行保存manifest/config/command/environment/metrics/status、输入hash及实际代码SHA；正式追溯运行在已提交/干净分支启动。本轮用户已授权1000—3000步诊断，--allow-long-run是工具显式预算开关，不代表无限训练。

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = 'E:\diffusion_new\src'
$dataset = 'artifacts/processed/stage2-accepted-20261004'
$scale = 'artifacts/runs/stage5-scales-accepted-20261005/scale_audit.json'
$old = 'artifacts/runs/stage5-train-accepted-20261005/checkpoint_step_000240.pt'
& '.\.venv-model\Scripts\python.exe' -m sumodiff.diffusion diagnose --dataset $dataset --checkpoint $old --config configs/experiments/training_diagnostic.yaml --output artifacts/runs/diagnosis-0240-20261007 --formal
& '.\.venv-model\Scripts\python.exe' -m sumodiff.diffusion train --dataset $dataset --model-config configs/model/initial.yaml --config configs/train/diagnostic_1000.yaml --scale-audit $scale --resume $old --output artifacts/runs/diagnostic-train-1000-20261007 --formal
```

后续2000/3000用对应diagnostic_N.yaml，分别从1000/2000末checkpoint恢复，传--allow-long-run。diagnose对各checkpoint分别执行，新目录不覆盖；sampling对照配置training_diagnostic_sampling.yaml。学习诊断关闭“必须相对本段起点下降5%”的短验收门槛，记录实际变化；不修改训练签名中的算法项。训练/评估互不修改权重或RNG checkpoint。

换成完整core84/95窗会改变训练签名，不能冒充严格resume。本轮未启动该训练，后续应新建训练实验或另行实现明确warm-start接口。阶段6模块/合成验证可继续开发，短模型质量不是硬门槛；真实风险—质量研究结论需独立证据。
