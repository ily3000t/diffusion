# 方向换道权限条件编码验收

模型条件的工程前置已补齐。新三维控制架构能显式接收左右乘用车换道权限，原训练、控制校准和基础采样入口已接入；本轮未训练新模型或更换正式数据。

## 实现与兼容

- 新架构sumodiff.model.control.config.v2；运行时lane_change_permissions[B,L,2]为左/右类别权限，经共享MLP加入车道token。原8维折线、地图栅格、三维控制目标及loss保持。新候选配置control_directional.yaml，旧control.yaml未改。
- 只用t0及之前观测、已知路线和精确地图，标签掩码/未来退出时刻不进入条件。权限是静态类别许可，不等同于相邻车道存在、动态安全或完整法规执行。
- 旧六维和控制v1仍支持原接口，受限地图继续明确拒绝。新版本缺字段、错误类型/形状、checkpoint权限权重缺失/错形状/伪装版本均拒绝；跨版本restore在修改权重或优化器前拒绝。
- 新checkpoint外层仍base.checkpoint.v1，架构版本进入完整签名。合成CPU测试以实际AdamW、噪声Generator及原日程核对损失、参数和优化器状态的精确续接；未额外执行新架构的CUDA优化恢复；未实现旧模型暖启动。

## 短验证结果

固定5个训练输入：上一阶段4个combined回合和1个无限制control回合，均t0=300，19条车道、每窗2或3辆真实选车。主探针认证完整原始文件，但构造条件前将帧截断到t0，不读取存储标签数组。

两个融合的车辆预测重排最大误差1.296e-06，条件重排最大9.54e-7；车道重排最大误差1.192e-06。填充NaN数据输出差0、无效车辆噪声梯度0，真实GPU所有模型参数梯度有限。合成专项测试另确认权限MLP梯度非零、无效车道权限不能传递信息。

保持当前交通与几何不变，放开原限制时，随机模型输出最大变化0.065055。它只验证条件通路有效，不代表已学习规则、碰撞改善或道路合规。模型参数哈希前后不变，优化更新0。

| 旧checkpoint | 实际步数 | 验证 |
| --- | --- | --- |
| 六维200轮 | 75000 | 严格加载、权重逐位相同、无限制地图前向有限、跨版本恢复拒绝 |
| 三维控制短模型 | 16 | 同上；不将短模型当作已收敛模型 |

独立复核重新认证106个主探针输入，从编译XML直接读取changeLeft/changeRight并核对5幅输入地图；4个受限输入共12项禁止方向、1个对照0项。两个旧checkpoint哈希与既有结果/运行记录匹配。

CPU以相同种子重建参数哈希，并重放GPU保存的噪声、条件和门控输出；最大差2.503e-06，在原atol=2e-5、rtol=5e-5内。CPU复核不重做GPU性能或完整训练。

506项全量测试通过（56.31s），本阶段新增19项；52项旧路径子集和18项初版专项均不重复累计。覆盖方向语义、无未来读取、两种融合、车辆/车道重排、掩码、原控制缓存、新checkpoint精确恢复/篡改拒绝、旧接口以及固定探针预算。

## 本机资源

128维、4头、float32、严格确定性，关闭AMP/TF32；每种配置预热1次、测2次。下列只含网络前后向，不含正式优化器状态、数据流水线、解码/扩散采样；合成12车负载不是真实SUMO密集回合。

| 输入/融合 | 单次前后向 s | allocated峰值 MiB | reserved峰值 MiB |
| --- | --- | --- | --- |
| five_observed_train_windows/hierarchical | 0.3109 | 179.52 | 206.00 |
| five_observed_train_windows/parallel | 0.3037 | 179.52 | 206.00 |
| synthetic_full_capacity_b4/hierarchical | 0.2374 | 240.55 | 314.00 |
| synthetic_full_capacity_b4/parallel | 0.2235 | 240.55 | 314.00 |

主探针主体12.88s，独立复核主体2.94s；环境准备/哈希另计。未测正式B16优化，不能据本表承诺正式训练吞吐或峰值。

## 来源与下一步

主探针与独立复核均formal/clean/completed，实际源SHA为bf74057839d7891986cf52bfb371ea5ee302b8b1；原始训练回合来源SHA为f44da9ca0dd3427423ac3a8860c80791e681d27c。完整配置、命令、环境、种子和输入身份保留，交付提交不回填运行SHA。

- gpu：[directional-conditioning-20261014-r1](E:/diffusion_new/artifacts/runs/directional-conditioning-20261014-r1/manifest.json)，输入身份106。

- review：[directional-conditioning-review-20261014-r1](E:/diffusion_new/artifacts/runs/directional-conditioning-review-20261014-r1/manifest.json)，输入身份115。

接口与复现命令见[协议](directional_conditioning.md)，源头质量证据见[汇入修复报告](ramp_alignment_results.md)。

下一步冻结旧/交互参考混合、三类道路运动及唯一事件配额，并在既定几何/回合划分内准备新数据。保留原质量筛选与失败分母；新数据冻结后重审控制尺度、组合梯度并制定一次固定正式训练协议。现有控制入口仍限64更新，test封存尚未扩展。本轮没有新增交通采集、模型优化、验证/测试标签读取、阶段6、push或tag。
