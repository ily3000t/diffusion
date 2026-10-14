# 方向换道权限条件编码

本阶段只补齐新地图条件，完成有界GPU前后向与旧checkpoint兼容检查，不替换数据、不训练新模型、不进入阶段6。源头证据见[ramp_alignment_results.md](ramp_alignment_results.md)。

## 接口与语义

新架构schema为sumodiff.model.control.config.v2；历史仍六维，未来仍三维[a,omega,r]，归一化和loss保持原合同。旧sumodiff.model.config.v1与model.control.config.v1不增加配置字段或权重键。新候选位于configs/model/control_directional.yaml，旧control.yaml保持。

运行时适配器显式接收model config，从精确地图生成lane_change_permissions[B,L,2]布尔量，通道为左/右。该方向按SUMO沿行驶方向的车道索引定义，不随参考车辆坐标旋转交换。缺少class限制表示乘用车允许；显式列表含passenger或all为允许，其他类别或空列表为禁止；填充车道置false。

这是乘用车类别权限，不代表存在相邻车道、当前可安全换道或路线所需动作。几何、连接及通行优先关系仍由原折线/图条件表达，不将静态权限称为完整交通规则执行器。地图通道和既有8维折线保持，新增2维经独立共享MLP加入车道token后再进行图消息和路线聚合；车辆槽位不绑定到地图通道。

prepare_conditioning只读取conditioning、当前路线与exact_map；future_mask和真实退出时刻仍不进入推理。旧架构遇到受限地图继续报错，新架构缺少权限字段/错误类型或形状报错。掩码贯穿投影、图消息与注意力。

原train/control calibration/sample入口均使用显式架构消费此条件；模型smoke探针根据架构生成3/6维噪声。checkpoint外层仍base.checkpoint.v1，架构版本进入原完整训练签名；读取时验证权限MLP权重存在性和形状，旧架构不得包含这些权重。跨版本restore在修改权重/优化器前拒绝。新模型从头初始化，未实现旧头暖启动。

## 固定短验证

现有模型CLI增加rules-smoke，配置configs/experiments/directional_conditioning.yaml。仅使用上一阶段4个combined训练回合及1个control训练回合，固定t0=300。认证完整原始文件，但构造特征时先截断回合到t0，未来观测不可用；不读取存储训练/验证/测试标签数组。

GPU保持128维/4头原候选，随机初始化、优化0；检查两融合的车辆重排/填充/有限梯度、车道重排、静态权限敏感性和参数不变。另测5真实输入批次与合成B4/12槽位前后向，各2次、预热1次；后者仅资源负载。加载旧六维200轮及旧三维16更新checkpoint，在无限制输入上检查严格加载/有限前向，拒绝跨版本恢复；新checkpoint实际AdamW/RNG/日程精确续接由合成单元测试核验。

敏感性只说明条件通路有效，随机模型不证明规则遵从或生成质量；资源不含正式优化器状态，不据此推断长训练吞吐。已有控制训练<=64更新限制及test封存限制保持。

~~~powershell
$env:PYTHONPATH='E:\\diffusion_new\\src'
& '.\\.venv-model\\Scripts\\python.exe' -m sumodiff.models rules-smoke --config configs/experiments/directional_conditioning.yaml --output artifacts/runs/directional-conditioning-20261014-r1 --formal
~~~

正式运行先提交并保持干净，完整配置、命令、种子、环境及源身份入清单。验收后再冻结参考混合、运动/唯一事件配额和新数据版本，重新审计尺度与完整组合梯度；不自动启动采集或长训练。
