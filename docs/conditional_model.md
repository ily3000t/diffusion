# 阶段4：共享条件编码与联合噪声预测

本阶段网络未训练，不提供DDIM或轨迹生成。配置initial.yaml已替换阶段0的候选占位文档，保存全部实际生效架构和固定条件特征单位。旧checkpoint不兼容。

## 输入与共享编码

`prepare_conditioning(collate_numpy(inference_samples), device)`只读取conditioning、exact_map、input_metadata；不读取targets、labels、future_mask或退出时刻。阶段2栅格是uint8的0/1，直接转float，禁止猜测为0/255。适配器保留逐车t0路线序列和route_index，不修改原数据。

| 条件 | 形状 |
|---|---|
| history / history_mask | [B,N,21,6] / [B,N,21] |
| agent_mask / attributes / initial_positions | [B,N] / [B,N,4] / [B,N,2] |
| map_raster / map_extent_m | [B,3,256,256] / [B,4] |
| lane_polylines / lane_point_mask / lane_mask | [B,L,64,8] / [B,L,64] / [B,L] |
| lane_adjacency | bool [B,L,L]，有向合法后继 |
| lane_rule_relations | bool [B,L,L,2]，让行方向/冲突 |
| route_lane_mask / route_lane_features | bool [B,N,L] / float [B,N,L,2] |

规则从已核对的SUMO请求关系生成：每个movement以首条via车道为代表，无via时使用source；request i让行给j为[i,j,0]，foes为[i,j,1]。不把相交或优先数字自动视为合法连接。完整连接、内部链和原始请求仍在exact_map。多个请求共用代表车道时关系做并集，因此这是车道条件摘要，不是显式车辆优先决策或完整连接级注意力。

路线特征为允许车道的相对计划次序与尚未经过标志。外部车道rank为已知路线edge次序，内部连接rank为相邻edge次序均值；与t0 route_index作差并按已知路线长度缩放。非路线/无效槽位屏蔽。重复edge路线明确不支持，不能猜测绕圈次数。

共享历史CNN使用8输入通道（六状态、有效点标志、固定历史时间坐标），三层kernel3、dilation1/2/4，每点LayerNorm、SiLU、逐层mask，池化有效点均值/最后有效点。已知属性、t0初始位置及历史有效比例另经MLP，合并为逐车128维历史特征。属性含长宽、乘用车码1和参考标记，没有攻击角色和车辆槽位embedding。当前历史全缺失仍有当前位置/属性，解码器需要的观测方向仍须另按阶段3合同提供，不能读取未来。

栅格CNN五层kernel4/stride2/padding1，通道16/32/64/128/128，每层GroupNorm8和SiLU；得到8×8的空间token，经1×1投影至128维，加入token对应的固定局部物理xy编码。行从+y到-y，位置依extent计算。栅格通道不与车辆槽位绑定。

车道逐点MLP编码8维车道特征和固定点序，按点mask池化；一层图消息聚合合法后继、前驱、让行对象、冲突对象四种邻域均值，再残差MLP/LayerNorm。路线编码对允许车道token及计划次序特征作共享MLP/掩码池化，不引入槽位ID。

条件特征单位候选：位置50m、速度/限速20m/s、车长10m、宽5m、优先数字3；sin/cos、方向、类型/标志保留。这是固定输入单位换算，不做逐场景标准化，不裁剪数据；阶段5仍须用train/validation检查并冻结模型未来状态归一化，checkpoint保存全部模型/尺度配置。

## 两种融合的公平接口

条件128维，4头，各融合阶段一层，dropout=0。道路交叉注意力query=历史+路线特征，keys=车道与64个栅格token；道路特征为路线残差加cross-attention后LayerNorm。所有车辆使用共享层。

hierarchical车辆交互query/keys为历史+道路条件；parallel为历史。两种模式使用同一组社会注意力/FFN和全部编码、门控、U-Net参数，仅改变依赖顺序。三路历史/道路/社会特征经逐车softmax门控混合，MLP/LayerNorm输出[B,N,128]。可用同一model的fusion参数切换；不同模式实例可直接加载相同state_dict，参数总数相同。

attention用显式Q/K/V投影和缩放点积，logits及softmax在float32并关闭该处autocast。无效键参与归一化前屏蔽，全部键屏蔽时更新为0，避免NaN；无效query输出0。有效车辆缺少可用计划路线token则报错，不静默退化为无路线模型。有效非有限输入报错；无效数据在所有运算前屏蔽。

层次设计只是可替换架构，目前没有训练或质量对照，不能宣称它优于parallel或形成新贡献。

## 联合时间U-Net

`ConditionalDenoiser(noisy_future, timestep, conditioning, fusion=..., return_details=...)`预测epsilon，输入/输出[B,N,40,6]。噪声未来的归一化及扩散日程留阶段5；本阶段使用随机单位高斯latent测试，不当作物理轨迹解码。

先把车辆作为共享时间CNN的独立batch项。时间维40→20→10，通道64/128/256。残差块为GroupNorm8、SiLU、kernel3卷积，加逐车条件与扩散整数步sinusoidal embedding的FiLM scale/shift。每个未来索引另有sinusoidal位置编码。embedding步范围0..999是当前接口候选，不等于已经实现1000步扩散。

瓶颈reshape成[B,10,N,256]，每个下采样未来时间位置独立进行车辆注意力/FFN，没有跨时间注意力或槽位位置embedding。时间卷积仍可跨邻近未来时刻。之后最近邻两倍重复+卷积上采样至20/40，接skip和残差块，输出六通道epsilon。无效车辆在输入和最终输出隔离，所有中间社会注意力使用agent mask。

本机torch2.5.1的linear1d上采样反向不支持严格确定性，失败预演保留；最终使用nearest_repeat并仍开启确定性，未使用warn_only或自动降级。

cfg、partial_diffusion、rolling_generation、attack_role_embedding默认false，启用明确NotImplementedError；当前只实现一层融合，不接受伪多层开关。固定21/40/6、256栅格、64折线点接口之外明确不支持。

## 正式短验证与重放

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
$env:SUMO_HOME = 'E:\Program Files\sumo-1.22.0'
& '.\.venv-model\Scripts\python.exe' -m pytest -q --basetemp artifacts/cache/pytest-stage4-final
& '.\.venv-model\Scripts\python.exe' -m sumodiff.models smoke --dataset artifacts/processed/stage2-accepted-20261004 --model-config configs/model/initial.yaml --probe-config configs/experiments/stage4_smoke.yaml --output artifacts/runs/stage4-accepted-20261005 --formal
& '.\.venv-model\Scripts\python.exe' -m sumodiff.models audit --run artifacts/runs/stage4-accepted-20261005 --output artifacts/runs/stage4-accepted-audit-20261005 --formal
```

sumodiff-model与python -m入口等价。正式运行必须已提交且干净；输出目录不可覆盖。smoke校验全部train/validation的当前输入，只按t0选车数为每类选择一个负载窗口；不使用test或未来完整性选择。另用同一个地图/已知路线的合成12车条件测试batch1/2/4，仅为资源负载，不声称真实12车SUMO数据。

保存标准运行六文件、完整配置、全部实际当前输入哈希、参数初始化种子/哈希、随机噪声种子、车辆重排/NaN padding/梯度结果、逐case时间显存与未训练输出NPZ。两种融合共享参数和初始噪声，模型没有optimizer或权重更新，结束时验证参数哈希未变。checkpoint=null；不会保存一个假训练模型。

CPU重放重新初始化同参数，检查所有输入/输出/config/environment哈希，按当前条件重算两种融合的epsilon、condition和gates，与GPU保存结果在公开容差内匹配。正式audit拒绝脏的非正式源运行；源文件被后来修改也报错。

计时在预热后同步CUDA，float32，TF32和AMP关闭，严格确定性。包含完整网络、输入及参数梯度和接口验证开销；不含optimizer状态、数据加载、DDIM、解码或正式训练。torch allocated/reserved是分配器峰值，不是驱动总占用。通过batch4不代表已测最大batch或正式训练吞吐量。
