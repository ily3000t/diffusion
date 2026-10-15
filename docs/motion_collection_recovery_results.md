# Motion 采集恢复验收（2026-10-15）

原stage5a入口已加入显式源隔离策略与只读检查。功能提交4772490；协议提交9ca104187000c9247e03950084d12f7f000838f7。后续结果文档及合并不回填实验源SHA。未启动剩余正式采集、Prepare长任务、训练或阶段6。

## 验证

- 全量540项测试通过（68.91s），包含19项恢复/拒绝证据/源身份/账本与旧模式回归。使用artifacts/cache/pytest-recovery-full-20261015-r1作为新的工程内临时目录；首次系统pytest临时目录权限失败属于环境setup错误，已通过显式工程内basetemp解决，不改全局环境。
- 合成元数据测试验证CheckOnly零调用、恢复仅运行未完成任务、不重试完整拒绝种子、默认Stop继续报错、执行失败保留failed、不生成最终来源registry；Prepare核对全任务结果并拒绝缺失、重复、改写理由、改写策略、失败控制运行和重新加入拒绝源。合成接线检查不声称新SUMO数据或7200窗准备已经完成。
- PowerShell语法解析及git diff --check通过。
- 真实只读运行artifacts/runs/stage5a-motion-v1-recovery-audit-20261015-collect，actual SHA=9ca104187000c9247e03950084d12f7f000838f7，feat/motion-collection-recovery，formal/clean/completed；867个输入身份进入manifest。
- 实际计数216计划、96已完整处理、95accepted、1rejected、120pending。直路48accepted，汇入47accepted/1rejected。唯一拒绝motion-train-ramp_merge-g07-s05，原seed20271906，理由collision及teleport_started。未新增attempt、未生成最终sources.json，原始来源不变。
- SUMO调用0，无模型训练/显存与训练吞吐测试。CheckOnly的status时段仅记录清单落盘阶段，不包含前置扫描/环境和输入哈希，不将其当作整体恢复扫描耗时。

## 用户执行

已完成CheckOnly，无需重复同一后缀。当前恢复输出recover1、processed/stage5a-motion-v1均不存在，可以执行：

~~~powershell
Set-Location -LiteralPath 'E:\diffusion_new'
.\scripts\stage5a.ps1 -Action Collect -DataProtocol Motion -Name stage5a-motion-v1 -ReuseCompleted -QualityPolicy Quarantine -RunSuffix recover1
~~~

Collect返回成功后：

~~~powershell
.\scripts\stage5a.ps1 -Action Prepare -DataProtocol Motion -Name stage5a-motion-v1
~~~

-Name保留原活动数据身份；新-RunSuffix只创建新控制运行。若中断再次恢复，换未占用的recover2等。已存在最终sources.json时禁止覆盖，应执行Prepare。原始回合无需删除。新策略仅允许完整异常源拒绝，配置/进程/证据错误仍停止；全部216任务结果留在sources.v2账本。
Prepare仍要通过原6000/600/600、参考/运动/唯一事件及几何/质量门槛；不足可能失败，不自动补采或降低阈值。该恢复没有消除汇入连接的潜在源问题，后续如异常集中，应根据分道路/几何拒绝统计做局部诊断。
