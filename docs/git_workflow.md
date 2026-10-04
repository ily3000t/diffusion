# Git 工作流程

目标远端：`https://github.com/ily3000t/diffusion.git`。阶段0以实际 `git ls-remote --symref` 成功但无输出确认远端没有引用。仅因此创建新历史。初始 main 是一个无功能的基线提交，用于保留阶段分支 `--no-ff` 合并记录；所有功能在阶段分支完成。

开始每阶段：读 AGENTS 和设计/进度，检查 `git status --short --branch`、远端和现有提交。已有项目沿用其历史。发现不属于本任务的修改须保留并隔离，禁止强行清理。

```powershell
git -C E:\diffusion_new switch main
git -C E:\diffusion_new switch -c feat/sumo-collection
# 实现一个完整单一职责的修改、验证，再显式 git add 对应文件
git -C E:\diffusion_new commit -m 'feat(simulation): collect vehicle states and lifecycle events'
# 当前阶段验收、更新 progress 后
git -C E:\diffusion_new switch main
git -C E:\diffusion_new merge --no-ff feat/sumo-collection -m 'chore(release): merge accepted SUMO collection stage'
```

一个阶段通常多个原子提交。大重构前已有可恢复提交，不使用无必要的 reset --hard、force push 或改写共享历史。文件清单和 staged diff 必须检查，数据/权重不入 Git。

正式运行前提交全部代码、配置，确认干净工作区；清单记录运行时 SHA。随后 summary 新提交保留原 SHA，不回填。推送认证/权限失败立即停止、不反复重试，保留本地并提供实际可用命令。

里程碑条件：

| Tag | 必须满足 |
|---|---|
| v0.1.0 | 小规模采集—切窗—训练—基础采样—评估完整可复现 |
| v0.2.0 | 固定引导与base的统一协议比较完整可复现 |
| v0.3.0 | 可选控制器及必要对照的完整研究实验可复现 |

版本可调整；代码写完或测试通过本身不能创建这些tag。阶段0不tag。
