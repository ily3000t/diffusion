# 实施进度

## 阶段 0：已验收

所有实现位于 `E:\diffusion_new`。旧目录仅只读检查，没有复制工程、权重或数据，没有修改参考目录。

### 交付
- 设计、数据/坐标合同、阶段验收、Git规则、实验协议、环境和来源记录落盘。
- 独立src包、候选配置分类、保留模块目录及README、pyproject、artifacts忽略规则。
- 清单工具：Git SHA/分支/准确工作区状态，完整实际配置，种子，完整argv/PowerShell命令/工作目录，输入与checkpoint哈希，真实环境及其哈希，指标与运行状态。
- 正式模式拒绝脏工作区、无提交仓库、未忽略的仓库内输出；拒绝覆盖已有运行、无效种子、未实现配置组合和科学流程。异常保留failed状态并抛出，初始metrics为空。

### 实际验证
验证解释器：`E:\Programs\EnvAnaconda3\envs\pytorch\python.exe`。

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe' -m pytest -q --basetemp artifacts/cache/pytest-stage0-final
& 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe' -m pip --disable-pip-version-check wheel --no-deps --no-build-isolation --no-index --wheel-dir artifacts/cache/wheels-final .
& 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe' -m sumodiff.experiments record --config configs/experiments/manifest_smoke.yaml --output artifacts/runs/stage0-accepted-20261004 --seed 20261004 --formal
```

- 最终19项测试通过，10.00s；配置YAML/TOML与UTF-8检查通过，git diff --check通过。
- 本地wheel构建成功，直接从wheel独立导入通过，没有安装/升级全局环境，没有下载依赖。
- 真实环境清单六文件完整，状态completed，配置与环境SHA-256核对通过；运行代码已提交、工作区干净。
- 已有运行目录不能重用；重跑请选新目录。
- 验收清单是metadata smoke，不是模型实验。metrics为空，没有科学结果或实验里程碑tag。

### 资源实测与待测
RTX 4070 Laptop GPU：8585216000 bytes / 8188 MiB；驱动610.47；PyTorch2.5.1/CUDA build12.4，CUDA可用。最终元数据清单记录耗时 `3.124` 秒，包括环境查询，不是生成或训练耗时。SUMO二进制1.22.0，现有TraCI/sumolib1.25.0。

没有模型单批峰值显存、batch size、吞吐量或正式训练时长测量，阶段4/5再测。总显存不能代替模型峰值。

### Git 与追溯
远端成功只读查询无引用，才创建新历史。main初始无功能基线 `e297483`，开发分支 `chore/project-foundation`。

| 提交 | 职责 |
|---|---|
| e5a3107 | 完整设计与实验合同 |
| 85451c2 | 独立包布局与依赖分组 |
| 45f25a5 | 环境与清单实现 |
| 6bdfd3f | 追溯、失败和保护测试 |
| 84ebd81 | 工作目录、准确状态和环境哈希 |

最终清单运行SHA为 `84ebd8192289a696de5419b3e52bcaefa5849106`，运行分支为chore/project-foundation，输出位置为artifacts/runs/stage0-accepted-20261004。本进度与summary随后提交，不回填该SHA。可提交的小型验收summary在 `stage0_validation.json`，原始环境和结果不进Git。

验收分支通过 `--no-ff` 合并main保留阶段记录。最终交付/合并SHA和远端推送状态以Git记录及本次交付报告为准，不预先假设推送成功。

### 尚未解决与下一阶段
- 阶段1—8均未实现；没有采集数据、模型、采样、评估或RL控制器。
- 阶段1先建立隔离的SUMO客户端环境，将客户端与二进制统一版本并短回合验证；不升级旧训练环境。
- 下一阶段分支建议 `feat/sumo-collection`：参数化三类场景、正常安全规则/连续换道、0.1s完整回合与生命周期记录，只运行少量短回合，验证换道、转弯、停车及进入退出。
- 不默认开展批量采集、长训练或阶段8研究，不创建版本tag。
