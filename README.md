# SUMODiff

独立的 SUMO 多车轨迹扩散生成工程。当前仅完成阶段0：设计/数据合同、工程结构、环境核对和运行清单工具。尚未实现采集、模型、训练、采样或评估，旧checkpoint不兼容。实施范围见 [设计](docs/design.md)、[数据合同](docs/data_schema.md)、[进度](docs/progress.md)。

## 目录
`configs/` 分类保存候选配置，`src/sumodiff/` 为新代码，`scripts/` 为入口辅助，`tests/` 为短验证，`docs/` 为合同和协议。`artifacts/{raw,processed,cache,checkpoints,runs}` 全部忽略，不提交数据或权重。保留模块目录标明后续阶段，不包含假实现。

## 阶段0短验证（PowerShell）
以下使用本机已核对的环境，不要求安装工程或修改全局环境：

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe' -m pytest -q
& 'E:\Programs\EnvAnaconda3\envs\pytorch\python.exe' -m sumodiff.experiments record --config configs/experiments/manifest_smoke.yaml --output artifacts/runs/manifest-smoke --seed 20261004
```

重复执行时改用新的输出目录；工具拒绝覆盖。命令只记录真实环境和追溯信息，不执行科学实验，metrics初始为空。正式实验入口在后续阶段实现；清单工具 `--formal` 会拒绝未提交或脏工作区。运行清单包含实际生效配置，不只保存文件路径。

## 环境与安装方案
已核对：Python 3.10.16、PyTorch 2.5.1/CUDA build 12.4、RTX 4070 Laptop GPU（8188 MiB）、驱动610.47、SUMO二进制1.22.0。已有TraCI/sumolib为1.25.0，与二进制不一致，阶段1先统一版本。默认Anaconda Python 3.12.7没有PyTorch。

当前阶段0只依赖PyYAML，测试额外依赖pytest，均已存在。未来使用项目隔离环境，不修改全局环境；先安装 `.[dev,sumo]`，SUMO Python客户端锁为1.22.0匹配现有二进制。进入模型阶段再安装并核对PyTorch/CUDA，不在阶段0下载大型依赖。具体方案见 [环境记录](docs/environment.md)。

## 继续实施
每次先读AGENTS、design、data_schema、progress，检查Git和已完成内容；只完成指定阶段。正式大采集和训练默认不启动。目标远端为 https://github.com/ily3000t/diffusion.git，阶段验收后再合并main，不根据代码完成自动打实验tag。
