# SUMODiff

独立的 SUMO 多车轨迹扩散生成工程。当前实现阶段0工程追溯、阶段1三类SUMO场景采集及阶段2固定坐标/地图/窗口数据。验收状态以progress为准。模型、训练、扩散采样及独立评估尚未实现，旧checkpoint不兼容。实施范围见 [设计](docs/design.md)、[数据合同](docs/data_schema.md)、[进度](docs/progress.md)。

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

基础追溯依赖PyYAML，测试依赖pytest；数据阶段额外依赖NumPy、Shapely、Pillow和psutil。阶段1使用项目 `.venv-sumo` 和1.22.0安装目录自带客户端，严格检查来源与版本，不修改全局包。`sumo`依赖组仅提供进程内存测量的psutil，不重复安装客户端。进入模型阶段再安装并核对PyTorch/CUDA，不在阶段0下载大型依赖。具体方案见 [环境记录](docs/environment.md)。

## 继续实施
每次先读AGENTS、design、data_schema、progress，检查Git和已完成内容；只完成指定阶段。正式大采集和训练默认不启动。目标远端为 https://github.com/ily3000t/diffusion.git，阶段验收后再合并main，不根据代码完成自动打实验tag。

## 阶段1短回合采集

```powershell
Set-Location E:\diffusion_new
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv-sumo\Scripts\python.exe' -m sumodiff.simulation collect --config configs/scenarios/ramp_a.yaml --output artifacts/raw/my-ramp-episode --seed 21 --formal
```

详见 [采集协议](docs/sumo_collection.md)。六个正常场景验收配置覆盖每类两个几何；诊断回合单独标记，不进入正常训练数据。达到限时但未排空会失败并保留输出，不能当作完整回合。

## 阶段2窗口数据

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv-sumo\Scripts\python.exe' -m sumodiff.data process --config configs/data/window.yaml --sources configs/data/sources_stage2.json --output artifacts/processed/my-windows --formal
```

须先提供清单中的原始回合，所有输出使用新目录。详见[切窗协议](docs/window_dataset.md)：按几何及回合先划分，t0选车，历史21点/未来40点，未来标签与推理条件分文件；独立审计核对原始状态及SUMO规则。不完整窗口保留，工程验收数据不能代替正式训练数据。
