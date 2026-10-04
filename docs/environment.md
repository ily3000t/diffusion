# 阶段0实际环境与后续方案

本记录来自实际只读查询，未安装、升级或删除全局包。原始查询输出在阶段0运行清单的environment.json中追溯。

| 项目 | 实测 |
|---|---|
| 操作系统 | Windows，PowerShell |
| Git | 2.50.1.windows.1 |
| 默认Python | E:\Programs\anaconda3\python.exe，3.12.7；未安装torch |
| 可用训练环境 | E:\Programs\EnvAnaconda3\envs\pytorch\python.exe，3.10.16 |
| 另一现有环境 | pytorch_copy，核对到相同主要版本 |
| PyTorch | 2.5.1，CUDA build 12.4，cuda_available=True |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| 总显存 | 8585216000 bytes / 8188 MiB；不是模型峰值实测 |
| NVIDIA驱动 | 610.47；驱动CUDA兼容信息与PyTorch build不同 |
| NumPy/PyYAML/pytest | 2.2.6 / 6.0.2 / 9.0.3（pytorch环境） |
| SUMO/netconvert | E:\Program Files\sumo-1.22.0\bin；SUMO 1.22.0 |
| SUMO_HOME | E:\Program Files\sumo-1.22.0 |
| 已安装客户端 | traci/sumolib 1.25.0，与二进制1.22.0不一致 |

## 分阶段安装方案（未执行）
阶段0现有环境可运行PyYAML清单和pytest，不安装工程到全局site-packages。通过临时进程PYTHONPATH指向本工程src，或后续在项目虚拟环境安装。

阶段1建议以明确的Python解释器创建 `E:\diffusion_new\.venv`，安装 `.[dev,sumo]`，客户端锁1.22.0以匹配二进制，执行客户端/二进制一致性与短回合连通测试。另一方案是使用SUMO_HOME/tools自带客户端，但必须明确来源和版本，不静默切换。未进行这项验证前，不宣称采集环境验收。

阶段3/4再配置NumPy与PyTorch，并核对CUDA可用性、数值解算精度和实际单批前后向峰值显存。已有pytorch环境可用于只读探测或经记录的短验证，但不升级它的包。

`pyproject.toml` 的model范围是初始兼容候选，不是完成验证的锁文件。正式实验保存全部实际依赖版本，阶段验收后再锁定实测组合。SUMO可执行文件不由pip依赖安装。

未实测：模型显存、batch size、训练吞吐量、训练时长、采样成本。后续按实际测量决定，不能以GPU总显存或旧项目耗时冒充。
