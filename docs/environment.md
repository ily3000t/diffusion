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

阶段1已选择下节的SUMO_HOME/tools随包客户端方案。后续纯隔离环境可安装 `.[dev,sumo]` 提供PyYAML、pytest和psutil；TraCI/sumolib来自同一SUMO安装，并核对实际bundle revision和协议。入口不静默切换到其他已安装客户端。

阶段3/4再配置NumPy与PyTorch，并核对CUDA可用性、数值解算精度和实际单批前后向峰值显存。已有pytorch环境可用于只读探测或经记录的短验证，但不升级它的包。

`pyproject.toml` 的model范围是初始兼容候选，不是完成验证的锁文件。正式实验保存全部实际依赖版本，阶段验收后再锁定实测组合。SUMO可执行文件不由pip依赖安装。

未实测：模型显存、batch size、训练吞吐量、训练时长、采样成本。后续按实际测量决定，不能以GPU总显存或旧项目耗时冒充。

## 阶段1采用的客户端隔离（已建立）

项目内 `.venv-sumo` 基于默认Python3.12.7，以 `--system-site-packages` 复用已有PyYAML/pytest，不安装或升级全局包。这是项目解释器和SUMO客户端来源隔离，其他依赖仍继承基础环境，全部版本进入运行清单。

采集入口显式加载SUMO_HOME/tools随包TraCI及sumolib；不使用pytorch环境里1.25.0的客户端。入口检查实际模块路径、二进制版本及运行时TraCI协议版本，记录客户端源码树哈希。不复制SUMO源代码进工程。阶段1无需PyTorch；该环境未安装torch会在环境记录中明确显示。

远端只读检查已确认main为14d4352，与本地阶段0交付一致，之前的网络推送问题已不影响阶段1沿用历史。
