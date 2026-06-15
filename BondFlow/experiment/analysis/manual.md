# Structure & Energy Analysis Pipeline

本项目提供一套完整的 **蛋白质结构比对、层级聚类、Rosetta 能量分析与结合能计算、Shannon entropy 计算和可视化** 的管线工具，可批量处理 PDB 文件，并支持多线程加速。

---

## 1.目录结构

```text
Nocta/
├── data/                  # 待分析 PDB 文件目录
├── results/               # 默认输出目录
├── main.py                # 主入口脚本
├── TMalign.py             # TMalign 批量比对模块
├── energy.py              # Rosetta 能量计算模块
├── shannon.py             # Shannon entropy 计算模块
├── cluster.py             # 层级聚类模块
├── visualize.py           # 可视化模块
├── README.md              # 使用说明
└── requirements.txt       # 需要安装的依赖包
└── setup.sh               #自动化部署环境脚本 
```

## 2.部署说明

### 2.1 脚本安装

使用 Conda 管理 Python 环境，Nocta 提供了自动化安装脚本`setup_env.sh`：

```bash
bash setup_env.sh
```

### 2.2 手动方式（可选）

如果不使用脚本，也可手动安装：

1.创建 Conda 环境：

```
conda create -n analysis python=3.10 -y
conda activate analysis
```

2.安装 Python 依赖：

```
pip install -r requirements.txt
```

3.安装 PyRosetta：

```
# 在 Python 交互式环境中运行，其中pyrosetta在requirements.txt中已经安装
import pyrosetta_installer
pyrosetta_installer.install_pyrosetta()
```

4.验证 PyRosetta 是否正常：

```
import pyrosetta
pyrosetta.init("-mute all")
print("PyRosetta successfully initialized!")
```

安装完成后，你就可以运行 `main.py` 进行结构分析、能量计算和 Shannon 熵计算。

## 各模块文件介绍

- **TMalign.py**：利用TMalign和USalign进行批量 PDB 比对，输出 RMSD 和 TM-score 矩阵。
- **energy.py**：使用 PyRosetta 中的计算单个 PDB 或批量 PDB 的总能量及结合能。
- **shannon.py**：计算每个位点的 Shannon entropy，可自动提取 PDB 序列。
- **cluster.py**：基于 RMSD 或 TM-score 的层级聚类，生成簇标签。
- **visualize.py**：提供可视化函数，包括热图、聚类树、能量分布和 entropy 图表。



num_WORKERS 选择建议

| num_workers | CPU 使用情况        | 并行度   | 建议             |
| ----------- | ------------------- | -------- | ---------------- |
| 1           | 顺序执行，最慢      | 无并行   | 调试用           |
| 4           | 同时跑 4 个 USalign | 4× 加速  | 轻量任务         |
| 8           | 同时跑 8 个         | 高并行度 | 16核机器推荐     |
| 16+         | CPU 饱和            | 最高并发 | 超算或多核服务器 |



RMSD一致但TM-score不同，归一化的长度不同

#### TM-score 的定义是不对称的

TMalign 的核心指标是 **TM-score（Template Modeling score）**，定义为：

```

```

$TM-score=max⁡[1Ltarget∑i=1Laligned11+(di/d0)2]TM\text{-score} = \max \left[\frac{1}{L_{\text{target}}} \sum_{i=1}^{L_{\text{aligned}}} \frac{1}{1 + (d_i/d_0)^2}\right]TM-score=maxLtarget1i=1∑Laligned1+(di/d0)21$

其中：

- LtargetL_{\text{target}}Ltarget：**归一化长度**（可以是结构 1 或结构 2 的长度）
- did_idi：第 i 对匹配原子间的距离
- d0d_0d0：距离归一化因子，依赖于 LtargetL_{\text{target}}Ltarget

这意味着：

- 如果你让 **A 对齐到 B**，那么 TM-score 就是用 **B 的长度**归一化；
- 如果你让 **B 对齐到 A**，归一化用的就是 **A 的长度**；
- 因此，**TM-score(A→B) ≠ TM-score(B→A)**。

# 

# 结构聚类 

