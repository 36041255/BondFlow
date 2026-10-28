# 远程与本地 BondFlow 筛选对比

审计日期：2026-10-28。范围为只读源码、环境元数据、目录容量和既有结果/日志；未安装环境、未迁移权重、未运行 Rosetta 或 HighFold。

远程根目录：`/home/fit/lulei/WORK/xjt/Protein_design/BondFlow`。
本地根目录：`/home/xjt/BondFlow`。
下文代码行号基于此次审计快照，后续修改可能使行号变化。

## 结论

本地具备运行同一筛选流程的硬件条件，但当前缺少远程的分析环境与 HighFold 环境，筛选源码也存在实质差异。只复制环境或执行本地 requirements.txt 不足以复现远程结果。应迁移远程筛选实现、定制 HighFold 代码、两个环境及模型参数，并统一参数后做对照。

当前远程运行日志确认使用 `single_sequence`，不需要迁移 MSA 数据库。两端数值是否可比仍需实测；FastRelax/PackStat 的随机性、CPU/GPU 数值差异和历史缓存可能影响临界样本的通过结果。

## 代码与默认参数

| 项目 | 本地 | 远程 |
|---|---|---|
| 统一筛选入口 | knottin_pipeline.py，1816 行 | 同名，3501 行 |
| Rosetta 模块 | energy.py，2244 行 | 同名，3512 行 |
| 主流程 MSA | mmseqs2_uniref，max_msa=256:512，填写远程路径 | single_sequence，清空 MSA 参数 |
| HighFold 二硫键信息 | 把检测到的配对传给 HighFold | shell 默认 no-disulfide-constraints=true，预测时清空配对 |
| 头尾环化预测 | 随检测结果 | 默认仍随检测结果，可显式禁用 |
| recycle / 模型数 | 主流程均为 3 / 3 | 主流程均为 3 / 3 |
| PLDDT 门槛 | >=70 | >=80 |
| 新增成键门槛 | 无 LINK 分类列 | Covalent_Bonds>=2；Disulfide/Amide/Lactone>=0 |
| 新增形状门槛 | 无 | Rg_Calpha<=50 |
| 其他评分能力 | 基础能量、SAP、PLDDT、RMSD、dslf | 另有 fa_dun、Packstat、氢键、链间二硫键评分等 |
| 并行 | 单节点脚本 | Slurm 多节点分片；当前脚本12节点、每节点24核 |
| 复合物预测 | 无 | 可选 Complex_iPTM/PLDDT/IPAE，当前默认关闭 |

远程和本地脚本的共同门槛：Binding_Energy<=-40、SAP_total<=35、scRMSD<=3、dslf_fa13<=-1、target_chain_Energy_per_Res<=10。远程氢键阈值目前为 >=0，不构成正数氢键要求；提取 Packstat/fa_dun 不等于默认按它们筛选。

`calc_sap.py` 与 `config/link.csv` 的 SHA256 相同。主链 scRMSD、侧链 RMSD、序列提取、最佳 PLDDT 选择和基础 LINK 化学键创建的对应函数相同；差异主要来自输入预测条件、Rosetta 优化约束、新指标和筛选门槛。

### Rosetta 的实质变化

- 远程 pipeline 在检测到 LINK 时自动打开 link_constraints；本地只在发现缺 LINK 的头尾环时打开。对于已有侧链 LINK 的输入，两者可能使用不同的优化约束。
- 远程支持羰基额外角度、第二二面角、平面二面角等价约束及原子别名，影响酰胺/内酯的优化几何。
- 远程 energy 默认距离/角度/二面角 sigma 为 0.15 Å /15° /10°；本地为 0.10 Å /10° /20°。远程启动脚本显式把二面角设为20°，因此不能直接用 Python 默认值代表实际脚本运行参数。
- 远程允许对 Binding_Energy 使用删除链间 LINK 几何约束的 pose 副本；这不会移除化学拓扑或全部共价能量贡献。复合物 relax 与 Total_Energy 仍使用原来的约束。
- 双方都支持提取指定链后单链 FastRelax；这不是远程新增功能。Binding_Energy 的界面均按首链对其余链计算，--chain 不会重排该界面。

源码定位（相对上述各自仓库根目录）：

- 本地 pipeline:1177、远程 pipeline:2107：自动启用 link_constraints 的条件。
- 远程 energy:559、602、731：原子别名、额外角度、平面/第二二面角约束。
- 远程 energy:2317、2759：Binding_Energy 删除链间几何约束的精确定义与实现。
- 远程 pipeline:1557、1560、1567：NumSS、LINK 计数、禁用预测二硫键约束。
- 本地 shell:56、81、远程 shell:99、114、162：阈值和预测条件。

独立入口 evaluate_knottins.py 也有差异：远程支持本地 MSA 搜索、recycle/模型数参数和 CPU 管理，本地更简化。不能把该独立入口的缺失能力归到本地统一 pipeline；后者仍支持 MSA 数据库路径和 recycle/模型数参数。

## 环境与存储

远程使用各环境 Python 实际读取 importlib.metadata 得到：

| 环境 | 关键版本 | du 容量 |
|---|---|---|
| analysis | Python3.10.19、PyRosetta2025.45+release.d79cb06334、NumPy2.2.6、Pandas2.3.3、SciPy1.15.3、Torch2.9.1 | 约11GiB |
| HighFold | Python3.10.19、ColabFold1.5.5、AlphaFold-ColabFold2.3.11、JAX/JAXlib0.5.3、OpenMM8.2.0 | 约14GiB |
| 完整模型参数 | HighFold2/localcolabfold/colabfold/params，含单体/pTM/multimer参数 | 5,587,958,764 字节，du约5.3GiB |

远程 Conda 元数据另显示 mmseqs2 18.8cc5c、cudatoolkit11.8 和 jax-cuda12-plugin0.5.3。这反映环境中混装组件，不能仅凭 cudatoolkit 一项判断实际 JAX CUDA 运行要求。analysis 中 SciPy 的 Conda 元数据与实际 Python distribution 元数据不完全一致，应以实际导入环境及文件为准。

远程 HighFold 的 site-packages/colabfold/batch.py 包含二硫键配对、head_to_tail 和环化位置编码定制。pip 元数据显示的 ColabFold1.5.5/上游 commit 并不能描述这些文件修改。迁移必须保留实际安装代码，不能只重新安装官方同版本包。

本地实测：8×A100-PCIE-40GB，驱动590.44.01；约503GiB内存；/home剩余约883GiB。本地 BondFlow 环境无 PyRosetta/JAX/ColabFold；名为 rosetta_env 的环境目前仅有基础 Python，不能按环境名认为其可用于筛选。

两端均为 x86_64 Linux；远程 glibc2.35，本地2.39。未发现硬件架构上的明显迁移障碍，但没有在本地执行迁移后的二进制测试。

远程公共路径 /home/fit/lulei/WORK/public/database 当前为空；项目 public_databases/AF2_DATABASE 下存在 UniRef30_2202 数据库，顶层普通文件逻辑大小合计768,750,971,074字节（约716GiB，不含更深层目录及重复数据优化）。当前 single_sequence 流程无需它，直接完整迁移会大幅消耗本地空间。

## 建议迁移与验证顺序

1. 保存当前本地修改，单独同步远程筛选 Python 文件、Rosetta residue params 和启动参数；避免覆盖生成模型/guidance 的本地改动。
2. 为 analysis 与 HighFold 建立独立本地环境。优先采用可修复环境前缀的打包方式，保留 HighFold 已安装补丁；同时保存实际 package 清单、代码校验和与模型参数校验和。
3. 修复 bin 脚本 shebang、环境前缀、Conda 激活脚本、数据路径及 editable 安装。远程 analysis 的 editable finder 含 /WORK/PUBLIC/lulei_work/... 绝对路径，直接复制目录不够。
4. 复制 HighFold2/localcolabfold/colabfold/params，并保留远程显式 --data 的行为；不迁移 MSA 库。环境、参数本体合计约30GiB，预留60–100GiB用于迁移临时文件/缓存较合理；批量结构产物另计。
5. 以少量相同输入和全新输出目录做对照，覆盖二硫键、头尾环与侧链酰胺/内酯；先统一CPU模式、随机种子及评分参数，再验证GPU模式。
6. 分别比较 relax 后几何、能量、SAP、PLDDT/scRMSD、指标缺失情况与最终通过名单；运行时间和并行量单独测试，不以 GPU 型号直接承诺全流程吞吐。

## 结果解读与缓存

远程 MC4R/MDM2 已有 all_metrics.csv 和 passed_structures.csv，确认新增指标实际进入过流程。但它们不构成所有行均由当前脚本版本生成的证据：

- pipeline 按名字/文件完整性复用 Energy_results.csv 与 all_metrics.csv；不核对参数或代码哈希。
- --resume-from-relaxed 可对已有 relaxed PDB 跳过 FastRelax，只重新评分。
- HighFold 目录和已有PLDDT/RMSD也会复用。改预测约束或MSA参数后沿用旧目录，可能没有重新预测。
- NumSS根据提取肽链的 SG 距离检测；Disulfide_Bonds仅计 relaxed PDB 的 SG-SG LINK，不读SSBOND或几何。实际CSV中存在NumSS=2但Disulfide_Bonds=0的记录，这不是同一指标的矛盾。Covalent_Bonds门槛实际上还依赖LINK记录是否保留。

本次只做可行性与差异审计，没有证明本地已经能够运行该流程，也没有证明两端分数逐项相同。
