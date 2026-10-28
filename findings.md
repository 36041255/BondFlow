# Findings & Decisions

## Requirements
- Explain why disulfide bonds appear too few and other bond types are not suppressed, using `/home/xjt/BondFlow/BondFlow/config/log.txt` and the current config.

## Research Findings
- Current config has one `type_soft_bond_count` block for disulfide `range_N: [1, 2]` with `weight: 5.0`, `seq_step: 0.4`, `entropy_weight: 0.05`.
- A second `type_soft_bond_count` block suppresses isopeptide and lactone with `target_N: 0`, `weight: 10.0`, `seq_step: 0.5`.
- In implementation, `range_N` ignores `loss_type` and uses squared penalties only outside the interval.
- `post_step` constructs `seq_logits` from discrete `aatypes_t_2` via one-hot probabilities clamped to `[eps, 1-eps]`, runs a few gradient steps, then converts back with `argmax`. This makes residue-type changes require crossing a large logit margin.
- The linear schedule multiplies `bond_step`/`seq_step` by roughly `1 - t_1`; late in sampling (`t_1 ~= 0.98-0.99`) the effective steps become tiny.
- Log summary: disulfide soft count averages about 0.004 early, 0.215 mid, 0.524 late, and 0.511 very late. It remains below the configured lower bound 1.0 on average.
- Log summary: isopeptide/lactone soft counts rise late/very late, while each 5-step inner loop only reduces them slightly.
- Final `post_refine` bond files contain 37 disulfide candidates (36 valid), 16 isopeptide candidates (15 valid), 20 lactone candidates (14 valid), and 4 other candidates.
- Per-sample `post_refine`: 30/62 samples with candidate bonds have disulfides, 29/62 have valid disulfides; 16/62 have isopeptides and 20/62 have lactones.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Treat log trends as primary evidence | The issue is observed during sampling and debug logs should show active losses/counts |
| Treat `post_refine/bonds_final_refined_structure_*.txt` as final evidence | These files directly classify candidate bonds after refinement |

## Issues Encountered
| Issue | Resolution |
|-------|------------|

## Resources
- `/home/xjt/BondFlow/BondFlow/config/MC4R_design3.yaml`
- `/home/xjt/BondFlow/BondFlow/config/log.txt`
- `/home/xjt/BondFlow/BondFlow/models/guidance.py`

## 2026-10-25: Unpaired CYS implementation
Reuse disulfide P_eff after region/spacing masking. Penalize relu(p_CYS - row_sum(P_eff)) on valid mutable region endpoints. Preserve fixed residues as possible partners and existing local one-of-N changes. Default weight 0; TL1A weight 0.5.

## 2026-10-29: 项目流程阅读
- 已识别入口 BondFlow/sample.py；采样核心 models/sampler.py、interpolant.py、adapter.py、guidance.py。
- 后处理主候选为 experiment/analysis/knottin_pipeline.py；另有 main.py 的统计评估管线。
- README 的链编号与 hotspot 说明需对照数据处理实现，不能直接当作实际语义。
- sample.py 在创建 Target 后按 num_cycle 循环采样；每个 worker 只创建一次 Target，随机 contig 长度因此在 worker 内复用。
- sample_step 顺序为模型前向 → pre_model → 全原子/骨架构建 → pre_interpolant → 插值更新 → post_step；ss 表示成键矩阵。
- knottin_pipeline.py 为设计筛选主流程，main.py 偏向结构集合统计。筛选把指定指标条件做 AND。
- 已核实 parse_hotspot 使用 full_origin_pdb_idx；bond_condition 使用 full_pdb_idx（重排编号）。terminal clone 是 N/C 功能节点，最终 PDB 中移除。
- 采样后先以 BondCoherence base_matrix < 0.65 剪除候选键，再做侧链/端点梯度优化（未配置侧链模型时），分别保存 pre_refine 与 post_refine。
- HighFold 对从 relaxed 复合物提取的指定链单独预测；scRMSD 使用 N/CA/C/O 主链原子对齐，sidechainRMSD 单列。因此该步骤评估肽自身构象一致性。
- 筛选实现：已存在列的 NaN 不通过；缺失整列的阈值被跳过。Total_Energy 没有 lower-is-better 分支，会落入 >= 默认分支。

- 四份 YAML 已用项目环境 OmegaConf 解析：MC4R6 为 13–22 aa、32×1000=32000 个名义样本；MC4R5 为 8–20 aa、32×6400=204800；MDM2 测试为 16–19 aa、8×7=56；TL1A 为 25–30 aa、32×2000=64000；均为 200 步。
- MC4R6：设计链 A，原始受体 A 的片段映射为设计链 B；hotspots 原始 A268/A129/A188/A126/A261；禁止头尾成键与肽-受体新键；二硫键 one_of_N=[1,2]，异肽键/内酯键 exact_N=0。软引导不保证最终成键数量。
- TL1A 文件名 design1，但 output_prefix 指向 condition_large_disulf_design2；MDM2 测试 guidance.list 为 null，固定头尾键。
- 筛选入口 knottin_pipeline.py：Rosetta relax/界面及单链评分 → 提取指定链做 SAP 与 HighFold → all_metrics.csv → 阈值交集 → passed_structures.csv 和通过的 PDB。
- target_chain_Energy 命名中的 target 是 --chain 指定链，默认 A（设计肽）；Binding_Energy 为 Rosetta 分数，不能直接视为实验亲和力。
- 筛选的输入目录扫描不递归，应直接指向 post_refine。run_knottin_pipeline.sh 的阈值只是示例，路径含集群配置；筛选阈值与设计 YAML 分开。
- 原 README 的 hotspot 编号与 partial_t 描述存在偏差：实际 hotspot 用原始编号；partial_t 越大加噪越少。

## 2026-10-28: 远程筛选对比
- 已通过 SSH 控制连接只读下载小型筛选源码快照至 /tmp；未保存密钥口令。主流程、energy.py、evaluate_knottins.py 有差异；calc_sap.py 和 link.csv SHA256 完全相同。
- 远程 HighFold2/localcolabfold/colabfold-conda 存在；远程 analysis 环境位于 WORK/.conda/envs/analysis。本地两者缺失。
- 启动一次独立只读代码差异审查；主代理继续检查环境与实际产物。

## 2026-10-28: 远程筛选与本地差异
- 远程分析代码明显更新：knottin_pipeline.py 约 3501 行、本地 1816 行；energy.py 约 3512 行、本地 2244 行；evaluate_knottins.py 远程 1001 行、本地 616 行。calc_sap.py 与 link.csv 完全一致，eval_bonds.py 预测算法主体一致。
- 远程新增 LINK 分类指标（Disulfide/Amide/Lactone/Covalent）、氢键、Rg_Calpha、Packstat、fa_dun、链间二硫键能量，以及可选复杂物 HighFold iPTM/IPAE；本地版本没有这些指标及对应筛选分支。
- 远程 energy 支持排除 Binding_Energy 计算中的链间共价几何约束、链间二硫键惩罚、更多约束几何测量和单链额外评分；远程约束默认 sigma 为 distance .15 Å / angle 15° / dihedral 10°，本地旧版为 .10 Å / 10° / 20°。
- 远程默认筛选脚本使用 PLDDT 80、成键/氢键/Rg 等阈值；本地脚本使用 PLDDT 70，且只筛选旧指标。远程默认 HighFold 为 single_sequence、num_recycle=3、num_samples=3；本地默认 mmseqs2_uniref，并指向远程不可用的 MSA 路径。
- 远程可以多节点 Slurm 分片并支持 resume-from-relaxed；本地脚本是单节点直接运行。远程实际结果 CSV 已包含新增指标列并成功产生 passed_structures.csv。
- 远程 analysis 环境约 Python 3.10.19、PyRosetta 2025.45、Torch 2.9.1、NumPy 2.2.6、SciPy 1.15.3、Pandas 2.3.3；HighFold 环境约 ColabFold 1.5.5、AlphaFold-ColabFold 2.3.11、JAX 0.5.3、CUDA toolkit 11.8、MMseqs2 18.8cc5c、OpenMM 8.2，约 14 GB。远程 analysis 环境约 11 GB。
- 本地 BondFlow 环境有 Python 3.10.12、Torch 2.6.0、NumPy 1.26.3、SciPy 1.11.4、Pandas 2.1.4，但没有 PyRosetta、JAX、ColabFold/HighFold；本地有 8×A100 40GB、约 503 GiB RAM、剩余磁盘约 883 GB。
- 远程项目包含 HighFold2 模型参数（约 5.3 GiB）和本地 ColabFold 环境；项目 AF2_DATABASE 顶层普通文件逻辑大小约716GiB，完整迁移会显著消耗本地空间。远程当前脚本默认 single_sequence，因此复现当前流程不必复制 MSA 库；若坚持 mmseqs2_uniref，则必须另行准备数据库。
- 可行性判断：本地可复制完整筛选能力，但需迁移远程更新后的 analysis/HighFold 代码、analysis 环境、HighFold 环境及模型参数；要得到数值可比结果还需固定脚本版本、参数、Rosetta/HighFold 版本和 CPU/GPU 模式。仅安装 requirements.txt 或沿用本地旧脚本不能达到相同效果。

- 复核：远程 shell 把二面角 sigma 显式设为20°，覆盖 Python默认10°；主流程双方 recycle/模型数均3。远程原地修改的 ColabFold site-packages 含环化/二硫键补丁，必须一起迁移。
- 远程既有结果可混有缓存；NumSS用几何距离，Disulfide_Bonds用relaxed PDB的LINK，两者不能混同。完整审计与验证建议：docs/reports/2026-10-28-remote-screening-comparison.md。
