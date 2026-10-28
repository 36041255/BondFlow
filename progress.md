# Progress Log

## Session: 2026-07-04

### Phase 1: Evidence Gathering
- **Status:** complete
- Actions taken:
  - Located `type_soft_bond_count` implementation.
  - Confirmed `loss_type` handling for `mse` vs non-`mse`.
  - Confirmed `range_N` uses squared interval penalty regardless of `loss_type`.
  - Confirmed `post_step` sequence guidance starts from discrete `aatypes_t_2` and returns to discrete `argmax`.
  - Confirmed the configured linear schedule strongly reduces effective guidance steps late in sampling.
  - Summarized `log.txt` by time bins and type.
  - Counted final bond types from `pre_refine` and `post_refine` bond reports.
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Log soft-count summary | `log.txt` type guidance debug lines | Quantify trends | Disulfide remains <1 on average; isopeptide/lactone rise late and are weakly reduced per inner loop | complete |
| Final bond report summary | `post_refine/bonds_final_refined_structure_*.txt` | Validate final hard bond types | 36 valid disulfide, 15 valid isopeptide, 14 valid lactone | complete |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|

## 2026-10-25: Unpaired CYS penalty
Baseline: 4 existing bond-count tests pass using Python 3.10 in /home/miniconda3/miniconda3/envs/BondFlow. Pytest is absent; unittest is available. Implementation authorized in current workspace.

Unpaired CYS: implemented loss, mask plumbing, debug output, and TL1A weight 0.5. RED observed missing-feature failures; GREEN 18/18 focused tests pass. Real OmegaConf config instantiation confirms disulfide 0.5 and other typed guidance 0.0. Independent review pending.

Independent review approved: no concrete issues; independently reran all 18 tests successfully. Unpaired CYS implementation complete. No GPU generation run.

## 2026-10-29: 项目流程阅读
- 已阅读 README、目录、现有工作记录和四个打开的 YAML。发现工作区已有改动，全部保留。
- 本次只增加阅读记录，不修改业务代码或采样配置。

- 完成采样入口、Target 编号/mask、模型与插值、guidance、成键剪枝及精修、统一筛选和集合统计入口的追踪。
- 使用 BondFlow 环境的 OmegaConf 成功解析四份配置，核实规模、路径、guidance 设置与 MC4R5/6 差异；核对关键函数入口和阈值判断源码。
- 阅读任务完成；未运行设计采样、Rosetta 或 HighFold，未修改业务代码/配置。仅追加/更新上述阅读记录。

## 2026-10-28: 远程筛选审计
- SSH BatchMode 因私钥已加密无法直接认证；已通过交互式密钥解密成功只读连接，未持久化凭据。
- 本地有 8×A100 40GB、约 503GiB RAM、/home 剩余约 883G；现有 conda 环境没有 analysis，但有 rosetta_env/BondFlow。

- 完成远程只读审计：远程分析脚本、Slurm 参数、HighFold/ColabFold 环境、模型参数、数据库挂载和已有 all_metrics/passed_structures 产物均已核对。
- 远程代码快照与本地按文件 SHA256/函数级差异比对；未复制远程环境、未安装依赖、未运行筛选。
- 结论：迁移后可在本地复现同一筛选流程；“完全相同结果”还受版本、随机性、MSA 和 CPU/GPU 模式影响，需小规模基准校验。
- 环境元数据补充检查首次因本地脚本三引号嵌套触发 SyntaxError（未执行远程命令）；已改用转义字符串重试。
- 最终核对了远程环境的实际Python元数据、定制ColabFold代码、参数目录容量、LINK约束启用逻辑与缓存条件；报告已保存。未开展迁移或运行筛选。
