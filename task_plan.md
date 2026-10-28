# Task Plan: Bond Guidance Log Analysis

## Goal
Explain why disulfide bonds appear underrepresented while other bond types are not suppressed, based on the provided config and log.

## Current Phase
Phase 3

## Phases

### Phase 1: Evidence Gathering
- [x] Inspect log for type-aware bond guidance trends
- [x] Inspect relevant config and implementation semantics
- **Status:** complete

### Phase 2: Root Cause Analysis
- [x] Compare observed log trends with expected guidance behavior
- [x] Identify likely root causes and confidence level
- **Status:** complete

### Phase 3: Recommendations
- [x] Provide targeted config/code recommendations without editing runtime config unless asked
- **Status:** complete

## Key Questions
1. Does the log show disulfide guidance is weak, inactive, or being counteracted?
2. Are isopeptide/lactone suppression losses configured in a way that can actually push counts down?
3. Do multiple guidance blocks interact through the same bond matrix/logits in a harmful way?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Analyze only; do not modify runtime config yet | User asked why behavior occurs, not to patch config |
| Prioritize config explanation over code patch | The observed behavior follows from current guidance semantics and competing objectives |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

## 2026-10-25: Unpaired CYS penalty
User approved implementation. Detailed checklist: `docs/superpowers/plans/2026-10-25-unpaired-cys.md`. Status: regression tests in progress.

Unpaired CYS penalty status: complete. Code/config implemented; all 18 focused tests pass; independent review approved. Full GPU generation remains unmeasured.

## 2026-10-29: 项目设计与筛选流程阅读
- [x] 追踪采样入口、配置与数据映射
- [x] 追踪模型、插值、guidance 与输出精修
- [x] 追踪筛选管线、指标、阈值和产物
- [x] 核实 MC4R/MDM2/TL1A 配置并汇总代码入口
范围：阅读当前工作区；保留已有未提交修改，不运行大规模设计。

## 2026-10-28: 远程筛选对比与本地复现可行性
- [x] 建立只读远程访问，盘点远程筛选入口及配置
- [x] 比较本地与远程代码、阈值和实际产物
- [x] 盘点两端环境、硬件、权重与数据库
- [x] 给出可行性、差异和迁移验证方案
范围：只读检查远程与本地环境；不安装、不迁移大文件、不修改业务代码。
