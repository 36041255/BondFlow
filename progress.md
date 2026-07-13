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
