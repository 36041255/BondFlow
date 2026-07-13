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
