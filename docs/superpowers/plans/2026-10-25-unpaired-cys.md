# Unpaired CYS Penalty Implementation Plan

**Goal:** Add the user-approved differentiable unpaired-CYS penalty and enable it at 0.5 for TL1A design 1.

**Architecture:** Extend `TypeAwareSoftBondCountGuidance._compute_energy` with `seq_mask`. For the disulfide type, after region/spacing masks, add `unpaired_cys_weight * sum(relu(p_cys - P_eff.sum(-1)) * residue_mask)`. The residue mask intersects valid residues, mutable sequence positions, and region endpoints. Keep fixed residues available as partners. Reuse this energy in both hooks; preserve the existing count loss and all unrelated local edits.

**Tech stack:** Python 3.10, PyTorch, unittest, OmegaConf.

## Constraints

- Default weight 0.0, finite and nonnegative; no extra loss for other bond types.
- Sum over residues and batches, independently of per-type count weights/top-k/power-sum.
- Gradients remain connected to sequence probabilities and bond probabilities.
- Keep existing sampling, final filtering, and scripts unchanged. No GPU job submission.
- Modify the current working files requested by the user; preserve their pre-existing changes.

## Task 1: Regression tests

- [x] Create `BondFlow/tests/test_unpaired_cys_guidance.py` using real `LinkInfo` from `BondFlow/config/link.csv`.
- [x] Cover paired versus unpaired CYS, disabled default, missing CYS, region/spacing/valid/mutable masks, fixed partners, batches, and finite gradients in both variables.
- [x] Exercise both sampling hooks to confirm mutable CYS probabilities can decrease and fixed positions remain unchanged.
- [x] Run `PYTHONDONTWRITEBYTECODE=1 /home/miniconda3/miniconda3/envs/BondFlow/bin/python -B -m unittest BondFlow.tests.test_unpaired_cys_guidance -v` and confirm failures caused by missing feature.

## Task 2: Implementation and configuration

- [x] Add validated `unpaired_cys_weight` to the constructor and document its scope.
- [x] Extend `_compute_energy(..., seq_mask=None)` and pass the mask from both hooks.
- [x] Add the masked differentiable penalty inside the disulfide branch after masking `P_eff`.
- [x] Include the unweighted soft unpaired count in existing debug output.
- [x] Add `unpaired_cys_weight: 0.5` only to the disulfide guidance block in `BondFlow/config/TL1A_large_disulf_design1.yaml`.

## Task 3: Verification and review

- [x] Run new tests and existing `BondFlow.tests.test_soft_bond_count_guidance` regressions with unittest.
- [x] Load the real TL1A YAML with OmegaConf and confirm the penalty is enabled only for disulfide guidance.
- [x] Review the incremental diff against the initial file snapshot and check whitespace.
- [x] Report verified scope and the absence of a full GPU sampling validation.

## Baseline

- Existing 4 bond count tests pass in `/home/miniconda3/miniconda3/envs/BondFlow/bin/python`.
- This environment lacks pytest, so use unittest directly; no dependency installation needed.
- `guidance.py` already contains local one-of-N changes; retain them.

## Verification results

- RED: new tests initially failed for missing penalty, missing seq_mask argument, and unchanged sampling outputs.
- GREEN: 14 new tests and 4 existing one-of-N tests pass (18 total).
- Real OmegaConf YAML loads both typed modules: disulfide weight 0.5; isopeptide/lactone weight 0.0.
- Incremental source diff preserves pre-existing local one-of-N work; git diff --check passes.
- Independent read-only review approved the incremental changes and independently reran all 18 tests successfully; no GPU sampling submitted.
