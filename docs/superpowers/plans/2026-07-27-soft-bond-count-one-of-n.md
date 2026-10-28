# Soft Bond Count One-of-N Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a discrete `one_of_N` target mode to `soft_bond_count`, initially allowing configurations such as `allowed_N: [1, 2]`.

**Architecture:** Keep counting, Sinkhorn projection, scheduling, and gradient updates unchanged. Add a small count-energy helper to `SoftBondCountGuidance`; it parses and validates target values once during construction, then computes each batch item's squared distance to its nearest allowed target.

**Tech Stack:** Python 3.10, PyTorch, standard-library `unittest`, OmegaConf-compatible configuration objects.

## Global Constraints

- Existing `exact_N` and `at_least_N` behavior must remain unchanged.
- `one_of_N` must give positive energy at an intermediate soft count such as `1.5` for `allowed_N: [1, 2]`.
- `allowed_N` is required, non-empty, numeric, and finite when `mode: one_of_N`.
- Do not create commits; repository changes stay in the user's worktree.

---

### Task 1: Add and verify discrete target energy

**Files:**
- Create: `BondFlow/tests/test_soft_bond_count_guidance.py`
- Modify: `BondFlow/models/guidance.py:535-1056`

**Interfaces:**
- Consumes: `SoftBondCountGuidance(cfg, device)` with `cfg.mode`, `cfg.weight`, and `cfg.allowed_N`.
- Produces: `SoftBondCountGuidance._count_energy(C: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]`, where the first tensor has one non-negative energy value per batch item.

- [ ] **Step 1: Write the failing tests**

```python
import unittest
from types import SimpleNamespace

import torch

from BondFlow.models.guidance import SoftBondCountGuidance


class SoftBondCountOneOfNTests(unittest.TestCase):
    def make_guidance(self, **overrides):
        values = {"mode": "one_of_N", "weight": 0.1, "allowed_N": [1, 2]}
        values.update(overrides)
        return SoftBondCountGuidance(SimpleNamespace(**values))

    def test_one_of_n_has_wells_only_at_allowed_counts(self):
        guidance = self.make_guidance()
        energy, _ = guidance._count_energy(torch.tensor([1.0, 1.5, 2.0]))
        self.assertTrue(torch.allclose(energy, torch.tensor([0.0, 0.25, 0.0])))

    def test_one_of_n_requires_nonempty_finite_allowed_targets(self):
        for allowed in (None, [], [1, float("inf")]):
            with self.subTest(allowed=allowed), self.assertRaisesRegex(ValueError, "allowed_N"):
                self.make_guidance(allowed_N=allowed)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `/home/xjt/.conda/envs/BondFlow/bin/python -m unittest BondFlow.tests.test_soft_bond_count_guidance`

Expected: FAIL because `_count_energy` does not yet exist and `one_of_N` does not validate `allowed_N`.

- [ ] **Step 3: Add the minimal implementation**

In `SoftBondCountGuidance.__init__`, parse `allowed_N` only for `mode == "one_of_n"`; reject a missing, empty, non-numeric, or non-finite sequence with `ValueError` containing `allowed_N`. Store the normalized targets as a tuple of floats.

Add this helper and use it in the current count-energy branch:

```python
def _count_energy(self, count: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if self.mode == "exact_n":
        return (count - self.target_N).square(), None
    if self.mode == "at_least_n":
        diff = torch.relu(self.target_N - count)
        return diff.square(), diff
    if self.mode == "one_of_n":
        targets = count.new_tensor(self.allowed_N)
        return (count.unsqueeze(-1) - targets).square().min(dim=-1).values, None
    raise ValueError(f"Unknown mode: {self.mode}")
```

Replace the inlined `exact_n` / `at_least_n` calculation with `energy_per_item, diff = self._count_energy(C)` followed by `energy = energy_per_item.sum()`. Extend debug output to print `allowed_N` for `one_of_n` without accessing `diff`.

- [ ] **Step 4: Run focused tests and confirm success**

Run: `/home/xjt/.conda/envs/BondFlow/bin/python -m unittest BondFlow.tests.test_soft_bond_count_guidance`

Expected: PASS with two test methods: one energy test and one validation test with three subcases.

- [ ] **Step 5: Run regression checks**

Run: `/home/xjt/.conda/envs/BondFlow/bin/python -m unittest BondFlow.tests.test_soft_bond_count_guidance`

Expected: PASS. The existing bond-coherence test is excluded because it fails on the unmodified baseline with an obsolete `d_far` constructor argument.
