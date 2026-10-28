# Soft Bond Count One-of-N Design

## Goal

Add a `one_of_N` mode to `soft_bond_count` so a soft bond count is encouraged to reach one of a discrete set of allowed targets, initially supporting the configuration `allowed_N: [1, 2]`.

## Semantics

For an estimated soft bond count `C` and allowed targets `T`, the energy is:

`min((C - t)^2 for t in T)`

Thus `C = 1` and `C = 2` have zero count energy for `allowed_N: [1, 2]`, while intermediate values such as `C = 1.5` have positive energy. This deliberately differs from a range constraint, which would have zero energy everywhere in `[1, 2]`.

## Configuration

```yaml
- name: soft_bond_count
  mode: one_of_N
  allowed_N: [1, 2]
```

`allowed_N` is required for `one_of_N`, must be a non-empty numeric sequence, and is normalized to floats. Existing `exact_N` and `at_least_N` configurations retain their current behavior.

## Implementation

`SoftBondCountGuidance` will parse and validate `allowed_N` during initialization. Its count-energy branch will compute the squared distance from `C` to every allowed target and select the minimum per batch item. Debug output will identify the new mode and print the configured targets.

## Verification

Focused tests will demonstrate that `one_of_N` gives zero energy at 1 and 2, positive energy at 1.5, and rejects a missing or empty `allowed_N` configuration. Existing supported modes will remain covered by the unchanged code path.
