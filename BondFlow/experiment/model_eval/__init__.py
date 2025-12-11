"""
Utilities for evaluating generative protein models.

This package is intentionally **analysis-only**: it should not import heavy
training code or assume any specific conda environment, beyond what is needed
to:

- locate generated structures (PDB + bond txt)
- read per-length metadata (timing logs, configs, etc.)
- orchestrate analysis/visualization scripts under `experiment/analysis`.

Model-specific sampling lives in separate environments and is expected to
write its outputs into a standard `artifacts/` directory layout.
"""




