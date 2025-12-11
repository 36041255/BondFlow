"""
Runner for BondFlow.models.mymodel.MySampler.

This file is meant to be executed in the *generation* conda environment where
BondFlow and Torch are installed. It provides a thin wrapper to:

- load a Hydra/OmegaConf config
- construct MySampler
- sample structures for different lengths
- write out PDB + bond text files using MySampler's existing logic
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from BondFlow.models.mymodel import MySampler


@dataclass
class GenerationConfig:
    config_path: str
    device: str = "cuda:0"
    # number of samples per length
    num_samples_per_length: int = 10
    # batch size when calling sampler
    batch_size: int = 1
    # whether to use post-refine (sidechain refinement) outputs
    use_post_refine: bool = True
    # root directory for artifacts
    out_root: str = "artifacts"
    # experiment name and model name used to organize artifacts
    experiment_name: str = "default_exp"
    model_name: str = "bondflow_cyclize"


class BondFlowSamplerRunner:
    """Wrapper around MySampler for multi-length sampling."""

    def __init__(self, gen_cfg: GenerationConfig):
        self.gen_cfg = gen_cfg
        self._device = torch.device(gen_cfg.device)
        # Load OmegaConf config
        cfg = OmegaConf.load(gen_cfg.config_path)
        self._sampler = MySampler(cfg, device=self._device)

    def _length_dir(self, length: int) -> str:
        root = os.path.abspath(self.gen_cfg.out_root)
        return os.path.join(
            root,
            self.gen_cfg.experiment_name,
            self.gen_cfg.model_name,
            f"len_{length}",
        )

    def sample_lengths(self, lengths: Iterable[int]) -> None:
        """
        For each length, run sampling until we have num_samples_per_length structures.

        Outputs:
          - PDB + bonds_* files are written by MySampler.sample_from_prior into
            `<len_dir>/pre_refine` and/or `<len_dir>/post_refine`.
          - A generation_log.csv is written per length with timing info.
        """

        for L in lengths:
            self._sample_for_length(L)

    def _sample_for_length(self, length: int) -> None:
        num_needed = self.gen_cfg.num_samples_per_length
        batch_size = self.gen_cfg.batch_size
        out_base = self._length_dir(length)
        # Let MySampler decide internal pre/post-refine subfolders via out_pdb_dir
        os.makedirs(out_base, exist_ok=True)

        log_rows: List[Dict] = []
        num_generated = 0
        batch_id = 0

        while num_generated < num_needed:
            cur_batch = min(batch_size, num_needed - num_generated)
            batch_id += 1
            t0 = time.time()

            # We always sample from prior here
            # Note: sample_from_prior already writes PDBs + bonds_* via _sample_loop.
            # We only care about timing and how many items we *intend* to generate.
            _ = self._sampler.sample_from_prior(
                num_batch=cur_batch,
                num_res=length,
                record_trajectory=False,
                out_pdb_dir=out_base,
            )

            t1 = time.time()
            elapsed = t1 - t0

            for i in range(cur_batch):
                num_generated += 1
                log_rows.append(
                    {
                        "length": length,
                        "sample_index": num_generated,
                        "batch_id": batch_id,
                        "batch_size": cur_batch,
                        "time_sec": elapsed,
                        "time_per_sample": elapsed / float(cur_batch),
                        "time_per_residue": elapsed / float(cur_batch * length),
                    }
                )

        # Save generation log
        import csv

        log_path = os.path.join(out_base, "generation_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "length",
                    "sample_index",
                    "batch_id",
                    "batch_size",
                    "time_sec",
                    "time_per_sample",
                    "time_per_residue",
                ],
            )
            writer.writeheader()
            for row in log_rows:
                writer.writerow(row)




