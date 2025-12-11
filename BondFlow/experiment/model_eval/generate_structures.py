"""
Entry script (to be run in the *generation* conda environment) that:

- selects a model from the registry (currently BondFlow MySampler)
- samples proteins for multiple lengths
- writes PDB + bond information into a standardized artifacts layout
- records generation speed for later analysis
"""

import argparse
from typing import List

from .model_registry import get_model_entry
from .runners.bondflow_sampler_runner import BondFlowSamplerRunner, GenerationConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate structures for evaluation.")
    p.add_argument(
        "--model_name",
        type=str,
        default="bondflow_cyclize",
        help="Model name registered in model_registry.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config (YAML). If not set, use registry default.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for generation, e.g., cuda:0 or cpu. "
        "If not set, use registry default.",
    )
    p.add_argument(
        "--lengths",
        type=str,
        required=True,
        help="Comma-separated list of sequence lengths, e.g. '50,100,150'.",
    )
    p.add_argument(
        "--num_samples_per_length",
        type=int,
        default=10,
        help="Number of samples per length.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for sampling.",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="artifacts",
        help="Root directory to store generated structures.",
    )
    p.add_argument(
        "--experiment_name",
        type=str,
        default="default_exp",
        help="Experiment name used to organize artifacts.",
    )
    return p.parse_args()


def _parse_lengths(lengths_str: str) -> List[int]:
    return [int(x) for x in lengths_str.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    entry = get_model_entry(args.model_name)

    cfg_path = args.config or entry.default_config
    if cfg_path is None:
        raise ValueError(
            f"No config provided and registry has no default_config for model {args.model_name}"
        )

    device = args.device or (entry.default_kwargs or {}).get("device", "cuda:0")

    gen_cfg = GenerationConfig(
        config_path=cfg_path,
        device=device,
        num_samples_per_length=args.num_samples_per_length,
        batch_size=args.batch_size,
        out_root=args.out_root,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
    )

    runner = BondFlowSamplerRunner(gen_cfg)
    lengths = _parse_lengths(args.lengths)
    runner.sample_lengths(lengths)


if __name__ == "__main__":
    main()




