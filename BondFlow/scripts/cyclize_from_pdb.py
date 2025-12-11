import os
import argparse
import torch
from omegaconf import OmegaConf

from BondFlow.models.mymodel import MySampler
import BondFlow.data.utils as iu


def main():
    parser = argparse.ArgumentParser(description="Cyclize a segment using only YAML configuration")
    parser.add_argument("--cfg", default="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/cyclize.yaml", help="Config yaml path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = OmegaConf.load(args.cfg)

    # Validate minimal required fields
    assert cfg.design_config.input_pdb is not None, "design_config.input_pdb must be set in YAML"
    assert cfg.design_config.contigs is not None and len(cfg.design_config.contigs) > 0, "design_config.contigs must be set in YAML"
    assert cfg.design_config.bond_condition is not None and len(cfg.design_config.bond_condition) > 0, "design_config.bond_condition must be set in YAML for cyclization"

    sampler = MySampler(cfg, device=device)

    # Build Target strictly from YAML
    pdb_parsed = iu.process_target(
        cfg.design_config.input_pdb,
        parse_hetatom=False,
        center=False,
        parse_link=True,
        link_csv_path=getattr(cfg.preprocess, 'link_config', None),
    )
    target = iu.Target(cfg.design_config, pdb_parsed)

    # Derive inputs from Target
    B = 1
    L = target.full_seq.shape[0]
    rf_idx = target.full_rf_idx[None, :].to(device)
    pdb_idx = [target.full_pdb_idx]
    res_mask = target.full_mask_seq[None, :].to(device)
    str_mask = target.full_mask_str[None, :].to(device)
    seq_mask = target.full_mask_seq[None, :].to(device)
    bond_mask = target.full_bond_mask[None, :, :].to(device)

    out_prefix = cfg.inference.output_prefix if hasattr(cfg, 'inference') else "outputs/cyclize/design"
    out_dir = os.path.dirname(out_prefix) if out_prefix else "outputs/cyclize"
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    num_designs = int(getattr(cfg.inference, 'num_designs', 1)) if hasattr(cfg, 'inference') else 1
    num_timesteps = int(cfg.interpolant.sampling.num_timesteps)

    for _ in range(num_designs):
        sampler.sample_from_prior(
            num_batch=B,
            num_res=L,
            num_timesteps=num_timesteps,
            rf_idx=rf_idx,
            pdb_idx=pdb_idx,
            res_mask=res_mask,
            str_mask=str_mask,
            seq_mask=seq_mask,
            bond_mask=bond_mask,
            record_trajectory=True,
            out_pdb_dir=out_dir,
        )


if __name__ == "__main__":
    main()


