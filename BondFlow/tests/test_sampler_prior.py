import os
import random
import numpy as np
import torch
from hydra import initialize, compose
from BondFlow.models.mymodel import MySampler


def set_reproducible(seed: int = 1234):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CuDNN determinism
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def test_apm_wrapper_sampler_execution():
    """Tests a full sampling run with the APM multi-model wrapper."""
    set_reproducible(780)
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="cyclize.yaml")
        print("start test_apm_wrapper_sampler_execution")
        sampler = MySampler(cfg,device='cuda')
        torch.set_printoptions(profile='full')
        B, L = 64, 10 # Small batch and length for a quick test
        num_timesteps = 500
        
        final_px0, final_x, final_seq, final_bond = sampler.sample_from_prior(
            num_batch=B,
            num_res=L,
            num_timesteps=num_timesteps,
            record_trajectory=False,
            out_pdb_dir="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/samples/apm_sample2"
        )


        # Check final output shapes
        assert final_px0.shape == (B, L, 14, 3)
        assert final_x.shape == (B, L, 14, 3)
        assert final_seq.shape == (B, L)
        assert final_bond.shape == (B, L, L)
        print(final_bond[0],final_bond[1],final_bond[2])
        # Check final output for NaNs
        # assert not torch.isnan(final_px0).any(), "NaN found in final_px0"
        # assert not torch.isnan(final_x).any(), "NaN found in final_x"
        # assert not torch.isnan(final_bond).any(), "NaN found in final_bond"

        # Check trajectory shapes
        assert traj['px0'].shape == (num_timesteps - 1, B, L, 14, 3)
        assert traj['x'].shape == (num_timesteps - 1, B, L, 14, 3)
        assert traj['seq'].shape == (num_timesteps - 1, B, L)
        assert traj['bond'].shape == (num_timesteps - 1, B, L, L)
        print(traj['px0'].shape)

if __name__ == "__main__":
    test_apm_wrapper_sampler_execution()

