import torch
from omegaconf import OmegaConf


def load_conf_from_yaml(path: str):
    from omegaconf import OmegaConf
    conf = OmegaConf.load(path)
    # Switch to APM
    conf.model.type = 'apm'
    # Disable non-seq losses by default for stability
    if 'apm' in conf.model and 'disable_losses' in conf.model.apm:
        conf.model.apm.disable_losses.frame = True
        conf.model.apm.disable_losses.fape = True
        conf.model.apm.disable_losses.clash = True
        conf.model.apm.disable_losses.torsion = True
        conf.model.apm.disable_losses.bond = False  # keep bond on to test head
    return conf


def test_apm_minibatch_forward():
    from BondFlow.models.mymodel import MySampler

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = load_conf_from_yaml('./config/base.yaml')
    sampler = MySampler(conf, device=device)
    model = sampler.model.eval()

    B, L = 2, 12
    # Synthetic inputs aligning with _preprocess_batch expectations
    
    seq = torch.randint(0, 21, (B, L))
    xyz_t = torch.randn(B, L, 3, 3)
    bond_mat = torch.rand(B, L, L)
    bond_mat = (bond_mat + bond_mat.transpose(1, 2)) / 2
    bond_mat = bond_mat / bond_mat.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    rf_idx = torch.arange(L)[None, :].repeat(B, 1)
    pdb_idx = torch.zeros(B, L, dtype=torch.long)
    alpha = torch.randn(B, L, 10, 2)
    alpha_tor_mask = torch.ones(B, L, 10)
    t = torch.rand(B)
    str_mask = torch.ones(B, L, dtype=torch.bool)
    seq_mask = torch.ones(B, L, dtype=torch.bool)
    bond_mask = torch.ones(B, L, L, dtype=torch.bool)
    res_mask = torch.ones(B, L, dtype=torch.bool)
    # 将张量转移到当前设备上
    seq = seq.to(device)
    xyz_t = xyz_t.to(device)
    bond_mat = bond_mat.to(device)
    rf_idx = rf_idx.to(device)
    pdb_idx = pdb_idx.to(device)
    alpha = alpha.to(device)
    alpha_tor_mask = alpha_tor_mask.to(device)
    t = t.to(device)
    str_mask = str_mask.to(device)
    seq_mask = seq_mask.to(device)
    bond_mask = bond_mask.to(device)
    res_mask = res_mask.to(device)

    # Preprocess using project pipeline
    msa_noised, xyz_noised, res_dist_matrix, alpha_t, idx_pdb, t1d, t2d, str_mask, seq_mask, bond_mask = sampler._preprocess_batch(
        seq, xyz_t, bond_mat, rf_idx, pdb_idx, alpha, alpha_tor_mask, t,
        str_mask=str_mask, seq_mask=seq_mask, bond_mask=bond_mask, res_mask=res_mask
    )


    with torch.no_grad():
        logits_aa, xyz_pred, alpha_s, bond_matrix = model(
            seq_noised=seq,
            xyz_noised=xyz_t,
            bond_noised=bond_mat,
            rf_idx=rf_idx,
            pdb_idx=pdb_idx,
            alpha_target=alpha,
            alpha_tor_mask=alpha_tor_mask,
            partial_T=t,
            str_mask=str_mask,
            seq_mask=seq_mask,
            bond_mask=bond_mask,
            res_mask=res_mask,
            use_checkpoint=False,
        )

    assert logits_aa.shape == (B, L, 21)
    assert xyz_pred.shape == (B, 1, L, 3, 3)
    assert alpha_s.shape == (B, 1, L, 10, 2)
    assert bond_matrix.shape == (B, L, L)
    for tns in [logits_aa, xyz_pred, alpha_s, bond_matrix]:
        assert not torch.isnan(tns).any()

if __name__ == "__main__":
    test_apm_minibatch_forward()
