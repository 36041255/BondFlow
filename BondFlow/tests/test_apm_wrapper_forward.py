
import torch


def test_apm_wrapper_forward_shapes():
    

    from BondFlow.models.adapter import build_design_model

    device = 'cpu'
    # d_t1d/d_t2d are unused by pure APM wrapper; provide reasonable dummies
    model = build_design_model('apm', device=device, d_t1d=53, d_t2d=61).eval()

    B, L = 2, 16

    # Synthetic BondFlow-style inputs
    seq_noised = torch.randint(0, 21, (B, L))
    xyz_noised = torch.randn(B, L, 14, 3)
    bond_noised = torch.rand(B, L, L)
    rf_idx = torch.arange(L).unsqueeze(0).repeat(B, 1)
    pdb_idx = torch.zeros(B, L, dtype=torch.long)  # Placeholder; unused in APMWrapper
    alpha_target = torch.zeros(B, L, 30)
    alpha_tor_mask = torch.ones(B, L, 10, 1)
    partial_T = torch.rand(B)

    str_mask = torch.ones(B, L, dtype=torch.bool)
    seq_mask = torch.ones(B, L, dtype=torch.bool)
    bond_mask = torch.ones(B, L, L, dtype=torch.bool)
    res_mask = torch.ones(B, L, dtype=torch.bool)

    with torch.no_grad():
        logits_aa, xyz_pred, alpha_s, bond_matrix = model(
            seq_noised=seq_noised,
            xyz_noised=xyz_noised,
            bond_noised=bond_noised,
            rf_idx=rf_idx,
            pdb_idx=pdb_idx,
            alpha_target=alpha_target,
            alpha_tor_mask=alpha_tor_mask,
            partial_T=partial_T,
            str_mask=str_mask,
            seq_mask=seq_mask,
            bond_mask=bond_mask,
            res_mask=res_mask,
            use_checkpoint=False,
        )

    assert logits_aa.shape == (B, L, 20)
    assert xyz_pred.shape == (B, 1, L, 14, 3)
    assert alpha_s.shape == (B, 1, L, 10, 2)
    assert bond_matrix.shape == (B, L, L)
    # Sanity: no NaNs
    for tns in [logits_aa, xyz_pred, alpha_s, bond_matrix]:
        assert not torch.isnan(tns).any()

if __name__ == "__main__":
    test_apm_wrapper_forward_shapes()


