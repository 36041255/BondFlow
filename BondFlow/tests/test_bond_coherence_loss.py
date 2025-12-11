import os
import math
import torch

from BondFlow.models.Loss import BondCoherenceLoss
from rfdiff.chemical import aa2num


def _link_csv_path():
    # Resolve to BondFlow/config/link.csv relative to this test file
    this_dir = os.path.dirname(__file__)
    cfg_path = os.path.abspath(os.path.join(this_dir, '../config/link.csv'))
    return cfg_path


def test_diagonal_only_bonds_zero_loss_components():
    B, L = 1, 4
    device = 'cpu'
    # bond_matrix = identity (all no-connections)
    bond = torch.eye(L, device=device).unsqueeze(0).repeat(B, 1, 1)
    # Coordinates arbitrary
    ca = torch.zeros(B, L, 3, device=device)
    res_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    loss_fn = BondCoherenceLoss(
        link_csv_path=_link_csv_path(),
        d_far=10.0,
        lambda_geom_hinge=1.0,
        lambda_seq=1.0,
        lambda_entropy=0.0,
        use_seq_logits=False,
    )

    # Provide labels but they should not matter because off-diagonal mass is zero
    labels = torch.full((B, L), int(aa2num['GLY']), device=device)
    loss = loss_fn(bond, ca, res_mask, seq_labels=labels)

    # With diagonal-only bonds and lambda_entropy=0, loss should be 0
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_hinge_penalty_increases_with_far_distance():
    B, L = 1, 4
    device = 'cpu'
    # Off-diagonal uniform DSM (rows/cols sum to 1, diagonal 0)
    off = torch.full((L, L), 1.0 / (L - 1), device=device)
    bond = off.clone()
    bond.fill_diagonal_(0.0)
    bond = bond.unsqueeze(0).repeat(B, 1, 1)

    res_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Near geometry: small distances
    ca_near = torch.zeros(B, L, 3, device=device)
    # Far geometry: spread points with large pairwise distances
    coords = torch.tensor([[0.0, 0.0, 0.0],
                           [50.0, 0.0, 0.0],
                           [0.0, 50.0, 0.0],
                           [0.0, 0.0, 50.0]], device=device)
    ca_far = coords.unsqueeze(0).repeat(B, 1, 1)

    labels = torch.full((B, L), int(aa2num['GLY']), device=device)

    loss_fn = BondCoherenceLoss(
        link_csv_path=_link_csv_path(),
        d_far=10.0,
        lambda_geom_hinge=1.0,
        lambda_seq=0.0,
        lambda_entropy=0.0,
        use_seq_logits=False,
    )

    loss_near = loss_fn(bond, ca_near, res_mask, seq_labels=labels)
    loss_far = loss_fn(bond, ca_far, res_mask, seq_labels=labels)

    assert torch.isfinite(loss_near) and torch.isfinite(loss_far)
    assert loss_far.item() > loss_near.item() + 1e-6


def test_sequence_penalty_reduces_with_compatible_pairs():
    B, L = 1, 4
    device = 'cpu'
    # Off-diagonal uniform DSM (diagonal 0)
    off = torch.full((L, L), 1.0 / (L - 1), device=device)
    bond = off.clone()
    bond.fill_diagonal_(0.0)
    bond = bond.unsqueeze(0).repeat(B, 1, 1)

    # Geometry irrelevant for this test (hinge disabled)
    ca = torch.zeros(B, L, 3, device=device)
    res_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Labels set A: use pairs less represented in link.csv (e.g., LYS/GLY)
    labels_a = torch.tensor([[aa2num['LYS'], aa2num['GLY'], aa2num['GLY'], aa2num['LYS']]], device=device)
    # Labels set B: include pairs present in link.csv (e.g., ASP–SER)
    labels_b = torch.tensor([[aa2num['ASP'], aa2num['SER'], aa2num['GLY'], aa2num['GLY']]], device=device)

    loss_fn = BondCoherenceLoss(
        link_csv_path=_link_csv_path(),
        d_far=10.0,
        lambda_geom_hinge=0.0,
        lambda_seq=1.0,
        lambda_entropy=0.0,
        use_seq_logits=False,
        compat_default=1e-4,
    )

    loss_a = loss_fn(bond, ca, res_mask, seq_labels=labels_a)
    loss_b = loss_fn(bond, ca, res_mask, seq_labels=labels_b)

    # Expect loss_b <= loss_a since more compatible pairs should reduce −bond·log(S)
    assert torch.isfinite(loss_a) and torch.isfinite(loss_b)
    assert loss_b.item() <= loss_a.item() + 1e-6

if __name__ == "__main__":
    test_diagonal_only_bonds_zero_loss_components()
    test_hinge_penalty_increases_with_far_distance()
    test_sequence_penalty_reduces_with_compatible_pairs()
