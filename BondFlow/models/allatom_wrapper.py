import torch
import torch.nn as nn
from typing import Optional, Tuple
from multiflow_data import utils as du
from rfdiff.util_module import ComputeAllAtomCoords
from BondFlow.data import utils as iu
from openfold.np import residue_constants as rc
from openfold.data import data_transforms
from rfdiff.chemical import aa2long, aa2num, num2aa

# APM utilities
try:
    from apm.apm.data.all_atom import atom37_from_trans_rot_torsion, create_denser_atom_position
    from apm.apm.data.utils import create_rigid
    APM_AVAILABLE = True
except Exception:
    APM_AVAILABLE = False

try:
    from BondFlow.data.link_utils import _get_bond_info
    LINK_UTILS_AVAILABLE = True
except Exception:
    LINK_UTILS_AVAILABLE = False


class AllAtomWrapper(nn.Module):
    """
    Unified interface for all-atom coordinate reconstruction.
    - backend='rfdiff': use rfdiffusion ComputeAllAtomCoords taking (seq, xyz_bb, alphas_sincos)
    - backend='apm': use APM's atom37_from_trans_rot_torsion with (trans, rots, torsion_angles)
    """
    def __init__(self, backend: str = 'rfdiff', device: Optional[torch.device] = None):
        super().__init__()
        self.backend = backend.lower()
        self.device = device
        if self.backend == 'rfdiff':
            self.rfdiff = ComputeAllAtomCoords()
        elif self.backend == 'apm':
            if not APM_AVAILABLE:
                raise ImportError("APM backend requested but APM modules are not available")
            self.rfdiff = None
        else:
            raise ValueError(f"Unsupported all-atom backend: {backend}")

    def forward(
        self,
        seq: torch.Tensor,
        xyz_bb: torch.Tensor,
        alphas_sincos: torch.Tensor,
        res_mask: Optional[torch.Tensor] = None,
        bond_mat: Optional[torch.Tensor] = None,
        link_csv_path: Optional[str] = None,
        use_H: bool = False,
        rotmats: Optional[torch.Tensor] = None,
        trans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
          - seq: [B, L] residue indices (0..20)
          - xyz_bb: [B, L, 3, 3] backbone N,CA,C coordinates
          - alphas_sincos: [B, L, 10, 2] torsions in [cos,sin] (统一内部标准)
          - res_mask: [B, L] optional residue mask (1=build, 0=skip); useful for N_C_add terminals/padding
          - rotmats/trans: 可选；APM 后端若未提供，将从 xyz_bb 自动推导
        Returns:
          - RTframes or dummy (for rfdiff), and all-atom coords [B, L, 14, 3]
        """
        aatype = seq.long().clone()
        aatype[aatype == du.MASK_TOKEN_INDEX] = 0 # ALA
        if self.backend == 'rfdiff':
            # rfdiff returns (RTframes, xyzs)
            RTs, xyz = self.rfdiff(
                seq, xyz_bb, alphas_sincos, bond_mat=bond_mat, link_csv_path=link_csv_path, use_H=use_H
            )
            # rfdiff already returns 14 or 3 depending on use_H
            if not use_H and xyz.shape[-2] == 27:
                xyz = xyz[..., :14, :]
            return RTs, xyz

        # APM path
        # Derive rotmats/trans from xyz_bb if not provided
        if rotmats is None:
            rotmats = iu.get_R_from_xyz(xyz_bb)
        if trans is None:
            trans = xyz_bb[:, :, 1, :]
        B, L = seq.shape[:2]
        
        # Safety: clamp residue types to supported range
        try:
            max_restype = len(rc.restype_atom37_mask) - 1
        except Exception:
            max_restype = 20
        aatype = aatype.clamp(0, max_restype)

        # Safety: translations fallback to CA when non-finite
        ca_pos = xyz_bb[:, :, 1, :]
        invalid_T = ~torch.isfinite(trans).all(dim=-1)
        trans = torch.where(invalid_T[..., None], ca_pos, trans)
        # Convert chi [cos,sin] to angles (radians) for 4 chis
        # alphas_sincos expected: [B, L, 10, 2]; use [:, :, 3:7]
        chi_cossin = alphas_sincos[:, :, 3:7, :]
        # Safety: normalize [cos, sin] pairs; fallback to [1,0] if degenerate
        chi_cossin = torch.nan_to_num(chi_cossin, nan=0.0, posinf=0.0, neginf=0.0)
        norms = torch.linalg.norm(chi_cossin, dim=-1, keepdim=True)
        degenerate = norms < 1e-6
        chi_cossin = chi_cossin / torch.clamp(norms, min=1e-6)
        chi_cossin = torch.where(
            degenerate,
            torch.tensor([1.0, 0.0], device=chi_cossin.device, dtype=chi_cossin.dtype).view(1, 1, 1, 2),
            chi_cossin,
        )
        # atan2(y=sin, x=cos)

        def sincos_to_angle(sincos: torch.Tensor) -> torch.Tensor:
            # sincos shape [..., 2], order = [sin, cos]
            sincos64 = sincos.to(torch.float64)
            norm = torch.clamp((sincos64**2).sum(dim=-1, keepdim=True), min=1e-12).sqrt()
            unit = sincos64 / norm
            angle = torch.atan2(unit[..., 0], unit[..., 1])  # y=sin (idx 0), x=cos (idx 1)
            return angle.to(sincos.dtype)
        #chi_angles = torch.atan2(chi_cossin[..., 1], chi_cossin[..., 0])  # [B, L, 4]
        chi_angles = sincos_to_angle(torch.flip(chi_cossin, dims=[-1]))  # chi_cossin[..., 0]=sin, [..., 1]=cos

        # Build residue mask (default all-ones)
        if res_mask is None:
            res_mask_in = torch.ones(B, L, device=seq.device)
        else:
            res_mask_in = res_mask.to(device=seq.device, dtype=torch.float32)
        atom37 = atom37_from_trans_rot_torsion(
            trans,
            rotmats,
            chi_angles,
            aatype,
            res_mask=res_mask_in,
        )  # [B, L, 37, 3]
        
        remove_mask_37 = torch.zeros_like(atom37[..., 0], dtype=torch.bool)
        # Optional: prune side-chain groups for bonded residues using link/bond specs
        if bond_mat is not None and link_csv_path is not None and LINK_UTILS_AVAILABLE:
            try:
                bonds, removals = _get_bond_info(link_csv_path)
            except Exception:
                bonds, removals = {}, {}

            b_ids, i_ids, j_ids = torch.where(torch.triu(bond_mat.bool(), diagonal=1))
            for b, i, j in zip(b_ids.tolist(), i_ids.tolist(), j_ids.tolist()):
                res1_num = int(aatype[b, i].item())
                res2_num = int(aatype[b, j].item())

                key = (res1_num, res2_num)
                if key not in bonds:
                    key = (res2_num, res1_num)
                    if key not in bonds:
                        continue

                atom1_name, atom2_name, ref_dist = bonds[key]
                a1 = atom1_name.strip() if atom1_name is not None else None
                a2 = atom2_name.strip() if atom2_name is not None else None

                try:
                    idx1_37 = rc.atom_order[a1]
                    idx2_37 = rc.atom_order[a2]
                except Exception:
                    continue

                xyz1 = atom37[b, i, idx1_37]
                xyz2 = atom37[b, j, idx2_37]
                dist = torch.linalg.norm(xyz1 - xyz2)

                if torch.isfinite(dist) and dist.item() < float(ref_dist) * 1.5:
                    rem_key = (res1_num, res2_num)
                    if rem_key not in removals:
                        rem_key = (res2_num, res1_num)
                    if rem_key in removals:
                        rem_info = removals[rem_key]
                        res1_name = num2aa[res1_num]
                        res2_name = num2aa[res2_num]
                        if res1_name in rem_info:
                            for rm_atom in rem_info[res1_name]:
                                name = (rm_atom or '').strip()
                                if name in rc.atom_order:
                                    remove_mask_37[b, i, rc.atom_order[name]] = True
                        if res2_name in rem_info:
                            for rm_atom in rem_info[res2_name]:
                                name = (rm_atom or '').strip()
                                if name in rc.atom_order:
                                    remove_mask_37[b, j, rc.atom_order[name]] = True

            # Do not inject NaNs into atom37 here to avoid NaN propagation in reductions.
            # We will exclude these atoms during 37→14 mapping and set their 14-atom outputs to NaN later.
            pass

        # Map atom37 -> atom14 using APM helper (build one-hot inverse mapping)
        restype_atom37_to_atom14, restype_atom37_mask = create_denser_atom_position()
        restype_atom37_to_atom14 = restype_atom37_to_atom14.to(device=atom37.device, dtype=torch.long)
        restype_atom37_mask = restype_atom37_mask.to(device=atom37.device, dtype=torch.float32)
        idx37_to_14 = restype_atom37_to_atom14[aatype]  # [B, L, 37]
        mask37 = restype_atom37_mask[aatype]            # [B, L, 37]
        
        # Exclude removed 37-atoms from contributing to 14-atom coordinates
        # Build a removal mask defaulting to all False if it does not exist


        dynamic_mask37 = mask37 * (~remove_mask_37).to(mask37.dtype)
        atom37_masked = atom37 * dynamic_mask37.unsqueeze(-1)
        one_hot_37_to_14 = torch.nn.functional.one_hot(idx37_to_14, num_classes=14).to(atom37_masked.dtype)  # [B, L, 37, 14]
        one_hot_14x37 = one_hot_37_to_14.permute(0, 1, 3, 2)  # [B, L, 14, 37]
        out14 = torch.matmul(one_hot_14x37, atom37_masked)  # [B, L, 14, 3]

        # mark non-existent heavy atoms as NaN so downstream writers skip them
        atom14_mask_table = torch.as_tensor(rc.restype_atom14_mask, device=out14.device, dtype=torch.bool)
        atom14_mask = atom14_mask_table[aatype]  # [B, L, 14]
        out14 = torch.where(atom14_mask.unsqueeze(-1), out14, torch.full_like(out14, float('nan')))

        # Map the 37-atom removal mask to atom14 indices and set only those to NaN
        remove_mask_14 = (torch.matmul(one_hot_14x37, remove_mask_37.to(out14.dtype).unsqueeze(-1)).squeeze(-1) > 0)
        out14 = torch.where(remove_mask_14.unsqueeze(-1) | (1-res_mask_in[...,None,None]).bool(), 
                            torch.full_like(out14, float('nan')), out14)
        return None, out14


def openfold_get_torsions(
    seq: torch.Tensor,
    xyz_27: torch.Tensor,
    mask_in: torch.Tensor = None,
):
    """
    OpenFold-based torsion extraction with rfdiff-compatible I/O signature.

    Inputs:
      - seq: [B, L] residue indices (same indexing used elsewhere in project)
      - xyz_27: [B, L, 27, 3] full-atom coordinates in rfdiff 27-atom scheme
      - mask_in: [B, L, 27] optional atom-existence mask (unused; OpenFold has its own)

    Returns (rfdiff-compatible shapes):
      - alpha:      [B, L, 10, 2] sin/cos for (omega, phi, psi, chi1..chi4, extras[3]=zeros)
      - alpha_alt:  [B, L, 10, 2] same as alpha (no flip handling here)
      - alpha_mask: [B, L, 10]    validity mask for first 7 torsions; extras False
      - tors_planar:[B, L, 10]    zeros (caller ignores in current pipeline)
    """
    assert seq.ndim == 2 and xyz_27.ndim == 4 and xyz_27.shape[-2:] == (27, 3)
    device = xyz_27.device
    B, L = seq.shape

    # Map rfdiff 27-atom names -> OpenFold atom37 indices per residue type
    restype_atom27_to_atom37 = torch.full((22, 27), -1, dtype=torch.long, device=device)
    for rt in range(22):
        names27 = aa2long[rt]
        for j, name in enumerate(names27):
            if name is None:
                continue
            nm = name.strip()
            try:
                idx37 = rc.atom_order[nm]
            except Exception:
                continue
            restype_atom27_to_atom37[rt, j] = idx37

    # Clamp unknown residue indices to a valid range and avoid UNK by mapping to ALA (0)
    try:
        unknown_idx = len(rc.restype_atom14_mask) - 1
    except Exception:
        unknown_idx = 20
    aatype = seq.long().clone()#.clamp(0, unknown_idx).clone()
    xyz_27 = xyz_27.clone()
    aatype = torch.where(aatype == unknown_idx, torch.zeros_like(aatype), aatype)

    idx_map = restype_atom27_to_atom37[aatype]  # [B, L, 27]
    atom37 = torch.zeros((B, L, 37, 3), dtype=xyz_27.dtype, device=device)
    atom37_mask = torch.zeros((B, L, 37), dtype=torch.bool, device=device)

    valid = idx_map >= 0
    if valid.any():
        b_idx, l_idx, j_idx = torch.where(valid)
        t_idx = idx_map[b_idx, l_idx, j_idx]
        atom37[b_idx, l_idx, t_idx] = xyz_27[b_idx, l_idx, j_idx]
        atom37_mask[b_idx, l_idx, t_idx] = True

    # Use OpenFold transform to extract torsions (omega, phi, psi, chi1..chi4)
    prot = {
        "aatype": aatype,
        "all_atom_positions": atom37,
        "all_atom_mask": atom37_mask,
    }
    outs = data_transforms.atom37_to_torsion_angles()(prot)
    tors_sincos = outs["torsion_angles_sin_cos"]  # [B, L, 7, 2]
    tors_mask = outs["torsion_angles_mask"]      # [B, L, 7]
    tors_alt = outs["alt_torsion_angles_sin_cos"]  # [B, L, 7, 2]

    alpha = torch.zeros((B, L, 10, 2), dtype=atom37.dtype, device=device)
    alpha_mask = torch.zeros((B, L, 10), dtype=torch.bool, device=device)
    alpha_alt = torch.zeros((B, L, 10, 2), dtype=atom37.dtype, device=device)
    alpha[:, :, :7, :] = tors_sincos
    alpha_mask[:, :, :7] = tors_mask
    alpha_alt[:, :, :7, :] = tors_alt


    # No alternative-flip handling here; keep identical
    # change to (cos,sin)
    alpha = torch.flip(alpha, dims=[-1])
    alpha_alt = torch.flip(alpha_alt, dims=[-1])

    tors_planar = torch.zeros((B, L, 10), dtype=torch.bool, device=device)
    return alpha, alpha_alt, alpha_mask, tors_planar
