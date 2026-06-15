import torch
import torch.nn as nn
from typing import Optional, Tuple
from multiflow_data import utils as du
#from rfdiff.util_module import ComputeAllAtomCoords
from BondFlow.data import utils as iu
from openfold.np import residue_constants as rc
from openfold.data import data_transforms
from rfdiff.chemical import aa2long, aa2num, num2aa


from multiflow_data.all_atom import atom37_from_trans_rot_torsion, create_denser_atom_position



from BondFlow.data.link_utils import _get_bond_info, LinkInfo



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
        # if self.backend == 'rfdiff':
        #     self.rfdiff = None#ComputeAllAtomCoords()
        # elif self.backend == 'apm':
        #     self.rfdiff = None
        # else:
        #     raise ValueError(f"Unsupported all-atom backend: {backend}")

    def _adjust_tail_oxygen(self, atom37, seq, bond_mat, link_csv_path, head_mask, tail_mask, N_C_anchor):
        """
        Explicitly adjust Oxygen atom positions for Tail Body residues based on their specific bond connectivity.
        Calculates O position based on the plane defined by CA, C, and the target atom (N/Sidechain-N) of the bonded residue.
        """
        
        try:
            link_info = LinkInfo(link_csv_path)
        except Exception:
            return atom37
        
        B, L = atom37.shape[:2]
        
        # Iterate over batch
        for b in range(B):
            # Identify Virtual Tail nodes
            t_indices = torch.where(tail_mask[b])[0]
            if len(t_indices) == 0:
                continue
                
            for t_idx in t_indices:
                # 1. Find Physical Body Tail Residue (i) via N_C_anchor
                # N_C_anchor: [B, L, L, 2], channel 1 is Tail anchor
                anchor_row = N_C_anchor[b, t_idx, :, 1] # [L]
                if not anchor_row.any():
                    continue
                i = torch.argmax(anchor_row.float()).item()
                
                # 2. Find connected node (k) in bond_mat
                # bond_mat: [B, L, L]
                bonds = torch.where(bond_mat[b, t_idx, :] > 0.5)[0]
                # Filter out self or anchor (though usually bond is virtual->virtual or virtual->body)
                candidates = [x.item() for x in bonds if x.item() != i and x.item() != t_idx]
                if not candidates:
                    continue
                
                # Assume first bonded partner is the relevant one
                k = candidates[0]
                
                # 3. Dereference k to Physical Body Target (j)
                if head_mask[b, k]:
                    # k is Virtual Head -> find its body anchor (channel 0)
                    head_anchor_row = N_C_anchor[b, k, :, 0]
                    if not head_anchor_row.any(): continue
                    j = torch.argmax(head_anchor_row.float()).item()
                elif tail_mask[b, k]:
                    # k is Virtual Tail -> find its body anchor (channel 1)
                    tail_anchor_row = N_C_anchor[b, k, :, 1]
                    if not tail_anchor_row.any(): continue
                    j = torch.argmax(tail_anchor_row.float()).item()
                else:
                    # k is Body Residue (e.g. sidechain connection)
                    j = k
                
                if i == j: continue

                # 4. Lookup Bond Rule for (Res[i], Res[j]) where Atom1='C'
                res1_num = int(seq[b, i].item())
                res2_num = int(seq[b, j].item())
                
                rules = link_info.bond_spec.get((res1_num, res2_num), [])
                target_atom_name = None
                
                # Find rule where atom1 is C (Backbone Carbon of Tail)
                for r in rules:
                    a1 = r.get('atom1', '').strip().upper()
                    a2 = r.get('atom2', '').strip().upper()
                    if a1 == 'C':
                        target_atom_name = a2
                        break
                
                if not target_atom_name:
                    continue
                    
                # 5. Calculate and Update O position
                try:
                    # Indices for residue i (Donor)
                    idx_CA = rc.atom_order['CA']
                    idx_C = rc.atom_order['C']
                    idx_O = rc.atom_order['O']
                    
                    pos_CA_i = atom37[b, i, idx_CA]
                    pos_C_i = atom37[b, i, idx_C]
                    
                    # Indices for residue j (Acceptor)
                    idx_target = rc.atom_order[target_atom_name]
                    pos_target_j = atom37[b, j, idx_target]
                    
                    # Check validity
                    if torch.isnan(pos_CA_i).any() or torch.isnan(pos_C_i).any() or torch.isnan(pos_target_j).any():
                        continue
                        
                    # Geometry calculation (logic mirrors apm.data.utils.adjust_oxygen_pos)
                    # Vector 1: CA(i) -> C(i)
                    v1 = pos_C_i - pos_CA_i
                    v1 = v1 / (torch.norm(v1) + 1e-7)
                    
                    # Vector 2: N_next(j) -> C(i)
                    v2 = pos_C_i - pos_target_j
                    v2 = v2 / (torch.norm(v2) + 1e-7)
                    
                    # Bisector direction (outward)
                    direction = v1 + v2
                    direction = direction / (torch.norm(direction) + 1e-7)
                    
                    new_O_pos = pos_C_i + direction * 1.23
                    
                    atom37[b, i, idx_O] = new_O_pos
                    
                except (KeyError, IndexError):
                    continue
                    
        return atom37

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
        head_mask: Optional[torch.Tensor] = None,
        tail_mask: Optional[torch.Tensor] = None,
        N_C_anchor: Optional[torch.Tensor] = None,
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
        if bond_mat is not None and link_csv_path is not None:
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


        # NEW: Adjust Oxygen positions for Tail Body residues based on specific bond connectivity
        # Must be done BEFORE atom37 -> atom14 mapping so it propagates to atom14
        if self.backend == 'apm' and bond_mat is not None and N_C_anchor is not None and tail_mask is not None and link_csv_path is not None:
             atom37 = self._adjust_tail_oxygen(atom37, aatype, bond_mat, link_csv_path, head_mask, tail_mask, N_C_anchor)

        dynamic_mask37 = mask37 * (~remove_mask_37).to(mask37.dtype)

        atom37_masked = atom37 * dynamic_mask37.unsqueeze(-1)
        one_hot_37_to_14 = torch.nn.functional.one_hot(idx37_to_14, num_classes=14).to(atom37_masked.dtype)  # [B, L, 37, 14]
        one_hot_14x37 = one_hot_37_to_14.permute(0, 1, 3, 2)  # [B, L, 14, 37]
        out14 = torch.matmul(one_hot_14x37, atom37_masked)  # [B, L, 14, 3]

        # mark non-existent heavy atoms as NaN so downstream writers skip them
        atom14_mask_table = torch.as_tensor(rc.restype_atom14_mask, device=out14.device, dtype=torch.bool)
        atom14_mask = atom14_mask_table[aatype]  # [B, L, 14]
        out14 = torch.where(atom14_mask.unsqueeze(-1), out14, torch.full_like(out14, float('nan')))

        # NEW: Adjust Oxygen positions for Tail Body residues based on specific bond connectivity
        # if self.backend == 'apm' and bond_mat is not None and N_C_anchor is not None and tail_mask is not None and link_csv_path is not None and LINK_UTILS_AVAILABLE:
        #      atom37 = self._adjust_tail_oxygen(atom37, aatype, bond_mat, link_csv_path, head_mask, tail_mask, N_C_anchor)

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


# --- Helper: Bi-Directional Update Function ---
def apply_bidirectional_anchor_update(xyz_in, delta, h_mask, t_mask):
    """
    h_mask (Head): Rotates N around CA-C axis.
    t_mask (Tail): Rotates C around N-CA axis.
    """
    xyz_out = xyz_in.clone()
    N_pos = xyz_in[..., 0:1, :] # Keep dim for axis_angle_rotation [B, L, 1, 3]
    CA_pos = xyz_in[..., 1, :]  # [B, L, 3]
    C_pos = xyz_in[..., 2:3, :] # [B, L, 1, 3]

    # --- 1. Head Update (Move N, Fix C) ---
    # Axis: CA -> C (Fixed vector)
    # Vector to rotate: CA -> N
    if h_mask.any():
        head_axis = xyz_in[..., 2, :] - CA_pos
        # Apply mask to delta (only calculate rotation for heads)
        delta_head = torch.where(h_mask, delta, torch.zeros_like(delta))
        
        N_rotated = iu.axis_angle_rotation(N_pos, head_axis, CA_pos, delta_head, h_mask)
        # Only update N positions where head_mask is True
        # axis_angle_rotation already handles masking internally, but we assign back
        xyz_out[..., 0:1, :] = N_rotated

    # --- 2. Tail Update (Move C, Fix N) ---
    # Axis: CA -> N (Fixed vector)
    # Vector to rotate: CA -> C
    if t_mask.any():
        tail_axis = xyz_in[..., 0, :] - CA_pos 
        # Apply mask to delta
        delta_tail = torch.where(t_mask, delta, torch.zeros_like(delta))
        
        C_rotated = iu.axis_angle_rotation(C_pos, tail_axis, CA_pos, delta_tail, t_mask)
        # Only update C positions where tail_mask is True
        xyz_out[..., 2:3, :] = C_rotated
    
    return xyz_out

# def apply_o_atom_rotation(xyz, psi_delta, mask):
#     """
#     xyz: (B, L, 14, 3)
#     psi_delta: (B, L)
#     mask: (B, L) - Body Tail Anchor Mask
#     """
#     CA_pos = xyz[..., 1, :]
#     C_pos = xyz[..., 2, :]
#     O_pos = xyz[..., 3:4, :] # Keep dim [B, L, 1, 3]
    
#     # Axis: CA -> C
#     axis = C_pos - CA_pos
#     pivot = C_pos

#     # Only apply where mask is True
#     delta = torch.where(mask, psi_delta, torch.zeros_like(psi_delta))
    
#     xyz_out = xyz.clone()
#     O_rotated = iu.axis_angle_rotation(O_pos, axis, pivot, delta, mask)
    
#     xyz_out[..., 3:4, :] = O_rotated
    
#     return xyz_out

def apply_head_phi_rotation(backbone_bb, phi_prev_delta, body_head_mask, chain_ids):
    """
    在 Head Anchor 附近，通过旋转同一条链上的上游残基 C(i-1) 及其上游原子来调节 phi(i)。
    
    修改点：增加了 chain_ids 参数，限制旋转只在同一条链内发生。
    """
    if not body_head_mask.any():
        return backbone_bb

    xyz = backbone_bb
    B, L = xyz.shape[:2]
    device = xyz.device

    # 全局索引 [0, 1, ..., L-1]
    idx = torch.arange(L, device=device)

    # 预分配旋转参数
    axis = torch.zeros((B, L, 3), dtype=xyz.dtype, device=device)
    pivot = torch.zeros((B, L, 3), dtype=xyz.dtype, device=device)
    angle = torch.zeros((B, L), dtype=xyz.dtype, device=device)
    mask = torch.zeros((B, L), dtype=torch.bool, device=device)

    b_idx, i_idx = torch.where(body_head_mask)
    for b, head_body_idx in zip(b_idx.tolist(), i_idx.tolist()):
        # i := head 对应 body 残基的 “后一个残基”
        i = head_body_idx + 1
        if i >= L:
            # 已经是链末尾，没有后一个残基，无法定义该 φ，跳过
            continue

        anchor_chain = chain_ids[b, head_body_idx]
        # 确保 i 与 head_body_idx 在同一条链上
        if chain_ids[b, i] != anchor_chain:
            continue

        # 取后一个残基 i 的主链三原子
        N_i = xyz[b, i, 0, :]   # N(i)
        CA_i = xyz[b, i, 1, :]  # CA(i)

        # 轴: N(i) -> CA(i)
        axis_vec = CA_i - N_i

        # 几何设定：
        #   - C(i)–CA(i) 以及下游固定；
        #   - N(i)–C(i-1) 以及更上游的残基作为刚体绕 N(i)–CA(i) 旋转。
        #
        # 因此旋转段应为：
        #   1. index <= i-1（包含 head_body_idx 及其上游）
        #   2. chain_id == anchor_chain
        seg_mask = (idx <= (i - 1)) & (chain_ids[b] == anchor_chain)
        if not seg_mask.any():
            continue

        axis[b, seg_mask, :] = axis_vec.unsqueeze(0)
        pivot[b, seg_mask, :] = N_i.unsqueeze(0)
        angle[b, seg_mask] = phi_prev_delta[b, head_body_idx]
        mask[b, seg_mask] = True

    # 调用通用的 axis-angle 旋转函数
    # 注意：这里假设 axis_angle_rotation 在当前命名空间或已导入
    xyz_rot = iu.axis_angle_rotation(xyz, axis, pivot, angle, mask)
    return xyz_rot


def apply_tail_psi_rotation(backbone_bb, psi_next_delta, body_tail_mask, chain_ids):
    """
    在 Tail Anchor 附近，通过旋转同一条链上的下游残基 N(j+1) 及其下游原子来调节 psi(j)。

    修改点：增加了 chain_ids 参数，限制旋转只在同一条链内发生。
    """
    if not body_tail_mask.any():
        return backbone_bb

    xyz = backbone_bb
    B, L = xyz.shape[:2]
    device = xyz.device

    idx = torch.arange(L, device=device)

    axis = torch.zeros((B, L, 3), dtype=xyz.dtype, device=device)
    pivot = torch.zeros((B, L, 3), dtype=xyz.dtype, device=device)
    angle = torch.zeros((B, L), dtype=xyz.dtype, device=device)
    mask = torch.zeros((B, L), dtype=torch.bool, device=device)

    b_idx, j_idx = torch.where(body_tail_mask)
    for b, tail_body_idx in zip(b_idx.tolist(), j_idx.tolist()):
        # j := tail 对应 body 残基的 “前一个残基”
        j = tail_body_idx - 1
        if j < 0:
            # 已经是链起点，没有前一个残基，无法定义该 ψ，跳过
            continue
        
        anchor_chain = chain_ids[b, tail_body_idx]
        # 确保 j 与 tail_body_idx 在同一条链上
        if chain_ids[b, j] != anchor_chain:
            continue

        CA_j = xyz[b, j, 1, :]  # CA(j)
        C_j = xyz[b, j, 2, :]   # C(j)

        # 轴: CA(j) -> C(j)
        axis_vec = C_j - CA_j

        # 几何设定：
        #   - N(j)–CA(j) 以及上游固定；
        #   - C(j)–N(j+1) 以及更下游的残基作为刚体绕 CA(j)–C(j) 旋转。
        #
        # 因此旋转段应为：
        #   1. index >= j+1（包含 tail_body_idx 及其下游）
        #   2. chain_id == anchor_chain
        seg_mask = (idx >= (j + 1)) & (chain_ids[b] == anchor_chain)
        if not seg_mask.any():
            continue

        axis[b, seg_mask, :] = axis_vec.unsqueeze(0)
        pivot[b, seg_mask, :] = C_j.unsqueeze(0)
        angle[b, seg_mask] = psi_next_delta[b, tail_body_idx]
        mask[b, seg_mask] = True

    xyz_rot = iu.axis_angle_rotation(xyz, axis, pivot, angle, mask)
    return xyz_rot

# ---------------------------------------------------------
# 辅助函数 (保持您提供的版本不变，确保放在同一文件中或正确导入)
# ---------------------------------------------------------
# def axis_angle_rotation(xyz, axis, pivot, angle, mask):
#     """
#     Rotates atoms in xyz around an axis defined by vector 'axis' and point 'pivot' by 'angle'.
#     """
#     # Normalize axis
#     axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8))
    
#     c = torch.cos(angle)
#     s = torch.sin(angle)
#     t = 1.0 - c
#     x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    
#     # Rodrigues' rotation matrix: [B, L, 3, 3]
#     R = torch.stack([
#         torch.stack([t*x*x + c,     t*x*y - s*z, t*x*z + s*y], dim=-1),
#         torch.stack([t*x*y + s*z,   t*y*y + c,   t*y*z - s*x], dim=-1),
#         torch.stack([t*x*z - s*y,   t*y*z + s*x, t*z*z + c], dim=-1),
#     ], dim=-2)
    
#     extra_dims = xyz.dim() - 3
    
#     if extra_dims > 0:
#         pivot_expanded = pivot
#         for _ in range(extra_dims):
#             pivot_expanded = pivot_expanded.unsqueeze(-2)
#     else:
#         pivot_expanded = pivot

#     rel_pos = xyz - pivot_expanded
    
#     # R maps vector j -> i
#     rot_pos = torch.einsum('blij,bl...j->bl...i', R, rel_pos)
    
#     new_xyz = rot_pos + pivot_expanded
    
#     mask_expanded = mask
#     for _ in range(extra_dims + 1): 
#         mask_expanded = mask_expanded.unsqueeze(-1)
        
#     return torch.where(mask_expanded, new_xyz, xyz)
    
# def apply_head_phi_rotation(backbone_bb, phi_prev_delta, body_head_mask):
#     """
#     在 Head Anchor 附近，通过旋转上游残基 C(i-1) 及其上游原子来调节 phi(i)=C(i-1)-N(i)-CA(i)-C(i)。

#     约定（与用户讨论后的版本）：
#       - 对于每条链仅有一个 body_head_anchor（body_head_mask 每个 batch 至多一个 True）。
#       - 固定 N(i)、CA(i)、C(i) 的位置不动。
#       - 使用 N(i)->CA(i) 作为旋转轴，围绕该轴旋转所有 index <= i-1 的残基原子。
#         这样：
#           * N(i)、CA(i) 在轴线上且不在旋转掩码中 => 不动；
#           * C(i) 不在旋转掩码中 => 不动；
#           * C(i-1) 以及更上游的残基被刚体旋转，从而改变 C(i-1)-N(i)-CA(i)-C(i) 的二面角。

#     Args:
#         backbone_bb: [B, L, 3, 3] (N, CA, C)
#         phi_prev_delta: [B, L] 旋转角度（只在 body_head_mask 为 True 的位置有效）
#         body_head_mask: [B, L] 布尔掩码，指示 Head Anchor 所在的 body 残基 i
#     """
#     if not body_head_mask.any():
#         return backbone_bb

#     xyz = backbone_bb
#     B, L = xyz.shape[:2]
#     device = xyz.device

#     # 全局索引 [0, 1, ..., L-1]
#     idx = torch.arange(L, device=device)

#     # 预分配旋转参数（默认不旋转）
#     axis = torch.zeros((B, L, 3), dtype=xyz.dtype, device=device)
#     pivot = torch.zeros((B, L, 3), dtype=xyz.dtype, device=device)
#     angle = torch.zeros((B, L), dtype=xyz.dtype, device=device)
#     mask = torch.zeros((B, L), dtype=torch.bool, device=device)

#     b_idx, i_idx = torch.where(body_head_mask)
#     for b, i in zip(b_idx.tolist(), i_idx.tolist()):
#         # 没有 i-1 时无法定义 phi(i) 的上游旋转，跳过
#         if i <= 0:
#             continue

#         # 取当前 batch/head anchor 的主链三原子
#         N_i = xyz[b, i, 0, :]   # N(i)
#         CA_i = xyz[b, i, 1, :]  # CA(i)

#         # 轴: N(i) -> CA(i)
#         axis_vec = CA_i - N_i

#         # 需要旋转的链段：所有 k <= i-1
#         seg_mask = idx <= (i - 1)

#         axis[b, seg_mask, :] = axis_vec.unsqueeze(0)
#         # 选择 N(i) 作为 pivot，保证 N(i) 在轴线上且不动
#         pivot[b, seg_mask, :] = N_i.unsqueeze(0)
#         angle[b, seg_mask] = phi_prev_delta[b, i]
#         mask[b, seg_mask] = True

#     # 使用通用的 axis-angle 旋转函数对 backbone 进行更新
#     xyz_rot = iu.axis_angle_rotation(xyz, axis, pivot, angle, mask)
#     return xyz_rot


# def apply_tail_psi_rotation(backbone_bb, psi_next_delta, body_tail_mask):
#     """
#     在 Tail Anchor 附近，通过旋转下游残基 N(j+1) 及其下游原子来调节
#     psi(j) = N(j) - CA(j) - C(j) - N(j+1)。

#     约定：
#       - 每条链仅有一个 body_tail_anchor（body_tail_mask 每个 batch 至多一个 True）。
#       - 固定 N(j)、CA(j)、C(j) 的位置不动。
#       - 使用 CA(j)->C(j) 作为旋转轴，围绕该轴旋转所有 index >= j+1 的残基原子。
#         这样：
#           * N(j)、CA(j)、C(j) 不在旋转掩码中 => 不动；
#           * N(j+1) 以及更下游的残基被刚体旋转，从而改变 N(j)-CA(j)-C(j)-N(j+1) 的二面角。

#     Args:
#         backbone_bb: [B, L, 3, 3] (N, CA, C)
#         psi_next_delta: [B, L] 旋转角度（只在 body_tail_mask 为 True 的位置有效）
#         body_tail_mask: [B, L] 布尔掩码，指示 Tail Anchor 所在的 body 残基 j
#     """
#     if not body_tail_mask.any():
#         return backbone_bb

#     xyz = backbone_bb
#     B, L = xyz.shape[:2]
#     device = xyz.device

#     idx = torch.arange(L, device=device)

#     axis = torch.zeros((B, L, 3), dtype=xyz.dtype, device=device)
#     pivot = torch.zeros((B, L, 3), dtype=xyz.dtype, device=device)
#     angle = torch.zeros((B, L), dtype=xyz.dtype, device=device)
#     mask = torch.zeros((B, L), dtype=torch.bool, device=device)

#     b_idx, j_idx = torch.where(body_tail_mask)
#     for b, j in zip(b_idx.tolist(), j_idx.tolist()):
#         # 没有 j+1 时无法定义 psi(j) 的下游旋转，跳过
#         if j >= L - 1:
#             continue

#         CA_j = xyz[b, j, 1, :]  # CA(j)
#         C_j = xyz[b, j, 2, :]   # C(j)

#         # 轴: CA(j) -> C(j)
#         axis_vec = C_j - CA_j

#         # 需要旋转的链段：所有 k >= j+1
#         seg_mask = idx >= (j + 1)

#         axis[b, seg_mask, :] = axis_vec.unsqueeze(0)
#         # 选择 C(j) 作为 pivot，保证 C(j) 在轴线上且不动
#         pivot[b, seg_mask, :] = C_j.unsqueeze(0)
#         angle[b, seg_mask] = psi_next_delta[b, j]
#         mask[b, seg_mask] = True

#     xyz_rot = iu.axis_angle_rotation(xyz, axis, pivot, angle, mask)
#     return xyz_rot

