import numpy as np
import os
import re
from omegaconf import DictConfig
import torch
import torch.nn.functional as nn
import networkx as nx
from Bio.PDB import MMCIFParser, MMCIF2Dict
from scipy.spatial.transform import Rotation as scipy_R
from rfdiff.util import rigid_from_3_points
from rfdiff import util
from rfdiff import chemical as che
import random
import logging

from BondFlow.models.allatom_wrapper import openfold_get_torsions


from rfdiff.chemical import one_aa2num,aa2num, num2aa, aa_321, aa_123, aabonds, aa2long
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
import pandas as pd
from .link_utils import load_allowed_bonds_from_csv
import networkit as nk
import BondFlow.data.SM_utlis as smu
import math

###########################################################
#### Functions which can be called outside of Denoiser ####
###########################################################
TOR_INDICES  = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES   = util.reference_angles
STANDARD_AMINO_ACIDS = num2aa
MAIN_CHAIN_ATOMS = {'N', 'CA', 'C', 'O', 'OXT'}
BOND_LENGTH_THRESHOLD = 5.0

# --- Constants for atom names ---
ATOM_CA = "CA"
ATOM_C = "C"
ATOM_N = "N"
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def get_R_from_xyz(xyz):
    """
    Get rotation matrix from xyz coordinates
    Args:
        xyz: coordinates of shape [B, L, 3, 3]
    Returns:
        R: rotation matrix of shape [B, L, 3, 3]
    """
    B, L = xyz.shape[:2]
    N_0, Ca_0, C_0 = xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :]

    # Build local right-handed frames via Gram–Schmidt (fully differentiable, no SVD/NumPy)
    eps = 1e-8

    v1 = C_0 - Ca_0  # primary axis (C -> CA)
    v2 = N_0 - Ca_0  # helper axis (N -> CA)

    def _normalize(vec):
        n = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(eps)
        return vec / n

    e1 = _normalize(v1)
    # remove component of v2 along e1
    v2_ortho = v2 - (e1 * (e1 * v2).sum(dim=-1, keepdim=True))
    e2 = _normalize(v2_ortho)
    # right-handed third axis
    e3 = torch.cross(e1, e2, dim=-1)
    e3 = _normalize(e3)
    # improve orthogonality of e2
    e2 = torch.cross(e3, e1, dim=-1)

    R = torch.stack([e1, e2, e3], dim=-1)  # [B, L, 3, 3]

    # Fallback to identity on non-finite
    invalid = ~torch.isfinite(R).all(dim=(-1, -2))
    if invalid.any():
        print("warning: get_R_from_xyz: invalid R")
        I = torch.eye(3, device=xyz.device, dtype=R.dtype).view(1, 1, 3, 3).expand(B, L, 3, 3)
        R = torch.where(invalid.unsqueeze(-1).unsqueeze(-1), I, R)

    return R.to(device=xyz.device, dtype=xyz.dtype)

def get_xyz_from_RT(R, T):
    """
    Get backbone xyz coordinates from rotation matrix R and translation T.
    Args:
        R: rotation matrix of shape [B, L, 3, 3]
        T: translation vector of shape [B, L, 3] (Ca coordinates)
    Returns:
        xyz: backbone coordinates of shape [B, L, 3, 3] (N, Ca, C)
    """
    # Ideal local coordinates of N and C relative to Ca
    # These values are from rfdiffusion.util
    N_ideal = che.init_N.to(device=R.device).float()
    C_ideal = che.init_C.to(device=R.device).float()
    R = R.float()
    # Transform local coordinates to global
    # R is [B, L, 3, 3], N_ideal is [3] -> N_global is [B, L, 3]

    N_global = torch.einsum('blij,j->bli', R, N_ideal).float() + T
    C_global = torch.einsum('blij,j->bli', R, C_ideal).float() + T
    Ca_global = T

    # Stack to get backbone coordinates
    # Unsqueeze to add atom dimension for stacking
    xyz = torch.stack([N_global, Ca_global, C_global], dim=-2)
    
    return xyz

def axis_angle_rotation(xyz, axis, pivot, angle, mask):
    """
    Rotates atoms in xyz around an axis defined by vector 'axis' and point 'pivot' by 'angle'.
    Only applies to atoms where 'mask' is True.
    
    Args:
        xyz: [B, L, ..., 3] or [..., 3] coordinates
        axis: [B, L, 3] Normalized or not (will be normalized)
        pivot: [B, L, 3]
        angle: [B, L]
        mask: [B, L] boolean mask
        
    Returns:
        xyz_out: [B, L, ..., 3] with rotation applied
    """
    # Normalize axis
    axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8))
    
    c = torch.cos(angle)
    s = torch.sin(angle)
    t = 1.0 - c
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    
    # Rodrigues' rotation matrix: [B, L, 3, 3]
    R = torch.stack([
        torch.stack([t*x*x + c,     t*x*y - s*z, t*x*z + s*y], dim=-1),
        torch.stack([t*x*y + s*z,   t*y*y + c,   t*y*z - s*x], dim=-1),
        torch.stack([t*x*z - s*y,   t*y*z + s*x, t*z*z + c], dim=-1),
    ], dim=-2)
    
    # Apply rotation
    # Ensure dimensions match
    # xyz can be [B, L, 3] or [B, L, 14, 3] or similar
    # pivot is [B, L, 3] -> expand to [B, L, 1, 3] if needed to match xyz's extra dims
    
    # Determine the number of extra dimensions between (B, L) and (3)
    extra_dims = xyz.dim() - 3 # e.g. if [B, L, 14, 3], extra=1; if [B, L, 3], extra=0
    
    if extra_dims > 0:
        pivot_expanded = pivot
        for _ in range(extra_dims):
            pivot_expanded = pivot_expanded.unsqueeze(-2)
    else:
        pivot_expanded = pivot

    rel_pos = xyz - pivot_expanded
    
    # Einsum: R[b,l,i,j] * rel_pos[b,l,...,j] -> rot_pos[b,l,...,i]
    # We can use matmul if we reshape properly, or einsum
    # To be generic with einsum string, let's assume standard broadcasting for matmul
    # R: [B, L, 3, 3]
    # rel_pos: [B, L, ..., 3] -> treat ... as batch for rotation? 
    # Or just use einsum with ellipsis
    
    # R maps vector j -> i
    rot_pos = torch.einsum('blij,bl...j->bl...i', R, rel_pos)
    
    new_xyz = rot_pos + pivot_expanded
    
    # Apply mask
    # mask: [B, L] -> expand to match xyz
    mask_expanded = mask
    for _ in range(extra_dims + 1): # +1 for the coordinates dim itself
        mask_expanded = mask_expanded.unsqueeze(-1)
        
    return torch.where(mask_expanded, new_xyz, xyz)

def parse_cif(filename, **kwargs):
    """
    Extract xyz coords for all heavy atoms from a cif file using BioPython.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("cif_structure", filename)
    cif_dict = MMCIF2Dict.MMCIF2Dict(filename)
    return parse_cif_structure(structure, cif_dict, **kwargs)


def parse_cif_structure(structure, cif_dict, parse_hetatom=False, ignore_het_h=True, 
                        parse_link=True, link_csv_path=None):
    """
    Extracts information from a BioPython structure object, mirroring parse_pdb_lines.
    """
    res, pdb_idx, rf_idx = [], [], []
    rf_index = 0
    last_index = None
    last_xyz = np.array([0, 0, 0], dtype=np.float32)
    
    residues_to_process = []
    
    # First pass: identify valid residues
    #for model in structure[0]:
    model = structure[0]
    for chain in model:
        for residue in chain:
            hetflag, resseq, icode = residue.get_id()
            if hetflag.strip() != '':  # Skip HETATM residues in this pass
                continue

            resname = residue.get_resname()
            if resname not in util.aa2num:
                continue
            
            # Skip glycine without sidechain
            if resname != 'GLY' and 'CB' not in residue.child_dict:
                continue
            
            # Check for complete backbone
            if 'N' in residue and 'CA' in residue and 'C' in residue:
                residues_to_process.append(residue)

    # Sort residues like in a PDB file (by chain then by residue number)
    residues_to_process.sort(key=lambda r: (r.get_parent().id, r.get_id()[1], r.get_id()[2]))
    for residue in residues_to_process:
        resname = residue.get_resname()
        chain_id = residue.get_parent().id
        _, resseq, icode = residue.get_id()
        
        resseq_str = str(resseq) + (icode if icode.strip() else "")
        current_index = (chain_id, resseq_str)
        current_xyz = residue['CA'].get_coord()

        if last_index is not None and last_index[0] == current_index[0]:
            try:
                # Attempt to parse residue numbers as integers for gap calculation
                last_res_num = int(re.match(r'^-?\d+', last_index[1]).group(0))
                current_res_num = int(re.match(r'^-?\d+', current_index[1]).group(0))
                distance = np.linalg.norm(current_xyz - last_xyz)
                
                res_diff = current_res_num - last_res_num
                if 1 < res_diff < 200 and distance > BOND_LENGTH_THRESHOLD:
                    rf_index += res_diff
                else:
                    rf_index += 1
            except (ValueError, TypeError):
                # Fallback if residue numbers are not simple integers
                rf_index += 1
        else:
            rf_index += 1
        res.append((resseq_str, resname))
        pdb_idx.append(current_index)
        rf_idx.append(rf_index)

        last_xyz = current_xyz
        last_index = current_index

    seq = [util.aa2num.get(r[1], 20) for r in res]
    
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    
    pdb_idx_to_res_idx = {pdb: i for i, pdb in enumerate(pdb_idx)}

    for residue in residues_to_process:
        chain_id = residue.get_parent().id
        _, resseq, icode = residue.get_id()
        resseq_str = str(resseq) + (icode if icode.strip() else "")
        current_pdb_idx = (chain_id, resseq_str)
        
        i = pdb_idx_to_res_idx.get(current_pdb_idx)
        if i is None:
            continue

        resname = residue.get_resname()
        if resname in util.aa2num:
            res_map_idx = util.aa2num[resname]
            for atom in residue:
                atom_name = atom.get_name()
                # Find atom index in the 14-atom representation
                try:
                    # aa2long is a list of lists, so we access the one for the current residue
                    atom_idx_in_14 = util.aa2long[res_map_idx].index(" "+atom_name.ljust(3))
                    xyz[i, atom_idx_in_14, :] = atom.get_coord()
                except (ValueError, IndexError):
                    continue # Atom not in the 14-atom list for this residue

    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz)] = 0.0

    # This part for removing duplicates is kept from parse_pdb, though less likely with BioPython
    new_idx, i_unique, new_rf_idx = [], [], []
    for i, idx_tuple in enumerate(pdb_idx):
        if idx_tuple not in new_idx:
            new_idx.append(idx_tuple)
            new_rf_idx.append(rf_idx[i])
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]
    seq = np.array(seq)[i_unique]

    out = {
        "xyz": xyz,
        "mask": mask,
        "idx": new_rf_idx,
        "seq": np.array(seq),
        "pdb_idx": pdb_idx,
    }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        model = structure[0]
        #for model in structure[:1]:
        for chain in model:
            for residue in chain:
                hetflag, _, _ = residue.get_id()
                if hetflag.strip() != '' and hetflag.strip() != 'W':
                    for atom in residue:
                        if ignore_het_h and atom.element == 'H':
                            continue
                        info_het.append(
                            dict(
                                idx=atom.get_serial_number(),
                                atom_id=atom.get_name(),
                                atom_type=atom.element,
                                name=residue.get_resname(),
                            )
                        )
                        xyz_het.append(atom.get_coord())

        out["xyz_het"] = np.array(xyz_het, dtype=np.float32)
        out["info_het"] = info_het
        
        
    if parse_link:
        allowed_link_types = load_allowed_bonds_from_csv(link_csv_path)
        links = []
        # Build a quick lookup for atom objects from the parsed structure
        atom_lookup = {}
        for r in residues_to_process:
            chain_id = r.get_parent().id
            _, resseq, icode = r.get_id()
            resseq_str = str(resseq) + (icode if icode.strip() else "")
            pdb_tuple = (chain_id, resseq_str)
            if pdb_tuple in pdb_idx:
                for atom in r:
                    atom_lookup[(chain_id, resseq_str, atom.get_name())] = atom

        # 1. 只检查最核心的、一定存在的列来确定循环次数
        #    几乎所有的 _struct_conn 记录都至少有 conn_type_id
        core_key = '_struct_conn.conn_type_id'
        if core_key in cif_dict:
            num_conns = len(cif_dict[core_key])
            
            # 2. 安全地获取每个数据列，如果列不存在，则提供一个默认的空列表
            conn_types = cif_dict.get('_struct_conn.conn_type_id', [])
            
            p1_chains = cif_dict.get('_struct_conn.ptnr1_auth_asym_id', []) or cif_dict.get('_struct_conn.ptnr1_label_asym_id', [])
            p1_res_names = cif_dict.get('_struct_conn.ptnr1_auth_comp_id', []) or cif_dict.get('_struct_conn.ptnr1_label_comp_id', [])
            p1_res_nums = cif_dict.get('_struct_conn.ptnr1_auth_seq_id', []) or cif_dict.get('_struct_conn.ptnr1_label_seq_id', [])
            p1_atom_names = cif_dict.get('_struct_conn.ptnr1_label_atom_id', [])
            # 对于可选的插入码列，如果不存在，我们让它返回一个空列表
            p1_ins_codes = cif_dict.get('_struct_conn.pdbx_ptnr1_PDB_ins_code', [])

            p2_chains = cif_dict.get('_struct_conn.ptnr2_auth_asym_id', []) or cif_dict.get('_struct_conn.ptnr2_label_asym_id', [])
            p2_res_names = cif_dict.get('_struct_conn.ptnr2_auth_comp_id', []) or cif_dict.get('_struct_conn.ptnr2_label_comp_id', [])
            p2_res_nums = cif_dict.get('_struct_conn.ptnr2_auth_seq_id', []) or cif_dict.get('_struct_conn.ptnr2_label_seq_id', [])
            p2_atom_names = cif_dict.get('_struct_conn.ptnr2_label_atom_id', [])
            p2_ins_codes = cif_dict.get('_struct_conn.pdbx_ptnr2_PDB_ins_code', [])
            # print(f"Processing {num_conns} connections")
            for i in range(num_conns):
                try:
                    conn_type = conn_types[i]
                    if conn_type not in ['disulf', 'covale']: # 增加对 isopeptide 的支持
                        continue

                    # 3. 在循环内部安全地获取每个值
                    #    如果列表为空（因为列不存在），则提供默认值 '?'
                    p1_chain = p1_chains[i]
                    p1_res_name = p1_res_names[i]
                    p1_res_num = p1_res_nums[i]
                    p1_atom_name = p1_atom_names[i]
                    p1_ins_code = p1_ins_codes[i] if i < len(p1_ins_codes) else '?'
                    
                    p2_chain = p2_chains[i]
                    p2_res_name = p2_res_names[i]
                    p2_res_num = p2_res_nums[i]
                    p2_atom_name = p2_atom_names[i]
                    p2_ins_code = p2_ins_codes[i] if i < len(p2_ins_codes) else '?'
                    
                    # Build residue identifiers robustly: keep insertion codes and avoid int() casts.
                    # Skip connections with missing residue numbers ('.' or '?').
                    p1_res_num_str = (p1_res_num or '').strip()
                    p1_ins_code_clean = (p1_ins_code or '').strip()
                    if p1_res_num_str in ('.', '?', ''):
                        continue
                    if p1_ins_code_clean in ('.', '?'):
                        p1_ins_code_clean = ''
                    p1_res_num_str = p1_res_num_str + p1_ins_code_clean
                    idx1 = (p1_chain, p1_res_num_str)

                    p2_res_num_str = (p2_res_num or '').strip()
                    p2_ins_code_clean = (p2_ins_code or '').strip()
                    if p2_res_num_str in ('.', '?', ''):
                        continue
                    if p2_ins_code_clean in ('.', '?'):
                        p2_ins_code_clean = ''
                    p2_res_num_str = p2_res_num_str + p2_ins_code_clean
                    idx2 = (p2_chain, p2_res_num_str)
                    # (后续的逻辑保持不变)
                    if idx1 == idx2: continue
                    if not (idx1 in pdb_idx and idx2 in pdb_idx): continue

                    atom1_obj = atom_lookup.get((p1_chain, p1_res_num_str, p1_atom_name))
                    atom2_obj = atom_lookup.get((p2_chain, p2_res_num_str, p2_atom_name))
                    if atom1_obj is None or atom2_obj is None: continue
                    
                    distance = np.linalg.norm(atom1_obj.get_coord() - atom2_obj.get_coord())
                    if distance > BOND_LENGTH_THRESHOLD: continue

                    # 仅当提供的 link_csv_path 指定了允许的键时才进行过滤
                    if allowed_link_types:
                        res1_up = (p1_res_name or '').upper()
                        res2_up = (p2_res_name or '').upper()
                        atom1_up = (p1_atom_name or '').upper()
                        atom2_up = (p2_atom_name or '').upper()
                        if (res1_up, res2_up, atom1_up, atom2_up) not in allowed_link_types:
                            # 不在允许列表中，跳过
                            continue
                    # Check if this link already exists in the links list
                    link_exists = False
                    for existing_link in links:
                        if ((existing_link["idx1"] == idx1 and existing_link["idx2"] == idx2 and
                             existing_link["res1"] == p1_res_name and existing_link["res2"] == p2_res_name) or
                            (existing_link["idx1"] == idx2 and existing_link["idx2"] == idx1 and
                             existing_link["res1"] == p2_res_name and existing_link["res2"] == p1_res_name)):
                            link_exists = True
                            break
                    
                    if link_exists:
                        continue
                    links.append({
                        "res1": p1_res_name, "idx1": idx1, "atom1": p1_atom_name,
                        "res2": p2_res_name, "idx2": idx2, "atom2": p2_atom_name,
                        "distance": distance
                    })

                except (ValueError, KeyError, IndexError) as e:
                    # INSERT_YOUR_CODE
                    # 打印出错的cif id（文件名），如果有
                    if 'cif_dict' in locals() and '_entry.id' in cif_dict:
                        print(f"Error in CIF id: {cif_dict['_entry.id'][0]}")
                    elif 'structure' in locals():
                        try:
                            print(f"Error in CIF structure id: {structure.id}")
                        except Exception:
                            pass
                    print(f"Could not parse a LINK/SSBOND record from CIF at index {i}: {e}")
                    continue
                    
        out["links"] = links

    return out


    

def process_target(
    pdb_path,
    parse_hetatom: bool = False,
    center: bool = True,
    parse_link: bool = True,
    parse_alpha: bool = True,
    link_csv_path=None,
    plm_encoder=None,
    plm_max_chain_length: int | None = None,
):
    """
    Parse a PDB/CIF file into a unified target representation used by BondFlow.

    Args:
        pdb_path: Path to input PDB/CIF file.
        parse_hetatom: Whether to parse hetero atoms.
        center: If True, zero-center CA coordinates.
        parse_link: Whether to parse LINK records into covalent links.
        parse_alpha: Whether to compute torsion-angle features (requires OpenFold/APM stack).
        link_csv_path: Optional CSV to constrain allowed LINK types.
        plm_encoder: Optional callable to compute per-chain PLM embeddings.
            Expected signature:
                plm_encoder(chain_seq: np.ndarray[int], chain_id: str, max_len: int | None) -> np.ndarray[float]
            where:
                - chain_seq: integer-encoded sequence for one chain (same coding as target_struct["seq"])
                - chain_id: chain ID, e.g. "A"
                - max_len: length cap for PLM model (caller should handle chunking / truncation if needed)
            The callable should return an array of shape [L_chain, D].
        plm_max_chain_length: Optional maximum length to pass to plm_encoder for a single chain.

    Returns:
        out: dict with keys:
            - xyz_14, mask_14, seq, pdb_idx, idx, chains, links (if parse_link),
              alpha / alpha_alt / alpha_tor_mask (if parse_alpha),
              pdb_id,
              plm_emb (if plm_encoder is provided).
    """
    target_struct = parse_cif(pdb_path, parse_hetatom=parse_hetatom, parse_link=parse_link, link_csv_path=link_csv_path)

    # Zero-center positions
    ca_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True)
    if not center:
        ca_center = 0
    xyz = torch.from_numpy(target_struct["xyz"] - ca_center)
    seq_orig = torch.from_numpy(target_struct["seq"])
    atom_mask = torch.from_numpy(target_struct["mask"])

    out = {
        "xyz_14": xyz,
        "mask_14": atom_mask,
        "seq": seq_orig,
        "pdb_idx": target_struct["pdb_idx"], #[('A', '100'), (A', '101'), ...]
        "idx": target_struct["idx"], #[0,1,2,3,4,...]
        "chains": list(set([i[0] for i in target_struct["pdb_idx"]])),
    }
    if parse_hetatom:
        out["xyz_het"] = target_struct["xyz_het"]
        out["info_het"] = target_struct["info_het"]
    if parse_link:
        out["links"] = target_struct["links"]
    if parse_alpha:
        if openfold_get_torsions is None:
            raise ImportError(
                "openfold_get_torsions is not available. "
                "Torsion-angle features require extra dependencies (OpenFold/APM/mdtraj)."
            )
        L = xyz.shape[0]
        xyz_27 = torch.cat((xyz, torch.full((L,13,3), float('nan'))), dim=1)
        #alpha, alpha_alt, alpha_mask, _ = util.get_torsions(xyz_27.unsqueeze(0), seq_orig.unsqueeze(0), TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)

        alpha, alpha_alt, alpha_mask, _ = openfold_get_torsions(
            seq_orig.unsqueeze(0), xyz_27.unsqueeze(0)
        )
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha_alt[torch.isnan(alpha_alt)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_alt = alpha_alt.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        out['alpha'] = alpha.squeeze()
        out['alpha_tor_mask'] = alpha_mask.squeeze()
        out['alpha_alt'] = alpha_alt.squeeze()

    out['pdb_id'] = os.path.basename(pdb_path).split('.')[0]  # pdb_id is the name of the pdb file without extension

    # Optional: compute per-chain PLM embeddings on the *full* parsed structure.
    # This allows downstream cropping modes (e.g. complex_space) to slice embeddings
    # by index without ever feeding broken/discontinuous fragments into the PLM.
    # if plm_encoder is not None:
    #     seq_np = target_struct["seq"]  # [L_total]
    #     pdb_idx = target_struct["pdb_idx"]
    #     chains = [ch for ch, _ in pdb_idx]
    #     unique_chains = []
    #     for ch in chains:
    #         if ch not in unique_chains:
    #             unique_chains.append(ch)

    #     # First pass: determine embedding dimensionality from the first non-empty chain
    #     plm_emb_full = None
    #     first_chain = None
    #     for ch in unique_chains:
    #         chain_positions = [i for i, (c, _) in enumerate(pdb_idx) if c == ch]
    #         if not chain_positions:
    #             continue
    #         chain_seq = seq_np[chain_positions]
    #         max_len = plm_max_chain_length
    #         emb_chain = plm_encoder(chain_seq, ch, max_len)
    #         if emb_chain is None:
    #             continue
    #         emb_chain = np.asarray(emb_chain, dtype=np.float32)
    #         if emb_chain.ndim != 2 or emb_chain.shape[0] != len(chain_seq):
    #             print(
    #                 f"plm_encoder for chain {ch} must return array of shape [L_chain, D], "
    #                 f"got {emb_chain.shape}"
    #             )
    #         D = emb_chain.shape[1]
    #         plm_emb_full = np.zeros((len(seq_np), D), dtype=np.float32)
    #         # Fill this first chain and break; remaining chains handled below.
    #         for pos, row in zip(chain_positions, emb_chain):
    #             plm_emb_full[pos] = row
    #         first_chain = ch
    #         break

    #     # If we successfully initialized plm_emb_full, fill remaining chains
    #     if plm_emb_full is not None:
    #         # We already processed one chain in the loop above, so skip it here
    #         processed_chains = set([first_chain]) if first_chain is not None else set()
    #         for ch in unique_chains:
    #             if ch in processed_chains:
    #                 continue
    #             chain_positions = [i for i, (c, _) in enumerate(pdb_idx) if c == ch]
    #             if not chain_positions:
    #                 continue
    #             chain_seq = seq_np[chain_positions]
    #             max_len = plm_max_chain_length
    #             emb_chain = plm_encoder(chain_seq, ch, max_len)
    #             if emb_chain is None:
    #                 continue
    #             emb_chain = np.asarray(emb_chain, dtype=np.float32)
    #             if emb_chain.shape[0] != len(chain_seq) or emb_chain.shape[1] != plm_emb_full.shape[1]:
    #                 raise ValueError(
    #                     f"plm_encoder for chain {ch} returned inconsistent shape {emb_chain.shape}; "
    #                     f"expected [{len(chain_seq)}, {plm_emb_full.shape[1]}]."
    #                 )
    #             for pos, row in zip(chain_positions, emb_chain):
    #                 plm_emb_full[pos] = row

    #         out["plm_emb"] = plm_emb_full
    # else:
    #     print("plm_encoder is None")
    return out


def update_nc_node_coordinates(xyz: torch.Tensor, nc_anchor: torch.Tensor, head_mask: torch.Tensor, tail_mask: torch.Tensor, apply_offset: bool = True) -> torch.Tensor:
    """
    根据 N_C_anchor 将 head_mask/tail_mask 对应的功能节点的坐标更新为其锚定 Body 残基的坐标。
    支持 (L, ...) 单样本输入或 (B, L, ...) 批次输入。

    Args:
        xyz: (..., L, 14, 3) 坐标张量
        nc_anchor: (..., L, L, 2) 锚定矩阵，[..., 0]为N端，[..., 1]为C端
        head_mask: (..., L) N端节点掩码
        tail_mask: (..., L) C端节点掩码
        apply_offset: 是否应用偏移 (模拟从锚定点延伸)
                      N节点偏移: +(N - CA)
                      C节点偏移: +(C - CA)

    Returns:
        xyz_new: 更新后的坐标张量
    """
    xyz_new = xyz.clone()

    # 标准化为 Batch 模式处理，如果输入没有 Batch 维度则 unsqueeze
    is_batched = xyz.dim() == 4
    if not is_batched:
        xyz_new = xyz_new.unsqueeze(0)
        nc_anchor = nc_anchor.unsqueeze(0)
        head_mask = head_mask.unsqueeze(0)
        tail_mask = tail_mask.unsqueeze(0)
        
    B, L = xyz_new.shape[:2]
    
    # 确保 mask 为 bool
    head_mask = head_mask.bool()
    tail_mask = tail_mask.bool()
    
    # --- 处理 N 端节点 (head_mask) ---
    batch_idx, n_idx = torch.where(head_mask)
    if len(batch_idx) > 0:
        # nc_anchor[b, n, :, 0] 应该有一个是 True
        anchor_row = nc_anchor[batch_idx, n_idx, :, 0] # (K, L)
        # 获取锚定节点索引，使用 argmax (假设每行只有一个 1)
        body_idx = anchor_row.float().argmax(dim=-1) # (K,)
        
        # 检查是否有效 (防止全0导致 argmax=0)
        valid_anchor = anchor_row.any(dim=-1)
        
        valid_b = batch_idx[valid_anchor]
        valid_n = n_idx[valid_anchor]
        valid_body = body_idx[valid_anchor]
        
        # 复制坐标
        xyz_new[valid_b, valid_n] = xyz_new[valid_b, valid_body]
        
        if apply_offset:
            # N节点偏移: v_N_CA = xyz[n, N_atom] - xyz[n, CA_atom]
            # 这里 xyz_new 已经是 body 的坐标
            # atom 0: N, atom 1: CA
            vec = xyz_new[valid_b, valid_n, 0, :] - xyz_new[valid_b, valid_n, 1, :]
            xyz_new[valid_b, valid_n] += vec.unsqueeze(1) # Broadcast over atoms

    # --- 处理 C 端节点 (tail_mask) ---
    batch_idx, c_idx = torch.where(tail_mask)
    if len(batch_idx) > 0:
        anchor_row = nc_anchor[batch_idx, c_idx, :, 1] # (K, L)
        body_idx = anchor_row.float().argmax(dim=-1) # (K,)
        
        valid_anchor = anchor_row.any(dim=-1)
        valid_b = batch_idx[valid_anchor]
        valid_c = c_idx[valid_anchor]
        valid_body = body_idx[valid_anchor]
        
        xyz_new[valid_b, valid_c] = xyz_new[valid_b, valid_body]
        
        if apply_offset:
            # C节点偏移: v_CA_C = xyz[c, C_atom] - xyz[c, CA_atom]
            # atom 2: C, atom 1: CA
            vec = xyz_new[valid_b, valid_c, 2, :] - xyz_new[valid_b, valid_c, 1, :]
            xyz_new[valid_b, valid_c] += vec.unsqueeze(1)

    if not is_batched:
        xyz_new = xyz_new.squeeze(0)
        
    return xyz_new


def update_nc_node_features(tensor: torch.Tensor, nc_anchor: torch.Tensor, head_mask: torch.Tensor, tail_mask: torch.Tensor) -> torch.Tensor:
    """
    Update features of functional nodes (head/tail) by gathering from their anchored body residues.
    This replaces the old _apply_mask_shift which assumed sequence adjacency.
    
    Args:
        tensor: [B, L, ...] Features to update (e.g. seq, xyz)
        nc_anchor: [B, L, L, 2] Anchor matrix
        head_mask: [B, L] Mask for head nodes
        tail_mask: [B, L] Mask for tail nodes
        
    Returns:
        tensor: Updated tensor
    """
    if nc_anchor is None or tensor is None:
        return tensor
        
    # Ensure masks are boolean
    head_mask = head_mask.bool()
    tail_mask = tail_mask.bool()
    
    # Helper to apply update for one layer (0 for head, 1 for tail)
    def _apply(mask, anchor_layer):
        batch_idx, node_idx = torch.where(mask)
        if len(batch_idx) == 0:
            return
            
        # Get anchor rows: [N_nodes, L]
        anchor_rows = nc_anchor[batch_idx, node_idx, :, anchor_layer]
        
        # Find body indices (argmax over L)
        # Check if valid anchor exists (row not all zeros)
        valid_anchor = anchor_rows.any(dim=-1)
        if not valid_anchor.any():
            return
            
        # Filter to valid updates
        batch_idx = batch_idx[valid_anchor]
        node_idx = node_idx[valid_anchor]
        anchor_rows = anchor_rows[valid_anchor]
        
        body_idx = anchor_rows.float().argmax(dim=-1)
        
        # Copy features: tensor[b, node] = tensor[b, body]
        # Handle arbitrary trailing dimensions
        tensor[batch_idx, node_idx] = tensor[batch_idx, body_idx]

    # Update Head nodes (layer 0)
    _apply(head_mask, 0)
    
    # Update Tail nodes (layer 1)
    _apply(tail_mask, 1)
    
    return tensor


class Target:
    """
    Class to handle targets (parsed chains).

    """

    def __init__(self, conf: DictConfig, pdb_parsed, N_C_add=True, nc_pos_prob = 0.3, inference: bool = False):
        self.inference = bool(inference)
        self.design_conf = conf
        self.N_C_add = N_C_add
        if self.design_conf.contigs is None:
            raise ValueError(
                "No design configuration provided"
            )
        self.chain_order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.pdb = pdb_parsed
        # if self.design_conf.input_pdb:      
        #     self.pdb = process_target(self.design_conf.input_pdb,
        #                             parse_hetatom=False, 
        #                             center=False,
        #                             parse_link=True)
        (
            self.full_seq,
            self.full_xyz,
            self.full_rf_idx,
            self.full_mask_str,  # False is fixed, True is changable
            self.full_mask_seq,  # False is fixed, True is changable
            self.full_pdb_idx,
            self.full_origin_pdb_idx,
            self.full_alpha,
            self.full_alpha_alt,
            self.full_alpha_tor_mask,  # False is no sidechain
            self.full_chain_ids,
            #self.full_plm_emb,
        ) = self.parse_contigs(
            self.design_conf.contigs,
            self.design_conf.length,
        )

        res_mask_full = torch.ones(len(self.full_origin_pdb_idx))
        pad_positions = [i for i, origin in enumerate(self.full_origin_pdb_idx) if origin == ('?', '-1')]
        if pad_positions and not self.inference:
            res_mask_full[pad_positions] = 0
        self.res_mask = res_mask_full

        # Initialize N/C anchor matrix (L, L, 2) if N_C_add is enabled.
        # 语义：
        #   - [..., 0] 层表示 N 功能节点与其锚定残基之间的隶属关系
        #   - [..., 1] 层表示 C 功能节点与其锚定残基之间的隶属关系
        #   - 只有功能 N / C 节点对应的行和列会出现 1，其余为 0
        #   - full_head_mask / full_tail_mask 保持原有语义，用于标识功能 N/C 节点
        L = len(self.full_seq)
        self.full_N_C_anchor = torch.zeros((L, L, 2), dtype=torch.bool)
        if self.N_C_add:
            self._init_default_nc_anchor()

        # 基础键矩阵（来自 PDB links + bond_condition）
        self.full_bond_matrix, self.full_bond_mask = self.parse_bond_condition()

        # 如果配置中启用了 N/C 训练样本，则在这里基于当前的 full_bond_matrix
        # 和 full_N_C_anchor 生成一批 N–C 正/负样本对，再做双随机化。
        #
        # 仅在训练阶段启用：
        #   - 推理 / 设计阶段（inference=True，例如 cyclize_from_pdb）不应再对 full_* 做重排，
        #     否则会破坏 YAML 指定的 contig 布局和 rf_idx / head_mask / tail_mask 语义。
        if self.N_C_add and (not getattr(self, "inference", False)) and nc_pos_prob is not None and float(nc_pos_prob) > 0.0:
            self.build_nc_training_pairs(pos_prob=float(nc_pos_prob))

        self.full_bond_matrix = smu.make_sub_doubly2doubly_stochastic(self.full_bond_matrix)

        # Initialize hotspot tensor
        self.full_hotspot = torch.zeros(len(self.full_seq), dtype=torch.float32)
        if self.design_conf.get('hotspots'):
            self.parse_hotspot(self.design_conf.hotspots)
        
        
        self.pdb_core_id = self.pdb['pdb_id']
        # Full-structure metadata (may be missing when running in pure-prior mode with no input PDB/CIF).
        # Convert seq to numpy array to avoid collate_fn stacking as a Tensor.
        if isinstance(self.pdb, dict):
            self.pdb_core_id = self.pdb.get('pdb_id', 'unknown')
            seq_full = self.pdb.get('seq', None)
            if seq_full is None:
                # No input structure: fall back to the designed (full) sequence.
                seq_full = self.full_seq.detach().cpu().numpy() if torch.is_tensor(self.full_seq) else self.full_seq
            if torch.is_tensor(seq_full):
                seq_full = seq_full.cpu().numpy()
            self.pdb_seq_full = seq_full
            self.pdb_idx_full = self.pdb.get('pdb_idx', None)
        else:
            # Unexpected type; keep safe defaults
            self.pdb_core_id = 'unknown'
            self.pdb_seq_full = self.full_seq.detach().cpu().numpy() if torch.is_tensor(self.full_seq) else self.full_seq
            self.pdb_idx_full = None

    def parse_hotspot(self, hotspot_list):
        """
        Parses the hotspot list and populates self.full_hotspot tensor.
        Args:
            hotspot_list (list): List of strings identifying hotspot residues (e.g. ["A/100", "B200"]).
        """
        origin_to_indices = {}
        for idx, origin in enumerate(self.full_origin_pdb_idx):
            if origin == ('?', '-1'):
                continue
            if origin not in origin_to_indices:
                origin_to_indices[origin] = []
            origin_to_indices[origin].append(idx)

        for item in hotspot_list:
            chain, res = None, None
            # Support "Chain/Residue" format
            if '/' in item:
                parts = item.split('/')
                if len(parts) == 2:
                    chain, res = parts
            else:
                # Support "ChainResidue" format (e.g. A100)
                # This regex captures chain (non-greedy) and residue number (including negative and insertion code)
                match = re.match(r"^([A-Za-z0-9]+?)(-?\d+[A-Za-z]*)$", item)
                if match:
                    chain, res = match.groups()
            
            if chain and res:
                key = (chain, res)
                if key in origin_to_indices:
                    for idx in origin_to_indices[key]:
                        self.full_hotspot[idx] = 1.0

    def _init_default_nc_anchor(self):
        """
        基于当前的 head/tail mask 与 origin 映射，构建一个最基础的 N_C_anchor：
          - 每条链的 N 功能节点锚定到该链的首个天然残基；
          - 每条链的 C 功能节点锚定到该链的尾部天然残基。

        注意：
          - full_head_mask / full_tail_mask 的语义保持不变：True 表示功能 N/C 节点；
          - 这里不做正负样本采样，也不改动 full_bond_matrix，只提供一个合理的默认锚定；
          - 之后可以在训练/推理阶段根据需要覆写 / 扩展 N_C_anchor。
        """
        if not hasattr(self, "full_head_mask") or not hasattr(self, "full_tail_mask"):
            return

        # head / tail 节点索引
        # full_head_mask / full_tail_mask 在 parse_contigs 中已经按链拼接好
        head_mask = self.full_head_mask
        tail_mask = self.full_tail_mask

        if head_mask.numel() == 0 or tail_mask.numel() == 0:
            return

        # 推理模式：不再依赖 origin，而是把整条设计链视作 body，
        #   - 每条链的 N 功能节点锚定该链第一个非 N/C 残基；
        #   - 每条链的 C 功能节点锚定该链最后一个非 N/C 残基。
        if getattr(self, "inference", False):
            self.full_N_C_anchor.fill_(False)
            chain_ids = self.full_chain_ids
            for ch in torch.unique(chain_ids):
                chain_idx = torch.where(chain_ids == ch)[0]
                if chain_idx.numel() == 0:
                    continue

                heads = chain_idx[head_mask[chain_idx]]
                tails = chain_idx[tail_mask[chain_idx]]
                if heads.numel() == 0 or tails.numel() == 0:
                    continue

                h = heads[0].item()
                t = tails[0].item()

                # body = 该链上除 head / tail 外的所有残基（包括 New_）
                body = [i for i in chain_idx.tolist() if i not in (h, t)]
                if not body:
                    continue
                first_body = body[0]
                last_body = body[-1]

                # N 层
                self.full_N_C_anchor[h, first_body, 0] = True
                self.full_N_C_anchor[first_body, h, 0] = True
                # C 层
                self.full_N_C_anchor[t, last_body, 1] = True
                self.full_N_C_anchor[last_body, t, 1] = True
            return

        # 训练模式：依赖 origin -> (body / nter / cter) 映射（保持原行为）
        # 依赖 parse_contigs 中已经构建好的 origin -> (body / nter / cter) 映射：
        #   - self._origin_to_body_idx:  (chain, res) -> 该残基在 full_* 中的主体 index
        #   - self._origin_to_nter_idx:  (chain, res) -> 该残基对应的 N 功能节点 index
        #   - self._origin_to_cter_idx:  (chain, res) -> 该残基对应的 C 功能节点 index
        #
        # 默认行为：N/C 功能节点各自锚定回同一残基的主体位置。
        # 这与之前 “在第一个/最后一个有效残基外侧再加一个 N/C 节点” 的行为是一致的，
        # 只是现在通过 full_N_C_anchor 显式表达出来。

        # N 层
        if hasattr(self, "_origin_to_nter_idx") and hasattr(self, "_origin_to_body_idx"):
            for origin, n_idx in self._origin_to_nter_idx.items():
                body_idx = self._origin_to_body_idx.get(origin, None)
                if body_idx is None:
                    continue
                # 只在 N 层上标记对称的 1
                self.full_N_C_anchor[n_idx, body_idx, 0] = True
                self.full_N_C_anchor[body_idx, n_idx, 0] = True

        # C 层
        if hasattr(self, "_origin_to_cter_idx") and hasattr(self, "_origin_to_body_idx"):
            for origin, c_idx in self._origin_to_cter_idx.items():
                body_idx = self._origin_to_body_idx.get(origin, None)
                if body_idx is None:
                    continue
                self.full_N_C_anchor[c_idx, body_idx, 1] = True
                self.full_N_C_anchor[body_idx, c_idx, 1] = True

    def _permute_target_data(self, perm_indices):
        """
        Helper to permute all full_* attributes according to perm_indices.
        Ensures all tensors and lists are reordered synchronously.
        """
        device = self.full_seq.device
        perm_indices = perm_indices.to(device)
        
        # 1. Permute 1D Tensors
        self.full_seq = self.full_seq[perm_indices]
        self.full_xyz = self.full_xyz[perm_indices]
        self.full_rf_idx = self.full_rf_idx[perm_indices]
        self.full_mask_str = self.full_mask_str[perm_indices]
        self.full_mask_seq = self.full_mask_seq[perm_indices]
        self.full_alpha = self.full_alpha[perm_indices]
        self.full_alpha_alt = self.full_alpha_alt[perm_indices]
        self.full_alpha_tor_mask = self.full_alpha_tor_mask[perm_indices]
        self.full_head_mask = self.full_head_mask[perm_indices]
        self.full_tail_mask = self.full_tail_mask[perm_indices]
        self.full_chain_ids = self.full_chain_ids[perm_indices]
        if hasattr(self, 'full_hotspot'):
            self.full_hotspot = self.full_hotspot[perm_indices]
        # if hasattr(self, 'full_plm_emb') and self.full_plm_emb is not None:
        #     self.full_plm_emb = self.full_plm_emb[perm_indices]
        
        # 2. Permute Lists
        perm_list = perm_indices.cpu().tolist()
        self.full_pdb_idx = [self.full_pdb_idx[i] for i in perm_list]
        self.full_origin_pdb_idx = [self.full_origin_pdb_idx[i] for i in perm_list]
        
        # 3. Permute 2D Matrices (Rows and Columns)
        # N_C_anchor, bond_matrix, bond_mask
        # matrix[i, j] -> matrix[perm[i], perm[j]]
        self.full_N_C_anchor = self.full_N_C_anchor[perm_indices][:, perm_indices]
        self.full_bond_matrix = self.full_bond_matrix[perm_indices][:, perm_indices]
        self.full_bond_mask = self.full_bond_mask[perm_indices][:, perm_indices]

    def build_nc_training_pairs(self, pos_prob):
        """
        基于当前 target 构造 N/C 功能节点的训练样本对（支持循环排列 Circular Permutation）：
        
        逻辑：
          - 遍历每条链，识别 N节点、C节点、Body残基。
          - 排除已参与真实 LINK 的功能节点。
          - 正样本 (Prob = pos_prob):
              - 随机选择 Body 中的切点 (i, j)，满足 j = i + 1。
              - 重排 Body 序列为: [j, j+1, ..., End, Start, ..., i]。
              - 链的新整体顺序: [N节点] + [新Body] + [C节点] + [Padding...]。
              - 这种排列下，N 锚定到 j (Body首)，C 锚定到 i (Body尾)，且 N-C 设为成键 (1)。
          - 负样本 (Prob = 1 - pos_prob):
              - 保持 Body 序列原序: [Start, ..., End]。
              - 链的新整体顺序: [N节点] + [Body] + [C节点] + [Padding...]。
              - N 锚定到 Start (Body首)，C 锚定到 End (Body尾)，且 N-C 设为断开 (0)。
          - 无论正负，N 总是锚定到重排后 Body 的第一个，C 锚定到最后一个。
        """
        if not self.N_C_add:
            return

        L = len(self.full_seq)
        if L == 0:
            return

        # if pos_prob == 0:
        #     return
        device = self.full_bond_matrix.device
        
        # 1. 识别真实 LINK，避免覆盖
        nc_link_chain_ids = set()
        if self.pdb and "links" in self.pdb and self.pdb["links"]:
            for link in self.pdb["links"]:
                idx1 = link.get("idx1") # (chain, res)
                idx2 = link.get("idx2")
                atom1 = (link.get("atom1") or "").upper()
                atom2 = (link.get("atom2") or "").upper()
                
                # Check endpoints
                for origin, atom in ((idx1, atom1), (idx2, atom2)):
                    if not origin or atom not in BACKBONE_ATOMS:
                        continue
                    # Check if this origin maps to a terminal node
                    if hasattr(self, "_origin_to_nter_idx") and origin in self._origin_to_nter_idx:
                        nc_link_chain_ids.add(origin[0])
                    if hasattr(self, "_origin_to_cter_idx") and origin in self._origin_to_cter_idx:
                        nc_link_chain_ids.add(origin[0])

        # 2. 建立数字链 ID 到原始链 ID 的映射（用于 nc_link_chain_ids 检查）
        # full_chain_ids 是数字编号 (1, 2, 3...)，而 nc_link_chain_ids 是原始链 ID ('A', 'B'...)
        chain_id_to_orig = {}
        for i, (orig_ch, _) in enumerate(self.full_origin_pdb_idx):
            num_ch = self.full_chain_ids[i].item()
            if num_ch not in chain_id_to_orig:
                chain_id_to_orig[num_ch] = orig_ch
        
        # 将 nc_link_chain_ids 转换为数字链 ID 集合
        nc_link_num_ids = set()
        for orig_ch in nc_link_chain_ids:
            for num_ch, mapped_orig in chain_id_to_orig.items():
                if mapped_orig == orig_ch:
                    nc_link_num_ids.add(num_ch)
                    break
        
        # 3. 准备全局重排索引列表
        all_new_indices = []
        # 标记哪些索引已经被处理进 all_new_indices，防止重复或遗漏
        processed_mask = torch.zeros(L, dtype=torch.bool, device=device)
        
        # 使用 full_chain_ids 获取所有链编号
        unique_chain_ids = torch.unique(self.full_chain_ids).cpu().tolist()
        rng = random.Random()

        for num_ch in unique_chain_ids:
            # 使用 torch.where 高效获取该链的所有索引
            chain_mask = (self.full_chain_ids == num_ch)
            chain_indices = torch.where(chain_mask)[0].cpu().tolist()
            if not chain_indices:
                continue

            # 如果该链的结构掩码全部为 False（即所有残基结构都固定），
            # 则该链在 N/C 训练中 **默认视为负样本**：
            #   - 仍然可以参与 N/C 训练对构建；
            #   - 但不允许被采样为正样本（不做 CP 重排）。
            # 这里通过 full_mask_str（False 表示结构固定，True 表示可扰动）
            # 判断“是否存在至少一个可扰动结构残基”；若不存在，则强制负样本。
            chain_str_mask = self.full_mask_str[chain_mask]
            chain_all_str_fixed = not bool(chain_str_mask.any())
            
            # 识别 N/C/Body
            head_mask = self.full_head_mask
            tail_mask = self.full_tail_mask
            
            n_idx = next((i for i in chain_indices if bool(head_mask[i])), None)
            c_idx = next((i for i in chain_indices if bool(tail_mask[i])), None)
            
            # 提取 Body (非 N/C 且非 padding)
            body_indices = [
                i for i in chain_indices
                if (not bool(head_mask[i]))
                and (not bool(tail_mask[i]))
                and i < len(self.full_origin_pdb_idx)
                and self.full_origin_pdb_idx[i] != ("?", "-1")
            ]
            
            # 提取 Padding (剩余部分，通常是 New_xxx)
            other_indices = [
                i for i in chain_indices 
                if i != n_idx and i != c_idx and i not in body_indices
            ]
            
            # 如果缺少关键节点或已被真实 LINK 占用，保持原样
            if n_idx is None or c_idx is None or len(body_indices) < 2 or (num_ch in nc_link_num_ids):
                all_new_indices.extend(chain_indices)
                processed_mask[chain_indices] = True
                continue

            # 决策正负样本
            # 若该链结构完全固定（chain_all_str_fixed=True），
            # 则强制视为负样本：不允许进入正样本（CP）分支。
            is_pos = (not chain_all_str_fixed) and (rng.random() < float(pos_prob))
            
            new_body_indices = list(body_indices)
            
            if is_pos:
                # 正样本：寻找序列相邻切点 (i, j)，即 rf_idx[i+1] == rf_idx[i] + 1
                candidates = []
                rf_idx = self.full_rf_idx
                for k in range(len(body_indices) - 1):
                    i_idx = body_indices[k]
                    j_idx = body_indices[k+1]
                    if int(rf_idx[j_idx].item()) == int(rf_idx[i_idx].item()) + 1:
                        candidates.append(k) # k 是 i 在 body_indices 中的下标
                
                if candidates:
                    # 随机选一个切点
                    cut_k = rng.choice(candidates) 
                    # 原序列: ... i(k), j(k+1) ...
                    # 新序列: j ... End, Start ... i
                    # split point is k+1
                    split_idx = cut_k + 1
                    new_body_indices = body_indices[split_idx:] + body_indices[:split_idx]
                else:
                    # 没找到连续片段，退化为负样本逻辑
                    is_pos = False
            
            # 构建该链的新顺序：[N] + [NewBody] + [C] + [Others]
            # 这样 N 紧邻 NewBody[0], C 紧邻 NewBody[-1]
            chain_new_order = [n_idx] + new_body_indices + [c_idx] + other_indices
            
            all_new_indices.extend(chain_new_order)
            processed_mask[chain_indices] = True
            
        # 3. 添加未处理的索引（如果有遗漏的）
        remaining_indices = torch.nonzero(~processed_mask).squeeze(-1).tolist()
        if remaining_indices:
            all_new_indices.extend(remaining_indices)

        # 4. 执行数据重排
        perm_indices = torch.tensor(all_new_indices, device=device, dtype=torch.long)
        self._permute_target_data(perm_indices)

        # 5. 重排后：统一设置锚定和成键
        # 此时 N_C_anchor 已经被重排（随着节点移动），我们需要清空并重新设置
        self.full_N_C_anchor.fill_(False)
        
        # 重新获取重排后的 mask
        head_mask = self.full_head_mask
        tail_mask = self.full_tail_mask
        rf_idx = self.full_rf_idx
        
        # 遍历每条链（使用 full_chain_ids）
        unique_chain_ids = torch.unique(self.full_chain_ids).cpu().tolist()
        
        for num_ch in unique_chain_ids:
            # 跳过有真实连接的链
            if num_ch in nc_link_num_ids:
                continue

            # 使用 torch.where 高效获取该链在新数据中的索引
            chain_mask = (self.full_chain_ids == num_ch)
            indices = torch.where(chain_mask)[0].cpu().tolist()
            if not indices:
                continue
            
            n_idx = next((i for i in indices if bool(head_mask[i])), None)
            c_idx = next((i for i in indices if bool(tail_mask[i])), None)
            
            # Body indices (excluding padding)
            body_indices = [
                i for i in indices
                if (not bool(head_mask[i]))
                and (not bool(tail_mask[i]))
                and i < len(self.full_origin_pdb_idx)
                and self.full_origin_pdb_idx[i] != ("?", "-1")
            ]
            
            if n_idx is None or c_idx is None or not body_indices:
                continue

            # 统一锚定逻辑：
            # N 锚定到 Body 的第一个
            # C 锚定到 Body 的最后一个
            first_body = body_indices[0]
            last_body = body_indices[-1]
            
            self.full_N_C_anchor[n_idx, first_body, 0] = True
            self.full_N_C_anchor[first_body, n_idx, 0] = True
            
            self.full_N_C_anchor[c_idx, last_body, 1] = True
            self.full_N_C_anchor[last_body, c_idx, 1] = True

            # 特征同步：功能节点复制其锚定对象的特征
            for node, anchor in [(n_idx, first_body), (c_idx, last_body)]:
                self.full_seq[node] = self.full_seq[anchor]
                self.full_mask_seq[node] = self.full_mask_seq[anchor]
                self.full_mask_str[node] = self.full_mask_str[anchor]
                self.full_xyz[node] = self.full_xyz[anchor]
                self.full_alpha[node] = self.full_alpha[anchor]
                self.full_alpha_alt[node] = self.full_alpha_alt[anchor]
                self.full_alpha_tor_mask[node] = self.full_alpha_tor_mask[anchor]
                self.full_rf_idx[node] = self.full_rf_idx[anchor]
                self.full_origin_pdb_idx[node] = self.full_origin_pdb_idx[anchor]

            # 坐标偏移更新
            self.full_xyz = update_nc_node_coordinates(self.full_xyz, self.full_N_C_anchor, head_mask, tail_mask)
            
            # 判断成键逻辑：
            # 如果是正样本（循环排列），则 last_body(旧i) 和 first_body(旧j) 在原序列中是相邻的
            # 即 rf_idx[first_body] == rf_idx[last_body] + 1
            # 注意：由于 rf_idx 也被重排了，所以 rf_idx[k] 仍然是残基 k 在原始 PDB 中的编号
            is_adjacent = (int(rf_idx[first_body].item()) == int(rf_idx[last_body].item()) + 1)
            
            if is_adjacent:
                self.full_bond_matrix[c_idx, n_idx] = 1
                self.full_bond_matrix[n_idx, c_idx] = 1
            else:
                self.full_bond_matrix[c_idx, n_idx] = 0
                self.full_bond_matrix[n_idx, c_idx] = 0

    def parse_bond_condition(self):

        # default perturbate all bond including in the res that contigs specify
        # the all pdb index below are remnamed, depend on the contigs, default chain index is ABCDEF... 
        # fix bond in chain A, B, A-B: ['A|A','B|B','A|B']
        # fix bond with in A100-200,B100-200 provied by PDB: ['A100-A200|B100-B200']
        # fix self-defined bond, 0 or 1: ['A100-A100|A200-A200:1:FIX'], mask is False, means the str is fixed
        # PNA(Partial Noise Addition) self-defined bond, 0 or 1: ['A100-A100|A200-A200:0:PNA'] mask is True, means the str is not fixed
        # fix start or end part: ['Astart-Astart|Aend-Aend:1:FIX'] or ['Astart-A2|A10-A10:1:FIX','B200-Bend|A100-A100:0:FIX']
        
        L = len(self.full_seq)
        bond_matrix = torch.zeros((L, L), dtype=torch.long)
        bond_mask = torch.ones((L, L), dtype=torch.bool)  # True means changeable
        #bond_mask = bond_mask ^ torch.eye(L, dtype=torch.bool) # diagonal is fixed, means no change
        # 全局的“可变/固定”掩码：后续所有 bond_condition 规则都会在这个掩码上累积效果，
        # 最终返回时使用 bond_mask & fixed_mask。
        fixed_mask = torch.ones((L, L), dtype=torch.bool)
        # 记录所有在「不带显式 value（仅 FIX 原始矩阵）」规则中出现过的残基索引，
        # 用于在最后根据行/列是否被锁死，决定是否恢复这些残基自身对角线 bond_mask[i, i]。
        keep_diag_true = torch.zeros(L, dtype=torch.bool)

        # If pdb input, check if links from pdb are preserved
        if self.pdb and 'links' in self.pdb and self.pdb['links']:
            N_C_add_enabled = self.N_C_add

            def resolve_endpoint(pdb_origin, atom_name):
                atom_up = (atom_name or '').upper()
                if not N_C_add_enabled:
                    # legacy behavior
                    if pdb_origin in self.full_origin_pdb_idx:
                        return self.full_origin_pdb_idx.index(pdb_origin)
                    return None

                # Prefer terminal node for first/last origins if backbone atom
                if hasattr(self, '_origin_to_nter_idx') and pdb_origin in getattr(self, '_origin_to_nter_idx') and atom_up in BACKBONE_ATOMS:
                    return self._origin_to_nter_idx[pdb_origin]
                if hasattr(self, '_origin_to_cter_idx') and pdb_origin in getattr(self, '_origin_to_cter_idx') and atom_up in BACKBONE_ATOMS:
                    return self._origin_to_cter_idx[pdb_origin]
                # Otherwise map to body residue index
                if hasattr(self, '_origin_to_body_idx') and pdb_origin in getattr(self, '_origin_to_body_idx'):
                    return self._origin_to_body_idx[pdb_origin]
                # Fallback to first occurrence
                if pdb_origin in self.full_origin_pdb_idx:
                    return self.full_origin_pdb_idx.index(pdb_origin)
                return None

            for link in self.pdb['links']:
                res1_pdb_idx = link['idx1']
                res2_pdb_idx = link['idx2']
                atom1 = link.get('atom1', None)
                atom2 = link.get('atom2', None)

                idx1 = resolve_endpoint(res1_pdb_idx, atom1)
                idx2 = resolve_endpoint(res2_pdb_idx, atom2)
                if idx1 is not None and idx2 is not None:
                    bond_matrix[idx1, idx2] = 1
                    bond_matrix[idx2, idx1] = 1

        if self.design_conf.bond_condition is None:
            return bond_matrix, bond_mask

        for bond_contig in self.design_conf.bond_condition:
            parts = bond_contig.split(':')
            res_parts = parts[0].split('|')
            
            res1_spec, res2_spec = res_parts[0], res_parts[1]

            def get_indices(spec):
                if len(spec) == 1: # Chain, e.g. 'A'
                    return [i for i, p_idx in enumerate(self.full_pdb_idx) if p_idx[0] == spec]
                
                range_parts = spec.split('-')

                def get_res_idx(res_spec):
                    # Support new style "A/2" as well as legacy "A2"
                    if '/' in res_spec:
                        chain_id, token = res_spec.split('/', 1)
                    else:
                        chain_id, token = res_spec[0], res_spec[1:]

                    if token == 'start':
                        chain_indices = [i for i, p_idx in enumerate(self.full_pdb_idx) if p_idx[0] == chain_id]
                        return min(chain_indices) if chain_indices else -1
                    elif token == 'end':
                        chain_indices = [i for i, p_idx in enumerate(self.full_pdb_idx) if p_idx[0] == chain_id]
                        return max(chain_indices) if chain_indices else -1
                    else:
                        res_num = int(token)
                        try:
                            return self.full_pdb_idx.index((chain_id, res_num))
                        except ValueError:
                            return -1

                start_idx = get_res_idx(range_parts[0])
                end_idx = get_res_idx(range_parts[1])
                
                if start_idx == -1 or end_idx == -1:
                    raise ValueError(f"Invalid residue specification for bond: {spec}")

                return list(range(start_idx, end_idx + 1))

            indices1 = get_indices(res1_spec)
            indices2 = get_indices(res2_spec)

            value = None  # Default to None, meaning no specific value set
            mask_value = False  # Default to FIX
                
            if len(parts) > 1:
                value = int(parts[1])
            if len(parts) > 2:
                if parts[2] == 'PNA' and self.design_conf.partial_t is None:
                    raise ValueError("Partial Noise Addition (PNA) requires partial_t to be set.")
                mask_value = (parts[2] == 'PNA')

            # 对于不带显式 value 的情况（例如 'A|A', 'B|B', 'A|B', 'A100-A200|B100-B200'），
            # 用户希望只是 FIX 原有的 bond_matrix，而这些残基对应的自连接项 (i,i)
            # 在 bond_mask 中仍然保持为 True（不被误锁死）。
            #
            # 因此这里记录所有受此类规则影响到的残基索引，稍后在行/列扩展之后，
            # 再把对应的对角线 bond_mask[i, i] / fixed_mask[i, i] 视情况恢复为 True。
            if value is None:
                if len(indices1) > 0:
                    keep_diag_true[indices1] = True
                if len(indices2) > 0:
                    keep_diag_true[indices2] = True

            # 在全局 fixed_mask 上累积当前规则的效果
            for i in indices1:
                for j in indices2:
                    # if i == j: continue
                    if value:
                        bond_matrix[i, j] = value
                        bond_matrix[j, i] = value
                    fixed_mask[i, j] = mask_value
                    fixed_mask[j, i] = mask_value

        # If any fixed entry equals 1 in a row/column, expand the fixed mask to the
        # whole row and column to respect the doubly-stochastic constraint.
        # Only entries explicitly fixed (mask False) and equal to 1 trigger expansion.
        fixed_one = (bond_matrix == 1) & (~fixed_mask)
        if fixed_one.any():
            rows_to_fix = fixed_one.any(dim=1)
            cols_to_fix = fixed_one.any(dim=0)
            if rows_to_fix.any():
                bond_mask[rows_to_fix, :] = False
            if cols_to_fix.any():
                bond_mask[:, cols_to_fix] = False

            # 恢复「仅 FIX 原有矩阵（不带 value）」规则中涉及的残基的自连接对角线可变性，
            # 但前提是这些残基所在的整行 / 整列没有因为某个固定为 1 的键而被锁死。
            # 一旦某行 / 某列已经被 fixed_one 触发为“整行/整列固定”，则对应残基的对角线
            # 也应保持为 False，不再覆盖为 True。
            if keep_diag_true.any():
                # 行或列被锁死的残基索引
                locked_rows = rows_to_fix
                locked_cols = cols_to_fix
                # 仅对既在 keep_diag_true 中、又不在锁死行/列中的残基恢复对角线 True
                diag_mask = keep_diag_true & (~locked_rows) & (~locked_cols)
                if diag_mask.any():
                    diag_idx = torch.where(diag_mask)[0]
                    bond_mask[diag_idx, diag_idx] = True
                    fixed_mask[diag_idx, diag_idx] = True
        else:
            # 没有任何 fixed_one：说明没有因为“固定为 1”的键触发行/列锁死，
            # 可以放心地为所有 keep_diag_true 的残基恢复对角线 True。
            if keep_diag_true.any():
                diag_idx = torch.where(keep_diag_true)[0]
                bond_mask[diag_idx, diag_idx] = True
                fixed_mask[diag_idx, diag_idx] = True

        return bond_matrix, bond_mask & fixed_mask

    def parse_contigs(self, contigs, length=None, chain_offset=200):
        """
        Parse a contig from the pdb file.

        Args:
            - contigs: list of contigs, each contig is a dict contain the following keys
                - seq: sequence of the contig
                - xyz: coordinates of the contig
                - rf_index: residue index of the contig
            - bond_mask: not used
            - length: specify the length of each chain
            - chain_offset: offset for the next chain's residue index

        Outputs:
            - maskseq and maskstr: mask for the contig
            - new pdb index
        """
        self.chain_list = []
        for chain in contigs:
            contig_list = []
            for i, contig in enumerate(chain):
                contig_list.append(self.single_parse_contig(contig))
            self.chain_list.append(contig_list)
        
        # specify the length of the contig
        # example [10,30,-1], while -1 represent for free length
        if length is None:
            length = [-1 for chain in self.chain_list]
 
        full_seq = []
        full_xyz = []
        full_mask_str = []
        full_mask_seq = []
        full_origin_pdb_idx = [] # genetated pdb index is ('?','-1')
        full_pdb_idx = []
        full_rf_idx = []
        full_alpha = []
        full_alpha_alt = []
        full_alpha_tor_mask = []
        full_head_mask = []  # True at NTER positions
        full_tail_mask = []  # True at CTER positions
        full_chain_ids = []

        # Prepare terminal/body origin mappings
        self.nter_indices = []
        self.cter_indices = []
        self._origin_to_body_idx = {}
        self._origin_to_nter_idx = {}
        self._origin_to_cter_idx = {}
        N_C_add_enabled = self.N_C_add
        
        current_offset = 0
        current_chain_id = 1 
        for i, chain in enumerate(self.chain_list):
            part_lengths = sample_parts([ran['length_range'] for ran in chain], length[i])
            if part_lengths is None:
                raise ValueError(f"Cannot satisfy length constraints for chain {i}")

            chain_seq = []
            chain_xyz = []
            chain_alpha = []
            chain_alpha_alt = []
            chain_alpha_tor_mask = []
            chain_rf_idx = []
            chain_origin_list = []
            chain_mask_str_list = []
            chain_mask_seq_list = []
            chain_start_idx = None
            # Chain-local cursor to ensure rf_idx increases across multiple New_/non-PDB segments
            # within the same chain (especially important in inference mode).
            chain_pos_cursor = 0

            for j, contig in enumerate(chain):
                part_len = part_lengths[j]
                if len(contig['seq']) > 0:
                    chain_seq.append(contig['seq'].clone())
                else:
                    chain_seq.append(torch.full((part_len,), 20, dtype=torch.long))

                if len(contig['xyz']) > 0:
                    chain_xyz.append(contig['xyz'].clone())
                elif contig.get('is_new', False):
                    # For newly created segments (New_), initialize ideal backbone heavy atoms (N, CA, C)
                    init_block = torch.full((part_len, 14, 3), float('nan'), dtype=torch.float32)
                    # N, CA, C from INIT_CRDS (indices 0,1,2)
                    init_block[:, 0, :] = che.INIT_CRDS[0]
                    init_block[:, 1, :] = che.INIT_CRDS[1]
                    init_block[:, 2, :] = che.INIT_CRDS[2]
                    chain_xyz.append(init_block)
                else:
                    chain_xyz.append(torch.full((part_len, 14, 3), np.nan, dtype=torch.float32))
                    
                if len(contig['alpha']) > 0:
                    chain_alpha.append(contig['alpha'].clone())
                else:
                    chain_alpha.append(torch.full((part_len,10,2), 0.0, dtype=torch.float32))

                if len(contig['alpha_alt']) > 0:
                    chain_alpha_alt.append(contig['alpha_alt'].clone())
                else:
                    chain_alpha_alt.append(torch.full((part_len,10,2), 0.0, dtype=torch.float32))

                if len(contig['alpha_tor_mask']) > 0:
                    chain_alpha_tor_mask.append(contig['alpha_tor_mask'].float().clone())
                else:
                    chain_alpha_tor_mask.append(torch.full((part_len,10), 0.0, dtype=torch.float32))

                if len(contig['origin_pdb_idx']) > 0:
                    chain_origin_list += contig['origin_pdb_idx']
                else:
                    chain_origin_list += [('?', '-1')] * part_len # new generated part in old pdb index is ('?','-1')
                
                if len(contig['rf_index']) > 0:
                    if chain_start_idx is None:
                        chain_start_idx = contig['rf_index'][0]
                    # In inference/design sampling, we want a stable per-chain positional index
                    # (and New_ segments must advance). Use chain_pos_cursor-based numbering.
                    if getattr(self, "inference", False):
                        chain_rf_idx.append(
                            torch.arange(
                                current_offset + chain_pos_cursor,
                                current_offset + chain_pos_cursor + part_len,
                                dtype=torch.long,
                            )
                        )
                    else:
                        chain_rf_idx.append(contig['rf_index'] + current_offset - chain_start_idx)
                else:
                    chain_rf_idx.append(
                        torch.arange(
                            current_offset + chain_pos_cursor,
                            current_offset + chain_pos_cursor + part_len,
                            dtype=torch.long,
                        )
                    )
                chain_pos_cursor += part_len
                                                
                chain_mask_str_list += [contig['mask_str_value']] * part_len   # True means changeable
                chain_mask_seq_list += [contig['mask_seq_value']] * part_len

            chain_seq = torch.cat(chain_seq)
            chain_xyz = torch.cat(chain_xyz)
            chain_alpha = torch.cat(chain_alpha)
            chain_alpha_alt = torch.cat(chain_alpha_alt)
            chain_alpha_tor_mask = torch.cat(chain_alpha_tor_mask)
            chain_rf_idx = torch.cat(chain_rf_idx)

            # Preserve original contig order, optionally adding N/C terminal clones
            chain_id = self.chain_order[i]

            # 训练阶段：只把有真实 PDB 起源的残基视为“天然 body”（origin != ('?','-1')）
            # 推理阶段（例如 cyclize_from_pdb）：New_ 也是要真实生成的主体，应视作 body；
            #   此时我们直接把整条链（后续若有专门用于 padding 的片段，可再细化）都当作 body。
            if getattr(self, "inference", False):
                body_indices = list(range(len(chain_origin_list)))
            else:
                is_body_mask = [origin != ('?', '-1') for origin in chain_origin_list]
                body_indices = [k for k, v in enumerate(is_body_mask) if v]

            add_terminals = N_C_add_enabled and len(body_indices) > 0

            # Start with original order
            chain_seq_new = chain_seq
            chain_xyz_new = chain_xyz
            chain_alpha_new = chain_alpha
            chain_alpha_alt_new = chain_alpha_alt
            chain_alpha_mask_new = chain_alpha_tor_mask
            chain_rf_idx_new = chain_rf_idx
            chain_origin_new = list(chain_origin_list)
            chain_mask_str_new = list(chain_mask_str_list)
            chain_mask_seq_new = list(chain_mask_seq_list)

            nter_local_idx = None
            cter_local_idx = None

            if add_terminals:
                first_idx = body_indices[0]
                last_idx = body_indices[-1]
                # NTER clone from first body residue (prepend)
                chain_seq_new = torch.cat([chain_seq[first_idx:first_idx+1], chain_seq_new])
                chain_xyz_new = torch.cat([chain_xyz[first_idx:first_idx+1], chain_xyz_new])
                chain_alpha_new = torch.cat([chain_alpha[first_idx:first_idx+1], chain_alpha_new])
                chain_alpha_alt_new = torch.cat([chain_alpha_alt[first_idx:first_idx+1], chain_alpha_alt_new])
                chain_alpha_mask_new = torch.cat([chain_alpha_tor_mask[first_idx:first_idx+1], chain_alpha_mask_new])
                chain_rf_idx_new = torch.cat([chain_rf_idx[first_idx:first_idx+1], chain_rf_idx_new])
                chain_origin_new = [chain_origin_list[first_idx]] + chain_origin_new
                chain_mask_str_new = [chain_mask_str_list[first_idx]] + chain_mask_str_new
                chain_mask_seq_new = [chain_mask_seq_list[first_idx]] + chain_mask_seq_new
                nter_local_idx = 0
            
                # CTER：训练 / 推理区分插入位置
                if getattr(self, "inference", False):
                    # 推理：克隆最后一个 body 残基，直接 append 到链末尾，
                    # 确保 C 功能节点总是处于整条设计链的最右端。
                    cter_insert_pos = chain_seq_new.shape[0]
                    chain_seq_new = torch.cat([
                        chain_seq_new,
                        chain_seq[last_idx:last_idx+1],
                    ])
                    chain_xyz_new = torch.cat([
                        chain_xyz_new,
                        chain_xyz[last_idx:last_idx+1],
                    ])
                    v_CA_C = chain_xyz_new[cter_insert_pos, 2, :] - chain_xyz_new[cter_insert_pos, 1, :]
                    chain_xyz_new[cter_insert_pos] += v_CA_C
                    v_N_CA = chain_xyz_new[nter_local_idx, 0, :] - chain_xyz_new[nter_local_idx, 1, :]
                    chain_xyz_new[nter_local_idx] += v_N_CA

                    chain_alpha_new = torch.cat([
                        chain_alpha_new,
                        chain_alpha[last_idx:last_idx+1],
                    ])
                    chain_alpha_alt_new = torch.cat([
                        chain_alpha_alt_new,
                        chain_alpha_alt[last_idx:last_idx+1],
                    ])
                    chain_alpha_mask_new = torch.cat([
                        chain_alpha_mask_new,
                        chain_alpha_tor_mask[last_idx:last_idx+1],
                    ])
                    chain_rf_idx_new = torch.cat([
                        chain_rf_idx_new,
                        chain_rf_idx[last_idx:last_idx+1],
                    ])
                    chain_origin_new = chain_origin_new + [chain_origin_list[last_idx]]
                    chain_mask_str_new = chain_mask_str_new + [chain_mask_str_list[last_idx]]
                    chain_mask_seq_new = chain_mask_seq_new + [chain_mask_seq_list[last_idx]]
                    cter_local_idx = cter_insert_pos
                else:
                    # 训练：恢复原始行为——CTER 插在“最后一个 body 残基后面一位”，在 padding / New_ 之前。
                    # After adding NTER, chain_seq_new = [NTER] + [body_residues] + [padding]
                    # last_idx is the index of last body residue in original chain_seq
                    # In chain_seq_new, last body residue is at position (last_idx + 1)
                    cter_insert_pos = last_idx + 2  # +1 for NTER prepended, +1 to insert after last body residue
                    chain_seq_new = torch.cat([
                        chain_seq_new[:cter_insert_pos],
                        chain_seq[last_idx:last_idx+1],
                        chain_seq_new[cter_insert_pos:]
                    ])
                    chain_xyz_new = torch.cat([
                        chain_xyz_new[:cter_insert_pos],
                        chain_xyz[last_idx:last_idx+1],
                        chain_xyz_new[cter_insert_pos:]
                    ])
                    v_CA_C = chain_xyz_new[cter_insert_pos, 2, :] - chain_xyz_new[cter_insert_pos, 1, :]
                    chain_xyz_new[cter_insert_pos] += v_CA_C
                    v_N_CA = chain_xyz_new[first_idx, 0, :] - chain_xyz_new[first_idx, 1, :]
                    chain_xyz_new[first_idx] += v_N_CA

                    chain_alpha_new = torch.cat([
                        chain_alpha_new[:cter_insert_pos],
                        chain_alpha[last_idx:last_idx+1],
                        chain_alpha_new[cter_insert_pos:]
                    ])
                    chain_alpha_alt_new = torch.cat([
                        chain_alpha_alt_new[:cter_insert_pos],
                        chain_alpha_alt[last_idx:last_idx+1],
                        chain_alpha_alt_new[cter_insert_pos:]
                    ])
                    chain_alpha_mask_new = torch.cat([
                        chain_alpha_mask_new[:cter_insert_pos],
                        chain_alpha_tor_mask[last_idx:last_idx+1],
                        chain_alpha_mask_new[cter_insert_pos:]
                    ])
                    chain_rf_idx_new = torch.cat([
                        chain_rf_idx_new[:cter_insert_pos],
                        chain_rf_idx[last_idx:last_idx+1],
                        chain_rf_idx_new[cter_insert_pos:]
                    ])
                    chain_origin_new = (
                        chain_origin_new[:cter_insert_pos] +
                        [chain_origin_list[last_idx]] +
                        chain_origin_new[cter_insert_pos:]
                    )
                    chain_mask_str_new = (
                        chain_mask_str_new[:cter_insert_pos] +
                        [chain_mask_str_list[last_idx]] +
                        chain_mask_str_new[cter_insert_pos:]
                    )
                    chain_mask_seq_new = (
                        chain_mask_seq_new[:cter_insert_pos] +
                        [chain_mask_seq_list[last_idx]] +
                        chain_mask_seq_new[cter_insert_pos:]
                    )
                    cter_local_idx = cter_insert_pos

            # Note: we no longer "force-renumber" rf_idx here. Instead, we ensure rf_idx is
            # constructed monotonically within the chain via `chain_pos_cursor` above.

            # Record global indices and mappings
            chain_global_start = sum(len(t) for t in full_seq)
            if add_terminals:
                nter_global = chain_global_start + (nter_local_idx or 0)
                cter_global = chain_global_start + (cter_local_idx or 0)
                self.nter_indices.append(nter_global)
                self.cter_indices.append(cter_global)
                # Map terminal origins to terminal indices
                self._origin_to_nter_idx[chain_origin_new[nter_local_idx]] = nter_global
                self._origin_to_cter_idx[chain_origin_new[cter_local_idx]] = cter_global

            # Map body origins to body indices (first occurrence only)
            # Body indices in new chain start at (1 if add_terminals else 0)
            body_start_local = 1 if add_terminals else 0
            for j, origin in enumerate(chain_origin_new):
                if origin == ('?', '-1'):
                    continue
                # Prefer mapping to body positions, not terminals
                if add_terminals and (j == nter_local_idx or j == cter_local_idx):
                    continue
                global_j = chain_global_start + j
                if origin not in self._origin_to_body_idx:
                    self._origin_to_body_idx[origin] = global_j

            # Center NEW segments to FIX centroid using CA atoms
            try:
                new_mask_bool = torch.tensor([orig == ('?', '-1') for orig in chain_origin_new], dtype=torch.bool)
                fix_mask_bool = torch.tensor([not m for m in chain_mask_str_new], dtype=torch.bool)
                if fix_mask_bool.any() and new_mask_bool.any():
                    fix_ca = chain_xyz_new[fix_mask_bool, 1, :]
                    valid_fix = ~torch.isnan(fix_ca).any(dim=-1)
                    if valid_fix.any():
                        fix_center = fix_ca[valid_fix].mean(dim=0)
                        new_ca = chain_xyz_new[new_mask_bool, 1, :]
                        valid_new = ~torch.isnan(new_ca).any(dim=-1)
                        if valid_new.any():
                            new_center = new_ca[valid_new].mean(dim=0)
                            delta = fix_center - new_center
                            new_xyz = chain_xyz_new[new_mask_bool]
                            valid_xyz = ~torch.isnan(new_xyz)
                            chain_xyz_new[new_mask_bool] = torch.where(valid_xyz, new_xyz + delta, new_xyz)
            except Exception:
                pass

            # Build per-chain head/tail masks
            chain_head_mask = torch.zeros(chain_seq_new.shape[0], dtype=torch.bool)
            chain_tail_mask = torch.zeros(chain_seq_new.shape[0], dtype=torch.bool)
            chain_num_id = torch.full((chain_seq_new.shape[0],), current_chain_id, dtype=torch.long)
            current_chain_id += 1
            if add_terminals:
                chain_head_mask[nter_local_idx] = True
                chain_tail_mask[cter_local_idx] = True
            full_head_mask.append(chain_head_mask)
            full_tail_mask.append(chain_tail_mask)

            # Append to full arrays
            full_seq.append(chain_seq_new)
            full_xyz.append(chain_xyz_new)
            full_alpha.append(chain_alpha_new)
            full_alpha_alt.append(chain_alpha_alt_new)
            full_alpha_tor_mask.append(chain_alpha_mask_new)
            full_rf_idx.append(chain_rf_idx_new)
            full_origin_pdb_idx += chain_origin_new
            full_mask_str += chain_mask_str_new
            full_mask_seq += chain_mask_seq_new
            full_chain_ids.append(chain_num_id)

            # Renumber designed pdb_idx per chain after reordering
            if add_terminals:
                pdb_idx = [(chain_id, k + 1) for k in range(len(chain_seq_new) - 2)]
                pdb_idx = [pdb_idx[0]] + pdb_idx + [pdb_idx[-1]]
            else:
                pdb_idx = [(chain_id, k + 1) for k in range(len(chain_seq_new))]
            full_pdb_idx.append(pdb_idx)
            # Advance global offset for the next chain.
            # - Training: keep legacy progression (based on PDB-like rf_idx ranges).
            # - Inference: use chain length cursor to avoid double-counting current_offset.
            if getattr(self, "inference", False):
                current_offset += chain_offset + int(chain_pos_cursor)
            else:
                current_offset += chain_offset
                # keep legacy offset progression
                current_offset += chain_rf_idx_new[-1].item() 
        
        full_rf_idx = torch.cat(full_rf_idx)
        full_rf_idx = full_rf_idx - full_rf_idx.min()  # make sure rf_idx starts from 0

        # Optionally build full PLM embeddings aligned with full_origin_pdb_idx
        # full_plm_emb = None
        # if getattr(self, "pdb", None) is not None and "plm_emb" in self.pdb:
        #     plm_arr = self.pdb["plm_emb"]
        #     if plm_arr is not None:
        #         plm_arr = np.asarray(plm_arr, dtype=np.float32)
        #         if plm_arr.ndim == 2 and len(self.pdb["pdb_idx"]) == plm_arr.shape[0]:
        #             D = plm_arr.shape[1]
        #             origin_to_idx = {p: i for i, p in enumerate(self.pdb["pdb_idx"])}
        #             buf = []
        #             for origin in full_origin_pdb_idx:
        #                 if origin in origin_to_idx:
        #                     buf.append(plm_arr[origin_to_idx[origin]])
        #                 else:
        #                     buf.append(np.zeros((D,), dtype=np.float32))
        #             full_plm_emb = torch.from_numpy(np.stack(buf, axis=0))

        # Expose head/tail masks (1D over full length)
        self.full_head_mask = torch.cat(full_head_mask) if len(full_head_mask) > 0 else torch.zeros(0, dtype=torch.bool)
        self.full_tail_mask = torch.cat(full_tail_mask) if len(full_tail_mask) > 0 else torch.zeros(0, dtype=torch.bool)
        return (
            torch.cat(full_seq),
            torch.cat(full_xyz).nan_to_num(0.0),
            full_rf_idx,
            torch.tensor(full_mask_str),
            torch.tensor(full_mask_seq),
            [item for sublist in full_pdb_idx for item in sublist],
            full_origin_pdb_idx,
            torch.cat(full_alpha),
            torch.cat(full_alpha_alt),  # [L,10,2] for alpha torsions, [L,10,2] for alpha torsion alt
            torch.cat(full_alpha_tor_mask),  # [L,10,2] for alpha torsions, [L,10,1] for alpha torsion mask
            torch.cat(full_chain_ids),
            #full_plm_emb,
        )

    def single_parse_contig(self, contig, inference=False):
        """
        Outputs:
            - length range: [min,max]
            - seq: sequence of the contig
            - xyz: xyz coordinates of the contig
            - maskseq and maskstr: mask for the contig
        """

        # select_range:fix_type:fix_type
        # fix_type: str_FIX, seq_FIX, seq_PNA, str_PNA, seq_DNV, str_DNV
        # FIX, DNV(De NoVo), PNA(Partial Noise Addition), PNA need partial_t, DNV and PNA cannot coexist
        # range chain: ['Chain_A:seq_FIX:str_FIX'','Chain_B:seq_FIX:str_DNV']
        # range res: 'A100-A200:seq_FIX:str_DNV','B100-B200:seq_PNA:str_PNA'
        # range 10-20, default and only seq_DNV, str_DNV
        # insert self-defined 'AGGGKI:seq_FIX:str_DNV', self-defined seq can only select str_DNV
        # example: [['Chain_A:seq_FIX:str_FIX'],
        #           ['New_30-40','B100-B120:seq_FIX:str_DNV','New_10-20',
        #           'B160-B200:seq_DNV:str_FIX','10-20','AGISHK:seq_FIX:str_DNV']]
        # example: [['Chain_A:seq_FIX:str_FIX'],
        #           ['B100-B120:seq_FIX:str_PNA'
        #          ]

        seq = []
        xyz = []
        alpha = []
        alpha_alt = []
        alpha_tor_mask = []
        mask_str_value = True  # default is True, means the str is not fixed
        mask_seq_value = True # default is True, means the seq is not fixed
        length_range = [0, 0] # min length, max length
        rf_index = []
        origin_pdb_idx = []
        index = [] # the index in the pdb_parsed list
        # Keep original string (for error messages / parsing decisions)
        contig_raw = contig
        contig = contig.split(":")

        # Decide whether this contig *requires* an input structure (PDB/CIF).
        # Only contigs that reference existing residues/chains need self.pdb.
        sel = contig[0]
        needs_pdb = False
        if "Chain_" in sel:
            needs_pdb = True
        elif "New_" in sel:
            needs_pdb = False
        elif sel.isalpha():
            needs_pdb = False
        elif "-" in sel:
            # Ranges like "A/100-A/200" require pdb to map indices.
            needs_pdb = True
        else:
            needs_pdb = True

        if needs_pdb and self.pdb is None:
            raise ValueError(f"No pdb/cif provided, cannot parse contig that references structure: {contig_raw}")

        ##########################################
        # parse the contig[0], namely select range
        ##########################################
        # if it is a chain, example: 'Chain_A'
        if "Chain_" in contig[0]:
            chain_id = contig[0].split("_")[1]  # e.g. 'A'
            if chain_id not in self.pdb["chains"]:
                raise ValueError(f"Chain {chain_id} not found in the pdb file")
            index = [i for i in range(len(self.pdb["pdb_idx"])) if self.pdb["pdb_idx"][i][0] == chain_id]
            length_range[0] = len(index)
            length_range[1] = len(index)
        # New_10-20
        elif "New_" in contig[0]:
            if self.design_conf.partial_t is not None and inference:
                raise ValueError(f"Partial t is not supported for contig {contig[0]}")
            if (len(contig) > 1 and contig[1].split("_")[1] != "DNV") or (len(contig) > 2 and contig[2].split("_")[1] != "DNV"):
                raise ValueError(f"Only DNV is supported for contig {contig[0]}")
            parts = contig[0].strip('New_').split("-")
            length_range[0] = int(parts[0])
            length_range[1] = int(parts[1])
            # mark this contig as newly generated for downstream ideal coord filling
            is_new = True
        # if it is a range residue, example: 'A100-A200' or '7100-7200' while '7' is chain id or 'A-10-B-20' -10 to -20
        elif "-" in contig[0]: #and contig[0][0] in self.pdb["chains"] and contig[0].split("-")[1][0] in self.pdb["chains"]:        
            # if it is a range, example: 'A100-A200'
            parts = self.parse_residue_range(contig[0])
            part1 = (parts['start_chain'], parts['start_res'])  # e.g. ('A', '100')
            part2 = (parts['end_chain'], parts['end_res'])
            if part1[0] not in self.pdb["chains"] or part2[0] not in self.pdb["chains"]:
                raise ValueError(f"Chain {part1[0]} or {part2[0]} not found in the {self.pdb['pdb_id']} pdb file")
            # check if the residue is in the pdb file
            if part1 not in self.pdb["pdb_idx"] or part2 not in self.pdb["pdb_idx"]:
                raise ValueError(f"Residue {part1} or {part2} not found in the {self.pdb['pdb_id']} pdb file")
            index1 = self.pdb["pdb_idx"].index(part1)
            index2 = self.pdb["pdb_idx"].index(part2)
            if index1 <= index2:                   
                index = [i for i in range(index1, index2 + 1)]
            else:
                raise ValueError(f"Residue {part1} is after {part2} in the pdb file")
            length_range[0] = len(index)
            length_range[1] = len(index)
        # e.x. AGISHK
        elif contig[0].isalpha():
            if self.design_conf.partial_t is not None:
                raise ValueError(f"Partial t is not supported for contig {contig[0]}")
            if contig[1] != "str_DNV" and contig[2] != "str_DNV":
                raise ValueError(f"Only str_DNV is supported for contig {contig[0]}")
            seq = [one_aa2num[c] for c in contig[0]] # interge seq
            seq = torch.tensor(seq)
            length_range[0] = len(seq)
            length_range[1] = len(seq)
        else:
            raise ValueError(f"Invalid contig format for range {contig[0]}")
        
        ##########################################
        # parse the contig[1] and contig[2], namely parse the type of fixed contig
        ########################################## 
        mask_type = {"PNA":True, "FIX":False, "DNV":True} 
        # parse_alpha = 0
        if len(contig) > 2:
            for con in [contig[1],contig[2]]:
                data_type = con.split("_")[0]
                process_type = con.split("_")[1]
                if process_type == "PNA" and self.design_conf.partial_t is None:
                    raise ValueError(f"Partial t is null , and not supported for contig {con}")
                # if process_type == "FIX":
                #     parse_alpha += 1
                if process_type == "FIX" or process_type == "PNA":
                    # if it is a fixed type, example: 'Chain_A:seq_FIX:str_FIX'
                    if data_type == "seq":
                        if len(index) > 0 and self.pdb is not None and "seq" in self.pdb:
                            seq = self.pdb["seq"][index]
                        else:
                            # Allow seq_FIX on explicit-seq contigs (seq already set).
                            # Disallow seq_FIX when we have neither pdb indices nor explicit seq.
                            if len(seq) == 0 and not torch.is_tensor(seq):
                                raise ValueError(
                                    f"seq_{process_type} requires an explicit sequence or a pdb-backed selection: {contig_raw}"
                                )
                        mask_seq_value = mask_type[process_type]  # True is changable, False is fixed
                    elif data_type == "str":
                        if len(index) > 0 and self.pdb is not None and "xyz_14" in self.pdb:
                            xyz = self.pdb["xyz_14"][index]
                        mask_str_value = mask_type[process_type]

        # Alpha torsion features are only available when parsing from a structure.
        # For New_/explicit-seq contigs (or when pdb is None), leave them empty and
        # parse_contigs() will fill default zeros.
        if len(index) > 0 and self.pdb is not None:
            if "alpha" in self.pdb and "alpha_alt" in self.pdb and "alpha_tor_mask" in self.pdb:
                alpha = self.pdb["alpha"][index]
                alpha_tor_mask = self.pdb["alpha_tor_mask"][index]
                alpha_alt = self.pdb["alpha_alt"][index]
        if len(index) > 0:    
            #start_rf_index = self.pdb["idx"][index[0]]  # residue 0-based index
            rf_index = [self.pdb["idx"][i] for i in index]  # residue 0-based index
            origin_pdb_idx = [self.pdb["pdb_idx"][i] for i in index]  # pdb index
        # default to False if not set
        if 'is_new' not in locals():
            is_new = False
        return {"seq": seq, "xyz": xyz, "length_range": length_range, 
                "rf_index": torch.tensor(rf_index), "origin_pdb_idx":origin_pdb_idx,
                "mask_seq_value":mask_seq_value, "mask_str_value":mask_str_value,
                "alpha": alpha,"alpha_alt":alpha_alt, "alpha_tor_mask": alpha_tor_mask,
                "is_new": is_new}
    
    def parse_residue_range(self,range_str):
        """
        健壮地解析使用 '/' 分隔符的残基范围字符串，支持负数残基。

        Args:
            range_str (str): 表示残基范围的字符串。
                            支持格式: 'A/100-A/200', 'A/100-200', 'B/-8-B/117'.

        Returns:
            dict: 包含起始和结束链ID及残基标识符的字典。
        """
        # Regex to handle negative residue numbers correctly.
        # It captures start_chain, start_res, optional end_chain, and end_res.
        pattern = re.compile(
            r"^(?P<start_chain>[^/]+)/(?P<start_res>-?\d+[A-Za-z]*)"  # Start: C/R
            r"-"                                                      # Separator
            r"(?:(?P<end_chain>[^/]+)/)?"                             # Optional End Chain: C/
            r"(?P<end_res>-?\d+[A-Za-z]*)$"                           # End Residue: R
        )
        match = pattern.match(range_str.strip())

        if not match:
            raise ValueError(f"Invalid range format: {range_str}. Expected 'chain/res-res' or 'chain/res-chain/res'.")

        parts = match.groupdict()
        start_chain = parts['start_chain']
        end_chain = parts['end_chain'] if parts['end_chain'] else start_chain

        return {
            'start_chain': start_chain,
            'start_res': parts['start_res'],
            'end_chain': end_chain,
            'end_res': parts['end_res']
        }

def sample_parts(intervals, total_length):
    """
    采样每个部分的值，使得总和等于 total_length，且每个值在对应区间内。
    如果 total_length 为 -1，则在每个区间内随机采样一个值，不考虑总和。
    
    参数:
        intervals: 列表的列表，每个子列表 [min, max] 表示一个部分的区间。
        total_length: 整数，目标总长度。如果为-1，则随机采样。
        
    返回:
        一个列表，包含每个部分采样后的值（整数），或 None（如果无解）。
    """
    # 验证输入区间
    min_vals = []
    max_vals = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("每个区间必须包含两个元素 [min, max]")
        min_val, max_val = interval
        if min_val > max_val:
            raise ValueError(f"区间 {interval} 无效：min 不能大于 max")
        min_vals.append(min_val)
        max_vals.append(max_val)

    # 如果 total_length 是 -1，则随机采样
    if total_length == -1:
        return [random.randint(min_v, max_v) for min_v, max_v in zip(min_vals, max_vals)]
    
    # 计算最小和、最大和
    min_sum = sum(min_vals)
    max_sum = sum(max_vals)
    
    # 可行性检查
    if total_length < min_sum or total_length > max_sum:
        return None
    
    n = len(intervals)
    slacks = [max_vals[i] - min_vals[i] for i in range(n)];  # 每个部分的松弛空间
    remaining = total_length - min_sum  # 剩余需要分配的长度
    
    # 初始化每个部分为最小值
    output = min_vals.copy()
    
    # 准备可增加部分的索引列表（只包含松弛空间大于 0 的部分）
    valid_indices = [i for i in range(n) if slacks[i] > 0]
    
    # 分配剩余长度：每次随机选择一个可增加部分，增加 1，直到剩余为 0
    for _ in range(remaining):
        if not valid_indices:
            break
        # 随机选择一个可增加的部分
        idx = random.choice(valid_indices)
        output[idx] += 1
        # 如果该部分达到最大值，从有效索引中移除
        if output[idx] == max_vals[idx]:
            valid_indices.remove(idx)
    
    return output


def _find_closest_chain_and_interface(pdb_parsed, target_chain_id, cut_off=10.0):
    """
    Finds the chain closest to the target chain and returns their interface residues.
    Uses cKDTree for efficient spatial searching.
    """
    target_chain_mask = (pdb_parsed['pdb_idx_to_chain_id'] == target_chain_id)
    target_indices = np.where(target_chain_mask)[0]
    target_coords = pdb_parsed['xyz'][target_chain_mask][:, 1]

    if target_coords.size == 0:
        return None, np.array([], dtype=int), np.array([], dtype=int)

    max_num_contacts = 0
    closest_chain_id = None
    best_interface1 = np.array([], dtype=int)
    best_interface2 = np.array([], dtype=int)

    chains = np.unique(pdb_parsed['pdb_idx_to_chain_id'])
    for chain_id in chains:
        if chain_id == target_chain_id:
            continue

        chain_mask = (pdb_parsed['pdb_idx_to_chain_id'] == chain_id)
        chain_indices = np.where(chain_mask)[0]
        chain_coords = pdb_parsed['xyz'][chain_mask][:, 1]

        if chain_coords.size == 0:
            continue

        # Use cKDTree for efficient distance calculation
        tree = cKDTree(chain_coords)
        contact_indices_list = tree.query_ball_point(target_coords, r=cut_off)
        
        num_contacts = sum(len(indices) for indices in contact_indices_list)

        if num_contacts > max_num_contacts:
            max_num_contacts = num_contacts
            closest_chain_id = chain_id
            
            # Determine interface residues based on contacts
            interface1_mask = np.array([len(indices) > 0 for indices in contact_indices_list], dtype=bool)
            best_interface1 = target_indices[interface1_mask]
            
            contact_indices_flat = np.unique([idx for sublist in contact_indices_list for idx in sublist])
            best_interface2 = chain_indices[contact_indices_flat]

    return closest_chain_id, best_interface1, best_interface2

def _generate_random_fixed_contigs(chain_id, res_indices, pdb_idx_map, proportion, segments):
    """
    Splits a designable contig into smaller designable and fixed contigs randomly.
    """
    num_residues = len(res_indices)
    if num_residues == 0:
        return []
    if proportion == 0 or segments == 0:
        start_res, end_res = pdb_idx_map[res_indices[0]][1], pdb_idx_map[res_indices[-1]][1]
        return [f"{chain_id}/{start_res}-{chain_id}/{end_res}:seq_PNA:str_PNA"]

    num_to_fix = int(num_residues * proportion)
    if num_to_fix == 0: 
        start_res, end_res = pdb_idx_map[res_indices[0]][1], pdb_idx_map[res_indices[-1]][1]
        return [f"{chain_id}/{start_res}-{chain_id}/{end_res}:seq_PNA:str_PNA"]

    # Generate random segment lengths
    fix_lengths = np.random.multinomial(num_to_fix, np.ones(segments)/segments)
    fix_lengths = fix_lengths[fix_lengths > 0]
    segments = len(fix_lengths)

    design_len = num_residues - num_to_fix
    # Allocate designable residues to gaps between fixed segments
    gaps = segments + 1
    design_lengths = np.random.multinomial(design_len, np.ones(gaps)/gaps)

    contigs = []
    current_idx = 0
    for i in range(segments):
        # Add design segment
        if design_lengths[i] > 0:
            start_res_idx = res_indices[current_idx]
            end_res_idx = res_indices[current_idx + design_lengths[i] - 1]
            start_pdb, end_pdb = pdb_idx_map[start_res_idx][1], pdb_idx_map[end_res_idx][1]
            contigs.append(f"{chain_id}/{start_pdb}-{chain_id}/{end_pdb}:seq_PNA:str_PNA")
            current_idx += design_lengths[i]

        # Add fixed segment
        if fix_lengths[i] > 0:
            start_res_idx = res_indices[current_idx]
            end_res_idx = res_indices[current_idx + fix_lengths[i] - 1]
            start_pdb, end_pdb = pdb_idx_map[start_res_idx][1], pdb_idx_map[end_res_idx][1]
            contigs.append(f"{chain_id}/{start_pdb}-{chain_id}/{end_pdb}:seq_FIX:str_FIX")
            current_idx += fix_lengths[i]
        
    # Add trailing design segment
    if design_lengths[gaps-1] > 0:
        start_res_idx = res_indices[current_idx]
        end_res_idx = res_indices[current_idx + design_lengths[gaps-1] - 1]
        start_pdb, end_pdb = pdb_idx_map[start_res_idx][1], pdb_idx_map[end_res_idx][1]
        contigs.append(f"{chain_id}/{start_pdb}-{chain_id}/{end_pdb}:seq_PNA:str_PNA")

    return contigs

def _split_indices_into_contiguous_blocks(indices):
    """Splits a list of indices into sub-lists of contiguous indices."""
    if len(indices) == 0:
        return []
    blocks = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    return [b.tolist() for b in blocks]

def generate_crop_contigs(
    pdb_parsed,
    target_chain_id,
    mode: str = 'monomer',
    crop_length: int = 100,
    fixed_res=None,
    expand_preference: str = "auto",
    target_expand_bias: float = 1.0,
    target_len_ratio: float = None,
    hotspot_k_range=None,
    Ca_threshold=10.0,
):
    """
    Generates contigs for cropping a protein from a parsed PDB file.

    Args:
        pdb_parsed (dict): Parsed PDB data from process_target.
        target_chain_id (str): The chain ID to crop from.
        mode (str): 'monomer' or 'complex'.
        crop_length (int): The desired length of the cropped segment.
        fixed_res (dict): Dictionary to specify random fixing, e.g. {'proportion': 0.2, 'segments': 2}.
        expand_preference (str): How to bias interface expansion in complex mode.
            - 'auto' (default): use room-based heuristic with a soft bias controlled by target_expand_bias.
            - 'target': always expand target chain when it still has room; fall back to neighbor only when needed.
            - 'neighbor': symmetric to 'target', but favor the partner chain.
        target_expand_bias (float): Only used when expand_preference == 'auto'.
            - > 1.0: softly prefer expanding the target chain when both chains have room.
            - = 1.0: no additional bias (legacy behaviour).
            - 0.0–1.0: softly prefer expanding the neighbour chain (0.0 ~ only neighbour, if possible).
        target_len_ratio (float, optional): Desired final fraction of residues on the target chain
            in complex mode, used only at the final cropping step when current_len > crop_length.
            - If in (0,1), we try to set len(target) ≈ crop_length * target_len_ratio (with clamping
              so that neither chain asks for more residues than it actually has).
            - If None, fall back to using the natural ratio len1 / (len1 + len2).
        hotspot_k_range (list or tuple, optional): Range [min, max] to sample K from.
            If provided, selects the top K contacting residues on the neighbor chain as hotspots.
            If None, all interface residues are marked as hotspots.

    Returns:
        tuple: A tuple containing:
            - contigs (list): The generated contigs.
            - res_mask (torch.Tensor): A mask indicating non-padded residues.
            - hotspots (list): List of hotspot residue strings (Chain/ResNum).
    """
    # Add pdb_idx_to_chain_id to pdb_parsed if it doesn't exist
    if 'pdb_idx_to_chain_id' not in pdb_parsed:
        pdb_parsed['pdb_idx_to_chain_id'] = np.array([i[0] for i in pdb_parsed['pdb_idx']])
    
    # Use the correct key for coordinates
    if 'xyz_14' in pdb_parsed and 'xyz' not in pdb_parsed:
        pdb_parsed['xyz'] = pdb_parsed['xyz_14']

    contigs = []
    hotspots = []
    final_indices = []

    if mode == 'monomer':
        chain_indices = np.where(pdb_parsed['pdb_idx_to_chain_id'] == target_chain_id)[0]
        if len(chain_indices) == 0:
            raise ValueError(f"Chain {target_chain_id} not found in PDB.")

        if len(chain_indices) > crop_length:
            # Check if there are any special connections (links) involving this chain
            special_pairs = []
            if 'links' in pdb_parsed and pdb_parsed['links']:
                # Map pdb_idx -> local index in chain_indices
                target_pdb_objs = [pdb_parsed['pdb_idx'][i] for i in chain_indices]
                target_pdb_map = {pidx: i for i, pidx in enumerate(target_pdb_objs)}
                
                for link in pdb_parsed['links']:
                    u_idx = link.get('idx1')
                    v_idx = link.get('idx2')
                    if u_idx in target_pdb_map and v_idx in target_pdb_map:
                        special_pairs.append((target_pdb_map[u_idx], target_pdb_map[v_idx]))
            
            valid_start_found = False
            if special_pairs:
                # Try pairs in random order until one fits
                random.shuffle(special_pairs)
                
                for pair in special_pairs:
                    p1, p2 = pair
                    min_p, max_p = min(p1, p2), max(p1, p2)
                    
                    # Check if pair span is within crop_length
                    # We need to include both min_p and max_p in [start, start + crop_length]
                    # Indices in window: [start, start+1, ..., start+crop_length-1]
                    # Condition: start <= min_p AND start + crop_length > max_p
                    # => start <= min_p
                    # => start >= max_p - crop_length + 1
                    # Range for start: [max(0, max_p - crop_length + 1), min(len - crop, min_p)]
                    
                    if max_p - min_p < crop_length:
                        min_s = max(0, max_p - crop_length + 1)
                        max_s = min(len(chain_indices) - crop_length, min_p)
                        
                        if min_s <= max_s:
                            start = np.random.randint(min_s, max_s + 1)
                            valid_start_found = True
                            break
            
            if not valid_start_found:
                start = np.random.randint(0, len(chain_indices) - crop_length + 1)
        else:
            start = 0
        end = start + crop_length
        
        cropped_indices = chain_indices[max(0, start):min(len(chain_indices), end)]
        final_indices = cropped_indices.tolist()
        
        if fixed_res:
            contigs.append(_generate_random_fixed_contigs(
                target_chain_id, final_indices, pdb_parsed['pdb_idx'], 
                fixed_res.get('proportion', 0), fixed_res.get('segments', 0)
            ))
        else:
            if final_indices:
                res_start = pdb_parsed['pdb_idx'][final_indices[0]][1]
                res_end = pdb_parsed['pdb_idx'][final_indices[-1]][1]
                contigs.append([f"{target_chain_id}/{res_start}-{target_chain_id}/{res_end}:seq_PNA:str_PNA"])

    elif mode == 'complex':
        neighbor_chain_id, interface1, interface2 = _find_closest_chain_and_interface(pdb_parsed, target_chain_id, cut_off=Ca_threshold)
        if neighbor_chain_id is None:
            raise ValueError(f"No neighboring chain found for chain {target_chain_id}.")
        
        # Expand if necessary
        current_len = len(interface1) + len(interface2)
        if current_len < crop_length:
            needed = crop_length - current_len
            
            chain1_indices = np.where(pdb_parsed['pdb_idx_to_chain_id'] == target_chain_id)[0]
            chain2_indices = np.where(pdb_parsed['pdb_idx_to_chain_id'] == neighbor_chain_id)[0]

            # Find interface boundaries within the full chain
            if len(interface1) > 0:
                min_idx1, max_idx1 = np.where(np.isin(chain1_indices, interface1))[0][[0, -1]]
            else:  # Handle case where one interface is empty
                min_idx1, max_idx1 = len(chain1_indices) // 2, len(chain1_indices) // 2

            if len(interface2) > 0:
                min_idx2, max_idx2 = np.where(np.isin(chain2_indices, interface2))[0][[0, -1]]
            else:
                min_idx2, max_idx2 = len(chain2_indices) // 2, len(chain2_indices) // 2

            # Expand outwards, preferring the chain that still has free residues.
            # 这里修正了原先 room 计算可能为负、导致提前停止扩展的问题，
            # 确保在有真实残基可以利用时尽量不用 New_ padding。
            expand1, expand2 = 0, 0
            len1 = len(chain1_indices)
            len2 = len(chain2_indices)

            # Sanitize bias
            if target_expand_bias is None or target_expand_bias < 0:
                target_expand_bias = 1.0

            for _ in range(needed):
                # 剩余可扩展空间（分别统计 N 端和 C 端，避免出现负数）
                left1 = max(0, min_idx1 - expand1)
                right1 = max(0, (len1 - 1) - (max_idx1 + expand1))
                room1 = left1 + right1

                left2 = max(0, min_idx2 - expand2)
                right2 = max(0, (len2 - 1) - (max_idx2 + expand2))
                room2 = left2 + right2

                # 两条链都没有空间了，提前停止
                if room1 <= 0 and room2 <= 0:
                    break

                # 根据策略和偏好选择扩展哪条链
                if expand_preference == "target":
                    # 尽量优先扩展目标链
                    if room1 > 0:
                        expand1 += 1
                    elif room2 > 0:
                        expand2 += 1
                elif expand_preference == "neighbor":
                    # 尽量优先扩展邻接链
                    if room2 > 0:
                        expand2 += 1
                    elif room1 > 0:
                        expand1 += 1
                else:
                    # 'auto'：按剩余 room，再叠加 target_expand_bias 作为软偏好
                    weighted_room1 = room1 * target_expand_bias
                    weighted_room2 = room2

                    # 优先扩展仍有空间且“加权后更宽松”的那条链；
                    # 如果 neighbor 链已经没有空间了，就只扩 target 链。
                    if weighted_room1 >= weighted_room2 and room1 > 0:
                        expand1 += 1
                    elif room2 > 0:
                        expand2 += 1

            interface1 = chain1_indices[
                max(0, min_idx1 - expand1) : min(len1, max_idx1 + expand1 + 1)
            ]
            interface2 = chain2_indices[
                max(0, min_idx2 - expand2) : min(len2, max_idx2 + expand2 + 1)
            ]

        # Crop if necessary
        current_len = len(interface1) + len(interface2)
        if current_len > crop_length:
            len1, len2 = len(interface1), len(interface2)

            # ---- Step 1: decide target / neighbour lengths ----
            if target_len_ratio is not None and 0.0 < float(target_len_ratio) < 1.0:
                # User-specified ratio: target ~ crop_length * r, neighbour ~ crop_length * (1-r)
                desired_t = int(round(crop_length * float(target_len_ratio)))
                desired_t = max(1, min(desired_t, crop_length - 1))

                # Clamp by what each chain actually has
                new_len1 = min(desired_t, len1)
                new_len2 = crop_length - new_len1
                if new_len2 > len2:
                    new_len2 = len2
                    new_len1 = crop_length - new_len2
                # Final safety clamp (in pathological cases fall back to natural ratio)
                if new_len1 <= 0 or new_len2 <= 0:
                    new_len1 = round(len1 / current_len * crop_length)
                    new_len2 = crop_length - new_len1
            else:
                # Legacy behaviour: use natural ratio len1 : len2
                new_len1 = round(len1 / current_len * crop_length)
                new_len2 = crop_length - new_len1

            # ---- Step 2: randomly crop contiguous windows on each chain ----
            max_s1 = len1 - new_len1
            max_s2 = len2 - new_len2
            s1 = np.random.randint(0, max_s1 + 1) if max_s1 > 0 else 0
            s2 = np.random.randint(0, max_s2 + 1) if max_s2 > 0 else 0
            interface1, interface2 = interface1[s1:s1+new_len1], interface2[s2:s2+new_len2]

        # Identify hotspots: residues on the neighbor chain that are part of the interface.
        # 约定：当 hotspot_k_range 为 None 时，不产生任何 hotspot（hotspots 为空）。
        selected_hotspot_indices = []
        
        if hotspot_k_range is not None and len(interface2) > 0 and len(interface1) > 0:
            # 1. Calculate contacts between the *cropped* target and neighbor segments
            # Use CA atoms ([:, 1, :]) for distance calculation
            target_coords = pdb_parsed['xyz'][interface1][:, 1]
            neighbor_coords = pdb_parsed['xyz'][interface2][:, 1]

            # Build KDTree on neighbor coordinates
            tree = cKDTree(neighbor_coords)
            # Query neighbors within 8.0 Angstroms from each target residue
            contact_indices_list = tree.query_ball_point(target_coords, r=Ca_threshold)
            
            # 2. Count contacts per neighbor residue
            # contact_indices_list contains indices relative to `neighbor_coords` (0..len(interface2)-1)
            contact_counts = {}
            for neighbors in contact_indices_list:
                for n_local_idx in neighbors:
                    contact_counts[n_local_idx] = contact_counts.get(n_local_idx, 0) + 1
            
            # 3. Sample K
            k_min, k_max = hotspot_k_range[0], hotspot_k_range[1]
            k = random.randint(k_min, k_max)
            
            # 4. Sort neighbor residues by contact count (descending)
            # Create a list of (global_idx, count)
            weighted_neighbors = []
            for i, global_idx in enumerate(interface2):
                count = contact_counts.get(i, 0)
                weighted_neighbors.append((global_idx, count))
            
            # Sort by count desc, then by global_idx for stability
            weighted_neighbors.sort(key=lambda x: (x[1], x[0]), reverse=True)
            
            # 5. Select top K
            selected_hotspot_indices = [x[0] for x in weighted_neighbors[:k] if x[1] > 0]
            # If no contacts found (unlikely if they are interface), selected might be empty. 
            # Fallback: if K>0 but we found 0 contacts, maybe just take random or keep empty?
            # Current logic: only pick those with > 0 contacts. 
            
        for idx in selected_hotspot_indices:
            chain, res = pdb_parsed['pdb_idx'][idx]
            hotspots.append(f"{chain}/{res}")

        final_indices = np.concatenate([interface1, interface2]).tolist()

        # Generate contigs for each chain
        target_blocks = _split_indices_into_contiguous_blocks(interface1)
        neighbor_blocks = _split_indices_into_contiguous_blocks(interface2)
        target_contigs_list = []
        if fixed_res:
             # Apply random fixing only on the designable chain (target)
            # target_contigs_list.extend(_generate_random_fixed_contigs(
            #     target_chain_id, interface1, pdb_parsed['pdb_idx'],
            #     fixed_res.get('proportion', 0), fixed_res.get('segments', 0)
            # ))
            for block in target_blocks:
                target_contigs_list.extend(_generate_random_fixed_contigs(
                    target_chain_id, block, pdb_parsed['pdb_idx'],
                    fixed_res['proportion'], fixed_res['segments']
                ))
        else:
            for block in target_blocks:
                start_res, end_res = pdb_parsed['pdb_idx'][block[0]][1], pdb_parsed['pdb_idx'][block[-1]][1]
                target_contigs_list.append(f"{target_chain_id}/{start_res}-{target_chain_id}/{end_res}:seq_PNA:str_PNA")
        contigs.append(target_contigs_list)

        neighbor_contigs_list = []
        for block in neighbor_blocks:
            start_res, end_res = pdb_parsed['pdb_idx'][block[0]][1], pdb_parsed['pdb_idx'][block[-1]][1]
            neighbor_contigs_list.append(f"{neighbor_chain_id}/{start_res}-{neighbor_chain_id}/{end_res}:seq_FIX:str_FIX")
        contigs.append(neighbor_contigs_list)

    elif mode == 'complex_space':
        # ----------------------------------------------------------
        # Spatially-biased complex cropping:
        #   - Target chain: pick a *sequence-contiguous* window that
        #     is as spatially compact as possible.
        #   - Neighbour chain: fill remaining length budget with
        #     residues that are closest in 3D to the chosen target
        #     window, *purely by distance order* (no radius cutoff).
        # ----------------------------------------------------------

        # 1) Locate target and neighbour chains
        chain_mask_t = (pdb_parsed['pdb_idx_to_chain_id'] == target_chain_id)
        target_indices = np.where(chain_mask_t)[0]
        if len(target_indices) == 0:
            raise ValueError(f"Chain {target_chain_id} not found in PDB.")

        # Choose neighbour chain the same way as in 'complex' mode
        neighbor_chain_id, _, _ = _find_closest_chain_and_interface(pdb_parsed, target_chain_id,cut_off=Ca_threshold)
        if neighbor_chain_id is None:
            raise ValueError(f"No neighboring chain found for chain {target_chain_id}.")

        chain_mask_n = (pdb_parsed['pdb_idx_to_chain_id'] == neighbor_chain_id)
        neighbor_indices = np.where(chain_mask_n)[0]
        if len(neighbor_indices) == 0:
            raise ValueError(f"Neighbor chain {neighbor_chain_id} has no residues.")

        # Coordinates (CA atoms)
        coords_all = pdb_parsed['xyz'][:, 1]  # [N, 3]
        coords_t = coords_all[target_indices]
        coords_n = coords_all[neighbor_indices]

        len_t_full = len(target_indices)
        len_n_full = len(neighbor_indices)

        # 2) Decide length budget between target and neighbour
        if target_len_ratio is not None and 0.0 < float(target_len_ratio) < 1.0:
            desired_t = int(round(crop_length * float(target_len_ratio)))
            desired_t = max(1, min(desired_t, crop_length - 1))
            L_t = min(desired_t, len_t_full)
        else:
            # Fallback: natural ratio
            L_t = round(len_t_full / (len_t_full + len_n_full) * crop_length)
            L_t = max(1, min(L_t, len_t_full))

        # Remaining budget for neighbour: L_n = crop_length - L_t
        L_n = crop_length - L_t
        # # 3) Target chain: pick the most compact contiguous window of length L_t
        # if len_t_full <= L_t:
        #     win_start_t = 0
        #     win_end_t = len_t_full
        # else:
        #     # Sliding window, choose minimal average squared distance from window center
        #     best_score = None
        #     best_start = 0
        #     for s in range(0, len_t_full - L_t + 1):
        #         e = s + L_t
        #         window = coords_t[s:e]  # [L_t, 3]
        #         center = window.mean(axis=0, keepdims=True)
        #         # radius of gyration (mean squared distance to center)
        #         rg2 = ((window - center) ** 2).sum(axis=1).mean()
        #         if best_score is None or rg2 < best_score:
        #             best_score = rg2
        #             best_start = s
        #     win_start_t = best_start
        #     win_end_t = best_start + L_t
        # 3) Target chain: pick the contiguous window closest to the neighbor chain
        if len_t_full <= L_t:
            win_start_t = 0
            win_end_t = len_t_full
        else:
            # 使用 cKDTree 计算每个 Target 残基到 Neighbor 链的最近距离
            # coords_t: [len_t_full, 3], coords_n: [len_n_full, 3]
            tree_n = cKDTree(coords_n)
            # query 返回两个数组：(distances, indices)，我们只需要 distances
            dists_t_to_n, _ = tree_n.query(coords_t)  # [len_t_full]
            
            best_score = None
            best_start = 0
            
            # Sliding window, choose minimal average distance to neighbor chain
            for s in range(0, len_t_full - L_t + 1):
                e = s + L_t
                # 计算该窗口内所有残基到 Neighbor 的平均距离
                current_score = dists_t_to_n[s:e].mean()
                
                if best_score is None or current_score < best_score:
                    best_score = current_score
                    best_start = s
            
            win_start_t = best_start
            win_end_t = best_start + L_t

        target_window_indices = target_indices[win_start_t:win_end_t]

        # 4) Neighbour chain: fill remaining budget by nearest distances to target window
        interface1 = target_window_indices
        interface2 = np.array([], dtype=int)

        if L_n > 0 and len_n_full > 0:
            # Compute minimal distance from each neighbour residue to any target-window residue
            target_win_coords = coords_t[win_start_t:win_end_t]  # [L_t, 3]
            # Use broadcasting to compute pairwise distances: [L_t, L_n]
            diff = target_win_coords[:, None, :] - coords_n[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            min_dists = dists.min(axis=0)  # [L_n]

            # Sort neighbour residues by distance and take as many as budget allows
            order = np.argsort(min_dists)
            k = min(L_n, len_n_full)
            chosen_local = order[:k]
            # NOTE: chosen_local 是按距离排序的局部索引，这里先选出对应的全局索引，
            # 再按序列顺序排序，保证后续按连续 index 合并成较长 contig。
            interface2 = neighbor_indices[chosen_local]
            interface2 = np.sort(interface2)

        # 5) Hotspots on neighbour chain (based on distance / contact count)
        # 约定：当 hotspot_k_range 为 None 时，不产生任何 hotspot。
        selected_hotspot_indices = []
        if hotspot_k_range is not None and len(interface2) > 0 and len(interface1) > 0:
            # Recompute contacts within 8Å between target window and chosen neighbour residues
            target_coords = coords_all[interface1]
            neighbor_coords = coords_all[interface2]

            tree = cKDTree(neighbor_coords)
            contact_indices_list = tree.query_ball_point(target_coords, r=Ca_threshold)

            contact_counts = {}
            for neighbors in contact_indices_list:
                for n_local_idx in neighbors:
                    contact_counts[n_local_idx] = contact_counts.get(n_local_idx, 0) + 1

            k_min, k_max = hotspot_k_range[0], hotspot_k_range[1]
            k_h = random.randint(k_min, k_max)

            weighted_neighbors = []
            for i, global_idx in enumerate(interface2):
                count = contact_counts.get(i, 0)
                weighted_neighbors.append((global_idx, count))

            weighted_neighbors.sort(key=lambda x: (x[1], x[0]), reverse=True)
            selected_hotspot_indices = [x[0] for x in weighted_neighbors[:k_h] if x[1] > 0]

        for idx in selected_hotspot_indices:
            ch, res = pdb_parsed['pdb_idx'][idx]
            hotspots.append(f"{ch}/{res}")

        final_indices = np.concatenate([interface1, interface2]).tolist()

        # 6) Build contigs for target and neighbour (sequence-contiguous blocks)
        # interface1 本身就是一个连续窗口；interface2 在上面已经按 index 排序，
        # 这里用 _split_indices_into_contiguous_blocks 自动把相邻残基合并成更长的区间。
        target_blocks = _split_indices_into_contiguous_blocks(interface1)
        neighbor_blocks = _split_indices_into_contiguous_blocks(interface2)

        target_contigs_list = []
        if fixed_res:
            # Apply random fixing only on the designable chain (target)
            for block in target_blocks:
                target_contigs_list.extend(_generate_random_fixed_contigs(
                    target_chain_id, block, pdb_parsed['pdb_idx'],
                    fixed_res.get('proportion', 0), fixed_res.get('segments', 0)
                ))
        else:
            for block in target_blocks:
                start_res, end_res = pdb_parsed['pdb_idx'][block[0]][1], pdb_parsed['pdb_idx'][block[-1]][1]
                target_contigs_list.append(f"{target_chain_id}/{start_res}-{target_chain_id}/{end_res}:seq_PNA:str_PNA")
        contigs.append(target_contigs_list)

        neighbor_contigs_list = []
        for block in neighbor_blocks:
            start_res, end_res = pdb_parsed['pdb_idx'][block[0]][1], pdb_parsed['pdb_idx'][block[-1]][1]
            neighbor_contigs_list.append(f"{neighbor_chain_id}/{start_res}-{neighbor_chain_id}/{end_res}:seq_FIX:str_FIX")
        contigs.append(neighbor_contigs_list)

    else:
        raise ValueError("Mode must be 'monomer', 'complex', or 'complex_space'.")

    total_len = len(final_indices)
    # Padding and res_mask
    padding = crop_length - total_len
    if padding > 0:
        # 为了不浪费设计位点：
        # - monomer: 仍然加在唯一一条链（最后一条链）的最后
        # - complex: 把 New 残基补在“设计链”（第一条链, contigs[0]）上
        if not contigs:
            contigs.append([])
        if mode == 'complex':
            # 设计链在 generate_crop_contigs 中总是作为第一条链加入 contigs
            contigs[0].append(f"New_{padding}-{padding}")
        else:
            contigs[-1].append(f"New_{padding}-{padding}")

        res_mask = torch.ones(crop_length)
        if total_len > 0:
            res_mask[total_len:] = 0  # Mark padding as 0
    else:
        res_mask = torch.ones(total_len)

    return contigs, res_mask, hotspots


def randomly_fix_bonds(target, fixed_bond_config=None):
    """
    Randomly fixes a fraction of existing bonds in the target object.

    This function identifies non-diagonal, non-padded bonds with a value of 1
    in the bond matrix and sets their corresponding mask value to False,
    effectively "fixing" them.

    Args:
        target (Target): The Target object to modify in-place.
        fixed_bond_config (float or dict, optional): Configuration for fixing.
            - If float: The exact fraction of bonds to fix.
            - If dict: Specifies a range {'ratio_min': float, 'ratio_max': float}
                        from which a random fraction is drawn.
            If None, no operation is performed.
    """
    if fixed_bond_config is None:
        return

    # Determine the ratio of bonds to fix
    if isinstance(fixed_bond_config, float):
        ratio = fixed_bond_config
    elif isinstance(fixed_bond_config, dict):
        min_r = fixed_bond_config.get('ratio_min', 0.0)
        max_r = fixed_bond_config.get('ratio_max', 1.0)
        ratio = random.uniform(min_r, max_r)
    else:
        return  # Invalid config

    if not 0 < ratio <= 1.0:
        return

    L = len(target.full_seq)
    device = target.full_bond_matrix.device

    # 1. Find candidate indices to fix (upper triangle only)
    # Candidates are: non-diagonal, bond_matrix==1, non-padded, and currently mutable.
    triu_indices = torch.triu_indices(L, L, offset=1, device=device)
    i_upper, j_upper = triu_indices[0], triu_indices[1]

    is_one = target.full_bond_matrix[i_upper, j_upper] >= 0.9999
    is_not_padded = (target.res_mask[i_upper] == 1) & (target.res_mask[j_upper] == 1)
    is_mutable = target.full_bond_mask[i_upper, j_upper]

    valid_mask = is_one & is_not_padded & is_mutable
    
    candidate_indices = torch.where(valid_mask)[0]
    
    num_candidates = len(candidate_indices)
    if num_candidates == 0:
        return

    # 2. Determine how many bonds to fix
    num_to_fix = math.ceil(num_candidates * ratio)
    
    # 3. Randomly select indices and apply the fix
    perm = torch.randperm(num_candidates, device=device)
    selected_indices_in_candidates = candidate_indices[perm[:num_to_fix]]

    fix_i = i_upper[selected_indices_in_candidates]
    fix_j = j_upper[selected_indices_in_candidates]

    # 4. Update the bond_mask in place, masking full rows and columns for all involved residues
    if fix_i.numel() > 0:
        # Collect all unique residue indices involved in the fixed bonds
        indices_to_fix = torch.unique(torch.cat([fix_i, fix_j]))
        
        # Mask the corresponding rows and columns
        target.full_bond_mask[indices_to_fix, :] = False
        target.full_bond_mask[:, indices_to_fix] = False