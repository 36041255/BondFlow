from rfdiff.chemical import aa2num, aa2long
import pandas as pd
import torch
import os
import logging
import csv
import math
from functools import lru_cache


def _pad_atom_name(a: str) -> str:
    """Pad atom name to AA-long format (e.g., 'CA' -> ' CA ')."""
    return ' ' + (a or '').strip().upper().ljust(3)


def _get_atom_coords(res_num: int, atom_name: str, res_coords: torch.Tensor):
    """Get atom coordinates by name for a given residue. Returns None on missing/NaN."""
    try:
        atom_idx = aa2long[res_num].index(_pad_atom_name(atom_name))
        coords = res_coords[atom_idx, :3]
        if torch.isnan(coords).any():
            return None
        return coords
    except (ValueError, IndexError):
        return None


def _calculate_angle(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor):
    """Computes angle p1-p2-p3 in radians."""
    v1 = p1 - p2
    v2 = p3 - p2
    v1_norm = torch.linalg.norm(v1)
    v2_norm = torch.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return None
    dot_prod = torch.dot(v1, v2)
    cos_angle = dot_prod / (v1_norm * v2_norm)
    return torch.acos(torch.clamp(cos_angle, -1.0, 1.0))


def _calculate_dihedral(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor):
    """Computes dihedral angle p1-p2-p3-p4 in radians in [-pi, pi]."""
    b1 = -1.0 * (p2 - p1)
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    
    b2_norm = torch.linalg.norm(b2)
    if b2_norm < 1e-6:
        return None

    m1 = torch.linalg.cross(n1, b2 / b2_norm)
    
    x = torch.dot(n1, n2)
    y = torch.dot(m1, n2)
    return torch.atan2(y, x)


def _periodic_angle_diff(a1_rad, a2_rad, is_planar: bool):
    """Periodic difference between two angles; planar dihedral uses pi-periodicity."""
    if a1_rad is None or a2_rad is None:
        return float('inf')
    period = math.pi if is_planar else 2 * math.pi
    diff = abs(a1_rad - a2_rad) % period
    return min(diff, period - diff)


class LinkInfo:
    """
    一个用于解析和存储 link.csv 文件中键合信息的统一接口类。
    """
    def __init__(self, link_csv_path: str, device: torch.device = 'cpu', compat_default: float = 1e-4):
        self.bond_spec = {}  # (res1_num, res2_num) -> list of {'atom1': str, 'atom2': str, 'dist': float}
        self.removals = {}
        self.allowed_bonds = set()
        
        all_aa_map = {name: num for name, num in aa2num.items() if name != 'MAS' and name != 'UNK'}
        aa_list = list(all_aa_map.keys())
        K = len(aa_list)
        self.K = K
        compat = torch.full((K, K), compat_default, device=device)

        if not link_csv_path or not os.path.exists(link_csv_path):
            logging.warning(f"Link CSV not found: {link_csv_path}. Continuing with empty link info.")
            self.compat_matrix = compat
            return
        

        with open(link_csv_path, 'r', encoding='utf-8') as f:
            # Filter out NUL characters that can cause csv.Error
            content = f.read().replace('\x00', '')
            reader = csv.DictReader(content.splitlines())
            for row in reader:
                res1_spec = (row.get('res1') or '').strip().upper()
                res2_spec = (row.get('res2') or '').strip().upper()
                atom1 = (row.get('atom1') or '').strip().upper()
                atom2 = (row.get('atom2') or '').strip().upper()
                
                try:
                    score = float(row.get('score')) if row.get('score') else 1.0
                except (ValueError, TypeError):
                    score = 1.0

                try:
                    avg_dist = float(row.get('avg_distance')) if row.get('avg_distance') else None
                except (ValueError, TypeError):
                    avg_dist = None

                if not (res1_spec and res2_spec and atom1 and atom2 and avg_dist is not None):
                    continue

                # Optional angle/dihedral fields in DEGREES (kept optional/backward-compatible)
                def _get_deg_float(key: str):
                    try:
                        val = row.get(key)
                        return float(val) if val not in (None, '') else None
                    except (ValueError, TypeError):
                        return None

                angle_i_ref_deg = _get_deg_float('angle_i_ref_deg')
                angle_j_ref_deg = _get_deg_float('angle_j_ref_deg')
                dihedral_1_ref_deg = _get_deg_float('dihedral_1_ref_deg')
                dihedral_2_ref_deg = _get_deg_float('dihedral_2_ref_deg')

                dihedral_1_planar = (row.get('dihedral_1_planar') or '').strip().upper() == 'TRUE'
                dihedral_2_planar = (row.get('dihedral_2_planar') or '').strip().upper() == 'TRUE'

                angle_i_anchor = (row.get('angle_i_anchor') or '').strip().upper()
                angle_j_anchor = (row.get('angle_j_anchor') or '').strip().upper()
                dihedral_1_anchor_i = (row.get('dihedral_1_anchor_i') or '').strip().upper()
                dihedral_1_anchor_j = (row.get('dihedral_1_anchor_j') or '').strip().upper()
                dihedral_2_anchor_i = (row.get('dihedral_2_anchor_i') or '').strip().upper()
                dihedral_2_anchor_j = (row.get('dihedral_2_anchor_j') or '').strip().upper()

                # Optional tolerances (currently not used in loss formula; for diagnostics)
                angle_tol_deg = _get_deg_float('angle_tol_deg')
                dihedral_tol_deg = _get_deg_float('dihedral_tol_deg')

                # Treat HEAD/TAIL and backbone N/C as termini-specific rules
                is_termini_rule = (
                        atom1 in ('N', 'C') or atom2 in ('N', 'C')
                )
                
                res1_names = all_aa_map.keys() if res1_spec == "ALL" else [res1_spec]
                res2_names = all_aa_map.keys() if res2_spec == "ALL" else [res2_spec]
                for r1_name in res1_names:
                    for r2_name in res2_names:
                        if r1_name not in all_aa_map or r2_name not in all_aa_map:
                            continue

                        self.allowed_bonds.add((r1_name, r2_name, atom1, atom2))
                        self.allowed_bonds.add((r2_name, r1_name, atom2, atom1))
                        
                        r1_num = all_aa_map[r1_name]
                        r2_num = all_aa_map[r2_name]

                        # 1. Update compat_matrix (only for non-termini sidechain-sidechain rules)
                        if not is_termini_rule:
                            compat[r1_num, r2_num] = max(score, compat_default)
                            compat[r2_num, r1_num] = max(score, compat_default)
                        
                        # 2. Update bond_spec with all rules (include optional angle/dihedral fields)
                        rule = {
                            'atom1': atom1,
                            'atom2': atom2,
                            'dist': avg_dist,
                            # Optional angle/dihedral definitions on i/j sides (converted to radians)
                            'angle_i_ref': math.radians(angle_i_ref_deg) if angle_i_ref_deg is not None else None,
                            'angle_i_anchor': angle_i_anchor if angle_i_anchor else None,
                            'angle_j_ref': math.radians(angle_j_ref_deg) if angle_j_ref_deg is not None else None,
                            'angle_j_anchor': angle_j_anchor if angle_j_anchor else None,
                            'dihedral_1_ref': math.radians(dihedral_1_ref_deg) if dihedral_1_ref_deg is not None else None,
                            'dihedral_1_anchor_i': dihedral_1_anchor_i if dihedral_1_anchor_i else None,
                            'dihedral_1_anchor_j': dihedral_1_anchor_j if dihedral_1_anchor_j else None,
                            'dihedral_1_planar': dihedral_1_planar,
                            'dihedral_2_ref': math.radians(dihedral_2_ref_deg) if dihedral_2_ref_deg is not None else None,
                            'dihedral_2_anchor_i': dihedral_2_anchor_i if dihedral_2_anchor_i else None,
                            'dihedral_2_anchor_j': dihedral_2_anchor_j if dihedral_2_anchor_j else None,
                            'dihedral_2_planar': dihedral_2_planar,
                            'angle_tol': math.radians(angle_tol_deg) if angle_tol_deg is not None else None,
                            'dihedral_tol': math.radians(dihedral_tol_deg) if dihedral_tol_deg is not None else None,
                        }
                        key = (r1_num, r2_num)
                        if key not in self.bond_spec:
                            self.bond_spec[key] = []
                        self.bond_spec[key].append(rule)
                        
                        # Add symmetric rule
                        rule_sym = {
                            'atom1': atom2,
                            'atom2': atom1,
                            'dist': avg_dist,
                            # Swap i/j sides for symmetric entry
                            'angle_i_ref': math.radians(angle_j_ref_deg) if angle_j_ref_deg is not None else None,
                            'angle_i_anchor': angle_j_anchor if angle_j_anchor else None,
                            'angle_j_ref': math.radians(angle_i_ref_deg) if angle_i_ref_deg is not None else None,
                            'angle_j_anchor': angle_i_anchor if angle_i_anchor else None,
                            'dihedral_1_ref': math.radians(dihedral_1_ref_deg) if dihedral_1_ref_deg is not None else None,  # same reference; anchors swap sides
                            'dihedral_1_anchor_i': dihedral_1_anchor_j if dihedral_1_anchor_j else None,
                            'dihedral_1_anchor_j': dihedral_1_anchor_i if dihedral_1_anchor_i else None,
                            'dihedral_1_planar': dihedral_1_planar,
                            'dihedral_2_ref': math.radians(dihedral_2_ref_deg) if dihedral_2_ref_deg is not None else None,  # same reference; anchors swap sides
                            'dihedral_2_anchor_i': dihedral_2_anchor_j if dihedral_2_anchor_j else None,
                            'dihedral_2_anchor_j': dihedral_2_anchor_i if dihedral_2_anchor_i else None,
                            'dihedral_2_planar': dihedral_2_planar,
                            'angle_tol': math.radians(angle_tol_deg) if angle_tol_deg is not None else None,
                            'dihedral_tol': math.radians(dihedral_tol_deg) if dihedral_tol_deg is not None else None,
                        }
                        key_sym = (r2_num, r1_num)
                        if key_sym not in self.bond_spec:
                            self.bond_spec[key_sym] = []
                        self.bond_spec[key_sym].append(rule_sym)

                        # 3. Update removals (accumulate from CSV if present)
                        rem1 = (row.get('remove_atom1') or '').strip().upper()
                        rem2 = (row.get('remove_atom2') or '').strip().upper()

                        key_rem = (r1_num, r2_num)
                        key_rem_sym = (r2_num, r1_num)

                        if key_rem not in self.removals:
                            self.removals[key_rem] = {r1_name: [], r2_name: []}
                        if key_rem_sym not in self.removals:
                            self.removals[key_rem_sym] = {r1_name: [], r2_name: []}
                        # Add CSV-specified removals (padded to AA long format)
                        def _pad_atom_name(a: str) -> str:
                            return (' ' + a.strip().upper().ljust(3)) if a else a

                        a1,a2 = '',''
                        if rem1:
                            a1 = _pad_atom_name(rem1)
                            self.removals[key_rem].setdefault(r1_name, [])
                            if a1 not in self.removals[key_rem][r1_name]:
                                self.removals[key_rem][r1_name].append(a1)
                            self.removals[key_rem_sym].setdefault(r1_name, [])
                            if a1 not in self.removals[key_rem_sym][r1_name]:
                                self.removals[key_rem_sym][r1_name].append(a1)

                        if rem2:
                            a2 = _pad_atom_name(rem2)
                            self.removals[key_rem].setdefault(r2_name, [])
                            if a2 not in self.removals[key_rem][r2_name]:
                                self.removals[key_rem][r2_name].append(a2)
                            self.removals[key_rem_sym].setdefault(r2_name, [])
                            if a2 not in self.removals[key_rem_sym][r2_name]:
                                self.removals[key_rem_sym][r2_name].append(a2)

        self.compat_matrix = compat


    def _expand_residue(self, res_str: str) -> list[int]:
        if res_str == 'ALL':
            return list(range(self.K)) # 0 to 19 for standard AAs
        
        res_num = aa2num.get(res_str)
        if res_num is not None and res_num < self.K:
            return [res_num]
        
        return []


def load_allowed_bonds_from_csv(link_csv_path):
    """
    (已重构) 从 CSV 加载允许的键类型。
    """
    if not link_csv_path:
        return set()
    link_info = LinkInfo(link_csv_path)
    return link_info.allowed_bonds


# def _get_bond_info(link_csv_path):
#     """
#     兼容旧接口：从 LinkInfo 适配出
#       - bonds[(res1,res2)] -> (atom1, atom2, dist) 三元组（选择第一条规则）
#       - removals: 直接转发 LinkInfo.removals
#     同时对原子名进行 AA-long 风格填充（如 ' CA ').
#     """
#     if not link_csv_path or not os.path.exists(link_csv_path):
#         return {}, {}

#     link = LinkInfo(link_csv_path)

#     def _pad_atom_name(a: str) -> str:
#         return ' ' + a.strip().upper().ljust(3)

#     bonds = {}
#     for key, rules in link.bond_spec.items():
#         if not rules:
#                 continue
#         r = rules[0]

#         dist_val = float(r.get('dist'))
#         bonds[key] = (_pad_atom_name(r.get('atom1', 'CA')),
#                       _pad_atom_name(r.get('atom2', 'CA')),
#                       dist_val)

#     return bonds, link.removals


def get_valid_links(
    seq,
    atom_coords,
    bond_mat,
    link_csv_path,
    head_mask=None,
    tail_mask=None,
    distance_scale: float = 2,
    distance_abs_tol: float = 0.2,
    check_angles: bool = True,
    angle_tolerance_deg: float = 20.0,
    check_dihedrals: bool = True,
    dihedral_tolerance_deg: float = 30.0,
    include_invalid: bool = False,
):
    """
    Determine valid inter-residue links from a bond adjacency matrix using the link spec,
    with optional checks for angles and dihedrals and per-rule tolerances.

    Args:
        seq: Tensor [L]
        atom_coords: Tensor [L, N_atoms, 3]
        bond_mat: Tensor [L, L] (nonzero upper triangle entries are considered)
        link_csv_path: str
        head_mask: Optional [L] bool tensor (override atom on head to N)
        tail_mask: Optional [L] bool tensor (override atom on tail to C)
        distance_scale: float, multiplicative slack on reference distance
        distance_abs_tol: float, absolute tolerance in Angstroms
        check_angles: bool, enable angle checks when rule provides anchors
        angle_tolerance_deg: default tolerance in degrees if per-rule is absent
        check_dihedrals: bool, enable dihedral checks when rule provides anchors
        dihedral_tolerance_deg: default tolerance in degrees if per-rule is absent
        include_invalid: include pairs without passing checks (or missing spec) with is_valid=False

    Returns:
        List[dict]: entries include at least keys
            'i','j','res1_num','res2_num','atom1_name','atom2_name','distance','is_valid'
            and diagnostic flags: 'distance_ok','angle_i_ok','angle_j_ok','dihedral_1_ok','dihedral_2_ok'
    """
    results = []
    if bond_mat is None or link_csv_path is None:
        return results

    link_info = LinkInfo(link_csv_path)
    
    # Global tolerances in radians
    angle_tol_rad = math.radians(angle_tolerance_deg)
    dihedral_tol_rad = math.radians(dihedral_tolerance_deg)

    scpu = seq if isinstance(seq, torch.Tensor) else torch.tensor(seq)
    if scpu.is_cuda:
        scpu = scpu.cpu()
    atomscpu = atom_coords if isinstance(atom_coords, torch.Tensor) else torch.tensor(atom_coords)
    if atomscpu.is_cuda:
        atomscpu = atomscpu.cpu()
    bond_cpu = bond_mat.cpu() if isinstance(bond_mat, torch.Tensor) else torch.tensor(bond_mat)

    i_indices, j_indices = torch.where(torch.triu(bond_cpu > 0.5, diagonal=1))

    def _override_atom(idx: int, orig_name: str) -> str:
        try:
            if head_mask is not None and bool(head_mask[idx]):
                return 'N'
            if tail_mask is not None and bool(tail_mask[idx]):
                return 'C'
        except Exception:
            pass
        return orig_name

    for i, j in zip(i_indices.tolist(), j_indices.tolist()):
        res1_num = int(scpu[i].item())
        res2_num = int(scpu[j].item())

        # Check if head_mask/tail_mask will override atoms to N/C
        # If so, treat as ALL,ALL (peptide bond) instead of residue-specific rules
        is_head_i = head_mask is not None and bool(head_mask[i]) if head_mask is not None else False
        is_tail_i = tail_mask is not None and bool(tail_mask[i]) if tail_mask is not None else False
        is_head_j = head_mask is not None and bool(head_mask[j]) if head_mask is not None else False
        is_tail_j = tail_mask is not None and bool(tail_mask[j]) if tail_mask is not None else False
        
        # If one is head (N) and the other is tail (C), use peptide bond rules (ALL,ALL,C,N)
        # Since ALL,ALL rules are expanded to all residue pairs in LinkInfo, 
        # we can find them by checking any residue pair for C-N rules
        will_use_peptide_bond = (is_head_i and is_tail_j) or (is_head_j and is_tail_i)
        
        # If peptide bond is expected, directly look for ALL,ALL rule (C-N) from any residue pair
        # This is equivalent to treating residue types as ALL when atoms are overridden to N/C
        if will_use_peptide_bond:
            # Find ALL,ALL,C,N rule by checking any residue pair (since it's expanded to all pairs)
            # We use the first residue pair (0, 0) as a reference to get the peptide bond geometry
            rules = None
            for test_res1 in range(link_info.K):
                for test_res2 in range(link_info.K):
                    test_rules = link_info.bond_spec.get((test_res1, test_res2))
                    if test_rules:
                        peptide_rules = [r for r in test_rules if 
                                        (r.get('atom1') == 'C' and r.get('atom2') == 'N') or 
                                        (r.get('atom1') == 'N' and r.get('atom2') == 'C')]
                        if peptide_rules:
                            rules = peptide_rules
                            break
                if rules:
                    break
        else:
            # Normal case: use residue-specific rules
            rules = link_info.bond_spec.get((res1_num, res2_num))
            if not rules:
                rules = link_info.bond_spec.get((res2_num, res1_num))

        # Prepare a default report template
        base_report = {
            'i': i,
            'j': j,
            'res1_num': res1_num,
            'res2_num': res2_num,
            'is_valid': False,
        }

        # Short-circuit when no rules
        if not rules:
            if include_invalid:
                # Fallback CA-CA geometry report
                p_i = _get_atom_coords(res1_num, 'CA', atomscpu[i])
                p_j = _get_atom_coords(res2_num, 'CA', atomscpu[j])
                if p_i is not None and p_j is not None:
                    dist = torch.linalg.norm(p_i - p_j)
                    rep = dict(base_report)
                    rep.update({
                        'atom1_name': 'CA',
                        'atom2_name': 'CA',
                        'distance': float(dist.item()),
                        'reason': 'no_spec',
                        'distance_ok': False,
                        'angle_i_ok': True,
                        'angle_j_ok': True,
                        'dihedral_1_ok': True,
                        'dihedral_2_ok': True,
                    })
                    results.append(rep)
            continue
        
        # Iterate rules until one satisfies all enabled checks
        chosen_report = None
        for rule_idx, rule in enumerate(rules):
            atom1_name = _override_atom(i, rule.get('atom1'))
            atom2_name = _override_atom(j, rule.get('atom2'))

            p_i = _get_atom_coords(res1_num, atom1_name, atomscpu[i])
            p_j = _get_atom_coords(res2_num, atom2_name, atomscpu[j])
            if p_i is None or p_j is None:
                continue

            # Compute atom indices for backward compatibility fields
            try:
                atom1_idx = aa2long[res1_num].index(_pad_atom_name(atom1_name))
                atom2_idx = aa2long[res2_num].index(_pad_atom_name(atom2_name))
            except (ValueError, IndexError):
                # If index mapping fails, skip this rule
                continue

            # Distance & geometry bookkeeping
            angle_i_val = None
            angle_j_val = None
            d1_val = None
            d2_val = None

            # Distance check
            dist = torch.linalg.norm(p_i - p_j)
            dist_ref = rule.get('dist')
            dist_ok = True
            if dist_ref is not None:
                dist_ok = (dist <= float(dist_ref) * float(distance_scale)) and (abs(float(dist.item()) - float(dist_ref)) <= float(distance_abs_tol))

            # Angle checks (optional)
            angle_i_ok = True
            angle_j_ok = True
            if check_angles:
                # i-side
                ai_ref = rule.get('angle_i_ref')
                ai_anchor = rule.get('angle_i_anchor')
                if ai_ref is not None and ai_anchor:
                    p_anchor = _get_atom_coords(res1_num, ai_anchor, atomscpu[i])
                    if p_anchor is not None:
                        ai_val = _calculate_angle(p_anchor, p_i, p_j)
                        angle_i_val = ai_val
                        tol_i_val = rule.get('angle_tol')
                        tol_i = float(tol_i_val) if tol_i_val is not None else float(angle_tol_rad)
                        angle_i_ok = (ai_val is not None) and (abs(ai_val - ai_ref) <= tol_i)
                # j-side
                aj_ref = rule.get('angle_j_ref')
                aj_anchor = rule.get('angle_j_anchor')
                if aj_ref is not None and aj_anchor:
                    p_anchor = _get_atom_coords(res2_num, aj_anchor, atomscpu[j])
                    if p_anchor is not None:
                        aj_val = _calculate_angle(p_i, p_j, p_anchor)
                        angle_j_val = aj_val
                        tol_j_val = rule.get('angle_tol')
                        tol_j = float(tol_j_val) if tol_j_val is not None else float(angle_tol_rad)
                        angle_j_ok = (aj_val is not None) and (abs(aj_val - aj_ref) <= tol_j)

            # Dihedral checks (optional)
            d1_ok = True
            d2_ok = True
            if check_dihedrals:
                # dihedral 1
                d1_ref = rule.get('dihedral_1_ref')
                d1_ai = rule.get('dihedral_1_anchor_i')
                d1_aj = rule.get('dihedral_1_anchor_j')
                if d1_ref is not None and d1_ai and d1_aj:
                    p_ai = _get_atom_coords(res1_num, d1_ai, atomscpu[i])
                    p_aj = _get_atom_coords(res2_num, d1_aj, atomscpu[j])
                    if p_ai is not None and p_aj is not None:
                        d1_val = _calculate_dihedral(p_ai, p_i, p_j, p_aj)
                        d1_val_local = d1_val
                        tol_d1_val = rule.get('dihedral_tol')
                        tol_d1 = float(tol_d1_val) if tol_d1_val is not None else float(dihedral_tol_rad)
                        d1_ok = (d1_val_local is not None) and (_periodic_angle_diff(d1_val_local, d1_ref, rule.get('dihedral_1_planar', False)) <= tol_d1)
                # dihedral 2 (optional second)
                d2_ref = rule.get('dihedral_2_ref')
                d2_ai = rule.get('dihedral_2_anchor_i')
                d2_aj = rule.get('dihedral_2_anchor_j')
                if d2_ref is not None and d2_ai and d2_aj:
                    p_ai2 = _get_atom_coords(res1_num, d2_ai, atomscpu[i])
                    p_aj2 = _get_atom_coords(res2_num, d2_aj, atomscpu[j])
                    if p_ai2 is not None and p_aj2 is not None:
                        d2_val = _calculate_dihedral(p_ai2, p_i, p_j, p_aj2)
                        d2_val_local = d2_val
                        tol_d2_val = rule.get('dihedral_tol')
                        tol_d2 = float(tol_d2_val) if tol_d2_val is not None else float(dihedral_tol_rad)
                        d2_ok = (d2_val_local is not None) and (_periodic_angle_diff(d2_val_local, d2_ref, rule.get('dihedral_2_planar', False)) <= tol_d2)

            is_valid = dist_ok and angle_i_ok and angle_j_ok and d1_ok and d2_ok

            report = dict(base_report)
            report.update({
                'atom1_name': atom1_name,
                'atom2_name': atom2_name,
                'atom1_idx': int(atom1_idx),
                'atom2_idx': int(atom2_idx),
                'distance': float(dist.item()),
                'distance_ref': float(dist_ref) if dist_ref is not None else None,
                'distance_ok': bool(dist_ok),
                'angle_i_ok': bool(angle_i_ok),
                'angle_j_ok': bool(angle_j_ok),
                'dihedral_1_ok': bool(d1_ok),
                'dihedral_2_ok': bool(d2_ok),
                'is_valid': bool(is_valid),
                'rule_idx': int(rule_idx),
                # Optional geometry values in radians (for diagnostics / downstream reporting)
                'angle_i': float(angle_i_val.item()) if angle_i_val is not None else None,
                'angle_i_ref': float(rule.get('angle_i_ref')) if rule.get('angle_i_ref') is not None else None,
                'angle_j': float(angle_j_val.item()) if angle_j_val is not None else None,
                'angle_j_ref': float(rule.get('angle_j_ref')) if rule.get('angle_j_ref') is not None else None,
                'dihedral_1': float(d1_val.item()) if d1_val is not None else None,
                'dihedral_1_ref': float(rule.get('dihedral_1_ref')) if rule.get('dihedral_1_ref') is not None else None,
                'dihedral_2': float(d2_val.item()) if d2_val is not None else None,
                'dihedral_2_ref': float(rule.get('dihedral_2_ref')) if rule.get('dihedral_2_ref') is not None else None,
            })
            if is_valid:
                chosen_report = report
                break
            if chosen_report is None:
                chosen_report = report  # keep first failing report for diagnostics

        if chosen_report is not None:
            if chosen_report['is_valid'] or include_invalid:
                results.append(chosen_report)

    return results
