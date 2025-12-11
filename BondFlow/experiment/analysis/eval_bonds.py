import argparse
import csv
import os
from typing import Dict, List, Set, Tuple, Optional

import torch
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser

from rfdiff.chemical import num2aa
from BondFlow.data.utils import parse_cif_structure
from BondFlow.data.link_utils import get_valid_links


def read_bonds_txt(path: str, pdb_idx: List[Tuple[str, str]]) -> Set[Tuple[int, int]]:
    """
    Reads bonds_x.txt saved by _save_bond_info and returns a set of residue index pairs (i,j), i<j.
    """
    pair_set: Set[Tuple[int, int]] = set()
    idx_map: Dict[Tuple[str, str], int] = {tuple(map(str, k)): int(k[1]) for i, k in enumerate(pdb_idx)}
    print(idx_map)
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            c1 = (row['res1_chain'] or '').strip()
            c2 = (row['res2_chain'] or '').strip()
            r1 = (row['res1_idx'] or '').strip()
            r2 = (row['res2_idx'] or '').strip()
            key1 = (c1, r1)
            key2 = (c2, r2)
            if key1 in idx_map and key2 in idx_map:
                i = idx_map[key1]
                j = idx_map[key2]
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                pair_set.add((i, j))
    return pair_set


def compute_scores_for_pairs(reports: List[dict]) -> Dict[Tuple[int, int], Tuple[float, int]]:
    """
    Build a map: (i,j) -> (score, label), with label=1 if is_valid else 0.
    Score is a continuous surrogate based on distance margin and geometry flags.
    """
    scores: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for rep in reports:
        i = int(rep['i'])
        j = int(rep['j'])
        if i > j:
            i, j = j, i
        dist = float(rep.get('distance', 1e9))
        dref = rep.get('distance_ref', None)
        if dref is not None:
            dref = float(dref)
            # Margin: higher is better, clipped at 0 for outside margin
            # This uses a soft allowance similar to get_valid_links thresholding
            margin = max(0.0, (dref * 1.2 + 0.5) - dist)
            base_score = margin
        else:
            # No spec rule: use inverse distance to allow relative ranking
            base_score = 1.0 / max(1e-6, dist)

        # Penalize failed geometry checks if present
        penalty = 1.0
        if rep.get('angle_i_ok') is False:
            penalty *= 0.8
        if rep.get('angle_j_ok') is False:
            penalty *= 0.8
        if rep.get('dihedral_1_ok') is False:
            penalty *= 0.7
        if rep.get('dihedral_2_ok') is False:
            penalty *= 0.7
        score = base_score * penalty
        label = 1 if bool(rep.get('is_valid', False)) else 0
        scores[(i, j)] = (float(score), int(label))
    return scores


def auc_from_scores(scores: List[float], labels: List[int]) -> float:
    """
    Compute ROC AUC using the rank statistic formula.
    """
    import numpy as np

    scores_np = np.array(scores, dtype=float)
    labels_np = np.array(labels, dtype=int)
    pos = labels_np == 1
    neg = labels_np == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    # argsort ranks (average ranks for ties)
    order = scores_np.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores_np) + 1)

    # handle ties: average ranks for equal scores
    # find groups of equal scores
    unique_scores, inverse, counts = np.unique(scores_np, return_inverse=True, return_counts=True)
    accum = 0
    for k, cnt in enumerate(counts):
        if cnt > 1:
            idxs = np.where(inverse == k)[0]
            avg_rank = ranks[idxs].mean()
            ranks[idxs] = avg_rank
        accum += cnt

    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _compute_chain_terminals(pdb_idx: List[Tuple[str, str]]) -> Set[int]:
    """Return indices that are chain termini (first and last residue of each chain)."""
    first_seen: Dict[str, int] = {}
    last_seen: Dict[str, int] = {}
    for k, (chain_id, _res) in enumerate(pdb_idx):
        if chain_id not in first_seen:
            first_seen[chain_id] = k
        last_seen[chain_id] = k
    terminals: Set[int] = set()
    for cid in first_seen:
        terminals.add(first_seen[cid])
        terminals.add(last_seen[cid])
    return terminals


def _classify_type(i: int, j: int, res1_num: int, res2_num: int, atom1_name: str, atom2_name: str, terminals: Set[int]) -> str:
    a1 = (atom1_name or '').strip().upper()
    a2 = (atom2_name or '').strip().upper()
    aa1 = num2aa[int(res1_num)] if 0 <= int(res1_num) < len(num2aa) else ''
    aa2 = num2aa[int(res2_num)] if 0 <= int(res2_num) < len(num2aa) else ''

    # Disulfide: CYS SG - CYS SG
    if aa1 == 'CYS' and aa2 == 'CYS' and a1 == 'SG' and a2 == 'SG':
        return 'disulfide'

    # Designed termini (head/tail): both residues must be chain termini AND atoms are backbone N/C
    if (i in terminals and j in terminals) and (a1 in {'N', 'C'} and a2 in {'N', 'C'}):
        return 'head_tail'

    # Isopeptide (amide sidechain crosslink): Lys NZ with acidic/amidic side-chain heavy atoms
    partner_acid_atoms = {'CG', 'CD', 'OE1', 'OE2', 'OD1', 'OD2', 'NE2', 'ND2'}
    partner_acid_res = {'ASP', 'GLU', 'ASN', 'GLN'}
    if (a1 == 'NZ' and aa1 == 'LYS' and (aa2 in partner_acid_res or a2 in partner_acid_atoms)) or \
       (a2 == 'NZ' and aa2 == 'LYS' and (aa1 in partner_acid_res or a1 in partner_acid_atoms)):
        return 'isopeptide'

    # Lactone-like (ester sidechain crosslink): Ser/Thr/Tyr O* with Asp/Glu carbonyl sidechain
    oxy_atoms = {'OG', 'OG1', 'OG2', 'OH'}
    acid_res = {'ASP', 'GLU'}
    acid_carb_atoms = {'CG', 'CD', 'OE1', 'OE2', 'OD1', 'OD2'}
    if ((a1 in oxy_atoms and aa1 in {'SER', 'THR', 'TYR'}) and (aa2 in acid_res and a2 in acid_carb_atoms)) or \
       ((a2 in oxy_atoms and aa2 in {'SER', 'THR', 'TYR'}) and (aa1 in acid_res and a1 in acid_carb_atoms)):
        return 'lactone'

    return 'other'


def _is_backbone_peptide_pair(i: int, j: int, atom1_name: str, atom2_name: str, pdb_idx: List[Tuple[str, str]]) -> bool:
    """Identify canonical backbone peptide bonds: same chain, |i-j|==1, atoms are C and N (either order)."""
    if abs(i - j) != 1:
        return False
    chain_i, _ = pdb_idx[i]
    chain_j, _ = pdb_idx[j]
    if chain_i != chain_j:
        return False
    a1 = (atom1_name or '').strip().upper()
    a2 = (atom2_name or '').strip().upper()
    return (a1 == 'C' and a2 == 'N') or (a1 == 'N' and a2 == 'C')


def compute_prf_per_type(reports: List[dict], pred_pairs: Set[Tuple[int, int]], terminals: Set[int], pdb_idx: List[Tuple[str, str]]):
    # Build truth per pair: label and type based on rule atoms
    truth: Dict[Tuple[int, int], Tuple[int, str]] = {}
    for rep in reports:
        i = int(rep['i']); j = int(rep['j'])
        if i > j: i, j = j, i
        label = 1 if bool(rep.get('is_valid', False)) else 0
        atom1 = rep.get('atom1_name'); atom2 = rep.get('atom2_name')
        # Exclude canonical backbone peptide bonds from per-type accounting
        if _is_backbone_peptide_pair(i, j, atom1, atom2, pdb_idx):
            continue
        t = _classify_type(i, j, int(rep['res1_num']), int(rep['res2_num']), atom1, atom2, terminals)
        truth[(i, j)] = (label, t)

    # Categories to report (fixed order)
    cats = ['disulfide', 'isopeptide', 'lactone', 'head_tail', 'other', 'all']
    counts = {c: {'tp':0, 'fp':0, 'fn':0, 'tn':0} for c in cats}
    pairs_all = list(truth.keys())
    for pair, (label, t) in truth.items():
        pred = pair in pred_pairs
        buckets = [t, 'all'] if t in counts else ['other', 'all']
        for cat in buckets:
            if pred and label == 1:
                counts[cat]['tp'] += 1
            elif pred and label == 0:
                counts[cat]['fp'] += 1
            elif (not pred) and label == 1:
                counts[cat]['fn'] += 1
            else:
                counts[cat]['tn'] += 1

    # Derive precision/recall/F1
    metrics = {}
    for cat, c in counts.items():
        tp, fp, fn = c['tp'], c['fp'], c['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[cat] = (prec, rec, f1, c['tp'], c['fp'], c['fn'], c['tn'])
    return metrics


def _load_structure_and_feats(struct_path: str):
    """
    读取结构文件并构建用于 get_valid_links 的输入：
      - 支持 .pdb / .cif / .mmcif
      - 返回 (seq_tensor, coords_tensor, pdb_idx_list)
    """
    ext = os.path.splitext(struct_path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("pdb_structure", struct_path)
    out = parse_cif_structure(structure, {}, parse_hetatom=False, parse_link=False, link_csv_path=None)

    import numpy as _np
    import torch as _torch

    seq = _torch.from_numpy(_np.array(out["seq"]))
    coords = _torch.from_numpy(out["xyz"])
    pdb_idx = out["pdb_idx"]
    # 前面添加与第一个元素一致，后面也添加一个与最后一个元素一致
    seq = _torch.cat([seq[:1], seq, seq[-1:]], dim=0)
    coords = _torch.cat([coords[:1], coords, coords[-1:]], dim=0)
    pdb_idx = [pdb_idx[0]] + pdb_idx + [pdb_idx[-1]]
    return seq, coords, pdb_idx


def eval_bonds_for_pair(
    struct_path: str,
    bonds_path: str,
    link_csv: str,
) -> Tuple[Dict, Dict[str, Tuple[float, float, float, int, int, int, int]]]:
    """
    对单个结构及其 bonds_txt 进行评估。

    返回:
        summary: dict，按结构整体汇总的 TP/FP/FN/TN、AUC、P/R/F1(all)
        prf: dict，键为类型（disulfide/isopeptide/.../all），值为
             (prec, rec, f1, tp, fp, fn, tn)
    """
    seq, coords, pdb_idx = _load_structure_and_feats(struct_path)
    L = len(seq)

    head_mask = torch.zeros(L, dtype=torch.bool)
    tail_mask = torch.zeros(L, dtype=torch.bool)
    head_mask[0] = True
    tail_mask[-1] = True

    pred_pairs = read_bonds_txt(bonds_path, pdb_idx)
    ones = torch.ones((L, L), dtype=torch.float32)
    ones.fill_diagonal_(0.0)

    reports = get_valid_links(
        seq,
        coords,
        ones,
        link_csv,
        head_mask=head_mask,
        tail_mask=tail_mask,
        include_invalid=True,
    )

    terminals = _compute_chain_terminals(pdb_idx)

    # 统计混淆矩阵与 AUC：
    # - AUC 仍然在所有候选 (i,j) 上计算（包括主链肽键）
    # - TP/FP/FN/TN 与 Precision/Recall/F1 则与 per-type 的 "all" 一致，
    #   即只统计被视为 LINK 候选的非主链对（排除 canonical backbone C–N）。
    score_map = compute_scores_for_pairs(reports)
    scores_all = [v[0] for v in score_map.values()]
    labels_all = [v[1] for v in score_map.values()]
    auc = auc_from_scores(scores_all, labels_all)

    prf = compute_prf_per_type(reports, pred_pairs, terminals, pdb_idx)
    prec_all, rec_all, f1_all, tp_all, fp_all, fn_all, tn_all = prf.get(
        "all", (0.0, 0.0, 0.0, 0, 0, 0, 0)
    )

    struct_name = os.path.splitext(os.path.basename(struct_path))[0]
    summary = {
        "Structure": struct_name,
        "TP": tp_all,
        "FP": fp_all,
        "FN": fn_all,
        "TN": tn_all,
        "AUC": auc,
        "Precision_all": prec_all,
        "Recall_all": rec_all,
        "F1_all": f1_all,
    }
    return summary, prf


def eval_bonds_folder(
    struct_dir: str,
    link_csv: str,
    output_dir: str,
    save_results: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    批量评估一个目录中的结构及其 bonds txt：
      - 支持 .pdb / .cif / .mmcif
      - 假定同名 txt：<name>.txt 或 <name>_bonds.txt 或 bonds_<name>.txt

    结果:
      - summary_df: 每个结构一行
      - per_type_df: 每个结构、每种类型一行
    """
    os.makedirs(output_dir, exist_ok=True)

    summaries = []
    per_type_rows = []

    for fname in sorted(os.listdir(struct_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".pdb", ".cif", ".mmcif"]:
            continue

        struct_path = os.path.join(struct_dir, fname)
        base = os.path.splitext(fname)[0]
        candidates = [
            os.path.join(struct_dir, base + ".txt"),
            os.path.join(struct_dir, base + "_bonds.txt"),
            os.path.join(struct_dir, "bonds_" + base + ".txt"),
        ]
        bonds_path = None
        for c in candidates:
            if os.path.exists(c):
                bonds_path = c
                break

        if bonds_path is None:
            print(f"[BondEval] 未找到 {fname} 对应的 bonds txt，跳过。")
            continue

        print(f"[BondEval] Evaluating {fname} with {os.path.basename(bonds_path)}")
        summary, prf = eval_bonds_for_pair(struct_path, bonds_path, link_csv)
        summaries.append(summary)

        for cat, (prec, rec, f1, tp, fp, fn, tn) in prf.items():
            per_type_rows.append(
                {
                    "Structure": summary["Structure"],
                    "Type": cat,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                }
            )

    if not summaries:
        print("[BondEval] 未找到任何可评估的结构/文本对。")
        return None, None

    summary_df = pd.DataFrame(summaries).sort_values("Structure")
    per_type_df = pd.DataFrame(per_type_rows)

    if save_results:
        summary_path = os.path.join(output_dir, "BondEval_summary.csv")
        per_type_path = os.path.join(output_dir, "BondEval_per_type.csv")
        summary_df.to_csv(summary_path, index=False)
        per_type_df.to_csv(per_type_path, index=False)
        print(f"[BondEval] 汇总结果已保存：\n  - {summary_path}\n  - {per_type_path}")

    return summary_df, per_type_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate bond predictions: TP/FP/FN/TN and AUC')
    parser.add_argument('--cif', required=True, help='Path to final_structure_X.cif (or PDB if supported by parser)')
    parser.add_argument('--bonds', required=True, help='Path to bonds_X.txt produced by _save_bond_info')
    parser.add_argument('--link_csv', default="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/link.csv", help='Path to link.csv specification')
    parser.add_argument('--no_angles', action='store_true', help='Disable angle checks')
    parser.add_argument('--no_dihedrals', action='store_true', help='Disable dihedral checks')
    parser.add_argument('--angle_tol_deg', type=float, default=20.0, help='Default angle tolerance (deg) if rule missing')
    parser.add_argument('--dihedral_tol_deg', type=float, default=35.0, help='Default dihedral tolerance (deg) if rule missing')
    parser.add_argument('--distance_scale', type=float, default=1.2, help='Allowed multiplicative slack on ref distance')
    parser.add_argument('--distance_abs_tol', type=float, default=0.5, help='Allowed absolute slack on ref distance (Å)')
    parser.add_argument('--chain_id', default='A', help='Chain ID')
    parser.add_argument('--target_mode', default="monomer", help='Target mode')
    parser.add_argument('--crop_length', type=int, default=20, help='Crop length')
    parser.add_argument('--fixed_res', type=dict, default=None, help='Fixed res')
    parser.add_argument('--fixed_bond', type=dict, default=None, help='Fixed bond')
    args = parser.parse_args()

    # try:
    # config_file = "/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/base.yaml"
    # config_path = os.path.dirname(config_file)
    # config_path = os.path.relpath(config_path)
    # config_name = os.path.basename(config_file).split(".yaml")[0]
    # with initialize(version_base=None, config_path=config_path):
    #     cfg = compose(config_name=config_name)
    # target, pdb_parsed, contig = generate_crop_target_pdb(
    #     args.cif, args.chain_id, args.target_mode, cfg, args.crop_length, args.fixed_res,fixed_bond=args.fixed_bond,N_C_add=True
    # )
    # #parsed = process_target(args.cif, parse_hetatom=False, center=False, parse_link=False, link_csv_path=args.link_csv)
    # seq = target.full_seq # pdb_parsed['seq']
    # coords = target.full_xyz # pdb_parsed['xyz_14']
    # pdb_idx = target.full_pdb_idx # pdb_parsed['pdb_idx']
    # except Exception:
    #     # Fallback: parse as PDB
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", args.cif)
    out = parse_cif_structure(structure, {}, parse_hetatom=False, parse_link=False, link_csv_path=None)
    import numpy as _np
    import torch as _torch
    seq = _torch.from_numpy(_np.array(out['seq']))
    coords = _torch.from_numpy(out['xyz'])
    pdb_idx = out['pdb_idx']
    L = len(seq)
    head_mask = torch.zeros(L, dtype=torch.bool)
    tail_mask = torch.zeros(L, dtype=torch.bool)
    head_mask[0] = True
    tail_mask[-1] = True


    pred_pairs = read_bonds_txt(args.bonds, pdb_idx)

    
    ones = torch.ones((L, L), dtype=torch.float32)
    ones.fill_diagonal_(0.0)


    reports = get_valid_links(
        seq, coords, ones, args.link_csv,
        head_mask=head_mask, tail_mask=tail_mask,
        # distance_scale=float(args.distance_scale),
        # distance_abs_tol=float(args.distance_abs_tol),
        # check_angles=(not args.no_angles), angle_tolerance_deg=float(args.angle_tol_deg),
        # check_dihedrals=(not args.no_dihedrals), dihedral_tolerance_deg=float(args.dihedral_tol_deg),
        include_invalid=True,
    )

    # Print all geometrically valid pairs (y=1)
    terminals = _compute_chain_terminals(pdb_idx)
    valid_reports = []
    for r in reports:
        if not bool(r.get('is_valid', False)):
            continue
        i = int(r['i']); j = int(r['j'])
        atom1 = (r.get('atom1_name') or '').strip()
        atom2 = (r.get('atom2_name') or '').strip()
        if _is_backbone_peptide_pair(i, j, atom1, atom2, pdb_idx):
            continue
        valid_reports.append(r)
    if len(valid_reports) > 0:
        print("\nGeometrically valid pairs (y=1):")
        for r in valid_reports:
            i = int(r['i']); j = int(r['j'])
            res1_num = int(r['res1_num']); res2_num = int(r['res2_num'])
            aa1 = num2aa[res1_num] if 0 <= res1_num < len(num2aa) else str(res1_num)
            aa2 = num2aa[res2_num] if 0 <= res2_num < len(num2aa) else str(res2_num)
            atom1 = (r.get('atom1_name') or '').strip()
            atom2 = (r.get('atom2_name') or '').strip()
            chain1, resi1 = pdb_idx[i]
            chain2, resi2 = pdb_idx[j]
            btype = _classify_type(i, j, res1_num, res2_num, atom1, atom2, terminals)
            rule_idx = r.get('rule_idx', '-')
            dist = float(r.get('distance', float('nan')))
            print(f"  ({i:3d},{j:3d}) {chain1}{resi1}:{aa1}-{atom1} -- {chain2}{resi2}:{aa2}-{atom2}  dist={dist:.3f}  rule={rule_idx}  type={btype}")

    score_map = compute_scores_for_pairs(reports)

    # Build confusion counts using predicted positives
    tp = fp = fn = tn = 0
    for (i, j), (score, label) in score_map.items():
        pred_pos = (i, j) in pred_pairs
        if pred_pos and label == 1:
            tp += 1
        elif pred_pos and label == 0:
            fp += 1
        elif (not pred_pos) and label == 1:
            fn += 1
        else:
            tn += 1

    # Compute AUC across all pairs present in score_map
    scores_all = [v[0] for v in score_map.values()]
    labels_all = [v[1] for v in score_map.values()]
    auc = auc_from_scores(scores_all, labels_all)

    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"AUC={auc:.6f}")

    # Per-type PRF
    prf = compute_prf_per_type(reports, pred_pairs, terminals, pdb_idx)
    def _fmt(x: float) -> str:
        return f"{x:.4f}"
    print("\nPer-type metrics (precision, recall, F1, TP, FP, FN, TN):")
    for cat in ['disulfide', 'isopeptide', 'lactone', 'head_tail', 'other', 'all']:
        prec, rec, f1, tp_c, fp_c, fn_c, tn_c = prf.get(cat, (0.0,0.0,0.0,0,0,0,0))
        print(f"{cat:10s} P={_fmt(prec)} R={_fmt(rec)} F1={_fmt(f1)} | TP={tp_c} FP={fp_c} FN={fn_c} TN={tn_c}")


if __name__ == '__main__':
    main()



