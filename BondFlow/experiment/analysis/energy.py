import os
import time
import pandas as pd
import numpy as np
import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.core.chemical import VariantType
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.id import AtomID
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from Bio.PDB import MMCIFParser, PDBIO

# ======================== 核心辅助函数 ========================

def _extra_res_fa_paths():
    """
    Extra residue params shipped with this repo (for covalent LINKs like SER/THR ester bonds).
    """
    base = os.path.join(os.path.dirname(__file__), "rosetta_params")
    return [
        os.path.join(base, "SERX.params"),
        os.path.join(base, "THRX.params"),
        os.path.join(base, "TYRX.params"),
    ]


def _init_pyrosetta_once():
    """
    Initialize PyRosetta exactly once per process, loading our extra residue types.
    Safe to call multiple times.
    """
    import pyrosetta
    from pyrosetta import rosetta

    if rosetta.basic.was_init_called():
        return

    flags = "-ignore_unrecognized_res true -mute all -multithreading:total_threads 1"
    for p in _extra_res_fa_paths():
        if os.path.exists(p):
            flags += f" -extra_res_fa {p}"
    pyrosetta.init(flags)

def _find_chain_index_by_pdb_letter(pose, chain_letter: str):
    """
    Return pose chain index (1..num_chains) whose PDB chain ID matches chain_letter.
    If pose has no pdb_info or not found, returns None.
    """
    try:
        pdb_info = pose.pdb_info()
    except Exception:
        pdb_info = None
    if not pdb_info:
        return None

    chain_letter = (chain_letter or "").strip()
    if not chain_letter:
        return None

    for chain_idx in range(1, pose.num_chains() + 1):
        start = pose.chain_begin(chain_idx)
        try:
            if pdb_info.chain(start) == chain_letter:
                return chain_idx
        except Exception:
            continue
    return None


def _ligand_pose_from_chainA_or_all(pose, ligand_chain_letter: str = "A"):
    """
    For complexes: return Pose corresponding to chain with PDB chain ID == ligand_chain_letter.
    For monomers (single chain) or if pdb_info missing: return full pose.
    """
    if pose.num_chains() <= 1:
        return pose.clone()

    chain_idx = _find_chain_index_by_pdb_letter(pose, ligand_chain_letter)
    if chain_idx is None:
        # Fallback: assume chain 1 is ligand if not annotated.
        chain_idx = 1
    return pose.split_by_chain(chain_idx)


def _make_ligand_movemap(pose, ligand_chain_letter: str = "A"):
    """
    MoveMap that allows backbone/sidechain moves only on ligand chain (PDB chain letter).
    If cannot identify, defaults to allowing moves on chain 1.
    """
    from pyrosetta import rosetta

    mm = rosetta.core.kinematics.MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    mm.set_jump(False)

    # Determine ligand residues in the full pose:
    chain_idx = _find_chain_index_by_pdb_letter(pose, ligand_chain_letter)
    if chain_idx is None:
        chain_idx = 1

    start = pose.chain_begin(chain_idx)
    end = pose.chain_end(chain_idx)
    for i in range(start, end + 1):
        mm.set_bb(i, True)
        mm.set_chi(i, True)
    return mm


def _compute_pnear_from_ensemble(energies, rmsds, kT: float = 1.0, rmsd_lambda: float = 1.5) -> float:
    """
    PNear = sum_i exp(-(E_i-Emin)/kT) * exp(-(r_i/lambda)^2) / sum_i exp(-(E_i-Emin)/kT)
    """
    if len(energies) == 0:
        return float("nan")
    if kT <= 0 or rmsd_lambda <= 0:
        return float("nan")

    E = np.asarray(energies, dtype=float)
    R = np.asarray(rmsds, dtype=float)
    Emin = np.min(E)
    w = np.exp(-(E - Emin) / float(kT))
    near = np.exp(-np.square(R / float(rmsd_lambda)))
    denom = np.sum(w)
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * near) / denom)

def _is_head_to_tail_backbone_cyclized(pose) -> bool:
    """
    Detect canonical N-terminus <-> C-terminus backbone cyclization (polymeric connection).
    """
    try:
        n = pose.total_residue()
        if n < 2:
            return False
        r1 = pose.residue(1)
        rn = pose.residue(n)
        # For a head-to-tail cyclized peptide, residue 1 has LOWER connected to n, and residue n has UPPER connected to 1.
        return (r1.connected_residue_at_lower() == n) and (rn.connected_residue_at_upper() == 1)
    except Exception:
        return False


def _default_link_csv_path():
    # energy.py is BondFlow/experiment/analysis/energy.py, so ../../config/link.csv is BondFlow/config/link.csv
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "link.csv"))


def _load_link_rules(link_csv_path: str | None = None):
    """
    Load link geometry rules from BondFlow/config/link.csv.
    Returns dict keyed by (res1, res2, atom1, atom2) with geometric params.
    """
    path = link_csv_path or _default_link_csv_path()
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    rules = {}
    for _, row in df.iterrows():
        try:
            key = (
                str(row["res1"]).strip().upper(),
                str(row["res2"]).strip().upper(),
                str(row["atom1"]).strip().upper(),
                str(row["atom2"]).strip().upper(),
            )
            rules[key] = {
                "avg_distance": float(row["avg_distance"]),
                "angle_i_ref_deg": float(row["angle_i_ref_deg"]) if str(row.get("angle_i_ref_deg", "")).strip() != "" else None,
                "angle_i_anchor": str(row.get("angle_i_anchor", "")).strip().upper(),
                "angle_j_ref_deg": float(row["angle_j_ref_deg"]) if str(row.get("angle_j_ref_deg", "")).strip() != "" else None,
                "angle_j_anchor": str(row.get("angle_j_anchor", "")).strip().upper(),
                "dihedral_1_ref_deg": float(row["dihedral_1_ref_deg"]) if str(row.get("dihedral_1_ref_deg", "")).strip() != "" else None,
                "dihedral_1_anchor_i": str(row.get("dihedral_1_anchor_i", "")).strip().upper(),
                "dihedral_1_anchor_j": str(row.get("dihedral_1_anchor_j", "")).strip().upper(),
            }
        except Exception:
            continue
    return rules


def _iter_pdb_link_records(pose, pdb_path):
    """
    Yield LINK-like bond specifications from a PDB file:
    (p1, atom1, p2, atom2, res1_name3, res2_name3)
    """
    if not pdb_path or not os.path.exists(pdb_path):
        return
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("LINK"):
                continue
            try:
                a1 = line[12:16].strip()
                c1 = line[21].strip() if line[21].strip() else line[20].strip()
                r1 = int(line[22:26].strip())

                a2 = line[42:46].strip()
                c2 = line[51].strip() if line[51].strip() else line[50].strip()
                r2 = int(line[52:56].strip())

                p1 = find_residue_index(pose, c1, r1)
                p2 = find_residue_index(pose, c2, r2)
                if p1 == 0 or p2 == 0:
                    continue
                res1 = pose.residue(p1).name3()
                res2 = pose.residue(p2).name3()
                yield (p1, a1, p2, a2, res1, res2)
            except Exception:
                continue


def _iter_bond_csv_records(pose, bond_file):
    """
    Yield custom bond specifications from the CSV bond file used by apply_custom_bonds():
    (p1, atom1, p2, atom2, res1_name3, res2_name3)
    """
    if not bond_file or not os.path.exists(bond_file):
        return
    try:
        df = pd.read_csv(bond_file)
    except Exception:
        return
    needed = ["res1_chain", "res1_idx", "atom1_name", "res2_chain", "res2_idx", "atom2_name"]
    if not all(c in df.columns for c in needed):
        return
    for _, row in df.iterrows():
        try:
            if "is_valid" in row and str(row["is_valid"]) == "0":
                continue
            c1, r1 = row["res1_chain"], int(row["res1_idx"])
            a1 = str(row["atom1_name"]).strip()
            c2, r2 = row["res2_chain"], int(row["res2_idx"])
            a2 = str(row["atom2_name"]).strip()
            p1 = find_residue_index(pose, str(c1).strip(), r1)
            p2 = find_residue_index(pose, str(c2).strip(), r2)
            if p1 == 0 or p2 == 0:
                continue
            res1 = pose.residue(p1).name3()
            res2 = pose.residue(p2).name3()
            yield (p1, a1, p2, a2, res1, res2)
        except Exception:
            continue


def _best_link_rule(rules, res1, res2, atom1, atom2):
    """
    Find the best-matching rule in link.csv, supporting ALL wildcards and swapped order.
    Returns (rule_dict, flipped_bool).
    """
    res1 = (res1 or "").strip().upper()
    res2 = (res2 or "").strip().upper()
    atom1 = (atom1 or "").strip().upper()
    atom2 = (atom2 or "").strip().upper()

    # exact first
    key = (res1, res2, atom1, atom2)
    if key in rules:
        return rules[key], False
    # wildcard variants
    candidates = [
        (("ALL", res2, atom1, atom2), False),
        ((res1, "ALL", atom1, atom2), False),
        (("ALL", "ALL", atom1, atom2), False),
    ]
    for k, flipped in candidates:
        if k in rules:
            return rules[k], flipped

    # swapped
    key2 = (res2, res1, atom2, atom1)
    if key2 in rules:
        return rules[key2], True
    candidates2 = [
        (("ALL", res1, atom2, atom1), True),
        ((res2, "ALL", atom2, atom1), True),
        (("ALL", "ALL", atom2, atom1), True),
    ]
    for k, flipped in candidates2:
        if k in rules:
            return rules[k], flipped

    return None, False


def _iter_crosslinks_in_pose(pose):
    """
    Detect covalent links in a pose by scanning residue pairs that are bonded but not polymer neighbors.
    Yields tuples: (p1, atom1, p2, atom2, res1_name3, res2_name3)
    """
    conf = pose.conformation()
    n = pose.total_residue()
    for i in range(1, n + 1):
        ri = pose.residue(i)
        for j in range(i + 1, n + 1):
            if j == i + 1:
                continue
            rj = pose.residue(j)
            if not ri.is_bonded(rj):
                continue
            try:
                conns = list(ri.connections_to_residue(rj))
            except Exception:
                conns = []
            for cid in conns:
                try:
                    cid = int(cid)
                    a1 = ri.atom_name(ri.residue_connect_atom_index(cid)).strip()
                    partner_atom = ri.inter_residue_connection_partner(cid, conf)  # AtomID
                    pj = int(partner_atom.rsd())
                    aj = pose.residue(pj).atom_name(int(partner_atom.atomno())).strip()
                    yield (i, a1, pj, aj, ri.name3(), pose.residue(pj).name3())
                except Exception:
                    continue


def add_link_constraints_from_rules(
    pose,
    link_specs,
    link_csv_path: str | None = None,
    dist_sigma: float = 0.10,
    angle_sigma_deg: float = 10.0,
    dihedral_sigma_deg: float = 20.0,
):
    """
    Add explicit geometric constraints for crosslinks, driven by BondFlow/config/link.csv rules.
    This helps prevent covalent links from drifting to unreasonable geometries during relax/sampling.

    Currently adds:
    - AtomPairConstraint on the bonded atom pair (distance)
    - AngleConstraint (optional if rule provides anchors)
    - DihedralConstraint (optional if rule provides anchors)
    """
    from pyrosetta import rosetta

    rules = _load_link_rules(link_csv_path)
    if not rules:
        return 0

    sigma_d = float(dist_sigma)
    sigma_a = float(angle_sigma_deg) * np.pi / 180.0
    sigma_t = float(dihedral_sigma_deg) * np.pi / 180.0

    added = 0
    for (p1, a1, p2, a2, r1n3, r2n3) in link_specs:
        rule, flipped = _best_link_rule(rules, r1n3, r2n3, a1, a2)
        if rule is None:
            continue

        # flip if the rule matched swapped order
        if flipped:
            p1, a1, p2, a2, r1n3, r2n3 = p2, a2, p1, a1, r2n3, r1n3

        r1 = pose.residue(int(p1))
        r2 = pose.residue(int(p2))
        a1u = a1.strip().upper()
        a2u = a2.strip().upper()
        if not r1.has(a1u) or not r2.has(a2u):
            continue

        id1 = AtomID(r1.atom_index(a1u), int(p1))
        id2 = AtomID(r2.atom_index(a2u), int(p2))

        # distance constraint
        d0 = float(rule["avg_distance"])
        func_d = rosetta.core.scoring.func.HarmonicFunc(d0, sigma_d)
        pose.add_constraint(rosetta.core.scoring.constraints.AtomPairConstraint(id1, id2, func_d))
        added += 1

        # optional angle constraints
        ai_anchor = (rule.get("angle_i_anchor") or "").strip().upper()
        aj_anchor = (rule.get("angle_j_anchor") or "").strip().upper()
        if ai_anchor and r1.has(ai_anchor):
            id0 = AtomID(r1.atom_index(ai_anchor), int(p1))
            ang0 = float(rule.get("angle_i_ref_deg") or 120.0) * np.pi / 180.0
            func_a = rosetta.core.scoring.func.HarmonicFunc(ang0, sigma_a)
            pose.add_constraint(rosetta.core.scoring.constraints.AngleConstraint(id0, id1, id2, func_a))
        if aj_anchor and r2.has(aj_anchor):
            id3 = AtomID(r2.atom_index(aj_anchor), int(p2))
            ang1 = float(rule.get("angle_j_ref_deg") or 120.0) * np.pi / 180.0
            func_a2 = rosetta.core.scoring.func.HarmonicFunc(ang1, sigma_a)
            pose.add_constraint(rosetta.core.scoring.constraints.AngleConstraint(id1, id2, id3, func_a2))

        # optional dihedral constraint
        di_ai = (rule.get("dihedral_1_anchor_i") or "").strip().upper()
        di_aj = (rule.get("dihedral_1_anchor_j") or "").strip().upper()
        if di_ai and di_aj and r1.has(di_ai) and r2.has(di_aj):
            id0 = AtomID(r1.atom_index(di_ai), int(p1))
            id3 = AtomID(r2.atom_index(di_aj), int(p2))
            tor0 = float(rule.get("dihedral_1_ref_deg") or 180.0) * np.pi / 180.0
            func_t = rosetta.core.scoring.func.CircularHarmonicFunc(tor0, sigma_t)
            pose.add_constraint(rosetta.core.scoring.constraints.DihedralConstraint(id0, id1, id2, id3, func_t))

    return added


def _measure_link_geometry(pose, link_specs, link_csv_path: str | None = None):
    """
    Measure crosslink bond distances (and compare to link.csv avg_distance if available).
    Returns summary dict for quick sanity checks.
    """
    rules = _load_link_rules(link_csv_path)
    dists = []
    devs = []
    matched = 0
    for (p1, a1, p2, a2, r1n3, r2n3) in link_specs:
        try:
            r1 = pose.residue(int(p1))
            r2 = pose.residue(int(p2))
            a1u = a1.strip().upper()
            a2u = a2.strip().upper()
            if not r1.has(a1u) or not r2.has(a2u):
                continue
            xyz1 = r1.xyz(a1u)
            xyz2 = r2.xyz(a2u)
            dist = float((xyz1 - xyz2).norm())
            dists.append(dist)

            rule, flipped = _best_link_rule(rules, r1n3, r2n3, a1, a2) if rules else (None, False)
            if rule is not None:
                matched += 1
                d0 = float(rule["avg_distance"])
                devs.append(abs(dist - d0))
        except Exception:
            continue

    def _stats(arr):
        if not arr:
            return {"min": float("nan"), "mean": float("nan"), "max": float("nan"), "std": float("nan")}
        a = np.asarray(arr, dtype=float)
        return {"min": float(np.min(a)), "mean": float(np.mean(a)), "max": float(np.max(a)), "std": float(np.std(a))}

    out = {}
    out["Crosslink_n"] = int(len(dists))
    out["Crosslink_rule_matched_n"] = int(matched)
    s1 = _stats(dists)
    out["Crosslink_dist_min"] = s1["min"]
    out["Crosslink_dist_mean"] = s1["mean"]
    out["Crosslink_dist_max"] = s1["max"]
    out["Crosslink_dist_std"] = s1["std"]
    s2 = _stats(devs)
    out["Crosslink_dev_mean"] = s2["mean"]
    out["Crosslink_dev_max"] = s2["max"]
    return out


def _inject_link_records_into_pdb(in_pdb_path: str, out_pdb_path: str):
    """
    Copy LINK records from the original PDB into the relaxed PDB (so viewers know covalent links).
    Safe no-op if input has no LINK lines.
    """
    try:
        if not (in_pdb_path and os.path.exists(in_pdb_path) and out_pdb_path and os.path.exists(out_pdb_path)):
            return False
        link_lines = []
        with open(in_pdb_path, "r") as f:
            for line in f:
                if line.startswith("LINK"):
                    link_lines.append(line.rstrip("\n"))
        if not link_lines:
            return False
        with open(out_pdb_path, "r") as f:
            out_lines = [ln.rstrip("\n") for ln in f]

        # insert after any header-like records; if none, insert at top
        insert_idx = 0
        for idx, ln in enumerate(out_lines[:50]):
            if ln.startswith(("HEADER", "TITLE", "REMARK", "EXPDTA", "CRYST1", "MODEL")):
                insert_idx = idx + 1
        new_lines = out_lines[:insert_idx] + link_lines + out_lines[insert_idx:]
        with open(out_pdb_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        return True
    except Exception:
        return False


def _format_pdb_link_line(atom1: str, res1: str, chain1: str, seq1: int, atom2: str, res2: str, chain2: str, seq2: int, dist: float | None = None) -> str:
    """
    Create a simple PDB LINK record line (for visualization / interoperability).
    This is not used by Rosetta internally, but helps external viewers keep covalent links.
    """
    atom1 = f"{atom1:>4s}"[:4]
    atom2 = f"{atom2:>4s}"[:4]
    res1 = f"{res1:>3s}"[:3]
    res2 = f"{res2:>3s}"[:3]
    chain1 = (chain1 or " ").strip()[:1] or " "
    chain2 = (chain2 or " ").strip()[:1] or " "
    dist_str = f"{dist:5.2f}" if dist is not None else "     "
    # PDB column-ish formatting (best-effort)
    return f"LINK        {atom1} {res1} {chain1}{seq1:4d}                {atom2} {res2} {chain2}{seq2:4d}                 {dist_str}"


def _iter_head_tail_candidates_from_pose(pose):
    """
    Yield head-tail candidate link specs for each chain in the pose.
    Output: (pC, 'C', pN, 'N', resC_name3, resN_name3, chain_letter, pdb_resC, pdb_resN)
    """
    try:
        pdb_info = pose.pdb_info()
    except Exception:
        pdb_info = None

    # If pdb_info exists, group by chain letter; else treat as one chain.
    if pdb_info:
        chain_to_idx = {}
        for i in range(1, pose.total_residue() + 1):
            ch = pdb_info.chain(i)
            chain_to_idx.setdefault(ch, []).append(i)
        for ch, idxs in chain_to_idx.items():
            if len(idxs) < 2:
                continue
            pN = idxs[0]
            pC = idxs[-1]
            rN = pose.residue(pN).name3()
            rC = pose.residue(pC).name3()
            seqN = pdb_info.number(pN)
            seqC = pdb_info.number(pC)
            yield (pC, "C", pN, "N", rC, rN, ch, seqC, seqN)
    else:
        if pose.total_residue() >= 2:
            pN = 1
            pC = pose.total_residue()
            rN = pose.residue(pN).name3()
            rC = pose.residue(pC).name3()
            yield (pC, "C", pN, "N", rC, rN, " ", pC, pN)


def _ensure_head_tail_cyclized(pose, pC: int, pN: int):
    """
    Attempt to create a head-tail peptide bond between residue pC atom C and residue pN atom N.
    Removes terminal variants that would otherwise block polymeric connectivity.
    """
    from pyrosetta import rosetta

    # Remove terminus variants on endpoints.
    # NOTE: In some Rosetta builds the termini are represented as NtermProteinFull/CtermProteinFull,
    # which must also be removed to restore polymeric connectivity at N/C.
    term_variants = [VariantType.LOWER_TERMINUS_VARIANT, VariantType.UPPER_TERMINUS_VARIANT]
    if hasattr(VariantType, "NtermProteinFull"):
        term_variants.append(VariantType.NtermProteinFull)
    if hasattr(VariantType, "CtermProteinFull"):
        term_variants.append(VariantType.CtermProteinFull)

    for v in term_variants:
        try:
            if pose.residue(pN).has_variant_type(v):
                rosetta.core.pose.remove_variant_type_from_pose_residue(pose, v, pN)
        except Exception:
            pass
        try:
            if pose.residue(pC).has_variant_type(v):
                rosetta.core.pose.remove_variant_type_from_pose_residue(pose, v, pC)
        except Exception:
            pass
    # Create bond.
    try:
        pose.conformation().declare_chemical_bond(int(pC), "C", int(pN), "N")
        # Verify bond was created successfully
        if not pose.residue(int(pC)).is_bonded(pose.residue(int(pN))):
            raise RuntimeError(f"Failed to create head-to-tail bond: residue {pC} C - residue {pN} N (bond declared but not verified)")
    except Exception as e:
        raise RuntimeError(f"Failed to create head-to-tail bond: residue {pC} C - residue {pN} N: {e}")


def _find_first_crosslink_in_pose(pose):
    """
    Identify a non-adjacent covalent link in a pose.
    Returns (i, ai, j, aj, res1_name3, res2_name3) or None.
    """
    conf = pose.conformation()
    n = pose.total_residue()
    candidates = []
    for i in range(1, n + 1):
        ri = pose.residue(i)
        for j in range(i + 1, n + 1):
            # skip polymer neighbors (i,i+1)
            if j == i + 1:
                continue
            rj = pose.residue(j)
            if not ri.is_bonded(rj):
                continue
            # Get which connection(s) on residue i connect to residue j:
            try:
                conns = list(ri.connections_to_residue(rj))
            except Exception:
                conns = []
            if not conns:
                continue
            cid = int(conns[0])
            try:
                ai = ri.atom_name(ri.residue_connect_atom_index(cid)).strip()
            except Exception:
                ai = ""
            try:
                partner_atom = ri.inter_residue_connection_partner(cid, conf)  # AtomID
                aj = pose.residue(int(partner_atom.rsd())).atom_name(int(partner_atom.atomno())).strip()
            except Exception:
                aj = ""
            if ai and aj:
                candidates.append((abs(i - j), i, ai, j, aj, ri.name3(), rj.name3()))

    if not candidates:
        return None
    # Heuristic: ring-closing bond tends to have largest sequence separation.
    candidates.sort(reverse=True, key=lambda x: x[0])
    _, i, ai, j, aj, n3i, n3j = candidates[0]
    return (i, ai, j, aj, n3i, n3j)
    return None


def _genkic_loop_and_tails(nres: int, i: int, j: int):
    """
    For GenKIC, the loop segment must have anchors (connections) to residues outside the loop.
    We treat the *endpoints* (i and j) as anchors, and the residues between them as the loop.
    This avoids GenKIC failures when the loop begins/ends at termini or when tails are misclassified as anchors.
    """
    lo, hi = (i, j) if i < j else (j, i)
    # Loop residues are strictly between anchors:
    loop_res = list(range(lo + 1, hi))
    tails = []  # keep empty for now; anchors are simply residues not in the loop.

    # Need enough residues for 3 pivots and meaningful closure; otherwise GenKIC is not suitable here.
    if len(loop_res) < 4:
        return [], []
    return loop_res, tails


def _pnear_genkic_sample_unbound(
    lig_pose,
    scorefxn,
    n_decoys: int,
    kT: float,
    rmsd_lambda: float,
    link_csv_path: str | None = None,
    closure_randomize: bool = True,
    debug_constraints: bool = False,
    debug_first_n: int = 3,
    debug_link_specs: list | None = None,
    debug_link_csv_path: str | None = None,
):
    """
    Experimental GenKIC sampler for cyclic/crosslinked ligands.
    Uses BondFlow/config/link.csv to parameterize the closure geometry for the first detected crosslink.
    """
    from pyrosetta import rosetta

    ref = lig_pose.clone()
    link = _find_first_crosslink_in_pose(lig_pose)
    if link is None:
        raise RuntimeError("GenKIC sampler requires a cyclic/crosslinked ligand (no non-adjacent covalent bond detected).")

    i, ai, j, aj, r1n3, r2n3 = link
    rules = _load_link_rules(link_csv_path)
    # Normalize residue names to match link.csv (ASP->ASP, Ser->SER etc): use 3-letter uppercase
    key = (r1n3.strip().upper(), r2n3.strip().upper(), ai.strip().upper(), aj.strip().upper())
    if key not in rules:
        # try swapped
        key2 = (r2n3.strip().upper(), r1n3.strip().upper(), aj.strip().upper(), ai.strip().upper())
        if key2 in rules:
            # swap everything
            i, ai, j, aj, r1n3, r2n3 = j, aj, i, ai, r2n3, r1n3
            key = key2
        else:
            raise RuntimeError(f"GenKIC sampler: no geometry rule in link.csv for {r1n3}:{ai} - {r2n3}:{aj}")

    rule = rules[key]
    # pick a loop segment and tail residues for GenKIC (handles links involving termini)
    loop_res, tail_res = _genkic_loop_and_tails(lig_pose.total_residue(), i, j)
    if not loop_res:
        raise RuntimeError("GenKIC sampler: loop segment too small after selecting anchors; use quick sampler for this case.")

    # Choose 3 pivot residues
    piv1 = loop_res[0]
    piv3 = loop_res[-1]
    piv2 = loop_res[len(loop_res) // 2]
    # Use CA atoms if present, else fall back to BB atoms
    def _pivot_atom(pose, rsd):
        r = pose.residue(rsd)
        for name in ("CA", "C", "N"):
            if r.has(name):
                return name
        return "CA"

    gkc = rosetta.protocols.generalized_kinematic_closure.GeneralizedKIC()
    gkc.clear_loop_residues()
    for r in loop_res:
        gkc.add_loop_residue(int(r))
    for t in tail_res:
        gkc.add_tail_residue(int(t))
    gkc.set_pivot_atoms(int(piv1), _pivot_atom(lig_pose, piv1), int(piv2), _pivot_atom(lig_pose, piv2), int(piv3), _pivot_atom(lig_pose, piv3))
    gkc.set_selector_scorefunction(scorefxn)
    gkc.set_closure_attempts(200)
    gkc.set_ntries_before_giving_up(200)
    gkc.set_dont_fail_if_no_solution_found(True)
    gkc.set_build_ideal_geometry(False)

    # Add a generic perturber and set its effect via enum (PyRosetta build-dependent).
    # We'll use rama-prepro aware randomization for backbone sampling.
    gkc.add_perturber("set_dihedral")  # create one perturber slot
    for r in loop_res:
        gkc.add_residue_to_perturber_residue_list(int(r))
    gkc.set_perturber_iterations(1)
    eff_enum = rosetta.protocols.generalized_kinematic_closure.perturber.perturber_effect
    gkc.set_perturber_effect(1, eff_enum.randomize_backbone_by_rama_prepro)

    # Define closure using link.csv anchors/angles (best-effort).
    # For bond angles/torsion, use anchors on the same residues as provided by link.csv.
    bondlen = float(rule["avg_distance"])
    ang1 = float(rule["angle_i_ref_deg"] or 120.0)
    ang2 = float(rule["angle_j_ref_deg"] or 120.0)
    tor = float(rule["dihedral_1_ref_deg"] or 180.0)
    at1_before = rule["angle_i_anchor"] or (rule["dihedral_1_anchor_i"] or "CA")
    at2_after = rule["angle_j_anchor"] or (rule["dihedral_1_anchor_j"] or "CA")

    # Ensure anchors exist; fall back to CA/C/N.
    def _ensure_atom(pose, rsd, name):
        r = pose.residue(rsd)
        if name and r.has(name):
            return name
        for cand in ("CA", "CB", "C", "N", "O"):
            if r.has(cand):
                return cand
        return name

    at1_before = _ensure_atom(lig_pose, i, at1_before)
    at2_after = _ensure_atom(lig_pose, j, at2_after)

    gkc.close_bond(
        int(i), ai.strip(),
        int(j), aj.strip(),
        int(i), at1_before,
        int(j), at2_after,
        float(bondlen),
        float(ang1),
        float(ang2),
        float(tor),
        bool(closure_randomize),
        True,
    )

    energies, rmsds = [], []
    for k in range(int(n_decoys)):
        decoy = lig_pose.clone()
        gkc.apply(decoy)
        # local minimization to remove any residual strain:
        mm = rosetta.core.kinematics.MoveMap()
        mm.set_bb(True)
        mm.set_chi(True)
        min_mover = rosetta.protocols.minimization_packing.MinMover()
        min_mover.movemap(mm)
        min_mover.score_function(scorefxn)
        min_mover.min_type("lbfgs_armijo_nonmonotone")
        min_mover.apply(decoy)

        if debug_constraints and (k < int(debug_first_n) or k == int(n_decoys) - 1):
            snap = _constraint_debug_snapshot(decoy, scorefxn)
            geom = {}
            try:
                if debug_link_specs:
                    geom = _measure_link_geometry(decoy, debug_link_specs, link_csv_path=debug_link_csv_path)
            except Exception:
                geom = {}
            print(f"  [PNear DEBUG] genkic decoy#{k} snapshot={snap} geom={geom}", flush=True)

        rmsd = rosetta.core.scoring.CA_rmsd(ref, decoy)
        e = float(scorefxn(decoy))
        energies.append(e)
        rmsds.append(float(rmsd))

    pnear = _compute_pnear_from_ensemble(energies, rmsds, kT=kT, rmsd_lambda=rmsd_lambda)
    E = np.asarray(energies, dtype=float)
    R = np.asarray(rmsds, dtype=float)
    return {
        "PNear": float(pnear),
        "PNear_Emin": float(np.min(E)),
        "PNear_Emean": float(np.mean(E)),
        "PNear_Estd": float(np.std(E)),
        "PNear_Rmin": float(np.min(R)),
        "PNear_Rmean": float(np.mean(R)),
        "PNear_Rstd": float(np.std(R)),
        "PNear_Rmax": float(np.max(R)),
    }


def _pnear_crankshaft_sample_unbound(
    lig_pose,
    scorefxn,
    n_decoys: int,
    kT: float,
    rmsd_lambda: float,
    n_moves_per_decoy: int = 5,
    debug_constraints: bool = False,
    debug_first_n: int = 3,
    debug_link_specs: list | None = None,
    debug_link_csv_path: str | None = None,
):
    """
    Cyclic-peptide-friendly sampler that preserves arbitrary covalent crosslinks by never breaking bonds.
    Uses CrankshaftFlipMover + minimization.
    """
    from pyrosetta import rosetta

    ref = lig_pose.clone()
    crank = rosetta.protocols.cyclic_peptide.CrankshaftFlipMover()

    mm = rosetta.core.kinematics.MoveMap()
    mm.set_bb(True)
    mm.set_chi(True)
    mm.set_jump(False)

    min_mover = rosetta.protocols.minimization_packing.MinMover()
    min_mover.movemap(mm)
    min_mover.score_function(scorefxn)
    min_mover.min_type("lbfgs_armijo_nonmonotone")

    energies, rmsds = [], []
    for i in range(int(n_decoys)):
        decoy = lig_pose.clone()
        for __ in range(int(n_moves_per_decoy)):
            crank.apply(decoy)
        min_mover.apply(decoy)

        if debug_constraints and (i < int(debug_first_n) or i == int(n_decoys) - 1):
            snap = _constraint_debug_snapshot(decoy, scorefxn)
            geom = {}
            try:
                if debug_link_specs:
                    geom = _measure_link_geometry(decoy, debug_link_specs, link_csv_path=debug_link_csv_path)
            except Exception:
                geom = {}
            print(f"  [PNear DEBUG] crankshaft decoy#{i} snapshot={snap} geom={geom}", flush=True)

        rmsd = rosetta.core.scoring.CA_rmsd(ref, decoy)
        e = float(scorefxn(decoy))
        energies.append(e)
        rmsds.append(float(rmsd))

    pnear = _compute_pnear_from_ensemble(energies, rmsds, kT=kT, rmsd_lambda=rmsd_lambda)
    E = np.asarray(energies, dtype=float)
    R = np.asarray(rmsds, dtype=float)
    return {
        "PNear": float(pnear),
        "PNear_Emin": float(np.min(E)),
        "PNear_Emean": float(np.mean(E)),
        "PNear_Estd": float(np.std(E)),
        "PNear_Rmin": float(np.min(R)),
        "PNear_Rmean": float(np.mean(R)),
        "PNear_Rstd": float(np.std(R)),
        "PNear_Rmax": float(np.max(R)),
    }


def _constraint_debug_snapshot(pose, scorefxn):
    """
    Compact snapshot for verifying constraints are present and contributing to the score.
    """
    from pyrosetta import rosetta

    # Ensure energies are computed
    try:
        float(scorefxn(pose))
    except Exception:
        pass

    # Constraint count (best-effort across builds)
    n_cst = 0
    try:
        cs = pose.constraint_set()
        allc = cs.get_all_constraints()
        # In PyRosetta, this is typically a vector1<shared_ptr<const Constraint>>
        # which supports __len__ but not .size().
        try:
            n_cst = int(len(allc))
        except Exception:
            # Fallback: manual iteration
            n = 0
            for _ in allc:
                n += 1
            n_cst = int(n)
    except Exception:
        n_cst = 0

    # Weights
    w_ap = float(scorefxn.get_weight(rosetta.core.scoring.atom_pair_constraint))
    w_ang = float(scorefxn.get_weight(rosetta.core.scoring.angle_constraint))
    w_dih = float(scorefxn.get_weight(rosetta.core.scoring.dihedral_constraint))

    # Unweighted energy components from pose energies map
    e_ap = 0.0
    e_ang = 0.0
    e_dih = 0.0
    try:
        emap = pose.energies().total_energies()
        e_ap = float(emap[rosetta.core.scoring.atom_pair_constraint])
        e_ang = float(emap[rosetta.core.scoring.angle_constraint])
        e_dih = float(emap[rosetta.core.scoring.dihedral_constraint])
    except Exception:
        pass

    return {
        "n_constraints": int(n_cst),
        "w_atom_pair": w_ap,
        "w_angle": w_ang,
        "w_dihedral": w_dih,
        "E_atom_pair": e_ap,
        "E_angle": e_ang,
        "E_dihedral": e_dih,
        "E_cst_weighted": float(w_ap * e_ap + w_ang * e_ang + w_dih * e_dih),
    }

def estimate_pnear(
    pose,
    scorefxn,
    ligand_chain_letter: str = "A",
    n_decoys: int = 50,
    kT: float = 1.0,
    rmsd_lambda: float = 1.5,
    method: str = "fastrelax",
    state: str = "unbound",
    sampler: str = "quick",
    seed: int | None = None,
    # ===== Constraints (optional) =====
    link_constraints: bool = False,
    link_csv_path: str | None = None,
    constraint_weight: float = 1.0,
    constraint_dist_sigma: float = 0.10,
    constraint_angle_sigma_deg: float = 10.0,
    constraint_dihedral_sigma_deg: float = 20.0,
    auto_head_tail_constraint: bool = False,
    # ===== Debug =====
    debug_constraints: bool = False,
    debug_first_n: int = 3,
):
    """
    Estimate PNear for cyclic peptide conformational entropy.

    **Reference**: ligand coordinates as present in the input pose (i.e., bound conformation if pose is a complex).

    **Sampling `state`:**
    - 'unbound': sample ligand-only (faster; best matches "unbound -> bound" conformational selection proxy)
    - 'bound'  : sample in complex with receptor frozen (captures induced-fit / bound-basin behavior)

    In both cases we compute RMSD to the reference ligand and score the ligand alone.
    """
    from pyrosetta import rosetta

    method = (method or "").lower().strip()
    if method not in ("fastrelax", "min"):
        method = "fastrelax"

    state = (state or "").lower().strip()
    if state not in ("unbound", "bound"):
        state = "unbound"

    sampler = (sampler or "").lower().strip()
    if sampler not in ("quick", "genkic", "crankshaft", "auto"):
        sampler = "quick"

    n_decoys = int(n_decoys)
    if n_decoys <= 0:
        return {"PNear": float("nan"), "PNear_n": 0}

    # Reference ligand (after any upstream relax / bond application):
    ref_lig = _ligand_pose_from_chainA_or_all(pose, ligand_chain_letter)

    # Choose sampling pose and movemap:
    if state == "unbound":
        # Sample ligand alone: no receptor present.
        base = ref_lig.clone()
        mm = None  # will build a movemap for the ligand-only pose below
    else:
        # Sample in complex but freeze receptor: ligand-only MoveMap on the full pose.
        base = pose.clone()
        mm = _make_ligand_movemap(base, ligand_chain_letter)

# Auto sampler selection:
    if sampler == "auto":
        # 1. 检查是否为头尾骨架环化
        is_ht = _is_head_to_tail_backbone_cyclized(base)
        # 2. 检查是否有任何非邻接的共价键 (二硫键、侧链交联等)
        #    注意：base 此时已经是 ligand_pose，如果有 crosslink，说明是环或二硫键
        n_crosslinks = len(list(_iter_crosslinks_in_pose(base)))
        
        if is_ht or n_crosslinks > 0:
        #     # 标准头尾环 -> 用 GenKIC (采样效率最高)
        #     sampler = "genkic"
        # elif n_crosslinks > 0:
            # 非头尾环，但有交联 (二硫键、侧链环) -> 用 Crankshaft (最稳健，不破坏复杂连接)
            sampler = "crankshaft"
        else:
            # 既不是头尾环，也没有交联 -> 线性肽 -> 用 Quick (SmallMover)
            sampler = "quick"
            
        print(f"  [Auto Sampler] Detected topology: HT={is_ht}, Crosslinks={n_crosslinks} -> Using '{sampler}'")
    # If requested, add explicit constraints to the sampling pose itself (important for unbound sampling).
    # NOTE: constraints are added to `base` (ligand-only or complex depending on state).
    debug_link_specs = []
    if link_constraints:
        try:
            link_specs = list(_iter_crosslinks_in_pose(base))
            if auto_head_tail_constraint:
                for (pC, aC, pN, aN, rC, rN, ch, seqC, seqN) in _iter_head_tail_candidates_from_pose(base):
                    link_specs.append((pC, aC, pN, aN, rC, rN))
            debug_link_specs = list(link_specs)
            add_link_constraints_from_rules(
                base,
                link_specs,
                link_csv_path=link_csv_path,
                dist_sigma=constraint_dist_sigma,
                angle_sigma_deg=constraint_angle_sigma_deg,
                dihedral_sigma_deg=constraint_dihedral_sigma_deg,
            )
            from pyrosetta import rosetta
            scorefxn.set_weight(rosetta.core.scoring.atom_pair_constraint, float(constraint_weight))
            scorefxn.set_weight(rosetta.core.scoring.angle_constraint, float(constraint_weight))
            scorefxn.set_weight(rosetta.core.scoring.dihedral_constraint, float(constraint_weight))
        except Exception:
            # Best-effort: if constraints fail, sampling still proceeds.
            pass

    if debug_constraints:
        try:
            snap = _constraint_debug_snapshot(base, scorefxn)
        except Exception:
            snap = {}
        print(
            f"  [PNear DEBUG] begin state={state} sampler={sampler} link_constraints={bool(link_constraints)} "
            f"n_specs={len(debug_link_specs)} base_snapshot={snap}",
            flush=True,
        )

    if state == "unbound" and sampler == "genkic":
        # GenKIC can fail for some topologies (e.g., head-to-tail + other crosslinks).
        # For robustness, fall back to crankshaft if GenKIC throws.
        try:
            stats = _pnear_genkic_sample_unbound(
                lig_pose=base,
                scorefxn=scorefxn,
                n_decoys=n_decoys,
                kT=kT,
                rmsd_lambda=rmsd_lambda,
                link_csv_path=link_csv_path,
                debug_constraints=debug_constraints,
                debug_first_n=debug_first_n,
                debug_link_specs=debug_link_specs,
                debug_link_csv_path=link_csv_path,
            )
            return {
                **stats,
                "PNear_n": int(n_decoys),
                "PNear_kT": float(kT),
                "PNear_lambda": float(rmsd_lambda),
                "PNear_chain": ligand_chain_letter,
                "PNear_method": method,
                "PNear_state": state,
                "PNear_sampler": sampler,
                "PNear_sampler_used": "genkic",
            }
        except Exception as e:
            if debug_constraints:
                print(f"  [PNear DEBUG] genkic failed, fallback to crankshaft: {e}", flush=True)
            stats = _pnear_crankshaft_sample_unbound(
                lig_pose=base,
                scorefxn=scorefxn,
                n_decoys=n_decoys,
                kT=kT,
                rmsd_lambda=rmsd_lambda,
                debug_constraints=debug_constraints,
                debug_first_n=debug_first_n,
                debug_link_specs=debug_link_specs,
                debug_link_csv_path=link_csv_path,
            )
            return {
                **stats,
                "PNear_n": int(n_decoys),
                "PNear_kT": float(kT),
                "PNear_lambda": float(rmsd_lambda),
                "PNear_chain": ligand_chain_letter,
                "PNear_method": method,
                "PNear_state": state,
                "PNear_sampler": sampler,
                "PNear_sampler_used": "crankshaft",
                "PNear_fallback_reason": str(e),
            }

    if state == "unbound" and sampler == "crankshaft":
        stats = _pnear_crankshaft_sample_unbound(
            lig_pose=base,
            scorefxn=scorefxn,
            n_decoys=n_decoys,
            kT=kT,
            rmsd_lambda=rmsd_lambda,
            debug_constraints=debug_constraints,
            debug_first_n=debug_first_n,
            debug_link_specs=debug_link_specs,
            debug_link_csv_path=link_csv_path,
        )
        return {
            **stats,
            "PNear_n": int(n_decoys),
            "PNear_kT": float(kT),
            "PNear_lambda": float(rmsd_lambda),
            "PNear_chain": ligand_chain_letter,
            "PNear_method": method,
            "PNear_state": state,
            "PNear_sampler": sampler,
            "PNear_sampler_used": "crankshaft",
        }

    if sampler == "genkic" and state != "unbound":
        # Keep behavior explicit: GenKIC sampling is only implemented for ligand-only sampling for now.
        sampler = "quick"

    if mm is None:
        # Ligand-only MoveMap: allow full bb/chi on all residues in the ligand pose.
        from pyrosetta import rosetta
        mm = rosetta.core.kinematics.MoveMap()
        mm.set_bb(True)
        mm.set_chi(True)
        mm.set_jump(False)

    # Movers for sampling:
    small = rosetta.protocols.simple_moves.SmallMover(mm, 1.0, 5)  # kT=1.0, nmoves=5
    small.angle_max("H", 25.0)
    small.angle_max("E", 25.0)
    small.angle_max("L", 25.0)

    min_mover = rosetta.protocols.minimization_packing.MinMover()
    min_mover.movemap(mm)
    min_mover.score_function(scorefxn)
    min_mover.min_type("lbfgs_armijo_nonmonotone")

    relaxer = None
    if method == "fastrelax":
        from pyrosetta.rosetta.protocols.relax import FastRelax
        relaxer = FastRelax()
        relaxer.set_scorefxn(scorefxn)
        try:
            relaxer.set_movemap(mm)
        except Exception:
            pass
        try:
            relaxer.set_repeat(1)
        except Exception:
            pass

    energies = []
    rmsds = []

    for i in range(n_decoys):
        decoy = base.clone()
        if seed is not None:
            # Best-effort reproducibility. Not all stochastic sources are covered, but this helps.
            try:
                rosetta.numeric.random.rg().set_seed(int(seed) + i)
            except Exception:
                pass

        # Diversify + minimize/relax ligand only:
        small.apply(decoy)
        min_mover.apply(decoy)
        if relaxer is not None:
            relaxer.apply(decoy)

        if state == "unbound":
            lig = decoy
        else:
            lig = _ligand_pose_from_chainA_or_all(decoy, ligand_chain_letter)
            # split-by-chain loses constraints; re-add ligand-local constraints so bound-state PNear can still feel crosslinks
            if link_constraints:
                try:
                    link_specs_lig = list(_iter_crosslinks_in_pose(lig))
                    if auto_head_tail_constraint:
                        for (pC, aC, pN, aN, rC, rN, ch, seqC, seqN) in _iter_head_tail_candidates_from_pose(lig):
                            link_specs_lig.append((pC, aC, pN, aN, rC, rN))
                    add_link_constraints_from_rules(
                        lig,
                        link_specs_lig,
                        link_csv_path=link_csv_path,
                        dist_sigma=constraint_dist_sigma,
                        angle_sigma_deg=constraint_angle_sigma_deg,
                        dihedral_sigma_deg=constraint_dihedral_sigma_deg,
                    )
                except Exception:
                    pass

        if debug_constraints and (i < int(debug_first_n) or i == n_decoys - 1):
            snap_decoy = _constraint_debug_snapshot(decoy, scorefxn)
            snap_lig = _constraint_debug_snapshot(lig, scorefxn)
            geom = {}
            try:
                if debug_link_specs and state == "unbound":
                    geom = _measure_link_geometry(lig, debug_link_specs, link_csv_path=link_csv_path)
            except Exception:
                geom = {}
            print(f"  [PNear DEBUG] quick decoy#{i} decoy_snapshot={snap_decoy} lig_snapshot={snap_lig} geom={geom}", flush=True)
        try:
            rmsd = rosetta.core.scoring.all_atom_rmsd(ref_lig, lig)
        except Exception:
            # fallback: CA rmsd
            rmsd = rosetta.core.scoring.CA_rmsd(ref_lig, lig)

        e = float(scorefxn(lig))
        energies.append(e)
        rmsds.append(float(rmsd))

    pnear = _compute_pnear_from_ensemble(energies, rmsds, kT=kT, rmsd_lambda=rmsd_lambda)
    E = np.asarray(energies, dtype=float)
    R = np.asarray(rmsds, dtype=float)
    return {
        "PNear": pnear,
        "PNear_n": int(n_decoys),
        "PNear_kT": float(kT),
        "PNear_lambda": float(rmsd_lambda),
        "PNear_chain": ligand_chain_letter,
        "PNear_method": method,
        "PNear_state": state,
        "PNear_sampler": sampler,
        "PNear_Emin": float(np.min(E)) if energies else float("nan"),
        "PNear_Emean": float(np.mean(E)) if energies else float("nan"),
        "PNear_Estd": float(np.std(E)) if energies else float("nan"),
        "PNear_Rmin": float(np.min(R)) if rmsds else float("nan"),
        "PNear_Rmean": float(np.mean(R)) if rmsds else float("nan"),
        "PNear_Rstd": float(np.std(R)) if rmsds else float("nan"),
        "PNear_Rmax": float(np.max(R)) if rmsds else float("nan"),
    }


def _swap_residue_type(pose, seqpos: int, new_name: str):
    """
    Replace pose residue type at seqpos with new_name (must be present in ResidueTypeSet),
    copying existing coordinates where possible.
    """
    from pyrosetta import rosetta

    rts = pose.residue_type_set_for_pose()
    if not rts.has_name(new_name):
        raise RuntimeError(f"ResidueType '{new_name}' not found in ResidueTypeSet (did PyRosetta init load extra_res_fa?)")
    new_rt = rts.name_map(new_name)
    rosetta.core.pose.replace_pose_residue_copying_existing_coordinates(pose, seqpos, new_rt)


def _ensure_ser_thr_sidechain_connectable(pose, seqpos: int, atom_name: str) -> bool:
    """
    Ensure SER/THR sidechain hydroxyl atoms have a Rosetta 'CONNECT' so declare_chemical_bond can work.
    We do this by swapping to custom residue types SERX/THRX (loaded via -extra_res_fa).
    """
    r = pose.residue(seqpos)
    name3 = r.name3()
    atom = atom_name.strip().upper()

    if name3 == "SER" and atom == "OG":
        _swap_residue_type(pose, seqpos, "SERX")
        return True
    if name3 == "THR" and atom in ("OG1", "OG"):
        # Some PDBs may write THR hydroxyl as OG (non-standard); accept both.
        _swap_residue_type(pose, seqpos, "THRX")
        return True
    if name3 == "TYR" and atom == "OH":
        _swap_residue_type(pose, seqpos, "TYRX")
        return True

    return False


def find_residue_index(pose, chain, res_seq):
    """
    辅助函数：根据 Chain 和 ResSeq 查找 Pose 索引
    """
    # 1. 尝试直接使用 pdb2pose
    if chain:
        p = pose.pdb_info().pdb2pose(chain, res_seq)
        if p != 0: return p
    
    # 2. 遍历查找 (处理 chain 为空或 pdb2pose 失败的情况)
    candidates = []
    for i in range(1, pose.total_residue() + 1):
        if pose.pdb_info().number(i) == res_seq:
            p_chain = pose.pdb_info().chain(i)
            candidates.append((i, p_chain))
    
    if len(candidates) == 1:
        return candidates[0][0]
    elif len(candidates) > 1:
        if chain:
            for idx, ch in candidates:
                if ch == chain: return idx
    return 0


def safe_declare_bond(pose, p1, a1, p2, a2, strict=True):
    """
    健壮的化学键创建函数 (核心修复版):
    1. 自动处理二硫键
    2. 自动剥离冲突的 N-term/C-term 变体 (解决 SER:Nterm 报错)
    3. 自动添加 SIDECHAIN_CONJUGATION 变体 (解决 LYS/ASP 连接报错)
    """
    if p1 == 0 or p2 == 0: return False
    
    conf = pose.conformation()
    r1 = pose.residue(p1)
    r2 = pose.residue(p2)
    
    # --- 1. 二硫键特判 ---
    if r1.name3() == "CYS" and r2.name3() == "CYS" and a1.strip() == "SG" and a2.strip() == "SG":
        if not r1.is_bonded(r2):
            try:
                conf.form_disulfide(p1, p2)
                print(f"  [Bond Info] 二硫键创建成功: {p1}(CYS)-{p2}(CYS)")
                return True
            except: pass

    # --- 2. 准备工作：判断是否为侧链原子 ---
    backbone_atoms = ["N", "CA", "C", "O", "H", "HA"]
    is_sidechain_1 = a1.strip() not in backbone_atoms
    is_sidechain_2 = a2.strip() not in backbone_atoms

    # --- 2.5. SER/THR 羟基侧链连接：从“跳过”改为“真修复” ---
    # Rosetta 默认数据库里 SER/THR 不提供 SIDECHAIN_CONJUGATION patch，
    # 因此 declare_chemical_bond 会报 "SER doesnt have connection at OG"。
    # 我们通过自定义 residue params (SERX/THRX) 来提供 CONNECT，并在这里自动替换。
    if is_sidechain_1:
        try:
            _ensure_ser_thr_sidechain_connectable(pose, p1, a1)
        except Exception:
            if strict:
                raise
    if is_sidechain_2:
        try:
            _ensure_ser_thr_sidechain_connectable(pose, p2, a2)
        except Exception:
            if strict:
                raise
    
    # 定义需要清理的终端变体
    term_variants = [
        VariantType.LOWER_TERMINUS_VARIANT, 
        VariantType.UPPER_TERMINUS_VARIANT
    ]
    # 尝试添加额外的变体类型（如果存在）
    if hasattr(VariantType, "NtermProteinFull"):
        term_variants.append(VariantType.NtermProteinFull)
    if hasattr(VariantType, "CtermProteinFull"):
        term_variants.append(VariantType.CtermProteinFull)

    # --- 3. 修复变体冲突 ---
    # 对两个残基分别处理
    for p_idx, is_sidechain in [(p1, is_sidechain_1), (p2, is_sidechain_2)]:
        if is_sidechain:
            # A. 移除冲突的终端变体 (解决 N 端 SER 无法侧链成键的问题)
            for v in term_variants:
                if pose.residue(p_idx).has_variant_type(v):
                    try:
                        pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue(pose, v, p_idx)
                    except: pass
            
            # B. 添加侧链连接变体 (解决 LYS/ASP/GLU 无法连接的问题)
            # 注意：某些残基（如 CYS:disulfide）可能已经通过二硫键变体“占位”，再加 SIDECHAIN_CONJUGATION 会失败
            # 另外，SER 等也可能需要特殊处理
            if pose.residue(p_idx).name3() in ("SER", "THR"):
                # SER/THR 我们使用自定义 residue params (SERX/THRX)，不走 patch 变体。
                continue
            if not pose.residue(p_idx).has_variant_type(VariantType.SIDECHAIN_CONJUGATION):
                # 只有当该残基还没连接或不是特殊类型时才尝试添加
                # 如果是 CYS 且参与二硫键，通常不需要再加 SIDECHAIN_CONJUGATION
                r_name = pose.residue(p_idx).name()
                if "CYS:disulfide" in r_name:
                    continue
                
                try:
                    pyrosetta.rosetta.core.pose.add_variant_type_to_pose_residue(pose, VariantType.SIDECHAIN_CONJUGATION, p_idx)
                except Exception as e:
                    # 如果添加变体失败，打印详细信息以便调试（特别是 SER 等特殊残基）
                    r_type = pose.residue(p_idx).type()
                    print(f"  [Variant Info] 无法添加 SIDECHAIN_CONJUGATION 到 {r_type.name()} (seq={p_idx}): {e}")
                    if strict:
                        raise

    # --- 4. 建立连接 ---
    try:
        # 再次检查是否已连接
        if pose.residue(p1).is_bonded(pose.residue(p2)):
            print(f"  [Bond Check] 已存在连接: {p1} {r1.name3()} {a1} - {p2} {r2.name3()} {a2}")
            return True
        
        # 针对 SER/THR 等含羟基侧链的特殊处理
        # 很多时候 Rosetta 需要显式的连接名称或特定的 Variant
        conf.declare_chemical_bond(p1, a1, p2, a2)
        # Verify bond was created successfully
        if not pose.residue(p1).is_bonded(pose.residue(p2)):
            raise RuntimeError(f"Bond declared but not verified: {p1} {r1.name3()} {a1} - {p2} {r2.name3()} {a2}")
        print(f"  [Bond Created] 新建连接成功: {p1} {r1.name3()} {a1} - {p2} {r2.name3()} {a2}")
        return True
    except Exception as e:
        # SER/THR: 尝试在失败后再做一次“替换 + 重试”
        msg = str(e)
        if "doesnt have connection" in msg and (" OG" in msg or " OG1" in msg):
            did_fix = False
            if is_sidechain_1:
                did_fix = _ensure_ser_thr_sidechain_connectable(pose, p1, a1) or did_fix
            if is_sidechain_2:
                did_fix = _ensure_ser_thr_sidechain_connectable(pose, p2, a2) or did_fix
            if did_fix:
                conf = pose.conformation()
                conf.declare_chemical_bond(p1, a1, p2, a2)
                # Verify bond was created successfully
                if not pose.residue(p1).is_bonded(pose.residue(p2)):
                    raise RuntimeError(f"Bond declared (retry) but not verified: {p1} {pose.residue(p1).name3()} {a1} - {p2} {pose.residue(p2).name3()} {a2}")
                print(f"  [Bond Created] (Retry) 新建连接成功: {p1} {pose.residue(p1).name3()} {a1} - {p2} {pose.residue(p2).name3()} {a2}")
                return True

        # 其他类型的错误仍然打印详细信息
        r1_type = pose.residue(p1).type()
        r2_type = pose.residue(p2).type()
        print(f"  [Bond Error] 连接失败 {p1}({r1_type.name()}):{a1} - {p2}({r2_type.name()}):{a2}")
        print(f"    Error Msg: {e}")
        if strict:
            raise
        return False


def apply_pdb_links(pose, pdb_path, strict=True):
    """
    解析 PDB 文件中的 LINK 记录并应用化学键。
    """
    if not os.path.exists(pdb_path):
        return 0

    count = 0
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("LINK"):
                    print(f"  [DEBUG] Found LINK line: {line.strip()}")
                    try:
                        # 解析 PDB LINK (列索引从0开始)
                        a1 = line[12:16].strip()
                        c1 = line[21].strip() if line[21].strip() else line[20].strip()
                        r1 = int(line[22:26].strip())
                        
                        a2 = line[42:46].strip()
                        c2 = line[51].strip() if line[51].strip() else line[50].strip()
                        r2 = int(line[52:56].strip())
                        
                        print(f"  [DEBUG] Parsed: {c1}.{r1}.{a1} -> {c2}.{r2}.{a2}")

                        # 查找残基并建立连接
                        p1 = find_residue_index(pose, c1, r1)
                        p2 = find_residue_index(pose, c2, r2)
                        print(f"  [DEBUG] Residue Indices: p1={p1}, p2={p2}")

                        # Special-case: head-to-tail backbone cyclization uses polymeric C--N,
                        # which will fail if the endpoints still carry N/C-terminus variants.
                        a1u = a1.strip().upper()
                        a2u = a2.strip().upper()
                        if (a1u, a2u) in [("C", "N"), ("N", "C")]:
                            pC = int(p1) if a1u == "C" else int(p2)
                            pN = int(p2) if a2u == "N" else int(p1)
                            same_chain = True
                            try:
                                pdb_info = pose.pdb_info()
                                if pdb_info:
                                    same_chain = (pdb_info.chain(pC) == pdb_info.chain(pN))
                            except Exception:
                                pass
                            # Only do this when the endpoints look like real termini.
                            try:
                                is_term_like = pose.residue(pN).is_lower_terminus() and pose.residue(pC).is_upper_terminus()
                            except Exception:
                                # Fallback: variant-type check
                                is_term_like = (
                                    pose.residue(pN).has_variant_type(VariantType.LOWER_TERMINUS_VARIANT)
                                    or (hasattr(VariantType, "NtermProteinFull") and pose.residue(pN).has_variant_type(VariantType.NtermProteinFull))
                                ) and (
                                    pose.residue(pC).has_variant_type(VariantType.UPPER_TERMINUS_VARIANT)
                                    or (hasattr(VariantType, "CtermProteinFull") and pose.residue(pC).has_variant_type(VariantType.CtermProteinFull))
                                )

                            if same_chain and is_term_like:
                                try:
                                    _ensure_head_tail_cyclized(pose, pC=pC, pN=pN)
                                    print(f"  [Bond Created] 头尾环化成功: {pC}({pose.residue(pC).name3()}) C - {pN}({pose.residue(pN).name3()}) N")
                                    count += 1
                                    continue
                                except Exception as e:
                                    print(f"  [Bond Error] 头尾环化失败: {e}")
                                    if strict:
                                        raise
                                    continue

                        if safe_declare_bond(pose, p1, a1, p2, a2, strict=strict):
                            count += 1
                    except Exception as e:
                        print(f"  [DEBUG Link Parse Error] {e}")
                        # 解析错误可以忽略；建键错误在 strict=True 时会在 safe_declare_bond 内抛出
                        if not strict:
                            pass
    except Exception as e:
        print(f"  [Link Read Error] {e}")

    if count > 0:
        print(f"  [Link Info] 从 PDB LINK 记录成功应用了 {count} 个化学键", flush=True)
    
    return count  # <--- 关键修复：确保有返回值


def apply_custom_bonds(pose, bond_file, strict=True):
    """
    读取键连接文件 (CSV) 并应用 chemical bonds 到 pose。
    """
    if not bond_file or not os.path.exists(bond_file):
        return 0

    count = 0
    try:
        df = pd.read_csv(bond_file)
        
        # 简单检查列名
        needed = ['res1_chain', 'res1_idx', 'atom1_name', 'res2_chain', 'res2_idx', 'atom2_name']
        if not all(c in df.columns for c in needed):
            print(f"  [Warning] Bond file {os.path.basename(bond_file)} 缺少必要列")
            return 0

        for i, row in df.iterrows():
            if 'is_valid' in row and str(row['is_valid']) == '0':
                continue
            
            # 读取信息
            c1, r1 = row['res1_chain'], int(row['res1_idx'])
            a1 = row['atom1_name']
            c2, r2 = row['res2_chain'], int(row['res2_idx'])
            a2 = row['atom2_name']
            
            # 查找残基
            p1 = find_residue_index(pose, c1, r1)
            p2 = find_residue_index(pose, c2, r2)
            
            # 使用统一的安全连接函数
            if safe_declare_bond(pose, p1, a1, p2, a2, strict=strict):
                count += 1
                
        if count > 0:
            print(f"  [Bond Info] 成功应用了 {count} 个自定义化学键 ({os.path.basename(bond_file)})", flush=True)
            
    except Exception as e:
        print(f"  [Bond Error] 读取或处理键文件出错: {e}")
        if strict:
            raise
        
    return count  # <--- 关键修复：确保有返回值


def _collect_pdb_like_files(struct_folder, output_dir):
    """
    收集 PDB/CIF 文件
    """
    struct_files = []
    cif_tmp_dir = os.path.join(output_dir, "_cif_to_pdb")

    for f in os.listdir(struct_folder):
        full = os.path.join(struct_folder, f)
        if not os.path.isfile(full):
            continue

        ext = os.path.splitext(f)[1].lower()
        if ext == ".pdb":
            struct_files.append(full)
        elif ext in [".cif", ".mmcif"]:
            os.makedirs(cif_tmp_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(f))[0]
            out_pdb = os.path.join(cif_tmp_dir, base + ".pdb")
            if not os.path.exists(out_pdb):
                try:
                    parser = MMCIFParser(QUIET=True)
                    structure = parser.get_structure(base, full)
                    io = PDBIO()
                    io.set_structure(structure)
                    io.save(out_pdb)
                except:
                    continue
            struct_files.append(out_pdb)

    return struct_files

# ======================== 计算逻辑 ========================

def compute_energy_for_pdb(
    pdb_path,
    relax=True,
    bond_file=None,
    compute_pnear: bool = False,
    pnear_n: int = 50,              # PNear: 采样 decoy 数量（越大越稳，但更慢）
    pnear_kT: float = 1.0,          # PNear: Boltzmann 权重的 kT
    pnear_lambda: float = 1.5,      # PNear: “near” 的 RMSD 尺度参数 λ（Å）
    pnear_chain: str = "A",         # 复合物中环肽配体链（PDB chain letter）；单体时忽略
    pnear_method: str = "fastrelax",# quick 采样内部用：fastrelax 或 min
    pnear_state: str = "unbound",   # unbound=ligand-only（用于构象熵损失）；bound=complex(受体冻结)
    pnear_sampler: str = "quick",   # PNear 采样器：auto/quick/crankshaft/genkic（见 batch_energy 注释）
    # ===== Crosslink constraints (link.csv) =====
    link_constraints: bool = False,         # 是否对交联键加显式几何约束（强烈建议在 PNear/Relax 时打开）
    link_csv_path: str | None = None,       # 规则文件路径；默认用 BondFlow/config/link.csv
    constraint_weight: float = 1.0,         # 约束项权重（同时作用于 atom_pair/angle/dihedral）
    constraint_dist_sigma: float = 0.10,    # 距离约束 sigma (Å)
    constraint_angle_sigma_deg: float = 10.0,   # 角度约束 sigma (deg)
    constraint_dihedral_sigma_deg: float = 20.0,# 二面角约束 sigma (deg)
    constrain_in_relax: bool = True,        # Relax 时是否启用约束
    constrain_in_pnear: bool = True,        # PNear 采样时是否启用约束（unbound 时对 ligand-only）
    auto_head_tail_constraint: bool = False,# 是否自动对每条链的“头尾 C-N”添加约束（即使 PDB 没有 LINK）
    auto_head_tail_bond: bool = False,      # 是否自动创建头尾 C-N 化学键（谨慎使用；适用于确实是头尾环但缺少 LINK 的输入）
    pnear_debug_constraints: bool = False,  # 是否打印 PNear 扰动阶段“约束是否生效”的调试信息
    pnear_debug_first_n: int = 3,           # 仅打印前 N 个 decoy + 最后 1 个
    # ===== Energy component extraction =====
    extract_dslf_fa13: bool = False,        # 是否提取二硫键评分 (dslf_fa13)
    target_chain_id: str | None = None,     # 目标 chain ID，用于计算该 chain 单独的能量（稳定性指标）
    # ===== Output =====
    save_relaxed_pdb: bool = False,         # 是否保存 relax 后结构
    relaxed_pdb_dir: str | None = None,     # 保存目录（默认 output_dir/energy_results/relaxed_structures）
    output_dir: str | None = None,          # batch_energy 传入，用于构造默认 relaxed 输出目录
):
    """
    计算单个结构的总能量、结合能；可选计算 PNear（用于估计环肽构象熵相关指标）。

    PNear 相关参数说明（高层语义）：
    - pnear_state='unbound'：在游离态对环肽本体采样（更贴近“unbound -> bound”的构象熵损失）
    - pnear_state='bound'  ：在复合物中采样（受体冻结），更偏 induced-fit / bound basin
    - pnear_sampler='auto' ：头尾环用 GenKIC，否则用 crankshaft（更适合内酯/异肽/二硫等交联环）
    """
    # 初始化 PyRosetta（多进程模式下每个进程都需要 init 一次；该函数保证每个进程只 init 一次）
    _init_pyrosetta_once()
    
    scorefxn = pyrosetta.rosetta.core.scoring.get_score_function()
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    start_time = time.time()
    print(f"启动计算: {pdb_name} | PID={os.getpid()}", flush=True)
    
    try:
        pose = pose_from_file(pdb_path)
        
        # ---------- 应用自定义键 (环化/二硫键等) ----------
        bond_count = 0
        
        # 1. 应用 PDB LINK
        link_cnt = apply_pdb_links(pose, pdb_path, strict=True)
        bond_count += link_cnt # 现在 apply_pdb_links 保证返回 int

        # 2. 应用 CSV Bond 文件
        if bond_file:
            csv_cnt = apply_custom_bonds(pose, bond_file, strict=True)
            bond_count += csv_cnt # 现在 apply_custom_bonds 保证返回 int

        # ---------- Optional: Add explicit constraints for crosslinks (from link.csv) ----------
        # 注意：这些约束只影响 relax / 采样，不会改变化学键本身。
        link_specs = []
        if link_constraints:
            link_specs.extend(list(_iter_pdb_link_records(pose, pdb_path)))
            if bond_file:
                link_specs.extend(list(_iter_bond_csv_records(pose, bond_file)))
            # 额外扫描 pose 中实际存在的 crosslink（覆盖没有 LINK 行但有二硫/头尾环等情况）
            link_specs.extend(list(_iter_crosslinks_in_pose(pose)))
        # 对缺少 LINK 的“头尾环”提供兜底：把 (last C) - (first N) 加入约束集合（不一定真的成键）
        if auto_head_tail_constraint:
            for (pC, aC, pN, aN, rC, rN, ch, seqC, seqN) in _iter_head_tail_candidates_from_pose(pose):
                link_specs.append((pC, aC, pN, aN, rC, rN))

        # 如用户明确要求，可尝试自动创建头尾化学键（用于输入缺少 LINK 但几何上已闭合的情况）
        if auto_head_tail_bond:
            for (pC, aC, pN, aN, rC, rN, ch, seqC, seqN) in _iter_head_tail_candidates_from_pose(pose):
                try:
                    _ensure_head_tail_cyclized(pose, int(pC), int(pN))
                    # Verify bond was created
                    if not pose.residue(int(pC)).is_bonded(pose.residue(int(pN))):
                        print(f"  [Warning] Head-to-tail bond creation failed for {pC}({rC}) C - {pN}({rN}) N (geometry may be too far)")
                except Exception as e:
                    print(f"  [Warning] Failed to create head-to-tail bond for {pC}({rC}) C - {pN}({rN}) N: {e}")
                    # Don't fail silently - this could cause issues during relax
                    # But we continue to allow processing of other structures

        # 真正添加约束（仅当 link_constraints=True）
        if link_constraints:
            # 去重
            seen = set()
            uniq = []
            for (p1, a1, p2, a2, r1, r2) in link_specs:
                k = (int(p1), a1.strip().upper(), int(p2), a2.strip().upper())
                k2 = (k[2], k[3], k[0], k[1])
                if k in seen or k2 in seen:
                    continue
                seen.add(k)
                uniq.append((p1, a1, p2, a2, r1, r2))
            link_specs = uniq

            added_cst = add_link_constraints_from_rules(
                pose,
                link_specs,
                link_csv_path=link_csv_path,
                dist_sigma=constraint_dist_sigma,
                angle_sigma_deg=constraint_angle_sigma_deg,
                dihedral_sigma_deg=constraint_dihedral_sigma_deg,
            )
            # record for debugging
            link_constraints_added = int(added_cst)
            if added_cst > 0:
                from pyrosetta import rosetta
                scorefxn.set_weight(rosetta.core.scoring.atom_pair_constraint, float(constraint_weight))
                scorefxn.set_weight(rosetta.core.scoring.angle_constraint, float(constraint_weight))
                scorefxn.set_weight(rosetta.core.scoring.dihedral_constraint, float(constraint_weight))
        else:
            link_constraints_added = 0

        # ---------- Relax ----------
        if relax:
            from pyrosetta.rosetta.protocols.relax import FastRelax
            relaxer = FastRelax()
            relaxer.set_scorefxn(scorefxn)
            # 若用户要求“只在采样时约束、relax 不约束”，则临时把约束权重置 0
            if link_constraints and not constrain_in_relax:
                from pyrosetta import rosetta
                scorefxn.set_weight(rosetta.core.scoring.atom_pair_constraint, 0.0)
                scorefxn.set_weight(rosetta.core.scoring.angle_constraint, 0.0)
                scorefxn.set_weight(rosetta.core.scoring.dihedral_constraint, 0.0)
            relaxer.apply(pose)
            # relax 后恢复约束权重（供后续 PNear 使用）
            if link_constraints and not constrain_in_relax:
                from pyrosetta import rosetta
                scorefxn.set_weight(rosetta.core.scoring.atom_pair_constraint, float(constraint_weight))
                scorefxn.set_weight(rosetta.core.scoring.angle_constraint, float(constraint_weight))
                scorefxn.set_weight(rosetta.core.scoring.dihedral_constraint, float(constraint_weight))

        # ---------- Post-relax geometry sanity check for crosslinks ----------
        crosslink_geom = {}
        if link_specs:
            crosslink_geom = _measure_link_geometry(pose, link_specs, link_csv_path=link_csv_path)

        # ---------- Optional: save relaxed structure ----------
        if save_relaxed_pdb:
            out_dir = relaxed_pdb_dir
            if out_dir is None:
                # default: <output_dir>/energy_results/relaxed_structures
                if output_dir is not None:
                    out_dir = os.path.join(str(output_dir), "energy_results", "relaxed_structures")
                else:
                    out_dir = os.path.join(os.path.dirname(os.path.abspath(pdb_path)), "_relaxed_structures")
            os.makedirs(out_dir, exist_ok=True)
            out_pdb = os.path.join(out_dir, f"{pdb_name}_relaxed.pdb")
            try:
                pose.dump_pdb(out_pdb)
                # Preserve LINK records so external viewers know about covalent links.
                _inject_link_records_into_pdb(pdb_path, out_pdb)
                # If user requested head-tail constraint/bond, also add a synthetic LINK line for visualization.
                if auto_head_tail_constraint or auto_head_tail_bond:
                    try:
                        # append head-tail LINKs (best-effort)
                        ht_lines = []
                        for (pC, aC, pN, aN, rC, rN, ch, seqC, seqN) in _iter_head_tail_candidates_from_pose(pose):
                            d = float((pose.residue(int(pC)).xyz("C") - pose.residue(int(pN)).xyz("N")).norm())
                            ht_lines.append(_format_pdb_link_line("C", rC, ch, int(seqC), "N", rN, ch, int(seqN), dist=d))
                        if ht_lines:
                            with open(out_pdb, "r") as f:
                                lines = [ln.rstrip("\n") for ln in f]
                            # insert after headers
                            insert_idx = 0
                            for idx, ln in enumerate(lines[:50]):
                                if ln.startswith(("HEADER", "TITLE", "REMARK", "EXPDTA", "CRYST1", "MODEL")):
                                    insert_idx = idx + 1
                            lines = lines[:insert_idx] + ht_lines + lines[insert_idx:]
                            with open(out_pdb, "w") as f:
                                f.write("\n".join(lines) + "\n")
                    except Exception:
                        pass
            except Exception:
                pass

        # ---------- 能量计算 ----------
        total_energy = scorefxn(pose)
        num_chains = pose.num_chains()

        binding_energy = 0.0
        E_protein = total_energy
        E_ligand = 0.0
        
        if num_chains >= 2:
            # 获取链ID列表
            chain_ids = []
            try:
                if pose.pdb_info():
                    for chain_num in range(1, num_chains + 1):
                        chain_start = pose.chain_begin(chain_num)
                        c_id = pose.pdb_info().chain(chain_start)
                        if c_id not in chain_ids: chain_ids.append(c_id)
                else:
                    chain_ids = [chr(64 + i) for i in range(1, num_chains + 1)]
            except:
                chain_ids = [chr(64 + i) for i in range(1, num_chains + 1)]

            # 计算结合能
            if len(chain_ids) >= 2:
                # 构造 Interface 配置 (e.g., A_B)
                interface_config = f"{chain_ids[0]}_{''.join(chain_ids[1:])}"
                try:
                    interface_analyzer = InterfaceAnalyzerMover(interface_config, False, scorefxn)
                    interface_analyzer.set_pack_rounds(1)
                    interface_analyzer.set_pack_input(True)
                    interface_analyzer.set_compute_packstat(False)
                    interface_analyzer.set_pack_separated(True)
                    interface_analyzer.apply(pose)
                    binding_energy = interface_analyzer.get_interface_dG()
                except Exception as e:
                    print(f"  [Warning] InterfaceAnalyzerMover 失败 ({e})，使用传统方法", flush=True)
                    # 传统方法后备
                    p_pose = pose.split_by_chain(1)
                    l_pose = pose.split_by_chain(2)
                    binding_energy = total_energy - (scorefxn(p_pose) + scorefxn(l_pose))
            else:
                p_pose = pose.split_by_chain(1)
                l_pose = pose.split_by_chain(2)
                binding_energy = total_energy - (scorefxn(p_pose) + scorefxn(l_pose))
        
        # ---------- 计算目标 chain 单独的能量（稳定性指标）----------
        # 注意：如果 extract_dslf_fa13=True，将从单独提取的 chain 中提取，而不是从复合物中提取
        # 这样更一致，因为二硫键是目标 chain 内部的键，应该反映该 chain 自身的稳定性
        target_chain_energy = None
        target_chain_energy_per_residue = None
        target_chain_num_residues = None
        dslf_fa13 = None  # 初始化，将从目标 chain 中提取（如果启用）
        if target_chain_id is not None:
            try:
                # 获取所有 chain 的 ID
                chain_ids = []
                if pose.pdb_info():
                    for chain_num in range(1, num_chains + 1):
                        chain_start = pose.chain_begin(chain_num)
                        c_id = pose.pdb_info().chain(chain_start)
                        if c_id not in chain_ids:
                            chain_ids.append(c_id)
                else:
                    chain_ids = [chr(64 + i) for i in range(1, num_chains + 1)]
                
                # 找到目标 chain 的索引
                target_chain_idx = None
                for idx, c_id in enumerate(chain_ids):
                    if c_id == target_chain_id:
                        target_chain_idx = idx + 1  # chain number is 1-based
                        break
                
                if target_chain_idx is not None:
                    # 提取目标 chain
                    target_pose = pose.split_by_chain(target_chain_idx)
                    
                    # 获取目标 chain 的残基范围（在原始 pose 中的索引）
                    chain_start = pose.chain_begin(target_chain_idx)
                    chain_end = pose.chain_end(target_chain_idx)
                    target_chain_residue_range = set(range(chain_start, chain_end + 1))
                    
                    # 如果之前进行了 relax，也需要对单独 chain 进行 relax
                    # 这样可以得到该 chain 在孤立状态下的最优能量
                    if relax:
                        # 创建新的 scorefxn 用于单独 chain（避免影响原始 scorefxn）
                        target_scorefxn = pyrosetta.rosetta.core.scoring.get_score_function()
                        
                        # 过滤 link_specs，只保留属于目标 chain 的约束
                        target_link_specs = []
                        if link_constraints and link_specs:
                            for (p1, a1, p2, a2, r1, r2) in link_specs:
                                # 检查两个残基是否都在目标 chain 中
                                if int(p1) in target_chain_residue_range and int(p2) in target_chain_residue_range:
                                    # 映射到新的 pose 中的残基索引（split_by_chain 后索引从1开始）
                                    new_p1 = int(p1) - chain_start + 1
                                    new_p2 = int(p2) - chain_start + 1
                                    target_link_specs.append((new_p1, a1, new_p2, a2, r1, r2))
                        
                        # 如果启用 auto_head_tail_constraint，也检查单独 chain 的头尾环
                        if auto_head_tail_constraint:
                            try:
                                for (pC, aC, pN, aN, rC, rN, ch, seqC, seqN) in _iter_head_tail_candidates_from_pose(target_pose):
                                    # 对于单独提取的 chain，所有残基都应该属于该 chain
                                    target_link_specs.append((int(pC), aC, int(pN), aN, rC, rN))
                            except Exception as e:
                                print(f"  [Warning] Failed to detect head-tail candidates for target chain: {e}", flush=True)
                        
                        # 应用约束到单独 chain（如果启用）
                        if link_constraints and target_link_specs:
                            try:
                                add_link_constraints_from_rules(
                                    target_pose,
                                    target_link_specs,
                                    link_csv_path=link_csv_path,
                                    dist_sigma=constraint_dist_sigma,
                                    angle_sigma_deg=constraint_angle_sigma_deg,
                                    dihedral_sigma_deg=constraint_dihedral_sigma_deg,
                                )
                                from pyrosetta import rosetta
                                if constrain_in_relax:
                                    target_scorefxn.set_weight(rosetta.core.scoring.atom_pair_constraint, float(constraint_weight))
                                    target_scorefxn.set_weight(rosetta.core.scoring.angle_constraint, float(constraint_weight))
                                    target_scorefxn.set_weight(rosetta.core.scoring.dihedral_constraint, float(constraint_weight))
                            except Exception as e:
                                print(f"  [Warning] Failed to apply constraints to target chain: {e}", flush=True)
                        
                        # 对单独 chain 进行 relax
                        try:
                            from pyrosetta.rosetta.protocols.relax import FastRelax
                            target_relaxer = FastRelax()
                            target_relaxer.set_scorefxn(target_scorefxn)
                            target_relaxer.apply(target_pose)
                            print(f"  [Info] Relaxed target chain {target_chain_id} separately", flush=True)
                        except Exception as e:
                            print(f"  [Warning] Failed to relax target chain separately: {e}", flush=True)
                    
                    # 计算该 chain 单独的能量（使用原始 scorefxn 或 target_scorefxn）
                    if relax:
                        target_chain_energy = target_scorefxn(target_pose)
                        # 使用 target_scorefxn 计算能量后，才能获取 energies()
                        target_pose_energies = target_pose.energies()
                    else:
                        target_chain_energy = scorefxn(target_pose)
                        # 使用 scorefxn 计算能量后，才能获取 energies()
                        target_pose_energies = target_pose.energies()
                    
                    # 从单独提取的 chain 中提取二硫键能量（如果启用）
                    # 这样更一致，因为二硫键是目标 chain 内部的键，应该反映该 chain 自身的稳定性
                    dslf_fa13 = None
                    if extract_dslf_fa13:
                        try:
                            from pyrosetta import rosetta
                            emap = target_pose_energies.total_energies()
                            dslf_fa13 = float(emap[rosetta.core.scoring.dslf_fa13])
                            print(f"  [Info] Extracted dslf_fa13 from target chain {target_chain_id}: {dslf_fa13:.3f}", flush=True)
                        except Exception as e:
                            print(f"  [Warning] 无法从目标 chain {target_chain_id} 提取 dslf_fa13: {e}", flush=True)
                            dslf_fa13 = None
                    
                    # 获取残基数量（只计算标准氨基酸残基）
                    target_chain_num_residues = target_pose.total_residue()
                    # 计算能量/残基数（每残基能量，更有说服力）
                    if target_chain_num_residues > 0:
                        target_chain_energy_per_residue = target_chain_energy / target_chain_num_residues
                        print(f"  [Info] Target chain {target_chain_id}: energy={target_chain_energy:.2f}, residues={target_chain_num_residues}, energy/residue={target_chain_energy_per_residue:.3f}", flush=True)
                    else:
                        print(f"  [Warning] Target chain {target_chain_id} has 0 residues, cannot calculate energy per residue", flush=True)
                else:
                    print(f"  [Warning] Target chain {target_chain_id} not found in structure", flush=True)
            except Exception as e:
                print(f"  [Warning] 无法计算目标 chain {target_chain_id} 的能量: {e}", flush=True)
                target_chain_energy = None
                target_chain_energy_per_residue = None
                target_chain_num_residues = None
                # 如果提取失败，dslf_fa13 保持为 None（已在前面初始化）
        
        # 如果 extract_dslf_fa13=True 但没有指定 target_chain_id，给出警告
        if extract_dslf_fa13 and target_chain_id is None:
            print(f"  [Warning] extract_dslf_fa13=True 但未指定 target_chain_id，无法提取二硫键能量", flush=True)
            dslf_fa13 = None
        
        elapsed = time.time() - start_time
        print(f" [{pdb_name}] 完成 | Relax={relax} | Total={total_energy:.2f} | dG={binding_energy:.2f}", end="", flush=True)
        if dslf_fa13 is not None:
            print(f" | dslf_fa13={dslf_fa13:.3f}", end="", flush=True)
        if target_chain_energy is not None:
            print(f" | target_chain_Energy={target_chain_energy:.2f}", end="", flush=True)
        if target_chain_energy_per_residue is not None:
            print(f" | target_chain_Energy_per_Res={target_chain_energy_per_residue:.3f}", end="", flush=True)
        print(f" | Time={elapsed:.2f}s", flush=True)

        out = {
            "PDB": pdb_name,
            "Binding_Energy": binding_energy,
            "Total_Energy": total_energy,
            "Has_Ligand": num_chains >= 2,
            "Relaxed": relax,
            "Time_sec": round(elapsed, 2)
        }
        # 添加二硫键评分（如果提取）
        if dslf_fa13 is not None:
            out["dslf_fa13"] = dslf_fa13
        # 添加目标 chain 单独的能量（如果计算）
        if target_chain_energy is not None:
            out["target_chain_Energy"] = target_chain_energy
        # 添加目标 chain 能量/残基数（如果计算，更有说服力）
        if target_chain_energy_per_residue is not None:
            out["target_chain_Energy_per_Res"] = target_chain_energy_per_residue
        if target_chain_num_residues is not None:
            out["target_chain_Num_Residues"] = target_chain_num_residues
        # debugging / validation fields
        out["Crosslink_constraints_added"] = int(link_constraints_added)
        if link_specs:
            out["Crosslink_specs_n"] = int(len(link_specs))
        out.update(crosslink_geom)
        if compute_pnear:
            try:
                # PNear sampling may run on ligand-only pose; optionally remove constraints for sampling.
                if link_constraints and not constrain_in_pnear:
                    from pyrosetta import rosetta
                    scorefxn.set_weight(rosetta.core.scoring.atom_pair_constraint, 0.0)
                    scorefxn.set_weight(rosetta.core.scoring.angle_constraint, 0.0)
                    scorefxn.set_weight(rosetta.core.scoring.dihedral_constraint, 0.0)
                pnear_res = estimate_pnear(
                    pose=pose,
                    scorefxn=scorefxn,
                    ligand_chain_letter=pnear_chain,
                    n_decoys=pnear_n,
                    kT=pnear_kT,
                    rmsd_lambda=pnear_lambda,
                    method=pnear_method,
                    state=pnear_state,
                    sampler=pnear_sampler,
                    link_constraints=(link_constraints and constrain_in_pnear),
                    link_csv_path=link_csv_path,
                    constraint_weight=constraint_weight,
                    constraint_dist_sigma=constraint_dist_sigma,
                    constraint_angle_sigma_deg=constraint_angle_sigma_deg,
                    constraint_dihedral_sigma_deg=constraint_dihedral_sigma_deg,
                    # unbound 采样时也可对“头尾 C-N”强制加约束（没有显式 crosslink 也能锁住）
                    # 这里复用 auto_head_tail_constraint 的语义（在 estimate_pnear 内部实现为对 base pose 加 spec）
                    auto_head_tail_constraint=auto_head_tail_constraint,
                    debug_constraints=bool(pnear_debug_constraints),
                    debug_first_n=int(pnear_debug_first_n),
                )
                out.update(pnear_res)
                if link_constraints and not constrain_in_pnear:
                    from pyrosetta import rosetta
                    scorefxn.set_weight(rosetta.core.scoring.atom_pair_constraint, float(constraint_weight))
                    scorefxn.set_weight(rosetta.core.scoring.angle_constraint, float(constraint_weight))
                    scorefxn.set_weight(rosetta.core.scoring.dihedral_constraint, float(constraint_weight))
            except Exception as e:
                out.update({
                    "PNear": float("nan"),
                    "PNear_error": str(e),
                    "PNear_n": int(pnear_n),
                    "PNear_kT": float(pnear_kT),
                    "PNear_lambda": float(pnear_lambda),
                    "PNear_chain": pnear_chain,
                    "PNear_method": pnear_method,
                    "PNear_state": pnear_state,
                    "PNear_sampler": pnear_sampler,
                })

        return out

    except Exception as e:
        elapsed = time.time() - start_time
        print(f" [{pdb_name}] 计算出错: {e} (耗时 {elapsed:.2f}s)")
        return {"PDB": pdb_name, "Error": str(e), "Time_sec": round(elapsed, 2)}


def batch_energy(
    pdb_folder,
    output_dir="results",
    num_workers=4,
    relax=True,
    save_results=False,
    compute_pnear: bool = True,     # 是否计算 PNear
    pnear_n: int = 50,               # PNear: 采样 decoy 数量
    pnear_kT: float = 1.0,           # PNear: Boltzmann 权重 kT
    pnear_lambda: float = 1.0,       # PNear: “near” 的 RMSD 尺度 λ（Å）
    pnear_chain: str = "A",          # 复合物：环肽配体链（PDB chain letter），你默认是 A
    pnear_method: str = "fastrelax", # quick 采样内部用：fastrelax 或 min
    pnear_state: str = "unbound",    # 推荐用于构象熵损失：unbound=ligand-only
    pnear_sampler: str = "crankshaft", # 默认改成 crankshaft：对内酯/异肽/二硫等“任意共价交联环”更稳
    # ===== Crosslink constraints (link.csv) =====
    link_constraints: bool = False,
    link_csv_path: str | None = None,
    constraint_weight: float = 1.0,
    constraint_dist_sigma: float = 0.10,
    constraint_angle_sigma_deg: float = 10.0,
    constraint_dihedral_sigma_deg: float = 20.0,
    constrain_in_relax: bool = True,
    constrain_in_pnear: bool = True,
    auto_head_tail_constraint: bool = False,
    auto_head_tail_bond: bool = False,
    pnear_debug_constraints: bool = False,
    pnear_debug_first_n: int = 3,
    # ===== Energy component extraction =====
    extract_dslf_fa13: bool = False,        # 是否提取二硫键评分 (dslf_fa13)
    target_chain_id: str | None = None,     # 目标 chain ID，用于计算该 chain 单独的能量（稳定性指标）
    # ===== Output =====
    save_relaxed_pdb: bool = False,
    relaxed_pdb_dir: str | None = None,
):
    """
    批量计算入口

    PNear sampler 含义（用于环肽采样生成 decoys）：
    - auto:
        - 若检测到头尾环（N-terminus <-> C-terminus 的 backbone polymeric cyclization）=> 用 genkic
        - 否则（内酯/异肽键/二硫等 sidechain 或 crosslink 闭环）=> 用 crankshaft
    - crankshaft:
        - 使用 cyclic_peptide.CrankshaftFlipMover 做 backbone 翻转扰动 + 最小化
        - 优点：不需要“环闭合求解”，对任意共价交联的环肽更通用（推荐用于 link.csv 的大多数情况）
    - genkic:
        - GeneralizedKIC（广义运动学闭合），更适合“头尾环”的 backbone 闭合采样
        - 对内酯/异肽/二硫等非头尾环交联：需要更复杂的 anchor/fold-tree，当前为实验能力
    - quick:
        - SmallMover + Min/（可选 FastRelax），快速基线；不破坏共价键，但环肽采样效率通常较低
    """
    # 全局初始化一次 PyRosetta（加载本 repo 的自定义 residue types）
    _init_pyrosetta_once()
    
    output_dir_energy = os.path.join(output_dir, "energy_results")
    os.makedirs(output_dir_energy, exist_ok=True)
    if save_relaxed_pdb and relaxed_pdb_dir is None:
        relaxed_pdb_dir = os.path.join(output_dir_energy, "relaxed_structures")
        os.makedirs(relaxed_pdb_dir, exist_ok=True)
    
    pdb_files = _collect_pdb_like_files(pdb_folder, output_dir)
    if not pdb_files:
        print("未找到任何 PDB 文件！")
        return pd.DataFrame()

    print(f"共检测到 {len(pdb_files)} 个结构，将使用 {num_workers} 个进程...\n")

    # 匹配 bond files
    pdb_to_bond = {}
    for pdb in pdb_files:
        d = os.path.dirname(pdb)
        base = os.path.splitext(os.path.basename(pdb))[0]
        candidates = [
            os.path.join(d, f"{base}.txt"),
            os.path.join(d, f"bonds_{base}.txt"),
        ]
        if base.startswith("partial_"):
            clean_base = base.replace("partial_", "")
            candidates.append(os.path.join(d, f"bonds_{clean_base}.txt"))
        
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        pdb_to_bond[pdb] = found

    start_all = time.time()
    results = []

    # 串行或并行执行
    # 如果 num_workers=1，直接循环以便于调试
    if num_workers <= 1:
        for pdb in tqdm(pdb_files, desc="Computing (Serial)"):
            results.append(
                compute_energy_for_pdb(
                    pdb,
                    relax,
                    pdb_to_bond.get(pdb),
                    compute_pnear=compute_pnear,
                    pnear_n=pnear_n,
                    pnear_kT=pnear_kT,
                    pnear_lambda=pnear_lambda,
                    pnear_chain=pnear_chain,
                    pnear_method=pnear_method,
                    pnear_state=pnear_state,
                    pnear_sampler=pnear_sampler,
                    link_constraints=link_constraints,
                    link_csv_path=link_csv_path,
                    constraint_weight=constraint_weight,
                    constraint_dist_sigma=constraint_dist_sigma,
                    constraint_angle_sigma_deg=constraint_angle_sigma_deg,
                    constraint_dihedral_sigma_deg=constraint_dihedral_sigma_deg,
                    constrain_in_relax=constrain_in_relax,
                    constrain_in_pnear=constrain_in_pnear,
                    auto_head_tail_constraint=auto_head_tail_constraint,
                    auto_head_tail_bond=auto_head_tail_bond,
                    pnear_debug_constraints=pnear_debug_constraints,
                    pnear_debug_first_n=pnear_debug_first_n,
                    extract_dslf_fa13=extract_dslf_fa13,
                    target_chain_id=target_chain_id,
                    save_relaxed_pdb=save_relaxed_pdb,
                    relaxed_pdb_dir=relaxed_pdb_dir,
                    output_dir=output_dir,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    compute_energy_for_pdb,
                    pdb,
                    relax,
                    pdb_to_bond.get(pdb),
                    compute_pnear=compute_pnear,
                    pnear_n=pnear_n,
                    pnear_kT=pnear_kT,
                    pnear_lambda=pnear_lambda,
                    pnear_chain=pnear_chain,
                    pnear_method=pnear_method,
                    pnear_state=pnear_state,
                    pnear_sampler=pnear_sampler,
                    link_constraints=link_constraints,
                    link_csv_path=link_csv_path,
                    constraint_weight=constraint_weight,
                    constraint_dist_sigma=constraint_dist_sigma,
                    constraint_angle_sigma_deg=constraint_angle_sigma_deg,
                    constraint_dihedral_sigma_deg=constraint_dihedral_sigma_deg,
                    constrain_in_relax=constrain_in_relax,
                    constrain_in_pnear=constrain_in_pnear,
                    auto_head_tail_constraint=auto_head_tail_constraint,
                    auto_head_tail_bond=auto_head_tail_bond,
                    pnear_debug_constraints=pnear_debug_constraints,
                    pnear_debug_first_n=pnear_debug_first_n,
                    extract_dslf_fa13=extract_dslf_fa13,
                    target_chain_id=target_chain_id,
                    save_relaxed_pdb=save_relaxed_pdb,
                    relaxed_pdb_dir=relaxed_pdb_dir,
                    output_dir=output_dir,
                ): pdb
                for pdb in pdb_files
            }
            for f in tqdm(as_completed(futures), total=len(futures), desc="Computing (Parallel)"):
                results.append(f.result())

    df = pd.DataFrame(results)
    elapsed_all = time.time() - start_all
    print(f"\n所有计算完成，总耗时 {elapsed_all / 60:.2f} 分钟")

    if save_results:
        output_path = os.path.join(output_dir_energy, "Energy_results.csv")
        df.to_csv(output_path, index=False)
        print(f"能量结果已保存到: {output_path}")

    return df
