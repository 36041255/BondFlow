#!/usr/bin/env python3
import os
import glob
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa, PPBuilder
from Bio.PDB.DSSP import DSSP


SS3 = ["H", "E", "C"]  # Helix / Strand / Coil


def _get_parser_by_ext(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def _collect_structure_files(folder: str):
    files = []
    for pat in ("*.pdb", "*.cif", "*.mmcif"):
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(files)


def _dssp8_to_ss3(ss8: str):
    if not ss8 or ss8 == " ":
        return "C"
    if ss8 in {"H", "G", "I"}:
        return "H"
    if ss8 in {"E", "B"}:
        return "E"
    return "C"


def _phi_psi_to_ss3(phi, psi):
    """Fallback coarse SS assignment from phi/psi in radians."""
    if phi is None or psi is None:
        return "C"
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)

    # Rough alpha-helix basin
    if -160.0 <= phi_deg <= -30.0 and -100.0 <= psi_deg <= 20.0:
        return "H"
    # Rough beta-sheet basin
    if -180.0 <= phi_deg <= -40.0 and (70.0 <= psi_deg <= 180.0 or -180.0 <= psi_deg <= -140.0):
        return "E"
    return "C"


def _extract_chain_ss_counter(file_path: str, chain_id: str = "A", prefer_dssp: bool = True):
    """
    Return:
      (counter_ss3, method) where method in {"dssp", "phi_psi", "none"}.
    """
    parser = _get_parser_by_ext(file_path)
    structure = parser.get_structure("struct", file_path)
    model = structure[0]
    chain = model[chain_id] if chain_id in model else None
    if chain is None:
        return Counter(), "none"

    # Try DSSP first
    if prefer_dssp:
        try:
            dssp = DSSP(model, file_path)
            c = Counter()
            for key in dssp.keys():
                dssp_chain = key[0]
                if dssp_chain != chain_id:
                    continue
                ss8 = dssp[key][2]
                c[_dssp8_to_ss3(ss8)] += 1
            if sum(c.values()) > 0:
                return c, "dssp"
        except Exception:
            pass

    # Fallback: phi/psi-based coarse assignment
    c = Counter()
    ppb = PPBuilder()
    peptides = ppb.build_peptides(chain)
    if not peptides:
        # No continuous peptide built; at least count residues as coil.
        for residue in chain:
            if is_aa(residue, standard=True):
                c["C"] += 1
        return c, "phi_psi" if sum(c.values()) > 0 else "none"

    for pep in peptides:
        for (phi, psi) in pep.get_phi_psi_list():
            c[_phi_psi_to_ss3(phi, psi)] += 1
    return c, "phi_psi" if sum(c.values()) > 0 else "none"


def _distribution_from_counter(counter: Counter):
    total = float(sum(counter.values()))
    if total <= 0:
        return {s: 0.0 for s in SS3}, 0
    return {s: float(counter.get(s, 0)) / total for s in SS3}, int(total)


def compare_ligand_ss_distribution(
    design_dir: str,
    benchmark_dir: str,
    chain_id: str = "A",
    output_dir: str = "results",
    save_results: bool = True,
    prefer_dssp: bool = True,
):
    """
    Compare ligand secondary-structure distribution (H/E/C) between design and benchmark.
    """
    design_files = _collect_structure_files(design_dir)
    benchmark_files = _collect_structure_files(benchmark_dir)
    if not design_files:
        raise ValueError(f"设计结构目录未找到结构文件: {design_dir}")
    if not benchmark_files:
        raise ValueError(f"Benchmark 目录未找到结构文件: {benchmark_dir}")

    design_counter = Counter()
    benchmark_counter = Counter()
    design_valid = 0
    benchmark_valid = 0
    design_method_counter = Counter()
    benchmark_method_counter = Counter()

    for fp in design_files:
        try:
            c, m = _extract_chain_ss_counter(fp, chain_id=chain_id, prefer_dssp=prefer_dssp)
            design_method_counter[m] += 1
            if sum(c.values()) > 0:
                design_counter.update(c)
                design_valid += 1
        except Exception:
            design_method_counter["none"] += 1

    for fp in benchmark_files:
        try:
            c, m = _extract_chain_ss_counter(fp, chain_id=chain_id, prefer_dssp=prefer_dssp)
            benchmark_method_counter[m] += 1
            if sum(c.values()) > 0:
                benchmark_counter.update(c)
                benchmark_valid += 1
        except Exception:
            benchmark_method_counter["none"] += 1

    design_freq, design_total = _distribution_from_counter(design_counter)
    benchmark_freq, benchmark_total = _distribution_from_counter(benchmark_counter)

    rows = []
    for s in SS3:
        d_cnt = int(design_counter.get(s, 0))
        b_cnt = int(benchmark_counter.get(s, 0))
        d_f = float(design_freq[s])
        b_f = float(benchmark_freq[s])
        rows.append(
            {
                "SS3": s,
                "Design_Count": d_cnt,
                "Design_Freq": d_f,
                "Benchmark_Count": b_cnt,
                "Benchmark_Freq": b_f,
                "Freq_Diff_DesignMinusBenchmark": d_f - b_f,
                "Abs_Freq_Diff": abs(d_f - b_f),
            }
        )
    df_compare = pd.DataFrame(rows).sort_values("Abs_Freq_Diff", ascending=False)

    p = np.array([design_freq[s] for s in SS3], dtype=float)
    q = np.array([benchmark_freq[s] for s in SS3], dtype=float)
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    m = 0.5 * (p_safe + q_safe)

    kl_pq = float(np.sum(p_safe * np.log2(p_safe / q_safe)))
    kl_qp = float(np.sum(q_safe * np.log2(q_safe / p_safe)))
    js_div = float(0.5 * np.sum(p_safe * np.log2(p_safe / m)) + 0.5 * np.sum(q_safe * np.log2(q_safe / m)))
    l1 = float(np.sum(np.abs(p - q)))
    l2 = float(np.sqrt(np.sum((p - q) ** 2)))

    summary = {
        "Chain_ID": chain_id,
        "Design_Structures_Total": len(design_files),
        "Design_Structures_Valid": design_valid,
        "Benchmark_Structures_Total": len(benchmark_files),
        "Benchmark_Structures_Valid": benchmark_valid,
        "Design_Total_SS_Assignments": design_total,
        "Benchmark_Total_SS_Assignments": benchmark_total,
        "L1_Distance": l1,
        "L2_Distance": l2,
        "KL_Design_to_Benchmark": kl_pq,
        "KL_Benchmark_to_Design": kl_qp,
        "JS_Divergence": js_div,
        "Design_Method_DSSP": int(design_method_counter.get("dssp", 0)),
        "Design_Method_PhiPsi": int(design_method_counter.get("phi_psi", 0)),
        "Benchmark_Method_DSSP": int(benchmark_method_counter.get("dssp", 0)),
        "Benchmark_Method_PhiPsi": int(benchmark_method_counter.get("phi_psi", 0)),
    }

    if save_results:
        out_dir = os.path.join(output_dir, "ligand_ss_diff_results")
        os.makedirs(out_dir, exist_ok=True)
        compare_csv = os.path.join(out_dir, "Ligand_SS_Distribution_Compare.csv")
        summary_csv = os.path.join(out_dir, "Ligand_SS_Distribution_Summary.csv")
        df_compare.to_csv(compare_csv, index=False)
        pd.DataFrame([summary]).to_csv(summary_csv, index=False)
        print("Ligand 二级结构分布差异结果已保存:")
        print(f"- 详细对比: {compare_csv}")
        print(f"- 汇总指标: {summary_csv}")

    return df_compare, summary


def main():
    parser = argparse.ArgumentParser(description="比较 benchmark ligand 与设计 ligand 的二级结构分布差异")
    parser.add_argument("--design_dir", required=True, help="设计结构目录")
    parser.add_argument("--benchmark_dir", required=True, help="benchmark 结构目录")
    parser.add_argument("--chain", default="A", help="ligand 链 ID（默认 A）")
    parser.add_argument("--output_dir", default="results", help="输出目录")
    parser.add_argument("--no_save", action="store_true", help="不保存 CSV，仅打印汇总")
    parser.add_argument("--no_dssp", action="store_true", help="禁用 DSSP，强制使用 phi/psi 近似分类")
    args = parser.parse_args()

    _, summary = compare_ligand_ss_distribution(
        design_dir=args.design_dir,
        benchmark_dir=args.benchmark_dir,
        chain_id=args.chain,
        output_dir=args.output_dir,
        save_results=(not args.no_save),
        prefer_dssp=(not args.no_dssp),
    )
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

