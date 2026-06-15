#!/usr/bin/env python3
import os
import glob
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1


AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def _get_parser_by_ext(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def _extract_chain_aa_counter(file_path: str, chain_id: str = "A") -> Counter:
    """Extract amino-acid counts for a specific chain from one structure file."""
    counter = Counter()
    parser = _get_parser_by_ext(file_path)
    structure = parser.get_structure("struct", file_path)
    model = structure[0]

    target_chain = None
    for ch in model:
        if ch.id == chain_id:
            target_chain = ch
            break
    if target_chain is None:
        return counter

    for residue in target_chain:
        if not is_aa(residue, standard=True):
            continue
        res_name = residue.get_resname().upper()
        if res_name in protein_letters_3to1:
            aa = protein_letters_3to1[res_name]
            if aa in AA20:
                counter[aa] += 1
    return counter


def _collect_structure_files(folder: str):
    files = []
    for pat in ("*.pdb", "*.cif", "*.mmcif"):
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(files)


def _distribution_from_counter(counter: Counter):
    total = float(sum(counter.values()))
    if total <= 0:
        return {aa: 0.0 for aa in AA20}, 0
    return {aa: float(counter.get(aa, 0)) / total for aa in AA20}, int(total)


def compare_ligand_aa_distribution(
    design_dir: str,
    benchmark_dir: str,
    chain_id: str = "A",
    output_dir: str = "results",
    save_results: bool = True,
):
    """
    Compare ligand amino-acid distribution between design outputs and benchmark structures.

    Returns:
      df_compare: per-amino-acid comparison table
      summary: dict with aggregate distance metrics
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
    bench_valid = 0

    for fp in design_files:
        try:
            c = _extract_chain_aa_counter(fp, chain_id=chain_id)
            if sum(c.values()) > 0:
                design_counter.update(c)
                design_valid += 1
        except Exception:
            continue

    for fp in benchmark_files:
        try:
            c = _extract_chain_aa_counter(fp, chain_id=chain_id)
            if sum(c.values()) > 0:
                benchmark_counter.update(c)
                bench_valid += 1
        except Exception:
            continue

    design_freq, design_total = _distribution_from_counter(design_counter)
    bench_freq, bench_total = _distribution_from_counter(benchmark_counter)

    rows = []
    for aa in AA20:
        d_cnt = int(design_counter.get(aa, 0))
        b_cnt = int(benchmark_counter.get(aa, 0))
        d_f = float(design_freq[aa])
        b_f = float(bench_freq[aa])
        rows.append(
            {
                "AA": aa,
                "Design_Count": d_cnt,
                "Design_Freq": d_f,
                "Benchmark_Count": b_cnt,
                "Benchmark_Freq": b_f,
                "Freq_Diff_DesignMinusBenchmark": d_f - b_f,
                "Abs_Freq_Diff": abs(d_f - b_f),
            }
        )

    df_compare = pd.DataFrame(rows).sort_values("Abs_Freq_Diff", ascending=False)

    p = np.array([design_freq[aa] for aa in AA20], dtype=float)
    q = np.array([bench_freq[aa] for aa in AA20], dtype=float)
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
        "Benchmark_Structures_Valid": bench_valid,
        "Design_Total_AA": design_total,
        "Benchmark_Total_AA": bench_total,
        "L1_Distance": l1,
        "L2_Distance": l2,
        "KL_Design_to_Benchmark": kl_pq,
        "KL_Benchmark_to_Design": kl_qp,
        "JS_Divergence": js_div,
    }

    if save_results:
        out_dir = os.path.join(output_dir, "ligand_aa_diff_results")
        os.makedirs(out_dir, exist_ok=True)
        compare_csv = os.path.join(out_dir, "Ligand_AA_Distribution_Compare.csv")
        summary_csv = os.path.join(out_dir, "Ligand_AA_Distribution_Summary.csv")
        df_compare.to_csv(compare_csv, index=False)
        pd.DataFrame([summary]).to_csv(summary_csv, index=False)
        print(f"Ligand AA 分布差异结果已保存:")
        print(f"- 详细对比: {compare_csv}")
        print(f"- 汇总指标: {summary_csv}")

    return df_compare, summary


def main():
    parser = argparse.ArgumentParser(description="比较 benchmark ligand 与设计 ligand 的氨基酸分布差异")
    parser.add_argument("--design_dir", required=True, help="设计结构目录")
    parser.add_argument("--benchmark_dir", required=True, help="benchmark 结构目录")
    parser.add_argument("--chain", default="A", help="ligand 链 ID（默认 A）")
    parser.add_argument("--output_dir", default="results", help="输出目录")
    parser.add_argument("--no_save", action="store_true", help="不保存 CSV，仅打印汇总")
    args = parser.parse_args()

    _, summary = compare_ligand_aa_distribution(
        design_dir=args.design_dir,
        benchmark_dir=args.benchmark_dir,
        chain_id=args.chain,
        output_dir=args.output_dir,
        save_results=(not args.no_save),
    )
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

