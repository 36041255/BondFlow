"""
Summarize evaluation metrics across lengths and models.

This script expects the following directory layout (as produced by
`run_demo_small.sh` + `generate_structures.py` + `analysis/main.py`):

root/
  <experiment>/
    <model_name>/
      len_<L>/
        generation_log.csv
        analysis/
          energy_results/Energy_results.csv
          bond_eval_results/BondEval_per_type.csv
          Shannon_entropy.csv
          tmalign_results/RMSD_matrix.csv
          tmalign_results/TMscore_matrix.csv

It computes for each (model, length):
  - average generation time per sample / per residue
  - average energy and energy per residue
  - cyclization statistics based on BondEval_per_type:
      * average head_tail predicted count
      * cyclization success rate (fraction of structures with TP>0 for head_tail)
  - sequence diversity: mean Shannon entropy across positions
  - structural diversity: mean pairwise RMSD / TM-score

Then produces:
  - a CSV `summary_per_model_length.csv`
  - several PNG plots under <root>/<experiment>/_summary
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _parse_list_ints(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _collect_models(base_dir: str, models_arg: Optional[str]) -> List[str]:
    if models_arg:
        return [m.strip() for m in models_arg.split(",") if m.strip()]
    out = []
    if not os.path.isdir(base_dir):
        return out
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            out.append(name)
    return sorted(out)


def _collect_lengths_for_model(model_dir: str) -> List[int]:
    lengths = []
    if not os.path.isdir(model_dir):
        return lengths
    for name in os.listdir(model_dir):
        if not name.startswith("len_"):
            continue
        try:
            L = int(name.split("len_")[1])
            lengths.append(L)
        except Exception:
            continue
    return sorted(set(lengths))


def _mean_from_generation_log(path: str) -> Tuple[Optional[float], Optional[float]]:
    if not os.path.isfile(path):
        return None, None
    df = pd.read_csv(path)
    if df.empty:
        return None, None
    return float(df["time_per_sample"].mean()), float(df["time_per_residue"].mean())


def _energy_stats(path: str, length: int) -> Tuple[Optional[float], Optional[float]]:
    if not os.path.isfile(path):
        return None, None
    df = pd.read_csv(path)
    if df.empty or "Total_Energy" not in df.columns:
        return None, None
    mean_E = float(df["Total_Energy"].mean())
    mean_E_per_res = mean_E / float(length) if length > 0 else None
    return mean_E, mean_E_per_res


def _cyclization_stats(summary_path: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Cyclization statistics based on *all* non-backbone links (all LINK types),
    aggregated over all structures for a given (model, length):

    - avg_count:    average number of predicted links per structure
                    (pred_pos = TP + FP, using per-structure TP/FP from BondEval_summary).
    - precision:    sum(TP) / sum(TP + FP)
    - recall:       sum(TP) / sum(TP + FN)
    - f1:           2 * P * R / (P + R)
    """
    if not os.path.isfile(summary_path):
        return None, None, None, None
    df = pd.read_csv(summary_path)
    required = {"TP", "FP", "FN"}
    if df.empty or not required.issubset(df.columns):
        return None, None, None, None

    df = df.copy()
    df["pred_pos"] = df["TP"] + df["FP"]

    total_tp = float(df["TP"].sum())
    total_fp = float(df["FP"].sum())
    total_fn = float(df["FN"].sum())

    avg_count = float(df["pred_pos"].mean())

    prec_den = total_tp + total_fp
    rec_den = total_tp + total_fn

    # Handle edge cases:
    # - If there are no positives at all (TP=FP=FN=0), treat precision/recall/F1 as 0.0
    #   so that plots show an explicit 0 performance instead of NaN.
    # - If only one of the denominators is zero, also treat the corresponding metric as 0.0.
    if (total_tp + total_fp + total_fn) == 0.0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = (total_tp / prec_den) if prec_den > 0 else 0.0
        recall = (total_tp / rec_den) if rec_den > 0 else 0.0
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

    return avg_count, precision, recall, f1


def _cyclization_stats_per_type(per_type_path: str, bond_types: List[str]) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]]:
    """
    Per-bond-type cyclization statistics using BondEval_per_type.csv.

    Returns a dict mapping bond type -> (avg_count, precision, recall, f1).
    """
    out: Dict[str, Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = {}

    if not os.path.isfile(per_type_path):
        for bt in bond_types:
            out[bt] = (None, None, None, None)
        return out

    df = pd.read_csv(per_type_path)
    required = {"Type", "TP", "FP", "FN"}
    if df.empty or not required.issubset(df.columns):
        for bt in bond_types:
            out[bt] = (None, None, None, None)
        return out

    df = df.copy()

    for bt in bond_types:
        sub = df[df["Type"] == bt]
        if sub.empty:
            out[bt] = (None, None, None, None)
            continue

        sub = sub.copy()
        sub["pred_pos"] = sub["TP"] + sub["FP"]

        total_tp = float(sub["TP"].sum())
        total_fp = float(sub["FP"].sum())
        total_fn = float(sub["FN"].sum())

        avg_count = float(sub["pred_pos"].mean())

        prec_den = total_tp + total_fp
        rec_den = total_tp + total_fn

        if (total_tp + total_fp + total_fn) == 0.0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            precision = (total_tp / prec_den) if prec_den > 0 else 0.0
            recall = (total_tp / rec_den) if rec_den > 0 else 0.0
            if (precision + recall) == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

        out[bt] = (avg_count, precision, recall, f1)

    return out


def _sequence_diversity(entropy_path: str) -> Optional[float]:
    if not os.path.isfile(entropy_path):
        return None
    df = pd.read_csv(entropy_path)
    if df.empty or "ShannonEntropy" not in df.columns:
        return None
    return float(df["ShannonEntropy"].mean())


def _structural_diversity(rmsd_path: str, tm_path: str) -> Tuple[Optional[float], Optional[float]]:
    if not (os.path.isfile(rmsd_path) and os.path.isfile(tm_path)):
        return None, None
    rmsd_df = pd.read_csv(rmsd_path, index_col=0)
    tm_df = pd.read_csv(tm_path, index_col=0)
    if rmsd_df.shape[0] < 2:
        return None, None

    # use upper triangle i<j to compute pairwise stats
    n = rmsd_df.shape[0]
    iu = np.triu_indices(n, k=1)
    rmsd_vals = rmsd_df.values[iu]
    tm_vals = tm_df.values[iu]
    # filter NaNs
    rmsd_vals = rmsd_vals[~np.isnan(rmsd_vals)]
    tm_vals = tm_vals[~np.isnan(tm_vals)]
    rmsd_mean = float(rmsd_vals.mean()) if rmsd_vals.size > 0 else None
    tm_mean = float(tm_vals.mean()) if tm_vals.size > 0 else None
    return rmsd_mean, tm_mean


def summarize(
    root: str,
    experiment: str,
    models: Optional[str],
    lengths_arg: Optional[str],
    out_dir: Optional[str],
    best_energy_frac: float = 0.2,
) -> pd.DataFrame:
    base = os.path.join(root, experiment)
    model_names = _collect_models(base, models)
    if not model_names:
        raise ValueError(f"No models found under {base}")

    lengths_filter = _parse_list_ints(lengths_arg)

    rows: List[Dict] = []
    # Collect per-sample energies for best-k% bar plot across models
    energy_samples: List[Dict] = []

    for model in model_names:
        model_dir = os.path.join(base, model)
        lengths = lengths_filter or _collect_lengths_for_model(model_dir)
        for L in lengths:
            len_dir = os.path.join(model_dir, f"len_{L}")
            gen_log = os.path.join(len_dir, "generation_log.csv")
            analysis_dir = os.path.join(len_dir, "analysis")

            time_ps, time_pr = _mean_from_generation_log(gen_log)

            energy_path = os.path.join(analysis_dir, "energy_results", "Energy_results.csv")
            mean_E, mean_E_per_res = _energy_stats(energy_path, L)

            # Accumulate per-sample energies for later best-k% analysis
            if os.path.isfile(energy_path):
                df_e = pd.read_csv(energy_path)
                if not df_e.empty and "Total_Energy" in df_e.columns:
                    for val in df_e["Total_Energy"]:
                        energy_samples.append(
                            {
                                "model": model,
                                "length": L,
                                "total_energy": float(val),
                            }
                        )

            bond_summary_path = os.path.join(analysis_dir, "bond_eval_results", "BondEval_summary.csv")
            avg_link_count, link_prec, link_rec, link_f1 = _cyclization_stats(bond_summary_path)

            # Per-bond-type LINK stats (e.g. disulfide, lactone, isopeptide)
            bond_per_type_path = os.path.join(analysis_dir, "bond_eval_results", "BondEval_per_type.csv")
            per_type_stats = _cyclization_stats_per_type(
                bond_per_type_path,
                bond_types=["disulfide", "lactone", "isopeptide"],
            )

            entropy_path = os.path.join(analysis_dir, "Shannon_entropy.csv")
            mean_entropy = _sequence_diversity(entropy_path)

            rmsd_path = os.path.join(analysis_dir, "tmalign_results", "RMSD_matrix.csv")
            tm_path = os.path.join(analysis_dir, "tmalign_results", "TMscore_matrix.csv")
            rmsd_mean, tm_mean = _structural_diversity(rmsd_path, tm_path)

            row = {
                "model": model,
                "length": L,
                "time_per_sample": time_ps,
                "time_per_residue": time_pr,
                "mean_total_energy": mean_E,
                "mean_energy_per_residue": mean_E_per_res,
                "avg_link_count": avg_link_count,
                "link_precision": link_prec,
                "link_recall": link_rec,
                "link_F1": link_f1,
                # Per-bond-type LINK metrics
                "disulfide_avg_link_count": per_type_stats["disulfide"][0],
                "disulfide_precision": per_type_stats["disulfide"][1],
                "disulfide_recall": per_type_stats["disulfide"][2],
                "disulfide_F1": per_type_stats["disulfide"][3],
                "lactone_avg_link_count": per_type_stats["lactone"][0],
                "lactone_precision": per_type_stats["lactone"][1],
                "lactone_recall": per_type_stats["lactone"][2],
                "lactone_F1": per_type_stats["lactone"][3],
                "isopeptide_avg_link_count": per_type_stats["isopeptide"][0],
                "isopeptide_precision": per_type_stats["isopeptide"][1],
                "isopeptide_recall": per_type_stats["isopeptide"][2],
                "isopeptide_F1": per_type_stats["isopeptide"][3],
                "mean_shannon_entropy": mean_entropy,
                "mean_pairwise_RMSD": rmsd_mean,
                "mean_pairwise_TM": tm_mean,
            }
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["model", "length"])

    # Output directory
    if out_dir is None:
        out_dir = os.path.join(base, "_summary")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "summary_per_model_length.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV written to: {csv_path}")

    # Plots
    sns.set(style="whitegrid")

    # 1) generation time vs length
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="length", y="time_per_sample", hue="model", marker="o")
    plt.ylabel("Time per sample (s)")
    plt.title("Generation time vs length")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_per_sample_vs_length.png"))
    plt.close()

    # 2) link count vs length
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="length", y="avg_link_count", hue="model", marker="o")
    plt.ylabel("Avg LINK count per structure")
    plt.title("LINK count vs length")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "link_count_vs_length.png"))
    plt.close()

    # 3) link precision/recall/F1 vs length
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="length", y="link_precision", hue="model", marker="o")
    plt.ylabel("Precision")
    plt.title("LINK precision vs length")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "link_precision_vs_length.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="length", y="link_recall", hue="model", marker="o")
    plt.ylabel("Recall")
    plt.title("LINK recall vs length")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "link_recall_vs_length.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="length", y="link_F1", hue="model", marker="o")
    plt.ylabel("F1")
    plt.title("LINK F1 vs length")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "link_F1_vs_length.png"))
    plt.close()

    # 4) energy per residue comparison across models
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="length", y="mean_total_energy", hue="model", marker="o")
    plt.ylabel("Mean total energy")
    plt.title("Total energy vs length")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy_per_residue_vs_length.png"))
    plt.close()

    # 5) diversity: RMSD / TM / entropy
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="length", y="mean_pairwise_RMSD", hue="model", marker="o")
    plt.ylabel("Mean pairwise RMSD")
    plt.title("Structural diversity (RMSD) vs length")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diversity_RMSD_vs_length.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="length", y="mean_pairwise_TM", hue="model", marker="o")
    plt.ylabel("Mean pairwise TM-score")
    plt.title("Structural diversity (TM-score) vs length")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diversity_TM_vs_length.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="length", y="mean_shannon_entropy", hue="model", marker="o")
    plt.ylabel("Mean Shannon entropy")
    plt.title("Sequence diversity vs length")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "diversity_entropy_vs_length.png"))
    plt.close()

    # 6) per-bond-type LINK metrics within each model:
    #    for each model, plot different bond types (all, isopeptide, disulfide, lactone)
    #    in a single figure, with hue = bond_type.
    bond_types = [
        ("all", "all"),
        ("isopeptide", "isopeptide"),
        ("disulfide", "disulfide"),
        ("lactone", "lactone"),
    ]

    metric_column_maps = {
        "count": {
            "all": "avg_link_count",
            "isopeptide": "isopeptide_avg_link_count",
            "disulfide": "disulfide_avg_link_count",
            "lactone": "lactone_avg_link_count",
        },
        "precision": {
            "all": "link_precision",
            "isopeptide": "isopeptide_precision",
            "disulfide": "disulfide_precision",
            "lactone": "lactone_precision",
        },
        "recall": {
            "all": "link_recall",
            "isopeptide": "isopeptide_recall",
            "disulfide": "disulfide_recall",
            "lactone": "lactone_recall",
        },
        "F1": {
            "all": "link_F1",
            "isopeptide": "isopeptide_F1",
            "disulfide": "disulfide_F1",
            "lactone": "lactone_F1",
        },
    }

    metric_ylabels = {
        "count": "Avg LINK count per structure",
        "precision": "Precision",
        "recall": "Recall",
        "F1": "F1",
    }

    for model_name in df["model"].unique():
        df_m = df[df["model"] == model_name]
        for metric_key, col_map in metric_column_maps.items():
            rows_bt: List[Dict] = []
            for bt_key, bt_label in bond_types:
                col = col_map.get(bt_key)
                if col not in df_m.columns:
                    continue
                if df_m[col].isna().all():
                    continue
                for _, r in df_m.iterrows():
                    rows_bt.append(
                        {
                            "length": r["length"],
                            "bond_type": bt_label,
                            "value": r[col],
                        }
                    )
            if not rows_bt:
                continue
            df_plot = pd.DataFrame(rows_bt)
            plt.figure(figsize=(6, 4))
            sns.lineplot(data=df_plot, x="length", y="value", hue="bond_type", marker="o")
            plt.ylabel(metric_ylabels.get(metric_key, metric_key))
            plt.title(f"LINK {metric_key} vs length by bond type (model={model_name})")
            plt.tight_layout()
            safe_model = str(model_name).replace("/", "_")
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"link_{metric_key}_vs_length_by_bond_type_model_{safe_model}.png",
                )
            )
            plt.close()

    # 7) Best-k% energy bar plot across models
    if energy_samples and best_energy_frac > 0.0:
        energy_df = pd.DataFrame(energy_samples)
        best_rows: List[Dict] = []
        for model_name in sorted(energy_df["model"].unique()):
            sub = energy_df[energy_df["model"] == model_name]["total_energy"].sort_values()
            if sub.empty:
                continue
            k = max(1, int(len(sub) * best_energy_frac))
            best_slice = sub.head(k)
            best_mean = float(best_slice.mean())
            best_std = float(best_slice.std()) if k > 1 else 0.0
            best_rows.append(
                {
                    "model": model_name,
                    "best_frac_mean_energy": best_mean,
                    "best_frac_std_energy": best_std,
                }
            )
        if best_rows:
            best_df = pd.DataFrame(best_rows)
            plt.figure(figsize=(6, 4))
            order = list(best_df["model"])
            ax = sns.barplot(data=best_df, x="model", y="best_frac_mean_energy", order=order, ci=None)
            x_coords = np.arange(len(order))
            ax.errorbar(
                x_coords,
                best_df["best_frac_mean_energy"],
                yerr=best_df["best_frac_std_energy"],
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=3,
            )
            plt.ylabel(f"Mean energy of best {int(best_energy_frac * 100)}% samples")
            plt.title("Best-energy comparison across models")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"energy_best_{int(best_energy_frac * 100)}pct_bar_by_model.png",
                )
            )
            plt.close()

    print(f"Plots written under: {out_dir}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize evaluation metrics across models and lengths.")
    ap.add_argument("--root", type=str, required=True, help="Root directory used as out_root for evaluation artifacts.")
    ap.add_argument("--experiment", type=str, required=True, help="Experiment name (subfolder under root).")
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model names. If not set, auto-detect subfolders under root/experiment.",
    )
    ap.add_argument(
        "--lengths",
        type=str,
        default=None,
        help="Optional comma-separated list of lengths to include. If not set, auto-detect len_* subfolders.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output directory for summary CSV and plots. "
        "Default: <root>/<experiment>/_summary",
    )
    ap.add_argument(
        "--best_energy_frac",
        type=float,
        default=0.2,
        help="Fraction (0-1) of best (lowest) energy samples per model used for bar plot comparison.",
    )
    args = ap.parse_args()

    summarize(
        root=args.root,
        experiment=args.experiment,
        models=args.models,
        lengths_arg=args.lengths,
        out_dir=args.output,
        best_energy_frac=args.best_energy_frac,
    )


if __name__ == "__main__":
    main()


