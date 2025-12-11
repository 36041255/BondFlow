#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
import pandas as pd

# 导入模块
from TMalign import batch_tmalign, run_usalign
from energy import batch_energy, compute_energy_for_pdb
from shannon import extract_sequence_from_pdb,compute_shannon_entropy,shannon_entropy_pipeline,save_sequences_to_fasta
from eval_bonds import eval_bonds_folder
from visualize import visualize_tmalign, visualize_cluster, visualize_energy, visualize_entropy, visualize_bonds, visualize_combined

def main():
    parser = argparse.ArgumentParser(description="结构分析管线主程序")

    # ========== 路径参数 ==========
    parser.add_argument("--input", type=str, default="data/", help="需要比对的 PDB 文件目录")
    parser.add_argument("--output", type=str, default="results", help="结果输出目录")

    # ========== 并行参数 ==========
    parser.add_argument("--alignnum_workers", type=int, default=8, help="TMalign 结构比对并行进程数")
    parser.add_argument("--energynum_workers", type=int, default=8, help="Rosetta 能量计算并行进程数")

    # ========== TMalign 相关 ==========
    parser.add_argument("--useTMalign", action="store_true", help="是否进行 TMalign 比对")
    parser.add_argument("--align_method", type=str, default="USalign", help="USalign/TMalign 可执行文件路径")
    parser.add_argument("--savetmalign_results", action="store_true", help="是否保存 TMalign 比对结果")

    # ========== 层级聚类相关 ==========
    parser.add_argument("--doClusterRMSD", action="store_true", help="是否基于 RMSD 进行层级聚类")
    parser.add_argument("--doClusterTMscore", action="store_true", help="是否基于 TM-score 进行层级聚类")
    parser.add_argument("--cluster_threshold", type=float, default=2.0, help="层级聚类阈值（distance 或 TM-score 转换阈值）")
    parser.add_argument("--cluster_method", type=str, default="average", help="聚类方式（single/complete/average/ward）")

    # ========== 能量计算相关 ==========
    parser.add_argument("--useEnergy", action="store_true", help="是否进行能量分析")
    parser.add_argument("--saveenergy_results", action="store_true", help="是否保存能量分析结果")

    # ========== 键预测评估相关 ==========
    parser.add_argument("--useBondEval", action="store_true", help="是否对预测键进行几何评估")
    parser.add_argument("--link_csv", type=str,
                        default="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/link.csv",
                        help="BondFlow link.csv 规则文件路径")
    parser.add_argument("--savebond_results", action="store_true", help="是否保存键评估结果（Excel）")
    # ======================== 计算位点香农熵 ==========
    parser.add_argument("--doentropy", action="store_true", help="是否计算序列 Shannon entropy")  
    # 可视化选项================================================================
    parser.add_argument("--visualize_tmalign", action="store_true", help="是否可视化 TMalign 热图")
    parser.add_argument("--visualize_cluster", action="store_true", help="是否可视化聚类树状图")
    parser.add_argument("--visualize_energy", action="store_true", help="是否可视化能量分布")
    parser.add_argument("--visualize_entropy", action="store_true", help="是否可视化香农熵")
    parser.add_argument("--visualize_bonds", action="store_true", help="是否可视化键评估结果")
    parser.add_argument("--visualize_combined", action="store_true", help="是否生成汇总 combined_plot")
    args = parser.parse_args()
    # ========== 路径检查 ==========
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists() or not input_path.is_dir():
        print(f"ERROR: 指定的输入目录不存在或不是目录: {input_path}")
        sys.exit(1)

    # =====================================================================
    # Step 1. TMalign 比对
    # =====================================================================
    if args.useTMalign:
        print(f"Start TMalign: input={input_path}, output={output_path}, workers={args.alignnum_workers}")

        df_tmalign, rmsd_matrix, tm_matrix = batch_tmalign(
            pdb_folder=str(input_path),
            output_dir=str(output_path),
            usalign_path=args.align_method,
            num_workers=args.alignnum_workers,
            save_results=args.savetmalign_results
        )

        if df_tmalign is not None:
            print(f"TMalign finished, pairs: {len(df_tmalign)}")
        else:
            print("TMalign returned no result (df is None).")

        # =====================================================================
        # Step 2. 层级聚类分析
        # =====================================================================
        if args.doClusterRMSD or args.doClusterTMscore:
            from cluster import hierarchical_clustering_from_matrix

            cluster_dir = output_path / "cluster_results"
            cluster_dir.mkdir(exist_ok=True)

            # --- RMSD 聚类 ---
            if args.doClusterRMSD:
                print("正在基于 RMSD 进行层级聚类...")
                df_labels_RMSD, Z_rmsd = hierarchical_clustering_from_matrix(
                    rmsd_matrix,
                    metric_type="RMSD",
                    method=args.cluster_method,
                    t=args.cluster_threshold
                )

            # --- TM-score 聚类 ---
            if args.doClusterTMscore:
                print("正在基于 TM-score 进行层级聚类...")
                df_labels_TM, Z_tm = hierarchical_clustering_from_matrix(
                    tm_matrix,
                    metric_type="TM",
                    method=args.cluster_method,
                    t=args.cluster_threshold
                )

    # =====================================================================
    # Step 3. Rosetta 能量计算
    # =====================================================================
    if args.useEnergy:
        print(f"Start Energy Calculation: input={input_path}, output={output_path}, workers={args.energynum_workers}")
        df_energy = batch_energy(
            pdb_folder=str(input_path),
            output_dir=str(output_path),
            num_workers=args.energynum_workers,
            relax=True,
            save_results=args.saveenergy_results
        )
        if df_energy is not None:
            print("能量计算完成。")
        else:
            print("未获得能量计算结果。")

    # ==================== Shannon Entropy ====================
    if args.doentropy:
        entropy_out = output_path / "Shannon_entropy.csv"
        print(f"计算序列 Shannon entropy，并保存到 {entropy_out} ...")
        df_entropy = shannon_entropy_pipeline(str(input_path), output_file=str(entropy_out))

    # ==================== Bond Evaluation ====================
    df_bond_summary = df_bond_per_type = None
    if args.useBondEval:
        bond_out_dir = output_path / "bond_eval_results"
        print(f"Start Bond Evaluation: input={input_path}, output={bond_out_dir}")
        df_bond_summary, df_bond_per_type = eval_bonds_folder(
            struct_dir=str(input_path),
            link_csv=args.link_csv,
            output_dir=str(bond_out_dir),
            save_results=args.savebond_results,
        )
        if df_bond_summary is not None:
            print("键评估完成。")
        else:
            print("未获得任何键评估结果（未找到成对的结构和 bonds txt）。")

    # ============================================================
    visual_files = {}

    if args.visualize_tmalign and args.useTMalign:
        visual_files["tmalign"] = visualize_tmalign(rmsd_matrix, tm_matrix, output_dir=str(output_path/"visualize"))

    if args.visualize_cluster and (args.doClusterRMSD or args.doClusterTMscore):
        cluster_files = []
        if args.doClusterRMSD:
            dend_file_rmsd = visualize_cluster(Z_rmsd, labels=list(rmsd_matrix.index), output_file=str(output_path/"visualize/Cluster_RMSD_dendrogram.png"))
            cluster_files.append(dend_file_rmsd)
        if args.doClusterTMscore:
            dend_file_tm = visualize_cluster(Z_tm, labels=list(tm_matrix.index), output_file=str(output_path/"visualize/Cluster_TMscore_dendrogram.png"))
            cluster_files.append(dend_file_tm)
        visual_files["cluster"] = cluster_files

    if args.visualize_energy and args.useEnergy:
        visual_files["energy"] = visualize_energy(df_energy, output_dir=str(output_path/"visualize"))

    if args.visualize_entropy and args.doentropy:
        visual_files["entropy"] = visualize_entropy(df_entropy, output_dir=str(output_path/"visualize"))

    if args.visualize_bonds and args.useBondEval and df_bond_summary is not None:
        visual_files["bonds"] = visualize_bonds(df_bond_summary, df_bond_per_type, output_dir=str(output_path/"visualize"))

    if args.visualize_combined:
        visualize_combined(
            tmalign_files=visual_files.get("tmalign"),
            cluster_files=visual_files.get("cluster"),
            energy_files=visual_files.get("energy"),
            entropy_file=visual_files.get("entropy"),
            bonds_files=visual_files.get("bonds"),
            output_file=str(output_path/"visualize/combined_plot.png")
    )
if __name__ == "__main__":
    main()
