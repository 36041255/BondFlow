#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
from multiprocessing import Pool

import pyrosetta
from pyrosetta import rosetta


# ============================================================
# Pool initializer：每个 worker 启动时初始化 PyRosetta
# ============================================================

def worker_init():
    pyrosetta.init(
        "-mute all "
        "-include_sugars false "
    )


# ============================================================
# 链提取
# ============================================================

def extract_chain(pose, chain_id):
    if not pose.pdb_info():
        rosetta.core.pose.PDBInfo(pose)

    keep = [
        i for i in range(1, pose.total_residue() + 1)
        if pose.pdb_info().chain(i) == chain_id
    ]

    if not keep:
        return None

    if len(keep) == pose.total_residue():
        return pose.clone()

    new_pose = pose.clone()
    delete = sorted(
        set(range(1, pose.total_residue() + 1)) - set(keep),
        reverse=True
    )
    for i in delete:
        new_pose.delete_residue_slow(i)

    return new_pose


# ============================================================
# 二硫键检测（版本兼容）
# ============================================================

def detect_disulfides_safe(pose):
    conf = pose.conformation()

    if hasattr(conf, "detect_disulfides"):
        try:
            conf.detect_disulfides()
            return True
        except Exception as e:
            print(f"Error detecting disulfides: {e}")
            pass

    try:
        rosetta.core.pose.initialize_disulfide_bonds(pose)
        return True
    except Exception as e:
        print(f"Error initializing disulfide bonds: {e}")
        return False


# ============================================================
# SAP 计算（Per-residue → total + mean）
# ============================================================

def calculate_sap_metrics(pose):
    """
    返回:
        SAP_total, SAP_mean, n_residues
    """
    # Per-residue SAP metric（Rosetta 官方）
    try:
        SapMetric = rosetta.core.pack.guidance_scoreterms.sap.PerResidueSapScoreMetric
    except Exception:
        raise RuntimeError("PerResidueSapScoreMetric not available in this PyRosetta")

    metric = SapMetric()

    # 兼容不同 cached_calculate 签名
    try:
        sap_map = metric.cached_calculate(pose, True)
    except TypeError:
        sap_map = metric.cached_calculate(pose, True, "", "", False)

    # 转成 python dict
    per_res_sap = {}
    try:
        per_res_sap = dict(sap_map)
    except Exception:
        for k in sap_map:
            per_res_sap[int(k)] = float(sap_map[k])

    if not per_res_sap:
        raise RuntimeError("Empty SAP result")

    sap_vals = list(per_res_sap.values())

    sap_total = float(sum(sap_vals))
    sap_mean  = float(sap_total / len(sap_vals))

    return sap_total, sap_mean, len(sap_vals)


# ============================================================
# 单个 PDB worker
# ============================================================

def calculate_for_pdb(args):
    pdb_path, chain_id = args
    pdb_name = os.path.basename(pdb_path)

    try:
        pose = pyrosetta.pose_from_pdb(pdb_path)

        # 二硫键
        detect_disulfides_safe(pose)

        # 链处理
        if pose.num_chains() > 1:
            sub_pose = extract_chain(pose, chain_id)
            if sub_pose is None:
                return (pdb_name, None, None, None, f"Chain {chain_id} not found")
        else:
            sub_pose = pose

        sap_total, sap_mean, nres = calculate_sap_metrics(sub_pose)

        return (
            pdb_name,
            sap_total,
            sap_mean,
            nres,
            "Success"
        )

    except Exception as e:
        return (pdb_name, None, None, None, str(e))


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parallel SAP calculator (total + mean)"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="包含 PDB 的目录"
    )
    parser.add_argument(
        "--chain", default="A",
        help="目标链 ID（默认 A）"
    )
    parser.add_argument(
        "--n_cores", type=int, default=4,
        help="并行核心数"
    )
    parser.add_argument(
        "--output", default="sap_results.csv",
        help="输出 CSV"
    )

    args = parser.parse_args()

    pdb_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pdb")))
    if not pdb_files:
        print("❌ 未找到 PDB 文件")
        return

    tasks = [(p, args.chain) for p in pdb_files]

    results = []
    with Pool(
        processes=args.n_cores,
        initializer=worker_init
    ) as pool:
        for i, r in enumerate(
            pool.imap_unordered(calculate_for_pdb, tasks), 1
        ):
            results.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"Processed {i}/{len(tasks)}", end="\r")

    df = pd.DataFrame(
        results,
        columns=[
            "PDB_Name",
            "SAP_total",
            "SAP_mean",
            "N_residues",
            "Status"
        ]
    )

    df.to_csv(args.output, index=False)
    print(f"\n✅ Results written to {args.output}")

    ok = df[df["Status"] == "Success"]
    if not ok.empty:
        print("📊 SAP_total stats:")
        print(
            f"  mean={ok['SAP_total'].mean():.2f}  "
            f"min={ok['SAP_total'].min():.2f}  "
            f"max={ok['SAP_total'].max():.2f}"
        )


if __name__ == "__main__":
    main()
