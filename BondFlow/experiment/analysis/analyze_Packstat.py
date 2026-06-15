import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp  # <--- 新增这行
# 为了在多进程中正常工作，PyRosetta 的 import 最好放在函数内或确保环境隔离
# 但通常只要在 worker 内 init 即可
import pyrosetta
from pyrosetta.rosetta.core.scoring import packstat
from pyrosetta.rosetta.core.pose import pdbslice
import os
from pathlib import Path
import logging

# configure logging at module level once
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def init_pyrosetta(init_options: str = "-mute all"):
    """
    Safe idempotent initializer for PyRosetta inside worker processes.
    Call this inside each worker/process before using PyRosetta APIs.
    """
    # import locally so this function is safe to import at top-level
    import pyrosetta
    if not getattr(init_pyrosetta, "_inited", False):
        pyrosetta.init(init_options)
        init_pyrosetta._inited = True
def process_single_pdb(args):
    """
    args: tuple (file_path: str, target_chain: str|None, do_relax: bool)
    returns: dict with keys: filename, chain, packstat, status, error_msg
    """
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    # 引入 pdbslice 用于切片
    from pyrosetta.rosetta.core.pose import pdbslice 
    from pyrosetta.rosetta.core.scoring import packstat
    # 移除不需要的 pdb_io 引用，避免误用
    # import pyrosetta.rosetta.core.io.pdb as pdb_io 

    file_path, target_chain, do_relax = args
    result = {
        "filename": os.path.basename(file_path),
        "chain": target_chain or "ALL",
        "packstat": None,
        "status": "Success",
        "error_msg": "",
    }

    try:
        # initialize PyRosetta only once per process
        init_pyrosetta()

        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        # 1. load
        pose = pyrosetta.pose_from_pdb(str(p))

        # 2. optional relax
        if do_relax:
            scorefxn = pyrosetta.get_score_function()
            relax = pyrosetta.rosetta.protocols.relax.FastRelax()
            relax.set_scorefxn(scorefxn)
            relax.constrain_relax_to_start_coords(True)
            relax.apply(pose)

        # 3. extract chain if requested
        if target_chain:
            res_indices = []
            for i in range(1, pose.total_residue() + 1):
                # 注意：pose.pdb_info() 可能为 None（虽然从 PDB 读取通常会有），加个安全检查
                if pose.pdb_info() and pose.pdb_info().chain(i) == target_chain:
                    res_indices.append(i)

            if not res_indices:
                raise ValueError(f"Chain '{target_chain}' not found in {file_path}")

            # 准备 indices vector
            v = pyrosetta.rosetta.utility.vector1_unsigned_long()
            for idx in res_indices:
                v.append(idx)

            # =========================
            # 修改核心：使用 pdbslice
            # =========================
            sliced_pose = Pose()
            # pdbslice 会将 pose 中 v 指定的残基复制到 sliced_pose 中
            pdbslice(sliced_pose, pose, v)
            
            # 更新 pose 变量
            pose = sliced_pose
            
            # 如果你想确保类似 "dump and reload" 的彻底清理效果（重置 numbering 等），
            # pdbslice 生成的 pose 已经是全新的对象，通常不需要再转 string。
            # 如果必须重置 PDBInfo，可以手动重置，但计算 PackStat 通常不需要。

        # 4. sync internal graphs and clear energies
        if hasattr(pose, "update_residue_neighbors"):
            pose.update_residue_neighbors()
        try:
            pose.energies().clear()
        except Exception:
            pass

        # 5. compute PackStat
        try:
            score = packstat.compute_packing_score(pose, oversample=100)
            result["packstat"] = round(float(score), 4)
        except Exception as pack_exc:
            logging.warning("packstat compute error on %s: %s", file_path, pack_exc)
            try:
                score = packstat.compute_packing_score(pose, oversample=10)
                result["packstat"] = round(float(score), 4)
            except Exception as pack_exc2:
                result["status"] = "Error"
                result["error_msg"] = f"Packstat failed: {pack_exc2}"
                return result

    except Exception as e:
        logging.exception("Error processing %s", file_path)
        result["status"] = "Error"
        result["error_msg"] = str(e)

    return result

def main():
    parser = argparse.ArgumentParser(description="并行计算 PDB 文件的 PackStat")
    parser.add_argument("folder", type=str, help="包含 PDB 文件的文件夹路径")
    parser.add_argument("--chain", type=str, default=None, help="指定计算的 Chain ID (例如 'A')。默认计算整体。")
    parser.add_argument("--output", type=str, default="packstat_results.csv", help="结果保存的 CSV 文件名")
    parser.add_argument("--workers", type=int, default=os.cpu_count() - 1, help="并行进程数")
    parser.add_argument("--relax", action="store_true", help="是否在计算前进行简单的 Relax (会显著变慢，慎用)")
    
    args = parser.parse_args()

    # 获取文件列表
    pdb_files = glob.glob(os.path.join(args.folder, "*.pdb"))
    if not pdb_files:
        print(f"在 {args.folder} 中未找到 PDB 文件。")
        return

    print(f"找到 {len(pdb_files)} 个 PDB 文件。开始处理...")
    print(f"目标 Chain: {args.chain if args.chain else 'Whole Structure'}")
    print(f"并行核心数: {args.workers}")
    if args.relax:
        print("警告: 已开启 Relax 预处理，速度将显著变慢。")

    # 准备任务参数
    tasks = [(f, args.chain, args.relax) for f in pdb_files]
    results = []

    # 并行执行
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 使用 tqdm 显示进度条
        futures = {executor.submit(process_single_pdb, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Calculating PackStat"):
            res = future.result()
            results.append(res)
            
            # (可选) 实时打印低分或报错的结构，方便调试
            if res["status"] == "Error":
                tqdm.write(f"Error processing {res['filename']}: {res['error_msg']}")
            # elif res["packstat"] is not None and res["packstat"] < 0.58:
            #     tqdm.write(f"Warning: {res['filename']} has low packing ({res['packstat']})")

    # 汇总并保存
    df = pd.DataFrame(results)
    
# --- 数据统计部分 ---
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    
    # 筛选计算成功的行
    success_df = df[df["status"] == "Success"]

    print("\n" + "="*40)
    print("           PACKSTAT 统计报告")
    print("="*40)

    if success_df.empty:
        print("没有成功计算任何文件的 PackStat。")
    else:
        # 计算统计量
        count = len(success_df)
        mean_val = success_df["packstat"].mean()
        median_val = success_df["packstat"].median()
        max_val = success_df["packstat"].max()
        min_val = success_df["packstat"].min()
        std_val = success_df["packstat"].std()

        # 找到最大最小值的对应文件名
        # idxmax/idxmin 返回索引，用 .loc 取值
        max_file = success_df.loc[success_df["packstat"].idxmax(), "filename"]
        min_file = success_df.loc[success_df["packstat"].idxmin(), "filename"]

        print(f"成功数量 : {count} / {len(df)}")
        print(f"平均值   : {mean_val:.4f}  (Std: {std_val:.4f})")
        print(f"中位数   : {median_val:.4f}")
        print("-" * 40)
        print(f"最大值   : {max_val:.4f} -> {max_file}")
        print(f"最小值   : {min_val:.4f} -> {min_file}")
    
    print("="*40)
    print(f"详细结果已保存至: {args.output}")
    
    df.to_csv(args.output, index=False)
    print(f"结果已保存至: {args.output}")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # 如果程序已经运行过一次（比如在 Jupyter 里），再次设置会报错，可以忽略
        pass
    main()