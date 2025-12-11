import os
import time
import pandas as pd
import numpy as np
from pyrosetta import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from Bio.PDB import MMCIFParser, PDBIO
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover


def _collect_pdb_like_files(struct_folder, output_dir):
    """
    收集可用于 PyRosetta 能量计算的结构文件：
    - 直接支持 .pdb
    - 若为 .cif / .mmcif，则先转成 PDB，保存在 output_dir 下的 `_cif_to_pdb` 目录中
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
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure(base, full)
                io = PDBIO()
                io.set_structure(structure)
                io.save(out_pdb)
            struct_files.append(out_pdb)

    return struct_files
# ======================== 单个 PDB 的能量计算 ========================
def compute_energy_for_pdb(pdb_path, relax=True):
    """
    计算单个结构的总能量和结合能
    使用 Rosetta 的 InterfaceAnalyzerMover 来计算结合能（更准确）
    如果有多条链：使用 InterfaceAnalyzerMover 分析所有链之间的接口
    """

    import pyrosetta
    pyrosetta.init("-ignore_unrecognized_res true -mute all -multithreading:total_threads 4")

    scorefxn = pyrosetta.rosetta.core.scoring.get_score_function()
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    start_time = time.time()
    print(f"启动计算: {pdb_name} | PID={os.getpid()}", flush=True)
    try:
        pose = pose_from_file(pdb_path)

        # ---------- Relax ----------
        if relax:
            from pyrosetta.rosetta.protocols.relax import FastRelax
            relaxer = FastRelax()
            relaxer.set_scorefxn(scorefxn)
            relaxer.apply(pose)

        # ---------- 能量计算 ----------
        total_energy = scorefxn(pose)
        num_chains = pose.num_chains()

        # 使用 InterfaceAnalyzerMover 计算结合能
        binding_energy = 0.0
        E_protein = total_energy
        E_ligand = 0.0
        
        if num_chains >= 2:
            # 构建接口配置字符串，例如 "A_B" 表示链A和链B之间的接口
            # 获取每个链的链ID（从每个链的第一个残基获取）
            chain_ids = []
            try:
                # 方法1: 通过 pdb_info 获取链ID
                if pose.pdb_info() is not None:
                    # 获取每个链的第一个残基索引
                    chain_start_residues = []
                    for chain_num in range(1, num_chains + 1):
                        chain_start = pose.chain_begin(chain_num)
                        chain_id = pose.pdb_info().chain(chain_start)
                        if chain_id not in chain_ids:
                            chain_ids.append(chain_id)
                else:
                    # 如果没有 pdb_info，使用链编号（A, B, C...）
                    chain_ids = [chr(64 + i) for i in range(1, num_chains + 1)]  # A, B, C...
            except:
                # 如果获取失败，使用链编号（A, B, C...）
                chain_ids = [chr(64 + i) for i in range(1, num_chains + 1)]  # A, B, C...
            
            # 如果至少有两条链，使用 InterfaceAnalyzerMover 分析接口
            if len(chain_ids) >= 2:
                # 构建接口配置：链1_链2 格式（例如 "A_B"）
                interface_config = f"{chain_ids[0]}_{chain_ids[1]}"
                
                try:
                    # 创建 InterfaceAnalyzerMover
                    interface_analyzer = InterfaceAnalyzerMover(interface_config, False, scorefxn)
                    interface_analyzer.set_pack_rounds(0)  # 不进行 repacking
                    interface_analyzer.set_pack_input(False)
                    interface_analyzer.set_compute_packstat(False)
                    interface_analyzer.set_pack_separated(False)
                    
                    # 应用 InterfaceAnalyzer
                    interface_analyzer.apply(pose)
                    
                    # 获取结合能（dG）
                    binding_energy = interface_analyzer.get_interface_dG()
                    
                    # 获取分离后的能量
                    try:
                        protein_pose = pose.split_by_chain(1)
                        ligand_pose = pose.split_by_chain(2)
                        E_protein = scorefxn(protein_pose)
                        E_ligand = scorefxn(ligand_pose)
                    except:
                        # 如果分离失败，使用总能量作为蛋白能量
                        E_protein = total_energy
                        E_ligand = 0.0
                except Exception as e:
                    # 如果 InterfaceAnalyzerMover 失败，使用旧方法作为后备
                    print(f"  [警告] InterfaceAnalyzerMover 失败 ({e})，使用传统方法计算结合能", flush=True)
                    protein_pose = pose.split_by_chain(1)
                    ligand_pose = pose.split_by_chain(2)
                    E_protein = scorefxn(protein_pose)
                    E_ligand = scorefxn(ligand_pose)
                    binding_energy = total_energy - (E_protein + E_ligand)
            else:
                # 无法确定链ID，使用旧方法作为后备
                protein_pose = pose.split_by_chain(1)
                ligand_pose = pose.split_by_chain(2)
                E_protein = scorefxn(protein_pose)
                E_ligand = scorefxn(ligand_pose)
                binding_energy = total_energy - (E_protein + E_ligand)
        else:
            # 单链结构
            E_protein = total_energy
            E_ligand = 0.0
            binding_energy = 0.0

        elapsed = time.time() - start_time
        print(f" [{pdb_name}] 完成计算 | Relax={relax} | Total={total_energy:.3f} | ΔG={binding_energy:.3f} | 耗时={elapsed:.2f}s",flush=True)

        return {
            "PDB": pdb_name,
            "E_complex": total_energy,
            "E_protein": E_protein,
            "E_ligand": E_ligand,
            "Binding_Energy": binding_energy,
            "Total_Energy": total_energy,
            "Has_Ligand": num_chains >= 2,
            "Relaxed": relax,
            "Time_sec": round(elapsed, 2)
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f" [{pdb_name}] 计算出错: {e} (耗时 {elapsed:.2f}s)")
        return {"PDB": pdb_name, "Error": str(e), "Time_sec": round(elapsed, 2)}


# ======================== 批量能量计算 ========================
def batch_energy(pdb_folder, output_dir="results", num_workers=4, relax=True, save_results=False):

    
    """
    批量计算结构文件的能量信息（支持多进程）。
    支持：
      - 直接读取 .pdb
      - 对 .cif / .mmcif 自动转换为临时 PDB 后再用 PyRosetta 计算
    """
    output_dir = os.path.join(output_dir, "energy_results")
    os.makedirs(output_dir, exist_ok=True)
    pdb_files = _collect_pdb_like_files(pdb_folder, output_dir)

    if not pdb_files:
        raise ValueError("未找到任何 PDB 文件！")

    print(f"共检测到 {len(pdb_files)} 个结构，将使用 {num_workers} 个进程进行能量分析...\n")

    start_all = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_energy_for_pdb, pdb_path, relax): pdb_path for pdb_path in pdb_files}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Computing energies"):
            results.append(f.result())

    df = pd.DataFrame(results)
    elapsed_all = time.time() - start_all
    print(f"\n所有计算完成，总耗时 {elapsed_all / 60:.2f} 分钟")

    # =============== 保存结果 ===============
    if save_results:
        output_path = os.path.join(output_dir, "Energy_results.csv")
        df.to_csv(output_path, index=False)
        # 仍然保留 numpy 版本以兼容可能的下游使用
        np.save(output_path.replace(".csv", ".npy"), df.to_numpy())
        print(f"能量结果已保存到: {output_path}")
    else:
        print("未保存结果（save_results=False）")

    return df
