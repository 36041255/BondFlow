#!/usr/bin/env python3
"""
分析PDB文件中两条链之间的极性互作个数（包括氢键、盐桥等）
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def analyze_hbonds_between_chains(
    pdb_path: str,
    chain1_id: str = None,
    chain2_id: str = None,
    distance_cutoff: float = 3.5,
    angle_cutoff: float = 120.0
) -> int:
    """
    分析PDB文件中两条链之间的氢键个数
    
    Args:
        pdb_path: PDB文件路径
        chain1_id: 第一条链的ID（如'A'），如果为None则使用第一条链
        chain2_id: 第二条链的ID（如'B'），如果为None则使用第二条链
        distance_cutoff: 氢键距离阈值（Å），默认3.5
        angle_cutoff: 氢键角度阈值（度），默认120.0
    
    Returns:
        两条链之间的氢键个数
    """
    # 解析PDB文件
    ext = os.path.splitext(pdb_path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('struct', pdb_path)
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return 0
    
    # 获取第一个模型
    model = structure[0]
    chains = list(model.get_chains())
    
    if len(chains) < 2:
        print(f"Warning: {pdb_path} has less than 2 chains, skipping.")
        return 0
    
    # 确定要分析的链
    if chain1_id is None:
        chain1 = chains[0]
        chain1_id = chain1.id
    else:
        chain1 = None
        for chain in chains:
            if chain.id == chain1_id:
                chain1 = chain
                break
        if chain1 is None:
            print(f"Warning: Chain {chain1_id} not found in {pdb_path}")
            return 0
    
    if chain2_id is None:
        # 找到不同于chain1_id的链
        chain2 = None
        for chain in chains:
            if chain.id != chain1_id:
                chain2 = chain
                chain2_id = chain.id
                break
        if chain2 is None:
            print(f"Warning: No second chain found in {pdb_path}")
            return 0
    else:
        chain2 = None
        for chain in chains:
            if chain.id == chain2_id:
                chain2 = chain
                break
        if chain2 is None:
            print(f"Warning: Chain {chain2_id} not found in {pdb_path}")
            return 0
    
    # 获取两条链的所有原子
    atoms_chain1 = []
    atoms_chain2 = []
    
    for residue in chain1:
        atoms_chain1.extend(list(residue.get_atoms()))
    
    for residue in chain2:
        atoms_chain2.extend(list(residue.get_atoms()))
    
    # 计算氢键数量
    hbond_count = 0
    
    # 遍历chain1中的每个原子
    for atom1 in atoms_chain1:
        # 遍历chain2中的每个原子
        for atom2 in atoms_chain2:
            # 检查是否是氢键供体-受体对
            # 氢键通常涉及：N-H...O, O-H...O, N-H...N等
            # 这里使用简化的距离和角度判断
            
            # 计算距离
            try:
                distance = atom1 - atom2
            except:
                continue
            
            if distance > distance_cutoff:
                continue
            
            # 检查原子类型是否可能形成氢键
            atom1_name = atom1.get_name().upper()
            atom2_name = atom2.get_name().upper()
            
            # 氢键供体通常是N或O（带H）
            # 氢键受体通常是N或O（不带H或带部分正电荷）
            is_donor1 = atom1.element in ['N', 'O']
            is_acceptor2 = atom2.element in ['N', 'O']
            
            is_donor2 = atom2.element in ['N', 'O']
            is_acceptor1 = atom1.element in ['N', 'O']
            
            # 如果满足氢键条件（距离和原子类型）
            if (is_donor1 and is_acceptor2) or (is_donor2 and is_acceptor1):
                # 进一步检查角度（简化版：检查是否有H原子参与）
                # 对于更精确的检测，需要检查H-X...Y的角度
                residue1 = atom1.get_parent()
                residue2 = atom2.get_parent()
                
                # 检查是否有H原子连接到供体原子
                has_hydrogen = False
                if is_donor1:
                    for atom in residue1:
                        if atom.element == 'H':
                            try:
                                dist_h = atom - atom1
                                if dist_h < 1.2:  # H原子应该在1Å左右
                                    has_hydrogen = True
                                    break
                            except:
                                pass
                
                if is_donor2 and not has_hydrogen:
                    for atom in residue2:
                        if atom.element == 'H':
                            try:
                                dist_h = atom - atom2
                                if dist_h < 1.2:
                                    has_hydrogen = True
                                    break
                            except:
                                pass
                
                if has_hydrogen or distance < 2.5:  # 如果距离很近，即使没有检测到H也认为是氢键
                    hbond_count += 1
    
    return hbond_count


def analyze_polar_interactions_between_chains(
    pdb_path: str,
    chain1_id: str = None,
    chain2_id: str = None,
    distance_cutoff: float = 3.5
) -> int:
    """
    分析PDB文件中两条链之间的极性互作个数（包括氢键、盐桥等）
    
    极性互作包括：
    1. 氢键：极性原子（N, O, S）之间的相互作用，距离 < 3.5 Å
    2. 盐桥：带正电残基（Lys, Arg, His）与带负电残基（Asp, Glu）之间的相互作用，距离 < 4.5 Å
    3. 其他极性相互作用：极性原子之间的相互作用，距离 < 4.5 Å
    
    Args:
        pdb_path: PDB文件路径
        chain1_id: 第一条链的ID（如'A'），如果为None则使用第一条链
        chain2_id: 第二条链的ID（如'B'），如果为None则使用第二条链
        distance_cutoff: 极性互作距离阈值（Å），默认4.5
    
    Returns:
        两条链之间的极性互作个数
    """
    # 解析PDB文件
    ext = os.path.splitext(pdb_path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('struct', pdb_path)
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return 0
    
    # 获取第一个模型
    model = structure[0]
    chains = list(model.get_chains())
    
    if len(chains) < 2:
        print(f"Warning: {pdb_path} has less than 2 chains, skipping.")
        return 0
    
    # 确定要分析的链
    if chain1_id is None:
        chain1 = chains[0]
        chain1_id = chain1.id
    else:
        chain1 = None
        for chain in chains:
            if chain.id == chain1_id:
                chain1 = chain
                break
        if chain1 is None:
            print(f"Warning: Chain {chain1_id} not found in {pdb_path}")
            return 0
    
    if chain2_id is None:
        # 找到不同于chain1_id的链
        chain2 = None
        for chain in chains:
            if chain.id != chain1_id:
                chain2 = chain
                chain2_id = chain.id
                break
        if chain2 is None:
            print(f"Warning: No second chain found in {pdb_path}")
            return 0
    else:
        chain2 = None
        for chain in chains:
            if chain.id == chain2_id:
                chain2 = chain
                break
        if chain2 is None:
            print(f"Warning: Chain {chain2_id} not found in {pdb_path}")
            return 0
    
    # 验证chain1和chain2是不同的链
    if chain1_id == chain2_id:
        print(f"Warning: chain1_id and chain2_id are the same ({chain1_id}). Cannot analyze interactions within the same chain.")
        return 0
    
    # 极性互作检测
    polar_count = 0
    polar_pairs = set()  # 使用set避免重复
    
    # 定义带正电和带负电的残基及其关键原子
    # 带正电残基
    POSITIVE_RESIDUES = {
        'LYS': ['NZ'],
        'ARG': ['NH1', 'NH2', 'NE'],
        'HIS': ['ND1', 'NE2']
    }
    
    # 带负电残基
    NEGATIVE_RESIDUES = {
        'ASP': ['OD1', 'OD2'],
        'GLU': ['OE1', 'OE2']
    }
    
    # 极性原子（可以形成氢键或极性相互作用）
    POLAR_ELEMENTS = ['N', 'O', 'S']
    
    # 遍历chain1中的残基
    for residue1 in chain1:
        resname1 = residue1.get_resname().upper()
        
        # 遍历chain2中的残基
        for residue2 in chain2:
            resname2 = residue2.get_resname().upper()
            
            # 获取残基中的所有原子
            atoms1 = list(residue1.get_atoms())
            atoms2 = list(residue2.get_atoms())
            
            # 验证：确保residue1和residue2来自不同的链
            residue1_chain = residue1.get_parent().id
            residue2_chain = residue2.get_parent().id
            
            if residue1_chain == residue2_chain:
                # 这不应该发生，因为我们已经确保chain1 != chain2
                # 但如果发生，跳过这个残基对
                continue
            
            # 检查每对原子
            for atom1 in atoms1:
                for atom2 in atoms2:
                    
                    try:
                        distance = atom1 - atom2
                    except:
                        continue
                    
                    # 距离过滤
                    if distance > distance_cutoff:
                        continue
                    
                    atom1_name = atom1.get_name().upper()
                    atom2_name = atom2.get_name().upper()
                    element1 = atom1.element
                    element2 = atom2.element
                    
                    # 检查1: 盐桥（带正电残基与带负电残基之间的相互作用）
                    is_salt_bridge = False
                    if resname1 in POSITIVE_RESIDUES and resname2 in NEGATIVE_RESIDUES:
                        if atom1_name in POSITIVE_RESIDUES[resname1] and atom2_name in NEGATIVE_RESIDUES[resname2]:
                            is_salt_bridge = True
                    elif resname2 in POSITIVE_RESIDUES and resname1 in NEGATIVE_RESIDUES:
                        if atom2_name in POSITIVE_RESIDUES[resname2] and atom1_name in NEGATIVE_RESIDUES[resname1]:
                            is_salt_bridge = True
                    
                    # 检查2: 极性原子之间的相互作用（氢键或其他极性相互作用）
                    is_polar_interaction = False
                    if element1 in POLAR_ELEMENTS and element2 in POLAR_ELEMENTS:
                        # 所有极性原子之间的相互作用，只要距离在阈值内都认为是极性互作
                        # 这包括：
                        # - 氢键（距离 < 3.5 Å，通常有H原子参与）
                        # - 其他极性相互作用（距离 < 4.5 Å，如偶极-偶极相互作用）
                        is_polar_interaction = True
                    
                    # 如果满足极性互作条件
                    if is_salt_bridge or is_polar_interaction:
                        # 避免重复计数（使用有序对）
                        pair = tuple(sorted([
                            atom1.get_full_id(),
                            atom2.get_full_id()
                        ]))
                        if pair not in polar_pairs:
                            polar_pairs.add(pair)
                            polar_count += 1
    
    return polar_count


def _calculate_angle(atom1, atom2, atom3):
    """计算三个原子之间的角度（度）"""
    import numpy as np
    from Bio.PDB.vectors import Vector
    
    v1 = Vector(atom1.get_coord()) - Vector(atom2.get_coord())
    v2 = Vector(atom3.get_coord()) - Vector(atom2.get_coord())
    
    v1_norm = v1.normalized()
    v2_norm = v2.normalized()
    
    cos_angle = v1_norm * v2_norm
    cos_angle = max(-1.0, min(1.0, cos_angle))  # 限制在[-1, 1]
    
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    return angle


def _process_single_pdb(
    pdb_file_path: str,
    chain1_id: str = None,
    chain2_id: str = None,
    distance_cutoff: float = 3.5
) -> dict:
    """
    处理单个PDB文件的辅助函数（用于并行处理）
    
    Args:
        pdb_file_path: PDB文件路径
        chain1_id: 第一条链的ID
        chain2_id: 第二条链的ID
        distance_cutoff: 极性互作距离阈值（Å）
    
    Returns:
        包含分析结果的字典
    """
    pdb_file = Path(pdb_file_path)
    try:
        polar_count = analyze_polar_interactions_between_chains(
            str(pdb_file),
            chain1_id=chain1_id,
            chain2_id=chain2_id,
            distance_cutoff=distance_cutoff
        )
        
        # 确定实际使用的链ID
        actual_chain1 = chain1_id
        actual_chain2 = chain2_id
        
        if actual_chain1 is None or actual_chain2 is None:
            # 读取文件获取链ID
            ext = os.path.splitext(pdb_file)[1].lower()
            if ext in [".cif", ".mmcif"]:
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)
            
            try:
                structure = parser.get_structure('temp', str(pdb_file))
                model = structure[0]
                chains = list(model.get_chains())
                if len(chains) >= 2:
                    if actual_chain1 is None:
                        actual_chain1 = chains[0].id
                    if actual_chain2 is None:
                        for chain in chains:
                            if chain.id != actual_chain1:
                                actual_chain2 = chain.id
                                break
            except:
                pass
        
        return {
            'pdb_file': pdb_file.name,
            'chain1_id': actual_chain1,
            'chain2_id': actual_chain2,
            'polar_interaction_count': polar_count
        }
    except Exception as e:
        return {
            'pdb_file': pdb_file.name,
            'chain1_id': None,
            'chain2_id': None,
            'polar_interaction_count': 0,
            'error': str(e)
        }


def analyze_folder_polar_interactions(
    folder_path: str,
    chain1_id: str = None,
    chain2_id: str = None,
    distance_cutoff: float = 3.5,
    num_workers: int = 4
) -> pd.DataFrame:
    """
    读取文件夹下所有PDB文件并分析极性互作（支持并行处理）
    
    Args:
        folder_path: 包含PDB文件的文件夹路径
        chain1_id: 第一条链的ID，如果为None则使用第一条链
        chain2_id: 第二条链的ID，如果为None则使用第二条链
        distance_cutoff: 极性互作距离阈值（Å），默认4.5
        num_workers: 并行处理的进程数，默认4。如果为1则串行处理
    
    Returns:
        包含分析结果的DataFrame，列包括：pdb_file, chain1_id, chain2_id, polar_interaction_count
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder_path}")
    
    # 查找所有PDB文件
    pdb_files = list(folder.glob("*.pdb")) + list(folder.glob("*.cif")) + list(folder.glob("*.mmcif"))
    
    if len(pdb_files) == 0:
        print(f"Warning: No PDB files found in {folder_path}")
        return pd.DataFrame(columns=['pdb_file', 'chain1_id', 'chain2_id', 'polar_interaction_count'])
    
    print(f"Found {len(pdb_files)} PDB files. Analyzing polar interactions...")
    
    results = []
    
    # 串行或并行执行
    if num_workers <= 1:
        # 串行处理（便于调试）
        for pdb_file in tqdm(pdb_files, desc="Analyzing (Serial)"):
            result = _process_single_pdb(
                str(pdb_file),
                chain1_id=chain1_id,
                chain2_id=chain2_id,
                distance_cutoff=distance_cutoff
            )
            results.append(result)
    else:
        # 并行处理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_pdb,
                    str(pdb_file),
                    chain1_id,
                    chain2_id,
                    distance_cutoff
                ): pdb_file
                for pdb_file in pdb_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing (Parallel)"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    pdb_file = futures[future]
                    print(f"Error processing {pdb_file.name}: {e}")
                    results.append({
                        'pdb_file': pdb_file.name,
                        'chain1_id': None,
                        'chain2_id': None,
                        'polar_interaction_count': 0,
                        'error': str(e)
                    })
    
    df = pd.DataFrame(results)
    return df


def main():
    """主函数：指定PDB文件夹路径并保存结果"""
    import argparse
    
    parser = argparse.ArgumentParser(description="分析PDB文件中两条链之间的极性互作个数（包括氢键、盐桥等）")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="包含PDB文件的文件夹路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="polar_interaction_analysis_results.csv",
        help="结果输出文件路径（CSV格式）"
    )
    parser.add_argument(
        "--chain1",
        type=str,
        default=None,
        help="第一条链的ID（如'A'），如果不指定则使用第一条链"
    )
    parser.add_argument(
        "--chain2",
        type=str,
        default=None,
        help="第二条链的ID（如'B'），如果不指定则使用第二条链"
    )
    parser.add_argument(
        "--distance_cutoff",
        type=float,
        default=3.5,
        help="极性互作距离阈值（Å），默认4.5"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行处理的进程数，默认4。如果为1则串行处理"
    )
    
    args = parser.parse_args()
    
    # 分析文件夹中的所有PDB文件
    print(f"Analyzing polar interactions in folder: {args.input}")
    print(f"Using {args.num_workers} worker(s)")
    df_results = analyze_folder_polar_interactions(
        args.input,
        chain1_id=args.chain1,
        chain2_id=args.chain2,
        distance_cutoff=args.distance_cutoff,
        num_workers=args.num_workers
    )
    
    # 保存结果
    output_path = Path(args.output)
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # 计算统计信息
    polar_counts = df_results['polar_interaction_count']
    mean_value = polar_counts.mean()
    median_value = polar_counts.median()
    max_value = polar_counts.max()
    min_value = polar_counts.min()
    
    print(f"\nSummary:")
    print(f"Total PDB files analyzed: {len(df_results)}")
    print(f"Total polar interactions found: {polar_counts.sum()}")
    print(f"\nStatistics:")
    print(f"  Average (mean): {mean_value:.2f}")
    print(f"  Median: {median_value:.2f}")
    print(f"  Maximum: {max_value}")
    print(f"  Minimum: {min_value}")
    print(f"\nFirst few results:")
    print(df_results.head())


if __name__ == "__main__":
    main()
