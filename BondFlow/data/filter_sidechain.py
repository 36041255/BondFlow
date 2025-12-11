# -*- coding: utf-8 -*-
"""
筛选CIF文件：找出名字不是"AF-"开头的文件，检查LINK是否为侧链-头、侧链-尾或头-尾连接
如果是，则复制到新文件夹
"""
import os
import sys
import gzip
import shutil
import multiprocessing as mp
from functools import partial
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from tqdm import tqdm

# 配置参数
CONFIG = {
    "input_dir": "/home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/ALL_MMCIF/train_data5/LINKAF_CIF",  # 输入文件夹路径，需要用户指定
    "output_dir": "/home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/ALL_MMCIF/train_data5/filtered_headtail_cifs2",  # 输出文件夹
    "threads": 95,  # 并行线程数
    "bond_length_threshold": 2,  # 键长阈值（Å）
    "max_chains": 12,  # 最大链数，超过此数量的结构将被跳过
}

# 主链原子集合
MAIN_CHAIN_ATOMS = {'N', 'CA', 'C', 'O', 'OXT'}

# 标准氨基酸
STANDARD_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}


def is_main_chain_atom(atom_name):
    """判断是否是主链原子"""
    return atom_name.strip() in MAIN_CHAIN_ATOMS


def is_sidechain_atom(atom_name):
    """判断是否是侧链原子"""
    return not is_main_chain_atom(atom_name)


def is_head_atom(atom_name):
    """判断是否是头原子（N）"""
    return atom_name.strip() == 'N'


def is_tail_atom(atom_name):
    """判断是否是尾原子（C或OXT）"""
    return atom_name.strip() in {'C', 'OXT'}


def calculate_distance(coord1, coord2):
    """计算两个坐标点之间的欧氏距离"""
    return ((coord1[0] - coord2[0])**2 + 
            (coord1[1] - coord2[1])**2 + 
            (coord1[2] - coord2[2])**2)**0.5


def build_coordinate_lookup(mmcif_dict):
    """
    构建原子坐标查找表
    键: (链ID, 残基号, 插入码, 原子名) -> 值: (x, y, z)
    """
    lookup = {}
    keys = [
        '_atom_site.auth_asym_id',
        '_atom_site.auth_seq_id',
        '_atom_site.pdbx_PDB_ins_code',
        '_atom_site.label_atom_id',
        '_atom_site.Cartn_x',
        '_atom_site.Cartn_y',
        '_atom_site.Cartn_z'
    ]
    
    if not all(key in mmcif_dict for key in keys):
        return None
    
    chain_ids = mmcif_dict.get('_atom_site.auth_asym_id', [])
    res_nums = mmcif_dict.get('_atom_site.auth_seq_id', [])
    ins_codes = mmcif_dict.get('_atom_site.pdbx_PDB_ins_code', [])
    atom_names = mmcif_dict.get('_atom_site.label_atom_id', [])
    xs = mmcif_dict.get('_atom_site.Cartn_x', [])
    ys = mmcif_dict.get('_atom_site.Cartn_y', [])
    zs = mmcif_dict.get('_atom_site.Cartn_z', [])
    
    n_atoms = len(chain_ids)
    for i in range(n_atoms):
        try:
            key = (chain_ids[i], res_nums[i], ins_codes[i] if ins_codes[i] else ' ', atom_names[i])
            coords = (float(xs[i]), float(ys[i]), float(zs[i]))
            lookup[key] = coords
        except (ValueError, IndexError, TypeError):
            continue
    
    return lookup


def check_sidechain_link_type(atom1_name, atom2_name):
    """
    检查连接类型
    返回: 'sidechain-head', 'sidechain-tail', 'head-tail', 或 None
    """
    is_sc1 = is_sidechain_atom(atom1_name)
    is_sc2 = is_sidechain_atom(atom2_name)
    is_head1 = is_head_atom(atom1_name)
    is_head2 = is_head_atom(atom2_name)
    is_tail1 = is_tail_atom(atom1_name)
    is_tail2 = is_tail_atom(atom2_name)
    
    # 侧链-头连接（侧链原子连接到N）
    if is_sc1 and is_head2:
        return 'sidechain-head'
    if is_sc2 and is_head1:
        return 'sidechain-head'
    
    # 侧链-尾连接（侧链原子连接到C或OXT）
    if is_sc1 and is_tail2:
        return 'sidechain-tail'
    if is_sc2 and is_tail1:
        return 'sidechain-tail'
    
    # 头-尾连接（N连接到C或OXT）
    if is_head1 and is_tail2:
        return 'head-tail'
    if is_head2 and is_tail1:
        return 'head-tail'
    
    return None


def is_consecutive_peptide_bond(
    chain1, resnum1, atom1_name,
    chain2, resnum2, atom2_name
):
    """
    判断是否为同一条链上、相邻残基之间的正常肽键 (C-N)，这种连接需要被排除。
    """
    # 只考虑同一条链
    if chain1 != chain2:
        return False
    
    # 残基号需能转换为整数，且相差为 1
    try:
        r1 = int(str(resnum1).strip())
        r2 = int(str(resnum2).strip())
    except (ValueError, TypeError):
        return False
    
    if abs(r1 - r2) != 1:
        return False
    
    a1 = atom1_name.strip()
    a2 = atom2_name.strip()
    
    # 正常肽键为主链 C-N 连接
    if (a1 == 'C' and a2 == 'N') or (a1 == 'N' and a2 == 'C'):
        return True
    
    return False


def process_single_cif(cif_file, input_dir, output_dir, bond_length_threshold, max_chains):
    """
    处理单个CIF文件
    返回: (文件名, 是否应该复制)
    """
    cif_path = os.path.join(input_dir, cif_file)
    
    try:
        # 读取CIF文件（支持压缩格式）
        if cif_path.endswith('.gz'):
            with gzip.open(cif_path, 'rt') as f:
                mmcif_dict = MMCIF2Dict(f)
        else:
            mmcif_dict = MMCIF2Dict(cif_path)
        
        # 检查链数量
        chain_ids = mmcif_dict.get('_atom_site.auth_asym_id', [])
        if chain_ids:
            unique_chains = set(chain_ids)
            if len(unique_chains) > max_chains:
                return (cif_file, False)  # 链太多，直接跳过
        
        # 构建坐标查找表
        coord_lookup = build_coordinate_lookup(mmcif_dict)
        if coord_lookup is None:
            return (cif_file, False)
        
        # 获取LINK信息
        conn_ids = mmcif_dict.get('_struct_conn.id', [])
        if not conn_ids:
            return (cif_file, False)
        
        # 获取连接数据
        conn_type_ids = mmcif_dict.get('_struct_conn.conn_type_id', [])
        p1_chains = mmcif_dict.get('_struct_conn.ptnr1_auth_asym_id', [])
        p1_res_names = mmcif_dict.get('_struct_conn.ptnr1_auth_comp_id', [])
        p1_res_nums = mmcif_dict.get('_struct_conn.ptnr1_auth_seq_id', [])
        p1_ins_codes = mmcif_dict.get('_struct_conn.pdbx_ptnr1_PDB_ins_code', [])
        p1_atom_names = mmcif_dict.get('_struct_conn.ptnr1_label_atom_id', [])
        
        p2_chains = mmcif_dict.get('_struct_conn.ptnr2_auth_asym_id', [])
        p2_res_names = mmcif_dict.get('_struct_conn.ptnr2_auth_comp_id', [])
        p2_res_nums = mmcif_dict.get('_struct_conn.ptnr2_auth_seq_id', [])
        p2_ins_codes = mmcif_dict.get('_struct_conn.pdbx_ptnr2_PDB_ins_code', [])
        p2_atom_names = mmcif_dict.get('_struct_conn.ptnr2_label_atom_id', [])
        
        # 检查每个连接
        found_target_link = False
        for i in range(len(conn_ids)):
            # 只检查共价连接（covale）
            conn_type = conn_type_ids[i] if i < len(conn_type_ids) else ''
            if conn_type != 'covale':
                continue
            
            # 只检查标准氨基酸之间的连接
            p1_res_name = p1_res_names[i] if i < len(p1_res_names) else ''
            p2_res_name = p2_res_names[i] if i < len(p2_res_names) else ''
            if p1_res_name not in STANDARD_AMINO_ACIDS or p2_res_name not in STANDARD_AMINO_ACIDS:
                continue
            
            # 获取原子坐标
            p1_chain = p1_chains[i] if i < len(p1_chains) else ''
            p1_res_num = p1_res_nums[i] if i < len(p1_res_nums) else ''
            p1_ins_code = p1_ins_codes[i] if i < len(p1_ins_codes) else ' '
            p1_atom_name = p1_atom_names[i] if i < len(p1_atom_names) else ''
            
            p2_chain = p2_chains[i] if i < len(p2_chains) else ''
            p2_res_num = p2_res_nums[i] if i < len(p2_res_nums) else ''
            p2_ins_code = p2_ins_codes[i] if i < len(p2_ins_codes) else ' '
            p2_atom_name = p2_atom_names[i] if i < len(p2_atom_names) else ''
            
            key1 = (p1_chain, p1_res_num, p1_ins_code, p1_atom_name)
            key2 = (p2_chain, p2_res_num, p2_ins_code, p2_atom_name)
            
            # 跳过自连接
            if key1 == key2:
                continue
            
            coord1 = coord_lookup.get(key1)
            coord2 = coord_lookup.get(key2)
            
            if not coord1 or not coord2:
                continue
            
            # 排除同一条链上、相邻残基之间的正常肽键 (C-N)
            if is_consecutive_peptide_bond(
                p1_chain, p1_res_num, p1_atom_name,
                p2_chain, p2_res_num, p2_atom_name
            ):
                continue
            
            # 可选：检查键长作为合理性验证（共价键通常在1-2Å之间）
            # 如果完全信任LINK记录，可以注释掉以下3行
            distance = calculate_distance(coord1, coord2)
            if distance > bond_length_threshold:
                continue  # 距离太远，可能是数据错误，跳过
            
            # 检查连接类型
            link_type = check_sidechain_link_type(p1_atom_name, p2_atom_name)
            if link_type:
                found_target_link = True
                break  # 找到一个符合条件的连接即可
        
        if found_target_link:
            # 复制文件到输出目录
            output_path = os.path.join(output_dir, cif_file)
            shutil.copy2(cif_path, output_path)
            return (cif_file, True)
        else:
            return (cif_file, False)
            
    except Exception as e:
        # 处理失败时返回False
        return (cif_file, False)


def main():
    """主函数"""
    # # 获取输入目录
    # if len(sys.argv) > 1:
    #     input_dir = sys.argv[1]
    # else:
    #     input_dir = input("请输入输入文件夹路径: ").strip()
    
    # if not os.path.isdir(input_dir):
    #     print(f"错误: 输入目录不存在: {input_dir}")
    #     sys.exit(1)
    
    # CONFIG["input_dir"] = input_dir
    input_dir = CONFIG["input_dir"]
    # 创建输出目录
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CIF文件（排除"AF-"开头的文件）
    all_files = os.listdir(input_dir)
    cif_files = [
        f for f in all_files 
        if (f.endswith('.cif') or f.endswith('.cif.gz')) 
        #and not f.startswith('AF-')
    ]
    
    if not cif_files:
        print(f"在目录 '{input_dir}' 中未找到符合条件的CIF文件（非'AF-'开头）")
        return
    
    print(f"找到 {len(cif_files)} 个符合条件的CIF文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"使用 {CONFIG['threads']} 个进程处理...")
    
    # 创建处理函数（固定参数）
    process_func = partial(
        process_single_cif,
        input_dir=input_dir,
        output_dir=output_dir,
        bond_length_threshold=CONFIG["bond_length_threshold"],
        max_chains=CONFIG["max_chains"]
    )
    
    # 并行处理
    copied_count = 0
    with mp.Pool(processes=CONFIG['threads']) as pool:
        results = list(tqdm(
            pool.imap(process_func, cif_files),
            total=len(cif_files),
            desc="处理CIF文件"
        ))
    
    # 统计结果
    for filename, should_copy in results:
        if should_copy:
            copied_count += 1
    
    print(f"\n处理完成!")
    print(f"总共处理: {len(cif_files)} 个文件")
    print(f"符合条件的文件: {copied_count} 个")
    print(f"已复制到: {output_dir}")


if __name__ == "__main__":
    main()

