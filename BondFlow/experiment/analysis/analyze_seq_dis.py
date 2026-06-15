import os
import glob
import argparse
import warnings
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial

# --- 修复点 1: 必须在导入 pyplot 之前设置后端 ---
import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端，防止在服务器上卡死

# 第三方库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from Bio import PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="统计PDB氨基酸分布 (修复服务器绘图卡死问题)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-i', '--input', required=True, type=str, help="PDB文件夹路径")
    parser.add_argument('-l', '--length', type=int, default=300, help="序列长度阈值")
    parser.add_argument('-c', '--chain', type=str, default=None, help="[可选] 指定链ID")
    parser.add_argument('-o', '--output', type=str, default="output/aa_distribution.png", help="输出图片路径")
    parser.add_argument('-j', '--jobs', type=int, default=max(1, cpu_count() - 1), help="并行核心数")
    parser.add_argument('-p', '--pattern', type=str, default="*.pdb", help="文件后缀模式")

    return parser.parse_args()

def process_single_pdb(file_path, max_len, target_chain=None):
    warnings.simplefilter('ignore', PDBConstructionWarning)
    parser = PDB.PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('struct', file_path)
        model = list(structure)[0]
        
        residues = []
        for chain in model:
            if target_chain is not None and chain.id != target_chain:
                continue

            for residue in chain:
                if is_aa(residue, standard=True):
                    res_name = residue.get_resname().upper()
                    if res_name in protein_letters_3to1:
                        one_letter = protein_letters_3to1[res_name]
                        residues.append(one_letter)
        
        seq_len = len(residues)
        
        if 0 < seq_len <= max_len:
            return Counter(residues)
        else:
            return None
            
    except Exception:
        return None

def main():
    args = parse_arguments()
    
    if not os.path.isdir(args.input):
        print(f"Error: 输入路径 '{args.input}' 不存在。")
        return

    pdb_files = glob.glob(os.path.join(args.input, args.pattern))
    total_files = len(pdb_files)
    
    if total_files == 0:
        print(f"未找到匹配 '{args.pattern}' 的文件。")
        return

    print(f"--- 任务开始 ---")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"筛选: 长度 <= {args.length}")
    if args.chain:
        print(f"筛选: Chain '{args.chain}'")
    print(f"并行: {args.jobs} 核心")

    func = partial(process_single_pdb, max_len=args.length, target_chain=args.chain)
    total_counter = Counter()
    valid_structures = 0
    
    with Pool(processes=args.jobs) as pool:
        iterator = pool.imap_unordered(func, pdb_files, chunksize=10)
        for result in tqdm(iterator, total=total_files, unit="pdb"):
            if result is not None:
                total_counter.update(result)
                valid_structures += 1

    print(f"\n处理完成。有效结构: {valid_structures} / {total_files}")

    if valid_structures == 0:
        print("无有效数据，程序结束。")
        return

    # 数据整理
    aa_data = pd.DataFrame.from_dict(total_counter, orient='index', columns=['Count']).reset_index()
    aa_data.columns = ['Amino Acid', 'Count']
    total_aa = aa_data['Count'].sum()
    aa_data['Frequency (%)'] = (aa_data['Count'] / total_aa) * 100
    aa_data = aa_data.sort_values(by='Count', ascending=False)

    print("\n=== 氨基酸分布统计表 ===")
    print(f"总氨基酸数: {total_aa}")
    print(aa_data.to_string(index=False, formatters={'Frequency (%)': '{:.2f}'.format}))
    print("========================\n")

    # --- 增加调试信息 ---
    print("正在初始化绘图画布...", end="", flush=True)
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    print(" 完成。")

    print("正在绘制柱状图...", end="", flush=True)
    ax = sns.barplot(x='Amino Acid', y='Frequency (%)', data=aa_data, palette='viridis', hue='Amino Acid', legend=False)
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.1, f'{height:.1f}%', ha="center", fontsize=10)

    title_str = f'AA Distribution (Len <= {args.length}, N={valid_structures}'
    if args.chain:
        title_str += f', Chain={args.chain}'
    title_str += ')'
    
    plt.title(title_str, fontsize=15)
    plt.xlabel('Amino Acid', fontsize=12)
    plt.ylabel('Frequency (%)', fontsize=12)
    plt.tight_layout()
    print(" 完成。")
    
    # 路径检查
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"创建输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    print(f"正在保存图片至: {args.output} ...", end="", flush=True)
    try:
        plt.savefig(args.output, dpi=300)
        print(" 成功！")
        print("\n所有任务已完成。")
    except Exception as e:
        print(f"\n保存图片失败: {e}")

if __name__ == "__main__":
    main()