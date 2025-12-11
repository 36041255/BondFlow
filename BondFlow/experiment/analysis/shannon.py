# shannon.py
import os
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from collections import Counter
import numpy as np

def extract_sequence_from_pdb(pdb_path):
    """
    从结构文件中提取氨基酸序列。
    支持：
      - PDB (.pdb)
      - mmCIF (.cif / .mmcif)
    """
    ext = os.path.splitext(pdb_path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("X", pdb_path)
    seq = ""

    for model in structure:
        for chain in model:
            for res in chain:
                resname = res.get_resname()
                try:
                    seq += protein_letters_3to1[resname.capitalize()]
                except KeyError:
                    seq += "X"
    return seq


def save_sequences_to_fasta(pdb_folder, fasta_file="sequences.fasta"):
    """
    遍历 pdb_folder 下的 PDB 文件，提取序列并保存为 fasta
    如果 fasta 已存在，直接读取
    """
    fasta_file = os.path.join(pdb_folder, fasta_file)

    # 如果 fasta 文件已存在，直接读取
    if os.path.exists(fasta_file):
        records = list(SeqIO.parse(fasta_file, "fasta"))
        return {rec.id: str(rec.seq) for rec in records}

    records = []
    seq_dict = {}
    for pdb_file in os.listdir(pdb_folder):
        ext = os.path.splitext(pdb_file)[1].lower()
        if ext not in [".pdb", ".cif", ".mmcif"]:
            continue
        pdb_path = os.path.join(pdb_folder, pdb_file)
        seq = extract_sequence_from_pdb(pdb_path)
        pdb_id = os.path.splitext(pdb_file)[0]
        seq_dict[pdb_id] = seq
        records.append(SeqRecord(Seq(seq), id=pdb_id, description=""))

    SeqIO.write(records, fasta_file, "fasta")
    return seq_dict


def compute_shannon_entropy(sequences):
    """
    计算每个位点的香农熵
    sequences: dict, {pdb_id: sequence}
    返回 DataFrame，每列为 PDB 名称，每行为位点，值为香农熵
    """
    seq_list = list(sequences.values())
    seq_ids = list(sequences.keys())

    # 检查序列长度一致性
    lengths = [len(s) for s in seq_list]
    if len(set(lengths)) != 1:
        print("⚠️ 序列长度不一致，跳过 Shannon Entropy 计算")
        return None

    L = lengths[0]
    entropy = []
    for i in range(L):
        residues = [s[i] for s in seq_list]
        counts = Counter(residues)
        probs = np.array(list(counts.values())) / sum(counts.values())
        shannon = -np.sum(probs * np.log2(probs))
        entropy.append(shannon)

    df_entropy = pd.DataFrame(entropy, columns=["ShannonEntropy"])
    df_entropy.index.name = "Position"
    return df_entropy


def shannon_entropy_pipeline(data_dir, fasta_file="sequences.fasta", output_file="ShannonEntropy.csv"):
    """
    主流程：读取序列（从 fasta 或 PDB），计算香农熵并保存 CSV
    """
    sequences = save_sequences_to_fasta(data_dir, fasta_file)
    df_entropy = compute_shannon_entropy(sequences)
    if df_entropy is not None:
        df_entropy.to_csv(output_file)
        print(f"Shannon Entropy 已保存：{output_file}")
    return df_entropy
