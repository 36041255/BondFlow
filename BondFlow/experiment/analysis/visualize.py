# visualize.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

plt.rcParams.update({'font.size': 10})
plt.switch_backend('Agg')  # 服务器上非交互模式
def visualize_tmalign(rmsd_matrix, tm_matrix, output_dir="visualize"):
    os.makedirs(output_dir, exist_ok=True)
    
    # RMSD 热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(rmsd_matrix, cmap="viridis", annot=True, fmt=".2f")
    plt.title("RMSD Heatmap")
    plt.tight_layout()
    rmsd_file = os.path.join(output_dir, "RMSD_heatmap.png")
    plt.savefig(rmsd_file)
    plt.close()
    
    # TM-score 热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(tm_matrix, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("TM-score Heatmap")
    plt.tight_layout()
    tm_file = os.path.join(output_dir, "TMscore_heatmap.png")
    plt.savefig(tm_file)
    plt.close()
    
    print(f"TMalign 热图已保存: {rmsd_file}, {tm_file}")
    return rmsd_file, tm_file

def visualize_cluster(Z, labels, output_file="cluster_dendrogram.png", figsize=(10, 6)):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.figure(figsize=figsize)
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=10, color_threshold=None)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Structures")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Dendrogram 已保存: {output_file}")
    return output_file

def visualize_energy(df_energy, output_dir="visualize"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Total Energy
    plt.figure(figsize=(8, 6))
    sns.barplot(x="PDB", y="Total_Energy", data=df_energy)
    plt.xticks(rotation=90)
    plt.title("Total Energy per Structure")
    plt.tight_layout()
    total_file = os.path.join(output_dir, "TotalEnergy.png")
    plt.savefig(total_file)
    plt.close()
    
    # Binding Energy
    plt.figure(figsize=(8, 6))
    sns.barplot(x="PDB", y="Binding_Energy", data=df_energy)
    plt.xticks(rotation=90)
    plt.title("Binding Energy per Structure")
    plt.tight_layout()
    bind_file = os.path.join(output_dir, "BindingEnergy.png")
    plt.savefig(bind_file)
    plt.close()
    
    print(f"能量图已保存: {total_file}, {bind_file}")
    return total_file, bind_file

def visualize_entropy(df_entropy, output_dir="visualize"):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    df_plot = df_entropy.reset_index()  # 索引 Position 变成一列
    sns.barplot(x="Position", y="ShannonEntropy", data=df_plot)
    plt.xticks(rotation=90)
    plt.title("Shannon Entropy per Position")
    plt.tight_layout()
    entropy_file = os.path.join(output_dir, "ShannonEntropy.png")
    plt.savefig(entropy_file)
    plt.close()
    
    print(f"香农熵图已保存: {entropy_file}")
    return entropy_file

def visualize_bonds(df_summary, df_per_type=None, output_dir="visualize"):
    """
    键评估结果可视化：
      - 每个结构的 AUC 条形图
      - 每个结构的 F1_all 条形图
    """
    os.makedirs(output_dir, exist_ok=True)

    # AUC per structure
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Structure", y="AUC", data=df_summary)
    plt.xticks(rotation=90)
    plt.title("Bond Evaluation AUC per Structure")
    plt.tight_layout()
    auc_file = os.path.join(output_dir, "BondEval_AUC.png")
    plt.savefig(auc_file)
    plt.close()

    # F1_all per structure
    if "F1_all" in df_summary.columns:
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Structure", y="F1_all", data=df_summary)
        plt.xticks(rotation=90)
        plt.title("Bond Evaluation F1 (all) per Structure")
        plt.tight_layout()
        f1_file = os.path.join(output_dir, "BondEval_F1_all.png")
        plt.savefig(f1_file)
        plt.close()
        files = [auc_file, f1_file]
    else:
        files = [auc_file]

    print(f"键评估图已保存: {', '.join(files)}")
    return files

def visualize_combined(tmalign_files=None, cluster_files=None, energy_files=None, entropy_file=None, bonds_files=None, output_file="combined_plot.png"):
    """
    简单组合展示所有已有图片
    """
    import matplotlib.image as mpimg
    
    files = []
    if tmalign_files: files.extend(tmalign_files)
    if cluster_files: files.extend(cluster_files)
    if energy_files: files.extend(energy_files)
    if entropy_file: files.append(entropy_file)
    if bonds_files: files.extend(bonds_files)
    
    n = len(files)
    if n == 0:
        print("No visualization files to combine.")
        return None
    
    cols = 2
    rows = (n + 1) // 2
    
    plt.figure(figsize=(cols*6, rows*5))
    
    for i, f in enumerate(files):
        img = mpimg.imread(f)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(f))
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Combined plot 已保存: {output_file}")
    return output_file
