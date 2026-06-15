# cluster.py
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import os

def hierarchical_clustering_from_matrix(
    mat_df,
    metric_type="RMSD",
    method="average",
    t=1.0,
    criterion="distance",
    max_clusters=None,
    output_dir="results",
    save_results=True
):
    """
    对相似性或距离矩阵执行层级聚类并输出聚类标签表格。

    参数:
        mat_df: pandas.DataFrame
            方阵（n x n），index 和 columns 为结构名。
        metric_type: str
            "RMSD" 或 "TM"，决定是否对矩阵进行 (1 - TM) 转换。
        method: str
            linkage 聚类方式 ('single', 'complete', 'average', 'ward' 等)。
        t: float
            阈值（用于 criterion='distance'）。
        criterion: str
            'distance' 或 'maxclust'。
        max_clusters: int
            若使用 maxclust，可手动指定簇数量。
        output_dir: str
            输出目录，默认 "results"。
        save_results: bool
            是否保存聚类结果为 Excel。

    返回:
        df_labels: DataFrame，列 ["Structure", "Cluster"]
        Z: linkage 矩阵
    """
    output_dir = os.path.join(output_dir, "cluster_results")
    os.makedirs(output_dir, exist_ok=True)

    names = list(mat_df.index)
    mat = mat_df.values.astype(float)

    # TM-score 转换为距离矩阵
    if metric_type.upper() == "TM":
        dist_mat = 1.0 - mat
    else:
        dist_mat = mat.copy()

    np.fill_diagonal(dist_mat, 0.0)
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method=method)

    # 聚类分配
    if criterion == "distance":
        labels = fcluster(Z, t=t, criterion='distance')
    else:
        k = max_clusters if max_clusters is not None else int(t)
        labels = fcluster(Z, t=k, criterion='maxclust')

    df_labels = pd.DataFrame({"Structure": names, "Cluster": labels})

    if save_results:
        output_path = os.path.join(output_dir, f"cluster_labels_{metric_type}.csv")
        df_labels.to_csv(output_path, index=False)
        print(f"聚类结果已保存: {output_path}")
    else:
        print("ℹ未保存聚类结果 (save_results=False)")

    return df_labels, Z