from dataloader import *
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
data_dir = "/home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/ALL_PDB"
cluster_file = os.path.join(data_dir,"tmp/cluster.tsv")
pdb_feature_dir = os.path.join(data_dir,"PDB_FEATURE")
pdb_dir =  os.path.join(data_dir,"final_cleaned_PDB")
link_file_path = os.path.join(data_dir,"LINK/results.txt")
cache_dir = os.path.join(data_dir,"dataset_cache")
crop_size = 180
val_ratio = 0.1
split = 'train'

# 第一部分：获取数据集
dataset = get_or_create_dataset(
    cluster_file=cluster_file,
    pdb_feature_dir=pdb_feature_dir,
    pdb_dir=pdb_dir,
    link_file_path=link_file_path,
    crop_size=crop_size,
    cache_dir=cache_dir,
    val_ratio=val_ratio,
    split=split 
)

split = 'split'
dataset = get_or_create_dataset(
    cluster_file=cluster_file,
    pdb_feature_dir=pdb_feature_dir,
    pdb_dir=pdb_dir,
    link_file_path=link_file_path,
    crop_size=crop_size,
    cache_dir=cache_dir,
    val_ratio=val_ratio,
    split=split 
)