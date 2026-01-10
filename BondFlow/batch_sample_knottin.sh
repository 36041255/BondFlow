#!/bin/bash
#SBATCH -J knottin_sample
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --no-requeue
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH -o slurm_knottin_sample.%j.out
#SBATCH -e slurm_knottin_sample.%j.err


# Set environment variables
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8

# Activate your environment
source /apps/soft/anaconda3/bin/activate
conda activate apm_env

# 切换到工作目录
cd /home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow

# 多GPU采样命令
# 使用2个GPU: cuda:0 和 cuda:1
python sample_knottin.py \
    --cfg config/cyclize.yaml \
    --device cuda:0,cuda:1,cuda:2,cuda:3 \
    --min_length 20 \
    --max_length 35 \
    --topology_seed 43

# 如果使用4个GPU，可以这样：
# --device cuda:0,cuda:1,cuda:2,cuda:3

# 如果只想预览拓扑（不实际采样），添加：
# --preview --max_preview 10

