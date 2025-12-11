#!/bin/bash
#SBATCH -J python
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --no-requeue
#SBATCH --cpus-per-task=32
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j


export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
# mlx5_5为存储口，不连接GPU卡

source /apps/soft/anaconda3/bin/activate
conda activate SE3nv

python dataset.py