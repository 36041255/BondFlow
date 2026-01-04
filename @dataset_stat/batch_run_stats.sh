#!/bin/bash
#SBATCH -J python
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --no-requeue
#SBATCH --cpus-per-task=65
#SBATCH -w g02

# Set environment variables
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8

# Activate your environment
source /apps/soft/anaconda3/bin/activate
conda activate cyc_stat

python run_stats.py   --cif_dir /home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/ALL_MMCIF/train_data5/COMPLEX_CIF  \
                   --out_dir /home/fit/lulei/WORK/xjt/Protein_design/BondFlow/@dataset_stat/COMPLEX_CIF3_output \
                   --skip_sasa  --n_workers 64
                   # --skip_secondary