#!/bin/bash
#SBATCH -J python
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --no-requeue
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1

# Set environment variables
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
export PATH="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/HighFold2/localcolabfold/colabfold-conda/bin:$PATH"
# Activate your environment
source /apps/soft/anaconda3/bin/activate
conda activate apm_env

time python evaluate_knottins.py --input_dir /WORK/PUBLIC/lulei_work/xjt/Protein_design/CyclicPeptide/Dataset/KNOTTIN/selected_knottin/ \
        --output_dir ./knottins_eval_result2 \
        --gpus 0 --jobs_per_gpu 8