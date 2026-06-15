#!/bin/bash
#SBATCH -J knottin_sample
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --no-requeue
#SBATCH --cpus-per-task=38
#SBATCH --gres=gpu:8
#SBATCH -o slurm_knottin_sample.%j.out
#SBATCH -e slurm_knottin_sample.%j.err


# Set environment variables
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
export HF_HOME="${HF_HOME:-/home/xjt/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

# Activate your environment
source /home/miniconda3/miniconda3/bin/activate
conda activate BondFlow

# 切换到工作目录
cd /home/xjt/BondFlow/BondFlow

# 多GPU采样命令
# 使用2个GPU: cuda:0 和 cuda:1
python sample_knottin.py \
    --cfg /home/xjt/BondFlow/BondFlow/config/Keap1_knottin_design8.yaml \
    --device cuda:4,cuda:5,cuda:6,cuda:7 \
    --min_length 20 \
    --max_length 26 \
    --topology_seed 56 \
    --terminal_bias_prob 0.8 \
    # --no_region_constraints \
    # --preview \
    # --max_preview 10 

# 如果使用4个GPU，可以这样：
# --device cuda:0,cuda:1,cuda:2,cuda:3

# 如果只想预览拓扑（不实际采样），添加：
# --preview --max_preview 10