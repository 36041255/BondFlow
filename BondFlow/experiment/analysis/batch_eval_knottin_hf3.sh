#!/bin/bash
#SBATCH -J python
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --no-requeue
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1

# Set environment variables
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8

export PATH=/apps/gpu/cuda/v12.6.1/bin:$PATH
export LD_LIBRARY_PATH=/apps/gpu/cuda/v12.6.1/lib64:$LD_LIBRARY_PATH
export PATH=/apps/soft/hmmer/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
# Memory optimization: reduce preallocation fraction
# IMPORTANT: If not set, JAX defaults to XLA_PYTHON_CLIENT_PREALLOCATE=true and XLA_CLIENT_MEM_FRACTION=0.75
# To reduce memory, you MUST explicitly set these values (uncomment and adjust as needed)
export XLA_PYTHON_CLIENT_PREALLOCATE=true
# XLA_CLIENT_MEM_FRACTION is automatically calculated based on --jobs_per_gpu in Python script
# You can override by setting environment variable here if needed:
# export XLA_CLIENT_MEM_FRACTION=0.124
module load soft/anaconda3/config
source activate
conda activate AF3
which python
 #/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/experiment/knottins_eval_result_hf3 \
 #/home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/KNOTTIN/selected_knottin/ \
python evaluate_knottins_hf3.py \
    --input_dir /home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/KNOTTIN/selected_knottin/ \
    --output_dir /home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/experiment/knottins_eval_result3_hf3 \
    --model_dir /WORK/PUBLIC/alphafold3/alphafold3_Weight \
    --gpus 0 \
    --jobs_per_gpu 8 \
    --num_recycles 5 \
    --num_diffusion_samples 3 \
    --bind_cpu