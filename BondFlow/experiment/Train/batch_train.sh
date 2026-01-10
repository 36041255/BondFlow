#!/bin/bash
#SBATCH -J python
#SBATCH -N 1
#SBATCH -p h01
#SBATCH --no-requeue
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4

CHECKPOINT_DIR="$1" 

# 检查参数是否为空
if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "Error: Checkpoint directory not provided to the job script."
    exit 1
fi

echo "Job started with CHECKPOINT_DIR = ${CHECKPOINT_DIR}"

# Set environment variables
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8

# Activate your environment
source /apps/soft/anaconda3/bin/activate
conda activate apm_env

# Run your application using the passed-in directory
echo "Starting python script..."
python ./train_com.py --checkpoint_dir "${CHECKPOINT_DIR}"
echo "Python script finished."
