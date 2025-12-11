#!/bin/bash

# ======================================================================
# 1. 在这里定义你的路径 (全局唯一的配置点)
# ======================================================================
CHECKPOINT_DIR="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/Train/weight_apm_backbone_monomer45"
#CHECKPOINT_DIR="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/Train/weight_apm_sidechain_monomer17"

# ======================================================================
# 2. 准备工作：基于该路径创建日志文件夹
# 在提交前就创建好，更加稳妥
# ======================================================================
LOG_DIR="${CHECKPOINT_DIR}/slurm_logs"
echo "Log directory will be: ${LOG_DIR}"
mkdir -p "${LOG_DIR}"


# ======================================================================
# 3. 使用 sbatch 命令提交作业脚本
#    - 通过命令行参数 --output 和 --error 动态设置日志路径
#    - 将 CHECKPOINT_DIR 变量作为参数传递给 job.slurm 脚本
# ======================================================================
sbatch \
  --job-name="train_job" \
  --output="${LOG_DIR}/%j.out" \
  --error="${LOG_DIR}/%j.err" \
  batch_train.sh "${CHECKPOINT_DIR}"

echo "Job submitted. Check status with squeue -u $(whoami)"