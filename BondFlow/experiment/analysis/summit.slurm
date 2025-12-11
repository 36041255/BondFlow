#!/bin/bash
#SBATCH --job-name=nocta_all
#SBATCH -o af3_job.%A.out          # 标准输出文件
#SBATCH -e af3_job.%A.err          # 标准错误输出文件
#SBATCH -p a01                     # 使用的计算分区（根据你们集群的实际分区修改）
#SBATCH --gres=gpu:1               # 请求 1 个 GPU
#SBATCH --cpus-per-task=13         # 每个任务请求 13 个 CPU 核心
#SBATCH --ntasks=1                 # 1 个任务
#SBATCH -t 72:00:00                # 最大运行时间 72 小时

# 加载环境
source /home/username/anaconda3/etc/profile.d/conda.sh
conda activate analysis

# 运行完整管线
python main.py --input data/ --output results/ \
               --useTMalign --savetmalign_results \
               --doClusterRMSD --doClusterTMscore \
               --useEnergy --saveenergy_results \
               --doentropy \
               --visualize_tmalign --visualize_cluster \
               --visualize_energy --visualize_entropy \
               --visualize_combined
