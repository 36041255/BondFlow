#!/bin/bash
#SBATCH -J python
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=64 #单节点CPU数
#SBATCH --gres=gpu:1

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
# mlx5_5为存储口，不连接GPU卡

source /apps/soft/anaconda3/bin/activate
conda activate base
n_cpu=64

# Small end-to-end demo:
# 1) Use the *generation* conda env to sample a few proteins of length 50 and 100
#    with BondFlow.models.mymodel.MySampler (cyclize config).
# 2) Use the *evaluation* conda env to run the analysis pipeline under
#    BondFlow/experiment/analysis on the generated structures.
#
# NOTE:
# - Please edit GEN_ENV / EVAL_ENV / DEVICE as appropriate for your machines.
# - This script is intentionally simple and uses only one model: bondflow_cyclize.

set -euo pipefail

############################
### User-editable config ###
############################

# Conda env used to run the generative model
GEN_ENV="apm_env"    # TODO: replace with your actual generation env name
# Conda env used to run analysis (PyRosetta, TMalign, etc.)
EVAL_ENV="analysis"  # TODO: replace with your actual evaluation env name

# Worker counts for analysis (inside analysis/main.py)
ALIGN_WORKERS=$n_cpu    # TMalign/USalign parallel processes
ENERGY_WORKERS=$n_cpu     # PyRosetta energy parallel processes

# Torch device for generation
DEVICE="cuda:0"           # or "cpu"

# Model + config
MODEL_NAME="bondflow_cyclize"
MODEL_CONFIG="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/cyclize.yaml"

# Lengths and sample counts for this tiny demo
LENGTHS_CSV="8,10,12,14,16,18,20"
LENGTHS_ARR=(8 10 12 14 16 18 20)
NUM_SAMPLES_PER_LENGTH=100
BATCH_SIZE=100

# Experiment + output root
EXP_NAME="demo_small"

# Resolve project root (this script lives in BondFlow/model_eval/)
PROJECT_ROOT="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/"
OUT_ROOT="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/experiment/model_eval/uncondition43_epoch42"

echo "Project root: ${PROJECT_ROOT}"
echo "Artifacts root: ${OUT_ROOT}"
echo "Experiment: ${EXP_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Lengths: ${LENGTHS_CSV}"
echo

# Ensure Python uses the source-tree BondFlow package (not an old installed one)
cd "${PROJECT_ROOT}"

##########################
### 1. Generation step ###
##########################

echo "=== [1/2] Running generation in conda env: ${GEN_ENV} ==="

conda run -n "${GEN_ENV}" \
  python -m BondFlow.experiment.model_eval.generate_structures \
    --model_name "${MODEL_NAME}" \
    --config "${MODEL_CONFIG}" \
    --device "${DEVICE}" \
    --lengths "${LENGTHS_CSV}" \
    --num_samples_per_length "${NUM_SAMPLES_PER_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --out_root "${OUT_ROOT}" \
    --experiment_name "${EXP_NAME}"

echo "Generation finished."
echo

#########################
### 2. Analysis step  ###
#########################

echo "=== [2/3] Running analysis in conda env: ${EVAL_ENV} ==="

ANALYSIS_MAIN="${PROJECT_ROOT}/BondFlow/experiment/analysis/main.py"

for L in "${LENGTHS_ARR[@]}"; do
  LEN_DIR="${OUT_ROOT}/${EXP_NAME}/${MODEL_NAME}/len_${L}"

  # Prefer post_refine outputs if they exist, otherwise fall back to pre_refine
  INPUT_DIR_POST="${LEN_DIR}/post_refine"
  INPUT_DIR_PRE="${LEN_DIR}/pre_refine"

  if [[ -d "${INPUT_DIR_POST}" ]]; then
    INPUT_DIR="${INPUT_DIR_POST}"
  elif [[ -d "${INPUT_DIR_PRE}" ]]; then
    INPUT_DIR="${INPUT_DIR_PRE}"
  else
    echo "WARNING: No pre_refine/post_refine directory found for length ${L}, skip."
    continue
  fi

  OUTPUT_DIR="${LEN_DIR}/analysis"
  mkdir -p "${OUTPUT_DIR}"

  echo "Running analysis for length ${L}:"
  echo "  input : ${INPUT_DIR}"
  echo "  output: ${OUTPUT_DIR}"

  conda run -n "${EVAL_ENV}" \
    python "${ANALYSIS_MAIN}" \
      --input "${INPUT_DIR}" \
      --output "${OUTPUT_DIR}" \
      --alignnum_workers "${ALIGN_WORKERS}" \
      --energynum_workers "${ENERGY_WORKERS}" \
      --useBondEval \
      --savebond_results \
      --visualize_bonds \
      --visualize_combined

  # 如需恢复结构比对 / 能量选项，可以在上面的命令块中按需加回：
  #   --useTMalign \
  #   --doClusterTMscore \
  #   --useEnergy \
  #   --savetmalign_results \
  #   --saveenergy_results \
  #   --visualize_tmalign \
  #   --visualize_cluster \
  #   --visualize_energy \
  #      --visualize_entropy \
  #      --doentropy \

  echo "Analysis for length ${L} done."
  echo
done

######################################
### 3. Cross-model/length summary  ###
######################################

echo "=== [3/3] Summarizing metrics across lengths/models ==="

conda run -n "${EVAL_ENV}" \
  python -m BondFlow.experiment.model_eval.summarize_across_models \
    --root "${OUT_ROOT}" \
    --experiment "${EXP_NAME}"

echo "Demo finished. Check results under:"
echo "  ${OUT_ROOT}/${EXP_NAME}/${MODEL_NAME}/len_*/analysis"
echo "  ${OUT_ROOT}/${EXP_NAME}/_summary"


