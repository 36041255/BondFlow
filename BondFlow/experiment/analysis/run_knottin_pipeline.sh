#!/bin/bash
#SBATCH -J knottin_eval
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --no-requeue
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH -o slurm_knottin_eval.%j.out
#SBATCH -e slurm_knottin_eval.%j.err
#SBATCH -w g01



export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8

# Activate your environment
source /apps/soft/anaconda3/bin/activate

# Set paths
PROJECT_ROOT="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow"
PIPELINE_SCRIPT="${PROJECT_ROOT}/BondFlow/experiment/analysis/knottin_pipeline.py"

# Configuration
# Usage: ./run_knottin_pipeline.sh [INPUT_DIR] [OUTPUT_DIR] [GEN_CONFIG] [GEN_OUTPUT_DIR] [GPUS] [JOBS_PER_GPU]
INPUT_DIR="${PROJECT_ROOT}/BondFlow/experiment/cyclize/MDM2/condition_knottin/post_refine"  # Directory with PDB files
OUTPUT_DIR="${PROJECT_ROOT}/BondFlow/experiment/cyclize/MDM2/MDM2_knottin_eval"   # Output directory for results
GEN_CONFIG= #"${3:-}"                       # YAML config for generation (optional, triggers --generate)
GEN_OUTPUT_DIR= #"${4:-}"                   # Output dir for generation (optional)

# Thresholds (JSON format) - Note: Binding_Energy threshold (lower is better, so use negative value)
# dslf_fa13: disulfide bond score (lower is better, typical good values < 1.0, acceptable < 2.0)
THRESHOLDS='{"Binding_Energy": -40, "SAP_total": 35, "PLDDT": 75, "scRMSD": 2.5, "dslf_fa13": 1.0}'

# Extract disulfide bond score (dslf_fa13) from Rosetta energy calculation
EXTRACT_DSLF_FA13=true  # Set to false to disable

# GPU configuration for HighFold
GPUS="0,1" #"${5:-0}"                    # Comma-separated GPU IDs for HighFold (e.g. "0,1,2,3")
JOBS_PER_GPU=16 #"${6:-1}"           # Number of concurrent HighFold jobs per GPU (default: 1)

# Number of CPU cores for energy/SAP calculation
N_CORES=63 #"${7:-32}"

# Chain ID (usually A for knottin binder)
CHAIN="A"

# HighFold MSA configuration (optional)
MSA_MODE="mmseqs2_uniref"  # Options: single_sequence, mmseqs2_uniref_env, mmseqs2_uniref
MSA_THREADS=  # online server is none buisness of local mmseqs2 MSA_THREADS
MAX_MSA="256:512"  # MSA depth in format "max-seq:max-extra-seq" (e.g., "512:5120", leave empty for default)

echo "=========================================="
echo "Knottin Binder Pipeline"
echo "=========================================="
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
if [ ! -z "${GEN_CONFIG}" ] && [ ! -z "${GEN_OUTPUT_DIR}" ]; then
    echo "Generation: ENABLED (config: ${GEN_CONFIG}, output: ${GEN_OUTPUT_DIR})"
else
    echo "Generation: DISABLED (evaluating existing structures)"
fi
echo "GPUs for HighFold: ${GPUS} (${JOBS_PER_GPU} jobs per GPU)"
echo "CPU cores: ${N_CORES}"
echo "Chain: ${CHAIN}"
echo "MSA mode: ${MSA_MODE}"
if [ ! -z "${MSA_THREADS}" ]; then
    echo "MSA threads: ${MSA_THREADS}"
fi
if [ ! -z "${MAX_MSA}" ]; then
    echo "Max MSA: ${MAX_MSA}"
fi
echo "Extract dslf_fa13: ${EXTRACT_DSLF_FA13}"
echo "=========================================="

# 记录脚本开始时间
start_time=$(date +%s)

# Build command arguments
PIPELINE_ARGS=(
    --input_dir "${INPUT_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --chain "${CHAIN}"
    --gpus "${GPUS}"
    --jobs_per_gpu "${JOBS_PER_GPU}"
    --n_cores "${N_CORES}"
    --msa-mode "${MSA_MODE}"
    --relax
    --thresholds "${THRESHOLDS}"
    --passed_dir "${OUTPUT_DIR}/passed/original"
    --passed_relax_dir "${OUTPUT_DIR}/passed/relaxed"
)

# Add extract_dslf_fa13 option if enabled
if [ "${EXTRACT_DSLF_FA13}" = "true" ]; then
    PIPELINE_ARGS+=(--extract_dslf_fa13)
fi

# Add optional MSA parameters if provided
if [ ! -z "${MSA_THREADS}" ]; then
    PIPELINE_ARGS+=(--msa-threads "${MSA_THREADS}")
fi
if [ ! -z "${MAX_MSA}" ]; then
    PIPELINE_ARGS+=(--max-msa "${MAX_MSA}")
fi
echo "PIPELINE_ARGS: ${PIPELINE_ARGS[@]}"
# Add generation arguments if provided
if [ ! -z "${GEN_CONFIG}" ] && [ ! -z "${GEN_OUTPUT_DIR}" ]; then
    PIPELINE_ARGS+=(
        --generate
        --gen_config "${GEN_CONFIG}"
        --gen_output_dir "${GEN_OUTPUT_DIR}"
    )
fi
echo "start to run pipeline"
# Run pipeline (Python script handles both generation and evaluation)
conda run --no-capture-output -n analysis python -u "${PIPELINE_SCRIPT}" "${PIPELINE_ARGS[@]}"

# 记录脚本结束时间并报告花费时间
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "=========================================="
printf "Pipeline complete!\n"
printf "Results saved to: ${OUTPUT_DIR}\n"
echo "Total time elapsed: ${duration} seconds."
echo "=========================================="

