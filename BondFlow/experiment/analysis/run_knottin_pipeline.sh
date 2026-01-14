#!/bin/bash
#SBATCH -J knottin_eval
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --no-requeue
#SBATCH --cpus-per-task=97
#SBATCH -w g02
#SBATCH -o slurm_knottin_eval.%j.out
#SBATCH -e slurm_knottin_eval.%j.err


export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8

# XLA/JAX optimization for multi-process HighFold execution
# These settings help avoid "Delay kernel timed out" and "Very slow compile" errors
# when running multiple HighFold processes on the same GPU
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=true

# IMPORTANT: Remove deprecated XLA_PYTHON_CLIENT_MEM_FRACTION if it exists
# JAX now only uses XLA_CLIENT_MEM_FRACTION, and having both causes an error:
# "ValueError: XLA_CLIENT_MEM_FRACTION is specified together with XLA_PYTHON_CLIENT_MEM_FRACTION"
unset XLA_PYTHON_CLIENT_MEM_FRACTION

# XLA memory base fraction: controls total GPU memory usage across all processes
# Formula: each_process = XLA_MEM_BASE_FRACTION / jobs_per_gpu
# - 0.75 (default): Safe, leaves 25% margin for system/overhead (recommended)
# - 0.9: Aggressive, uses 90% GPU memory, only 10% margin (may cause OOM with many processes)
# - 0.95: Very aggressive, minimal margin (not recommended for production)
# XLA_CLIENT_MEM_FRACTION will be auto-calculated by Python script based on JOBS_PER_GPU and this value
# With JOBS_PER_GPU=3: each process gets 0.75/3 = 0.25 (safe)
# With JOBS_PER_GPU=8: each process gets 0.85/8 = 0.106 (may be insufficient, can cause hanging)
export XLA_MEM_BASE_FRACTION=0.85  # Recommended: 0.75 for stability, 0.85 for more aggressive usage
# You can also directly override XLA_CLIENT_MEM_FRACTION if needed (e.g., export XLA_CLIENT_MEM_FRACTION=0.1)

# Activate your environment
source /apps/soft/anaconda3/bin/activate

# Set paths
PROJECT_ROOT="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow"
PIPELINE_SCRIPT="${PROJECT_ROOT}/BondFlow/experiment/analysis/knottin_pipeline.py"

#INPUT_DIR="${PROJECT_ROOT}/BondFlow/experiment/cyclize/MDM2/MDM2_baseline"  # Directory with PDB files
#OUTPUT_DIR="${PROJECT_ROOT}/BondFlow/experiment/cyclize/MDM2/MDM2_baseline_evals2"   # Output directory for results
INPUT_DIR="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/experiment/cyclize/MDM2/condition_knottin2/post_refine"  # Directory with PDB files
OUTPUT_DIR="${PROJECT_ROOT}/BondFlow/experiment/cyclize/MDM2/MDM2_knottin_evals2"   # Output directory for results
GEN_CONFIG= #"${3:-}"                       # YAML config for generation (optional, triggers --generate)
GEN_OUTPUT_DIR= #"${4:-}"                   # Output dir for generation (optional)

# Thresholds (JSON format) - Note: Binding_Energy threshold (lower is better, so use negative value)
# dslf_fa13: disulfide bond score (lower is better, typical good values < 1.0, acceptable < 2.0)
# target_chain_Energy: target chain stability energy (lower is better, more stable)
# target_chain_Energy_per_Res: target chain energy per residue (lower is better, more comparable across different lengths)
#   This is more meaningful than total energy as it normalizes by chain length
# Example: target_chain_Energy_per_Res threshold of 2.0 means only structures with target chain energy/residue <= 2.0 will pass
THRESHOLDS='{"Binding_Energy": -40, "SAP_total": 35, "PLDDT": 70, "scRMSD": 3, "dslf_fa13": -1, "target_chain_Energy_per_Res": 10}'

# Extract disulfide bond score (dslf_fa13) from Rosetta energy calculation
EXTRACT_DSLF_FA13=true  # Set to false to disable

# GPU configuration for HighFold
GPUS="" #"${5:-0}"                    # Comma-separated GPU IDs for HighFold (e.g. "0,1,2,3")
                                        # If empty and no GPU available, will use CPU mode (JAX_PLATFORMS=cpu)
# JOBS_PER_GPU: Number of concurrent HighFold jobs per GPU (or per CPU in CPU mode)
# WARNING: High values (e.g., 8) may cause XLA compilation conflicts and "Delay kernel timed out" errors
# CRITICAL: Each MSA search task estimates ~71GB memory usage. With 8 jobs = 568GB total, causing OOM and system hang!
# Recommended: 1-2 for MSA search (mmseqs2_uniref), 2-3 for single_sequence mode
# The script now uses shared XLA compilation cache to reduce conflicts, but still recommend lower values
# CPU MODE NOTE: In CPU mode, this controls concurrent HighFold prediction tasks. Each task will use all available
# CPU cores by default (JAX/XLA behavior). To limit CPU usage per task, set OMP_NUM_THREADS environment variable.
JOBS_PER_GPU=48 #"${6:-1}"           # Number of concurrent HighFold jobs per GPU (default: 1, recommended: 1-2 for MSA mode)
                                      # In CPU mode: number of concurrent prediction tasks

# Number of CPU cores for energy/SAP calculation
N_CORES=96 #"${7:-32}"

# Chain ID (usually A for knottin binder)
CHAIN="A"

# HighFold MSA configuration (optional)
MSA_MODE="mmseqs2_uniref"  # Options: single_sequence, mmseqs2_uniref_env, mmseqs2_uniref
# MSA_THREADS: Number of threads per MSA search job (MMseqs2)
# IMPORTANT: Each MSA search estimates ~71GB memory. With JOBS_PER_GPU=1, can use more threads safely
# Recommended: 8-16 threads per job when JOBS_PER_GPU=1, 4-8 when JOBS_PER_GPU=2
# CPU MODE NOTE: In CPU mode, if JAX_CPU_THREADS is set, it will override MSA_THREADS for consistency.
# If JAX_CPU_THREADS is not set, MSA_THREADS will be used for MSA search, and JAX will use all available cores.
MSA_THREADS=2  # Number of threads for local MSA search (colabfold_search). Default: 32
MAX_MSA="256:512"  # MSA depth in format "max-seq:max-extra-seq" (e.g., "512:5120", leave empty for default)
MSA_DB_PATH="/home/fit/lulei/WORK/public/database"  # Path to local MSA database. If set, uses colabfold_search for local MSA instead of server search.

# HighFold prediction configuration (optional)
NUM_RECYCLE=3  # Number of recycle iterations (default: None, uses ColabFold default, typically 3)
NUM_SAMPLES=3  # Number of model samples/predictions (default: None, uses ColabFold default, typically 1)

# CPU mode configuration (only applies when GPUS="" and using CPU mode)
# JAX_CPU_THREADS: Number of CPU threads per HighFold prediction task (JAX/XLA)
# If not set, JAX will use all available CPU cores per task
# Recommended: Set based on total CPU cores and JOBS_PER_GPU
# Example: With 80 CPU cores and JOBS_PER_GPU=8, set JAX_CPU_THREADS=10 (80/8=10 per task)
JAX_CPU_THREADS=2  # Uncomment and set to limit CPU threads per prediction task

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
if [ -z "${GPUS}" ]; then
    echo "GPUs for HighFold: Not specified (will use CPU mode if no GPU available)"
    # Force CPU mode when no GPUs specified
    export CUDA_VISIBLE_DEVICES=""
    export JAX_PLATFORMS=cpu
    echo "  → CPU mode enabled (CUDA_VISIBLE_DEVICES=\"\", JAX_PLATFORMS=cpu)"
    echo "  → JOBS_PER_GPU=${JOBS_PER_GPU} (controls concurrent prediction tasks in CPU mode)"
    echo "  → MSA_THREADS=${MSA_THREADS} (controls MMseqs2 threads for MSA search)"
    if [ ! -z "${JAX_CPU_THREADS}" ]; then
        export JAX_CPU_THREADS="${JAX_CPU_THREADS}"
        echo "  → JAX_CPU_THREADS=${JAX_CPU_THREADS} (CPU threads per HighFold prediction task)"
    else
        echo "  → JAX_CPU_THREADS not set (each prediction will use all available CPU cores)"
    fi
else
    echo "GPUs for HighFold: ${GPUS} (${JOBS_PER_GPU} jobs per GPU)"
fi
echo "CPU cores: ${N_CORES}"
echo "Chain: ${CHAIN}"
echo "MSA mode: ${MSA_MODE}"
if [ ! -z "${MSA_THREADS}" ]; then
    echo "MSA threads: ${MSA_THREADS}"
fi
if [ ! -z "${MAX_MSA}" ]; then
    echo "Max MSA: ${MAX_MSA}"
fi
if [ ! -z "${MSA_DB_PATH}" ]; then
    echo "MSA database path: ${MSA_DB_PATH} (local MSA search enabled)"
fi
if [ ! -z "${NUM_RECYCLE}" ]; then
    echo "HighFold num-recycle: ${NUM_RECYCLE}"
fi
if [ ! -z "${NUM_SAMPLES}" ]; then
    echo "HighFold num-samples: ${NUM_SAMPLES}"
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
if [ ! -z "${MSA_DB_PATH}" ]; then
    PIPELINE_ARGS+=(--msa-db-path "${MSA_DB_PATH}")
fi
if [ ! -z "${NUM_RECYCLE}" ]; then
    PIPELINE_ARGS+=(--num-recycle "${NUM_RECYCLE}")
fi
if [ ! -z "${NUM_SAMPLES}" ]; then
    PIPELINE_ARGS+=(--num-samples "${NUM_SAMPLES}")
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

