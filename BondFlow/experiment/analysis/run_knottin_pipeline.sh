#!/bin/bash
# Example script to run the knottin binder pipeline

# Set paths
PROJECT_ROOT="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow"
PIPELINE_SCRIPT="${PROJECT_ROOT}/BondFlow/experiment/analysis/knottin_pipeline.py"

# Configuration
# Usage: ./run_knottin_pipeline.sh [INPUT_DIR] [OUTPUT_DIR] [GEN_CONFIG] [GEN_OUTPUT_DIR] [GPUS] [JOBS_PER_GPU]
INPUT_DIR="${PROJECT_ROOT}/BondFlow/experiment/cyclize/MDM2/condition_knottin/post_refine"  # Directory with PDB files
OUTPUT_DIR="${PROJECT_ROOT}/BondFlow/experiment/MDM2_knottin_results"   # Output directory for results
GEN_CONFIG= #"${3:-}"                       # YAML config for generation (optional, triggers --generate)
GEN_OUTPUT_DIR= #"${4:-}"                   # Output dir for generation (optional)

# Thresholds (JSON format) - Note: Binding_Energy threshold (lower is better, so use negative value)
THRESHOLDS='{"Binding_Energy": -10, "SAP_total": 50, "PLDDT": 70, "scRMSD": 2.0}'

# GPU configuration for HighFold
GPUS=0 #"${5:-0}"                    # Comma-separated GPU IDs for HighFold (e.g. "0,1,2,3")
JOBS_PER_GPU=8 #"${6:-1}"           # Number of concurrent HighFold jobs per GPU (default: 1)

# Number of CPU cores for energy/SAP calculation
N_CORES=16 #"${7:-32}"

# Chain ID (usually A for knottin binder)
CHAIN="A"

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
echo "=========================================="

# Build command arguments
PIPELINE_ARGS=(
    --input_dir "${INPUT_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --chain "${CHAIN}"
    --gpus "${GPUS}"
    --jobs_per_gpu "${JOBS_PER_GPU}"
    --n_cores "${N_CORES}"
    --relax
    --thresholds "${THRESHOLDS}"
    --passed_dir "${OUTPUT_DIR}/passed/original"
    --passed_relax_dir "${OUTPUT_DIR}/passed/relaxed"
)
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

echo "=========================================="
echo "Pipeline complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

