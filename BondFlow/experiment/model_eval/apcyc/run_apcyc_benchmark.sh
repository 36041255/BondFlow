#!/bin/bash
# Multi-GPU cyclic-peptide benchmark runner using APCyc codesign.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PY_SCRIPT="${PROJECT_ROOT}/BondFlow/experiment/model_eval/apcyc/design_cyclic_peptides_apcyc.py"

CONDA_ENV="APCyc"
GPUS="0,1,2,3,4,5,6,7"
INPUT_DIR="/data/public/BondFlow/Highfold_cycpep_benchmark/clean_pdbs"
OUT_ROOT="/data/public/BondFlow/Highfold_cycpep_benchmark/apcyc_cyclic_binders"
APCYC_ROOT="${PROJECT_ROOT}/APCyc"
CKPT=""
NUM_DESIGNS=8
DRY_RUN="false"

print_help() {
  cat <<EOF
Usage:
  bash BondFlow/experiment/model_eval/apcyc/run_apcyc_benchmark.sh [options]

Options:
  --env <name>               Conda env name for APCyc (default: APCyc; use none for current env)
  --gpus <ids>               Comma-separated GPU ids (default: 0,1,2,3,4,5,6,7)
  --input-dir <dir>          Benchmark PDB directory
  --out-root <dir>           Output directory root
  --apcyc-root <dir>         APCyc repository root
  --ckpt <path>              APCyc codesign checkpoint (default: <apcyc-root>/checkpoints/codesign.ckpt)
  --num-designs <int>        Designs per PDB (default: 8)
  --dry-run                  Write inferred cases and commands without running APCyc
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) CONDA_ENV="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --apcyc-root) APCYC_ROOT="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --num-designs) NUM_DESIGNS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift 1 ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

echo "=============================================="
echo "APCyc cyclic benchmark"
echo "=============================================="
echo "Conda env:   ${CONDA_ENV}"
echo "GPUs:        ${GPUS}"
echo "Input dir:   ${INPUT_DIR}"
echo "Output root: ${OUT_ROOT}"
echo "APCyc root:  ${APCYC_ROOT}"
echo "Checkpoint:  ${CKPT:-<apcyc-root>/checkpoints/codesign.ckpt}"
echo "Designs/PDB: ${NUM_DESIGNS}"
echo "Dry run:     ${DRY_RUN}"
echo "=============================================="

CMD=(
  python -u "${PY_SCRIPT}"
  --input_dir "${INPUT_DIR}"
  --out_root "${OUT_ROOT}"
  --apcyc_root "${APCYC_ROOT}"
  --conda_env "${CONDA_ENV}"
  --gpus "${GPUS}"
  --num_designs_per_target "${NUM_DESIGNS}"
)

if [[ -n "${CKPT}" ]]; then
  CMD+=(--ckpt "${CKPT}")
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  CMD+=(--dry_run)
fi

cd "${PROJECT_ROOT}"
"${CMD[@]}"
