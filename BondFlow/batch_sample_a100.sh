#!/bin/bash
#SBATCH -J bondflow_sample_a100
#SBATCH -N 1
#SBATCH -p A100
#SBATCH -w n7
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:8
#SBATCH --no-requeue
#SBATCH --chdir=/home/xjt/BondFlow/BondFlow
#SBATCH -o slurm_sample_a100.%j.out
#SBATCH -e slurm_sample_a100.%j.err

set -euo pipefail

# Direct Bash invocation must submit the script for #SBATCH directives to apply.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if (( $# > 0 )) && [[ -f "$1" ]]; then
        set -- "$(realpath -- "$1")" "${@:2}"
    fi
    exec sbatch "$(realpath -- "${BASH_SOURCE[0]}")" "$@"
fi

# On n11: sbatch batch_sample_a100.sh /absolute/path/to/config.yaml
# Longer runs: sbatch --time=24:00:00 batch_sample_a100.sh /absolute/path/to/config.yaml
# CUDA check: sbatch --export=ALL,CHECK_TORCH_CUDA_ONLY=1 batch_sample_a100.sh

# Slurm executes a spool copy, so resolve the shared project path explicitly.
export REPO_ROOT="${REPO_ROOT:-/home/xjt/BondFlow}"
export SCRIPT_DIR="${SCRIPT_DIR:-${REPO_ROOT}/BondFlow}"
export WORKER_CPUS_PER_TASK="${WORKER_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${WORKER_CPUS_PER_TASK}}"

exec bash "${SCRIPT_DIR}/batch_sample.sh" "$@"
