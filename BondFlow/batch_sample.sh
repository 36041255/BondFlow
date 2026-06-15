#!/bin/bash
#SBATCH -J bondflow_sample
#SBATCH -N 1
#SBATCH -p V100
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -o slurm_sample.%j.out
#SBATCH -e slurm_sample.%j.err

set -euo pipefail

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8

INITIAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INITIAL_REPO_ROOT="$(cd "${INITIAL_SCRIPT_DIR}/.." && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-${INITIAL_SCRIPT_DIR}}"
REPO_ROOT="${REPO_ROOT:-${INITIAL_REPO_ROOT}}"
DEFAULT_CFG="${SCRIPT_DIR}/config/cyclize.yaml"
CONDA_BIN="${CONDA_BIN:-}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-BondFlow}"
MERGE_SHARDS="${MERGE_SHARDS:-1}"
WORKER_CPUS_PER_TASK="${WORKER_CPUS_PER_TASK:-1}"
CHECK_TORCH_CUDA_ONLY="${CHECK_TORCH_CUDA_ONLY:-0}"
TORCH_CUDA_PREFLIGHT="${TORCH_CUDA_PREFLIGHT:-1}"
HF_HOME="${HF_HOME:-/home/xjt/.cache/huggingface}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

die() {
    echo "[batch_sample] ERROR: $*" >&2
    exit 1
}

configure_offline_hf() {
    export HF_HOME
    export HF_HUB_OFFLINE
    export TRANSFORMERS_OFFLINE
    export HF_HUB_DISABLE_TELEMETRY

    if [[ "${HF_HUB_OFFLINE}" == "1" || "${TRANSFORMERS_OFFLINE}" == "1" ]]; then
        unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
    fi
}

activate_env() {
    local conda_bin="${CONDA_BIN}"
    local hook=""
    local candidate=""
    local had_nounset=0
    local -a conda_bin_candidates=()
    local -a activate_candidates=()

    case $- in
        *u*) had_nounset=1 ;;
    esac

    set +u

    if [[ -n "${conda_bin}" ]]; then
        conda_bin_candidates+=("${conda_bin}")
    fi
    if command -v conda >/dev/null 2>&1; then
        conda_bin_candidates+=("$(command -v conda)")
    fi
    conda_bin_candidates+=(
        "/home/miniconda3/miniconda3/bin/conda"
        "/apps/soft/anaconda3/bin/conda"
    )

    for candidate in "${conda_bin_candidates[@]}"; do
        if [[ -x "${candidate}" ]] && hook="$("${candidate}" shell.bash hook 2>/dev/null)"; then
            eval "${hook}"
            conda activate "${CONDA_ENV_NAME}"
            if (( had_nounset )); then
                set -u
            fi
            return
        fi
    done

    if [[ -n "${CONDA_ACTIVATE}" ]]; then
        activate_candidates+=("${CONDA_ACTIVATE}")
    fi
    activate_candidates+=(
        "/home/miniconda3/miniconda3/bin/activate"
        "/apps/soft/anaconda3/bin/activate"
    )

    for candidate in "${activate_candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            # shellcheck source=/dev/null
            source "${candidate}"
            conda activate "${CONDA_ENV_NAME}"
            if (( had_nounset )); then
                set -u
            fi
            return
        fi
    done

    if (( had_nounset )); then
        set -u
    fi
    die "Unable to initialize conda. Set CONDA_BIN or CONDA_ACTIVATE explicitly."
}

resolve_cfg_path() {
    local input_path="$1"
    local submit_dir="${SLURM_SUBMIT_DIR:-}"

    if [[ -z "${input_path}" ]]; then
        die "Configuration path is empty."
    fi

    if [[ "${input_path}" = /* ]]; then
        printf '%s\n' "${input_path}"
        return
    fi

    if [[ -f "${input_path}" ]]; then
        printf '%s\n' "$(cd "$(dirname "${input_path}")" && pwd)/$(basename "${input_path}")"
        return
    fi

    if [[ -n "${submit_dir}" && -f "${submit_dir}/${input_path}" ]]; then
        printf '%s\n' "${submit_dir}/${input_path}"
        return
    fi

    if [[ -f "${SCRIPT_DIR}/${input_path}" ]]; then
        printf '%s\n' "${SCRIPT_DIR}/${input_path}"
        return
    fi

    if [[ -f "${REPO_ROOT}/${input_path}" ]]; then
        printf '%s\n' "${REPO_ROOT}/${input_path}"
        return
    fi

    die "Config file not found: ${input_path}"
}

find_script_dir_from_cfg() {
    local cfg_path="$1"
    local current parent

    current="$(cd "$(dirname "${cfg_path}")" && pwd)"
    while true; do
        if [[ -f "${current}/sample.py" && -d "${current}/config" ]]; then
            printf '%s\n' "${current}"
            return 0
        fi
        if [[ -f "${current}/BondFlow/sample.py" && -d "${current}/BondFlow/config" ]]; then
            printf '%s\n' "${current}/BondFlow"
            return 0
        fi

        parent="$(dirname "${current}")"
        if [[ "${parent}" == "${current}" ]]; then
            break
        fi
        current="${parent}"
    done

    return 1
}

set_runtime_dirs_from_cfg() {
    local cfg_path="$1"
    local candidate=""
    local submit_dir="${SLURM_SUBMIT_DIR:-}"

    if candidate="$(find_script_dir_from_cfg "${cfg_path}" 2>/dev/null)"; then
        SCRIPT_DIR="${candidate}"
        REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
        DEFAULT_CFG="${SCRIPT_DIR}/config/cyclize.yaml"
        return
    fi

    if [[ -n "${submit_dir}" && -f "${submit_dir}/sample.py" && -d "${submit_dir}/config" ]]; then
        SCRIPT_DIR="$(cd "${submit_dir}" && pwd)"
        REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
        DEFAULT_CFG="${SCRIPT_DIR}/config/cyclize.yaml"
        return
    fi

    if [[ -f "${INITIAL_SCRIPT_DIR}/sample.py" && -d "${INITIAL_SCRIPT_DIR}/config" ]]; then
        SCRIPT_DIR="${INITIAL_SCRIPT_DIR}"
        REPO_ROOT="${INITIAL_REPO_ROOT}"
        DEFAULT_CFG="${SCRIPT_DIR}/config/cyclize.yaml"
        return
    fi
}

detect_gpus_per_node() {
    local raw=""
    local parsed=""

    parse_gpu_count() {
        local value="$1"
        local -a parts=()

        if [[ -z "${value}" ]]; then
            return 1
        fi

        if [[ "${value}" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "${value}"
            return 0
        fi

        if [[ "${value}" =~ ^([0-9]+)\(x[0-9]+\)$ ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
            return 0
        fi

        if [[ "${value}" == *,* ]]; then
            IFS=',' read -r -a parts <<< "${value}"
            if (( ${#parts[@]} > 0 )); then
                printf '%s\n' "${#parts[@]}"
                return 0
            fi
        fi

        if [[ "${value}" =~ :([0-9]+)$ ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
            return 0
        fi

        return 1
    }

    if [[ -n "${GPUS_PER_NODE:-}" ]]; then
        printf '%s\n' "${GPUS_PER_NODE}"
        return
    fi

    for raw in \
        "${SLURM_GPUS_PER_NODE:-}" \
        "${SLURM_GPUS_ON_NODE:-}" \
        "${SBATCH_GPUS_PER_NODE:-}" \
        "${SLURM_STEP_GPUS:-}" \
        "${SLURM_JOB_GPUS:-}"
    do
        if parsed="$(parse_gpu_count "${raw}" 2>/dev/null)"; then
            printf '%s\n' "${parsed}"
            return
        fi
    done

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L | wc -l | tr -d ' '
        return
    fi

    python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
}

compute_cycle_shard() {
    local shard_rank="$1"
    local shard_count="$2"
    local total_cycles="$3"
    local base_cycles remainder local_cycles start_cycle

    base_cycles=$((total_cycles / shard_count))
    remainder=$((total_cycles % shard_count))
    local_cycles="${base_cycles}"
    if (( shard_rank < remainder )); then
        local_cycles=$((local_cycles + 1))
    fi

    start_cycle=$((shard_rank * base_cycles))
    if (( shard_rank < remainder )); then
        start_cycle=$((start_cycle + shard_rank))
    else
        start_cycle=$((start_cycle + remainder))
    fi

    printf '%s %s\n' "${start_cycle}" "${local_cycles}"
}

read_cfg_meta() {
    local cfg_path="$1"
    local job_id="$2"

    python - "${cfg_path}" "${REPO_ROOT}" "${SCRIPT_DIR}" "${job_id}" <<'PY'
import os
import sys
from omegaconf import OmegaConf

cfg_path, repo_root, script_dir, job_id = sys.argv[1:]
cfg = OmegaConf.load(cfg_path)
cfg_dir = os.path.dirname(os.path.abspath(cfg_path))

def resolve_output_root(path: str) -> str:
    if not path:
        raise ValueError("inference.output_prefix must be set")
    if os.path.isabs(path):
        return path

    candidates = []
    if path.startswith("./") or path.startswith("../"):
        candidates.append(os.path.normpath(os.path.join(script_dir, path)))
    candidates.append(os.path.normpath(os.path.join(repo_root, path)))
    candidates.append(os.path.normpath(os.path.join(cfg_dir, path)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    return os.path.abspath(candidates[0])

num_cycle = int(cfg.inference.num_cycle)
num_designs = int(getattr(cfg.inference, "num_designs", 1))
base_output_root = resolve_output_root(cfg.inference.output_prefix).rstrip("/")
job_output_root = os.path.join(base_output_root, f"slurm_{job_id}")

print(num_cycle)
print(num_designs)
print(job_output_root)
PY
}

write_shard_cfg() {
    local src_cfg="$1"
    local dst_cfg="$2"
    local shard_cycles="$3"
    local start_cycle="$4"
    local num_designs="$5"
    local shard_output_root="$6"

    python - "${src_cfg}" "${dst_cfg}" "${shard_cycles}" "${start_cycle}" "${num_designs}" "${shard_output_root}" "${REPO_ROOT}" "${SCRIPT_DIR}" <<'PY'
import os
import sys
from omegaconf import OmegaConf

src_cfg, dst_cfg, shard_cycles, start_cycle, num_designs, shard_output_root, repo_root, script_dir = sys.argv[1:]
cfg = OmegaConf.load(src_cfg)
cfg_dir = os.path.dirname(os.path.abspath(src_cfg))

def resolve_path(path: str, prefer_config_dir: bool = False) -> str:
    if not path or os.path.isabs(path):
        return path

    candidates = []
    if prefer_config_dir:
        candidates.append(os.path.normpath(os.path.join(cfg_dir, path)))
    if path.startswith("./") or path.startswith("../"):
        candidates.append(os.path.normpath(os.path.join(script_dir, path)))
    candidates.append(os.path.normpath(os.path.join(repo_root, path)))
    if not prefer_config_dir:
        candidates.append(os.path.normpath(os.path.join(cfg_dir, path)))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    return os.path.abspath(candidates[0])

cfg.inference.num_cycle = int(shard_cycles)
cfg.inference.output_prefix = os.path.join(os.path.abspath(shard_output_root), "")

base_seed = getattr(cfg.inference, "seed", None)
if base_seed is not None:
    cfg.inference.seed = int(base_seed) + int(start_cycle) * int(num_designs)

if getattr(cfg.model, "model_config_path", None):
    cfg.model.model_config_path = resolve_path(cfg.model.model_config_path, prefer_config_dir=True)

if getattr(cfg.model, "ckpt_path", None):
    cfg.model.ckpt_path = resolve_path(cfg.model.ckpt_path)

if getattr(cfg.design_config, "input_pdb", None):
    cfg.design_config.input_pdb = resolve_path(cfg.design_config.input_pdb)

if getattr(cfg.preprocess, "link_config", None):
    cfg.preprocess.link_config = resolve_path(cfg.preprocess.link_config, prefer_config_dir=True)

OmegaConf.save(cfg, dst_cfg)
PY
}

merge_shards() {
    local total_tasks="$1"
    local total_cycles="$2"
    local num_designs="$3"
    local shards_root="$4"
    local merged_root="$5"
    local rank start_cycle local_cycles shard_dir subdir merged_subdir
    local base_name prefix index ext new_name

    mkdir -p "${merged_root}/pre_refine" "${merged_root}/post_refine"

    for ((rank = 0; rank < total_tasks; rank++)); do
        read -r start_cycle local_cycles < <(compute_cycle_shard "${rank}" "${total_tasks}" "${total_cycles}")
        if (( local_cycles == 0 )); then
            continue
        fi

        shard_dir="${shards_root}/rank_$(printf '%04d' "${rank}")"
        for subdir in pre_refine post_refine; do
            if [[ ! -d "${shard_dir}/${subdir}" ]]; then
                continue
            fi

            merged_subdir="${merged_root}/${subdir}"
            shopt -s nullglob
            for file_path in "${shard_dir}/${subdir}"/*; do
                base_name="$(basename "${file_path}")"
                if [[ "${base_name}" =~ ^(.*_)([0-9]+)\.(pdb|txt)$ ]]; then
                    prefix="${BASH_REMATCH[1]}"
                    index="${BASH_REMATCH[2]}"
                    ext="${BASH_REMATCH[3]}"
                    new_name="${prefix}$((start_cycle * num_designs + index)).${ext}"
                    mv "${file_path}" "${merged_subdir}/${new_name}"
                else
                    echo "[batch_sample] Skip unrecognized file during merge: ${file_path}" >&2
                fi
            done
            shopt -u nullglob
        done
    done
}

worker_main() {
    local cfg_path="$1"
    local total_tasks="$2"
    local total_cycles="$3"
    local num_designs="$4"
    local shards_root="$5"
    local rank="${SLURM_PROCID:?SLURM_PROCID is required in worker mode}"
    local start_cycle local_cycles cfg_dir tmp_cfg shard_output_root tmp_cfg_escaped

    set_runtime_dirs_from_cfg "${cfg_path}"

    read -r start_cycle local_cycles < <(compute_cycle_shard "${rank}" "${total_tasks}" "${total_cycles}")
    shard_output_root="${shards_root}/rank_$(printf '%04d' "${rank}")"

    echo "[batch_sample][rank ${rank}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} start_cycle=${start_cycle} local_cycles=${local_cycles}"

    if (( local_cycles == 0 )); then
        echo "[batch_sample][rank ${rank}] No cycles assigned, exiting."
        return 0
    fi

    cfg_dir="$(cd "$(dirname "${cfg_path}")" && pwd)"
    tmp_cfg="${cfg_dir}/.slurm_sample_${SLURM_JOB_ID}_rank$(printf '%04d' "${rank}").yaml"
    mkdir -p "${shard_output_root}"

    printf -v tmp_cfg_escaped '%q' "${tmp_cfg}"
    trap "rm -f -- ${tmp_cfg_escaped}" EXIT

    if [[ "${TORCH_CUDA_PREFLIGHT}" == "1" || "${CHECK_TORCH_CUDA_ONLY}" == "1" ]]; then
        python - <<'PY'
import os
import sys
import torch

print(f"[torch_cuda_check] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
print(f"[torch_cuda_check] torch={torch.__version__}")
print(f"[torch_cuda_check] torch.version.cuda={torch.version.cuda}")
print(f"[torch_cuda_check] is_available={torch.cuda.is_available()}")
count = torch.cuda.device_count()
print(f"[torch_cuda_check] device_count={count}")
if count <= 0:
    raise RuntimeError("torch.cuda.device_count() == 0")
for idx in range(count):
    props = torch.cuda.get_device_properties(idx)
    print(
        "[torch_cuda_check] "
        f"device[{idx}] name={props.name} capability={props.major}.{props.minor}"
    )
PY
    fi

    if [[ "${CHECK_TORCH_CUDA_ONLY}" == "1" ]]; then
        echo "[batch_sample][rank ${rank}] torch.cuda preflight finished; skipping sample.py because CHECK_TORCH_CUDA_ONLY=1"
        return 0
    fi

    write_shard_cfg "${cfg_path}" "${tmp_cfg}" "${local_cycles}" "${start_cycle}" "${num_designs}" "${shard_output_root}"

    cd "${SCRIPT_DIR}"
    python "${SCRIPT_DIR}/sample.py" \
        --cfg "${tmp_cfg}" \
        --device cuda:0
}

coordinator_main() {
    local cfg_input="${1:-${CFG_PATH:-${DEFAULT_CFG}}}"
    local cfg_path total_nodes gpus_per_node total_tasks
    local total_cycles num_designs job_output_root shards_root merged_root
    local -a meta

    cfg_path="$(resolve_cfg_path "${cfg_input}")"
    set_runtime_dirs_from_cfg "${cfg_path}"
    activate_env
    configure_offline_hf

    gpus_per_node="$(detect_gpus_per_node)"
    [[ "${gpus_per_node}" =~ ^[0-9]+$ ]] || die "Failed to detect GPU count per node: ${gpus_per_node}"
    (( gpus_per_node > 0 )) || die "No GPU detected on the allocated node."

    total_nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
    total_tasks=$((total_nodes * gpus_per_node))

    mapfile -t meta < <(read_cfg_meta "${cfg_path}" "${SLURM_JOB_ID:-manual}")
    total_cycles="${meta[0]}"
    num_designs="${meta[1]}"
    job_output_root="${meta[2]}"
    shards_root="${job_output_root}/shards"
    merged_root="${job_output_root}/merged"

    mkdir -p "${shards_root}"

    echo "[batch_sample] cfg=${cfg_path}"
    echo "[batch_sample] nodes=${total_nodes} gpus_per_node=${gpus_per_node} total_tasks=${total_tasks}"
    echo "[batch_sample] num_cycle=${total_cycles} num_designs=${num_designs}"
    echo "[batch_sample] shard outputs=${shards_root}"

    cd "${SCRIPT_DIR}"
    srun \
        --ntasks="${total_tasks}" \
        --ntasks-per-node="${gpus_per_node}" \
        --cpus-per-task="${WORKER_CPUS_PER_TASK}" \
        --gpus-per-task=1 \
        --kill-on-bad-exit=1 \
        bash "${SCRIPT_DIR}/batch_sample.sh" __worker__ "${cfg_path}" "${total_tasks}" "${total_cycles}" "${num_designs}" "${shards_root}"

    if [[ "${MERGE_SHARDS}" == "1" ]]; then
        echo "[batch_sample] Merging shard outputs into ${merged_root}"
        merge_shards "${total_tasks}" "${total_cycles}" "${num_designs}" "${shards_root}" "${merged_root}"
        echo "[batch_sample] Merged outputs ready under ${merged_root}"
    fi
}

if [[ "${1:-}" == "__worker__" ]]; then
    shift
    activate_env
    configure_offline_hf
    worker_main "$@"
else
    coordinator_main "${1:-}"
fi
