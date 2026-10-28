#!/bin/bash
#SBATCH -J mdm2_motif_sample
#SBATCH -N 1
#SBATCH -p V100
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -o slurm_mdm2_motif.%j.out
#SBATCH -e slurm_mdm2_motif.%j.err

set -euo pipefail

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8

INITIAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INITIAL_REPO_ROOT="$(cd "${INITIAL_SCRIPT_DIR}/.." && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-${INITIAL_SCRIPT_DIR}}"
REPO_ROOT="${REPO_ROOT:-${INITIAL_REPO_ROOT}}"
DEFAULT_CFG="${SCRIPT_DIR}/config/MDM2_cyc_desgin_hd3.yaml"
CONDA_BIN="${CONDA_BIN:-}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-BondFlow}"
MERGE_SHARDS="${MERGE_SHARDS:-1}"
WORKER_CPUS_PER_TASK="${WORKER_CPUS_PER_TASK:-1}"
HF_HOME="${HF_HOME:-/home/xjt/.cache/huggingface}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
MOTIF_PROBABILITY="${MOTIF_PROBABILITY:-0.5}"
MOTIF_SEED="${MOTIF_SEED:-${SLURM_JOB_ID:-manual}}"
MOTIF_TEMPLATE_MODE="${MOTIF_TEMPLATE_MODE:-both}"
DRY_RUN="${DRY_RUN:-0}"
LOCAL_RUN="${LOCAL_RUN:-0}"

die() {
    echo "[batch_sample_mdm2_motif] ERROR: $*" >&2
    exit 1
}

configure_offline_hf() {
    export HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_DISABLE_TELEMETRY
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

    [[ -n "${conda_bin}" ]] && conda_bin_candidates+=("${conda_bin}")
    command -v conda >/dev/null 2>&1 && conda_bin_candidates+=("$(command -v conda)")
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

    [[ -n "${CONDA_ACTIVATE}" ]] && activate_candidates+=("${CONDA_ACTIVATE}")
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

    [[ -n "${input_path}" ]] || die "Configuration path is empty."
    if [[ "${input_path}" = /* ]]; then
        [[ -f "${input_path}" ]] || die "Config file not found: ${input_path}"
        printf '%s\n' "${input_path}"
        return
    fi
    if [[ -f "${input_path}" ]]; then
        printf '%s\n' "$(cd "$(dirname "${input_path}")" && pwd)/$(basename "${input_path}")"
        return
    fi
    if [[ -n "${submit_dir}" && -f "${submit_dir}/${input_path}" ]]; then
        printf '%s\n' "$(cd "$(dirname "${submit_dir}/${input_path}")" && pwd)/$(basename "${input_path}")"
        return
    fi
    if [[ -f "${SCRIPT_DIR}/${input_path}" ]]; then
        printf '%s\n' "$(cd "$(dirname "${SCRIPT_DIR}/${input_path}")" && pwd)/$(basename "${input_path}")"
        return
    fi
    if [[ -f "${REPO_ROOT}/${input_path}" ]]; then
        printf '%s\n' "$(cd "$(dirname "${REPO_ROOT}/${input_path}")" && pwd)/$(basename "${input_path}")"
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
        [[ "${parent}" != "${current}" ]] || break
        current="${parent}"
    done
    return 1
}

set_runtime_dirs_from_cfg() {
    local cfg_path="$1"
    local candidate=""
    if candidate="$(find_script_dir_from_cfg "${cfg_path}" 2>/dev/null)"; then
        SCRIPT_DIR="${candidate}"
        REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
        DEFAULT_CFG="${SCRIPT_DIR}/config/MDM2_cyc_desgin_hd3.yaml"
    fi
}

detect_gpus_per_node() {
    local raw=""
    local parsed=""

    parse_gpu_count() {
        local value="$1"
        local -a parts=()
        [[ -n "${value}" ]] || return 1
        if [[ "${value}" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "${value}"
            return
        fi
        if [[ "${value}" =~ ^([0-9]+)\(x[0-9]+\)$ ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
            return
        fi
        if [[ "${value}" == *,* ]]; then
            IFS=',' read -r -a parts <<< "${value}"
            (( ${#parts[@]} > 0 )) && printf '%s\n' "${#parts[@]}" && return
        fi
        if [[ "${value}" =~ :([0-9]+)$ ]]; then
            printf '%s\n' "${BASH_REMATCH[1]}"
            return
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
    (( shard_rank < remainder )) && local_cycles=$((local_cycles + 1))
    start_cycle=$((shard_rank * base_cycles))
    if (( shard_rank < remainder )); then
        start_cycle=$((start_cycle + shard_rank))
    else
        start_cycle=$((start_cycle + remainder))
    fi
    printf '%s %s\n' "${start_cycle}" "${local_cycles}"
}

prepare_cycle_configs() {
    local cfg_path="$1"
    local job_id="$2"

    python - "${cfg_path}" "${REPO_ROOT}" "${SCRIPT_DIR}" "${job_id}" "${MOTIF_PROBABILITY}" "${MOTIF_SEED}" "${MOTIF_TEMPLATE_MODE}" <<'PY'
import math
import os
import random
import re
import sys

from omegaconf import OmegaConf

(
    cfg_path,
    repo_root,
    script_dir,
    job_id,
    probability_text,
    motif_seed,
    template_mode,
) = sys.argv[1:]


def fail(message: str) -> None:
    raise ValueError(message)


def resolve_output_root(path: str, cfg_dir: str) -> str:
    if not path:
        fail("inference.output_prefix must be set")
    if os.path.isabs(path):
        return os.path.abspath(path)
    candidates = []
    if path.startswith("./") or path.startswith("../"):
        candidates.append(os.path.normpath(os.path.join(script_dir, path)))
    candidates.extend(
        [
            os.path.normpath(os.path.join(repo_root, path)),
            os.path.normpath(os.path.join(cfg_dir, path)),
        ]
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def resolve_existing_path(path, cfg_dir: str, *, prefer_cfg_dir: bool = False):
    if not path or os.path.isabs(str(path)):
        return path
    path = str(path)
    candidates = []
    if prefer_cfg_dir:
        candidates.append(os.path.normpath(os.path.join(cfg_dir, path)))
    if path.startswith("./") or path.startswith("../"):
        candidates.append(os.path.normpath(os.path.join(script_dir, path)))
    candidates.append(os.path.normpath(os.path.join(repo_root, path)))
    if not prefer_cfg_dir:
        candidates.append(os.path.normpath(os.path.join(cfg_dir, path)))
    for candidate in dict.fromkeys(candidates):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def normalize_runtime_paths(cfg, cfg_dir: str) -> None:
    model = getattr(cfg, "model", None)
    if model is not None:
        if getattr(model, "model_config_path", None):
            model.model_config_path = resolve_existing_path(
                model.model_config_path, cfg_dir, prefer_cfg_dir=True
            )
        if getattr(model, "ckpt_path", None):
            model.ckpt_path = resolve_existing_path(model.ckpt_path, cfg_dir)
    design_config = getattr(cfg, "design_config", None)
    if design_config is not None and getattr(design_config, "input_pdb", None):
        design_config.input_pdb = resolve_existing_path(design_config.input_pdb, cfg_dir)
    preprocess = getattr(cfg, "preprocess", None)
    if preprocess is not None and getattr(preprocess, "link_config", None):
        preprocess.link_config = resolve_existing_path(
            preprocess.link_config, cfg_dir, prefer_cfg_dir=True
        )


try:
    probability = float(probability_text)
except ValueError as error:
    fail(f"MOTIF_PROBABILITY must be a number in [0, 1], got {probability_text!r}")
if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
    fail(f"MOTIF_PROBABILITY must be in [0, 1], got {probability_text!r}")
if template_mode not in {"both", "pmi", "p53"}:
    fail("MOTIF_TEMPLATE_MODE must be one of: both, pmi, p53")
if not motif_seed:
    fail("MOTIF_SEED must be nonempty")

cfg_path = os.path.abspath(cfg_path)
cfg_dir = os.path.dirname(cfg_path)
cfg = OmegaConf.load(cfg_path)
if getattr(cfg, "inference", None) is None:
    fail("Configuration requires an inference section")
if getattr(cfg, "design_config", None) is None:
    fail("Configuration requires a design_config section")
if not getattr(cfg.inference, "output_prefix", None):
    fail("inference.output_prefix must be set")

try:
    num_cycles = int(cfg.inference.num_cycle)
    num_designs = int(getattr(cfg.inference, "num_designs", 1))
except (AttributeError, TypeError, ValueError) as error:
    fail("inference.num_cycle and inference.num_designs must be integers")
if num_cycles <= 0 or num_designs <= 0:
    fail("inference.num_cycle and inference.num_designs must be positive")

contigs = getattr(cfg.design_config, "contigs", None)
if not OmegaConf.is_list(contigs) or not contigs:
    fail("design_config.contigs must contain a design chain")
design_chain = contigs[0]
if not OmegaConf.is_list(design_chain) or len(design_chain) != 1:
    fail("the first design contig must be exactly one New_lower-upper range")
new_contig = str(design_chain[0])
match = re.fullmatch(r"New_(\d+)-(\d+)", new_contig)
if not match:
    fail("the first design contig must match New_lower-upper")
minimum_length, maximum_length = map(int, match.groups())
if minimum_length <= 0 or minimum_length > maximum_length:
    fail(f"invalid design-length range: {new_contig}")

base_output_root = resolve_output_root(str(cfg.inference.output_prefix), cfg_dir).rstrip(os.sep)
job_output_root = os.path.join(base_output_root, f"slurm_{job_id}")
configs_root = os.path.join(job_output_root, "cycle_configs")
shards_root = os.path.join(job_output_root, "shards")
os.makedirs(configs_root, exist_ok=True)
os.makedirs(shards_root, exist_ok=True)
for entry in os.scandir(configs_root):
    if entry.is_file() and re.fullmatch(r"cycle_\d{4}\.yaml", entry.name):
        os.unlink(entry.path)

rng = random.Random(motif_seed)
templates = {
    "pmi": [
        "F:seq_FIX:str_DNV",
        "New_2-2",
        "Y:seq_FIX:str_DNV",
        "W:seq_FIX:str_DNV",
        "New_2-2",
        "L:seq_FIX:str_DNV",
    ],
    "p53": [
        "F:seq_FIX:str_DNV",
        "New_3-3",
        "W:seq_FIX:str_DNV",
        "New_2-2",
        "L:seq_FIX:str_DNV",
    ],
}
motif_length = 8
base_seed = getattr(cfg.inference, "seed", None)
if base_seed is not None:
    try:
        base_seed = int(base_seed)
    except (TypeError, ValueError) as error:
        fail("inference.seed must be an integer or null")

manifest_path = os.path.join(job_output_root, "motif_manifest.tsv")
with open(manifest_path, "w", encoding="utf-8", newline="") as manifest:
    manifest.write("cycle\tmotif_applied\ttemplate\tdesign_length\tcontigs\tconfig_path\n")
    for cycle in range(num_cycles):
        design_length = rng.randint(minimum_length, maximum_length)
        motif_applied = rng.random() < probability
        template = "none"
        if motif_applied:
            if design_length < motif_length:
                fail(
                    f"design length {design_length} is shorter than the {motif_length}-residue motif"
                )
            if template_mode == "both":
                template = rng.choice(("pmi", "p53"))
            else:
                template = template_mode
            remaining = design_length - motif_length
            prefix_length = remaining // 2
            suffix_length = remaining - prefix_length
            cycle_design_chain = []
            if prefix_length:
                cycle_design_chain.append(f"New_{prefix_length}-{prefix_length}")
            cycle_design_chain.extend(templates[template])
            if suffix_length:
                cycle_design_chain.append(f"New_{suffix_length}-{suffix_length}")
        else:
            cycle_design_chain = [f"New_{design_length}-{design_length}"]

        cycle_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cycle_cfg.inference.num_cycle = 1
        cycle_cfg.inference.output_prefix = os.path.join(
            os.path.abspath(shards_root), f"cycle_{cycle:04d}", ""
        )
        if base_seed is not None:
            cycle_cfg.inference.seed = base_seed + cycle * num_designs
        cycle_cfg.design_config.contigs[0] = cycle_design_chain
        normalize_runtime_paths(cycle_cfg, cfg_dir)

        config_path = os.path.join(configs_root, f"cycle_{cycle:04d}.yaml")
        OmegaConf.save(cycle_cfg, config_path)
        manifest.write(
            "\t".join(
                (
                    str(cycle),
                    "1" if motif_applied else "0",
                    template,
                    str(design_length),
                    ",".join(cycle_design_chain),
                    os.path.abspath(config_path),
                )
            )
            + "\n"
        )

print(num_cycles)
print(num_designs)
print(job_output_root)
print(configs_root)
print(shards_root)
PY
}

merge_cycle_outputs() {
    local total_cycles="$1"
    local num_designs="$2"
    local shards_root="$3"
    local merged_root="$4"
    local cycle cycle_dir subdir merged_subdir file_path base_name prefix index ext new_name

    mkdir -p "${merged_root}/pre_refine" "${merged_root}/post_refine"
    for ((cycle = 0; cycle < total_cycles; cycle++)); do
        cycle_dir="${shards_root}/cycle_$(printf '%04d' "${cycle}")"
        for subdir in pre_refine post_refine; do
            [[ -d "${cycle_dir}/${subdir}" ]] || continue
            merged_subdir="${merged_root}/${subdir}"
            shopt -s nullglob
            for file_path in "${cycle_dir}/${subdir}"/*; do
                base_name="$(basename "${file_path}")"
                if [[ "${base_name}" =~ ^(.*_)([0-9]+)\.(pdb|txt)$ ]]; then
                    prefix="${BASH_REMATCH[1]}"
                    index="${BASH_REMATCH[2]}"
                    ext="${BASH_REMATCH[3]}"
                    if (( index >= num_designs )); then
                        die "Unexpected local output index ${index} in ${file_path}"
                    fi
                    new_name="${prefix}$((cycle * num_designs + index)).${ext}"
                    mv "${file_path}" "${merged_subdir}/${new_name}"
                else
                    echo "[batch_sample_mdm2_motif] Skip unrecognized file during merge: ${file_path}" >&2
                fi
            done
            shopt -u nullglob
        done
    done
}

worker_main() {
    local configs_root="$1"
    local total_tasks="$2"
    local total_cycles="$3"
    local rank="${SLURM_PROCID:?SLURM_PROCID is required in worker mode}"
    local start_cycle local_cycles cycle cfg_path

    read -r start_cycle local_cycles < <(compute_cycle_shard "${rank}" "${total_tasks}" "${total_cycles}")
    echo "[batch_sample_mdm2_motif][rank ${rank}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} start_cycle=${start_cycle} local_cycles=${local_cycles}"
    (( local_cycles > 0 )) || return 0

    cd "${SCRIPT_DIR}"
    for ((cycle = start_cycle; cycle < start_cycle + local_cycles; cycle++)); do
        cfg_path="${configs_root}/cycle_$(printf '%04d' "${cycle}").yaml"
        [[ -f "${cfg_path}" ]] || die "Missing generated cycle config: ${cfg_path}"
        python "${SCRIPT_DIR}/sample.py" --cfg "${cfg_path}" --device cuda:0
    done
}

run_local_workers() {
    local configs_root="$1"
    local total_cycles="$2"
    local gpu_list="${CUDA_VISIBLE_DEVICES:-}"
    local rank total_tasks worker_pid status
    local -a gpu_ids=()
    local -a worker_pids=()

    [[ -n "${gpu_list}" ]] || die "Local mode requires CUDA_VISIBLE_DEVICES, e.g. CUDA_VISIBLE_DEVICES=1,5,6,7."
    IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
    total_tasks="${#gpu_ids[@]}"
    (( total_tasks > 0 )) || die "No GPUs found in CUDA_VISIBLE_DEVICES=${gpu_list}."

    echo "[batch_sample_mdm2_motif] Local mode: GPUs=${gpu_list} workers=${total_tasks}"
    for ((rank = 0; rank < total_tasks; rank++)); do
        [[ -n "${gpu_ids[rank]}" ]] || die "Invalid CUDA_VISIBLE_DEVICES=${gpu_list}."
        CUDA_VISIBLE_DEVICES="${gpu_ids[rank]}" \
            SLURM_PROCID="${rank}" \
            bash "${SCRIPT_DIR}/batch_sample_mdm2_motif.sh" __worker__ \
                "${configs_root}" "${total_tasks}" "${total_cycles}" &
        worker_pids+=("$!")
    done

    status=0
    for worker_pid in "${worker_pids[@]}"; do
        wait "${worker_pid}" || status=1
    done
    (( status == 0 )) || die "At least one local sampling worker failed."
}

coordinator_main() {
    local cfg_input="${1:-${CFG_PATH:-${DEFAULT_CFG}}}"
    local cfg_path job_id total_nodes gpus_per_node total_tasks
    local total_cycles num_designs job_output_root configs_root shards_root merged_root
    local meta_output
    local -a meta

    cfg_path="$(resolve_cfg_path "${cfg_input}")"
    set_runtime_dirs_from_cfg "${cfg_path}"
    job_id="${SLURM_JOB_ID:-manual}"
    if ! meta_output="$(prepare_cycle_configs "${cfg_path}" "${job_id}")"; then
        die "Failed to generate per-cycle motif configurations."
    fi
    mapfile -t meta <<< "${meta_output}"
    (( ${#meta[@]} == 5 )) || die "Invalid metadata returned while generating cycle configurations."
    total_cycles="${meta[0]}"
    num_designs="${meta[1]}"
    job_output_root="${meta[2]}"
    configs_root="${meta[3]}"
    shards_root="${meta[4]}"
    merged_root="${job_output_root}/merged"

    echo "[batch_sample_mdm2_motif] cfg=${cfg_path}"
    echo "[batch_sample_mdm2_motif] motif_probability=${MOTIF_PROBABILITY} motif_seed=${MOTIF_SEED} template_mode=${MOTIF_TEMPLATE_MODE}"
    echo "[batch_sample_mdm2_motif] num_cycle=${total_cycles} num_designs=${num_designs}"
    echo "[batch_sample_mdm2_motif] configs=${configs_root}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[batch_sample_mdm2_motif] DRY_RUN=1: generated configs and motif manifest only."
        return
    fi

    activate_env
    configure_offline_hf
    gpus_per_node="$(detect_gpus_per_node)"
    [[ "${gpus_per_node}" =~ ^[0-9]+$ ]] || die "Failed to detect GPU count per node: ${gpus_per_node}"
    (( gpus_per_node > 0 )) || die "No GPU detected on the allocated node."
    total_nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
    total_tasks=$((total_nodes * gpus_per_node))

    if [[ "${LOCAL_RUN}" == "1" ]] || ! command -v srun >/dev/null 2>&1; then
        run_local_workers "${configs_root}" "${total_cycles}"
    else
        srun \
            --ntasks="${total_tasks}" \
            --ntasks-per-node="${gpus_per_node}" \
            --cpus-per-task="${WORKER_CPUS_PER_TASK}" \
            --gpus-per-task=1 \
            --kill-on-bad-exit=1 \
            bash "${SCRIPT_DIR}/batch_sample_mdm2_motif.sh" __worker__ "${configs_root}" "${total_tasks}" "${total_cycles}"
    fi

    if [[ "${MERGE_SHARDS}" == "1" ]]; then
        merge_cycle_outputs "${total_cycles}" "${num_designs}" "${shards_root}" "${merged_root}"
        echo "[batch_sample_mdm2_motif] Merged outputs ready under ${merged_root}"
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
