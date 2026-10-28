import os
import subprocess
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "batch_sample_a100.sh"


def wrapper_env(tmp_path: Path) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "sbatch" "$@"\nexit "${SUBMIT_EXIT:-0}"\n')
    sbatch.chmod(0o755)
    delegate = tmp_path / "batch_sample.sh"
    delegate.write_text(
        '#!/bin/bash\nprintf "%s\\n" "delegate" "$WORKER_CPUS_PER_TASK" "$@"\n'
    )
    env = os.environ.copy()
    for name in ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_CPUS_PER_TASK", "WORKER_CPUS_PER_TASK"):
        env.pop(name, None)
    env.update(PATH=f"{bin_dir}:{env['PATH']}", SCRIPT_DIR=str(tmp_path))
    return env


@pytest.mark.parametrize("submit_exit", [0, 23])
def test_bash_outside_slurm_submits_itself(tmp_path: Path, submit_exit: int) -> None:
    env = wrapper_env(tmp_path)
    env["SUBMIT_EXIT"] = str(submit_exit)
    config = tmp_path / "sampling config.yaml"
    config.touch()

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), config.name],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == submit_exit
    assert result.stdout.splitlines() == ["sbatch", str(SCRIPT_PATH), str(config)]


def test_inside_slurm_runs_sampling_without_resubmitting(tmp_path: Path) -> None:
    env = wrapper_env(tmp_path)
    env.update(SLURM_JOB_ID="12345", SLURM_CPUS_PER_TASK="8")
    config = "/shared/sampling config.yaml"

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), config],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["delegate", "8", config]
