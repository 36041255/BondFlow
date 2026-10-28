import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "BondFlow" / "batch_sample_mdm2_motif.sh"


def write_config(path: Path, output_prefix: Path, num_cycle: int = 6) -> None:
    path.write_text(
        "\n".join(
            [
                "inference:",
                "  num_designs: 2",
                f"  num_cycle: {num_cycle}",
                f"  output_prefix: {output_prefix}",
                "  seed: null",
                "design_config:",
                "  use_partial_diffusion: false",
                "  input_pdb: BondFlow/experiment/cyclize/MDM2/4HFZ_MDM2_target.cif",
                "  contigs: [[New_16-19], [Chain_A:seq_FIX:str_FIX]]",
                "preprocess:",
                "  link_config: BondFlow/config/link.csv",
                "",
            ]
        )
    )


def run_dry_run(config_path: Path, **env_overrides: str) -> Path:
    env = os.environ.copy()
    env.update(
        {
            "CFG_PATH": str(config_path),
            "DRY_RUN": "1",
            "SLURM_JOB_ID": "testjob",
            **env_overrides,
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    return Path(OmegaConf.load(config_path).inference.output_prefix) / "slurm_testjob"


def design_chain_length(contigs) -> int:
    length = 0
    for contig in contigs[0]:
        selection = str(contig).split(":", 1)[0]
        if selection.startswith("New_"):
            lower, upper = selection.removeprefix("New_").split("-")
            assert lower == upper
            length += int(lower)
        else:
            assert len(selection) == 1
            length += 1
    return length


def test_dry_run_writes_one_motif_config_per_cycle(tmp_path: Path) -> None:
    config_path = tmp_path / "mdm2.yaml"
    output_prefix = tmp_path / "outputs"
    write_config(config_path, output_prefix)

    run_root = run_dry_run(
        config_path,
        MOTIF_PROBABILITY="1",
        MOTIF_SEED="11",
        MOTIF_TEMPLATE_MODE="both",
    )

    manifest_rows = (run_root / "motif_manifest.tsv").read_text().splitlines()
    assert manifest_rows[0].split("\t") == [
        "cycle",
        "motif_applied",
        "template",
        "design_length",
        "contigs",
        "config_path",
    ]
    assert len(manifest_rows) == 7

    config_paths = sorted((run_root / "cycle_configs").glob("cycle_*.yaml"))
    assert len(config_paths) == 6
    templates = set()
    for config_file in config_paths:
        config = OmegaConf.load(config_file)
        assert config.inference.num_cycle == 1
        assert 16 <= design_chain_length(config.design_config.contigs) <= 19
        contigs_text = ",".join(str(contig) for contig in config.design_config.contigs[0])
        if "F:seq_FIX:str_DNV,New_2-2,Y:seq_FIX:str_DNV,W:seq_FIX:str_DNV,New_2-2,L:seq_FIX:str_DNV" in contigs_text:
            templates.add("pmi")
        if "F:seq_FIX:str_DNV,New_3-3,W:seq_FIX:str_DNV,New_2-2,L:seq_FIX:str_DNV" in contigs_text:
            templates.add("p53")

    assert templates <= {"pmi", "p53"}
    assert templates


def test_dry_run_can_skip_motif_for_every_cycle(tmp_path: Path) -> None:
    config_path = tmp_path / "mdm2.yaml"
    output_prefix = tmp_path / "outputs"
    write_config(config_path, output_prefix, num_cycle=4)

    run_root = run_dry_run(
        config_path,
        MOTIF_PROBABILITY="0",
        MOTIF_SEED="11",
    )

    for config_file in sorted((run_root / "cycle_configs").glob("cycle_*.yaml")):
        config = OmegaConf.load(config_file)
        design_contigs = list(config.design_config.contigs[0])
        assert len(design_contigs) == 1
        assert str(design_contigs[0]).startswith("New_")
        assert 16 <= design_chain_length(config.design_config.contigs) <= 19


def test_dry_run_manifest_is_reproducible_for_the_same_seed(tmp_path: Path) -> None:
    config_path = tmp_path / "mdm2.yaml"
    output_prefix = tmp_path / "outputs"
    write_config(config_path, output_prefix, num_cycle=8)

    run_root = run_dry_run(config_path, MOTIF_PROBABILITY="0.5", MOTIF_SEED="41")
    first_manifest = (run_root / "motif_manifest.tsv").read_text()

    run_root = run_dry_run(config_path, MOTIF_PROBABILITY="0.5", MOTIF_SEED="41")
    assert (run_root / "motif_manifest.tsv").read_text() == first_manifest


def test_script_has_a_local_worker_fallback_without_srun() -> None:
    script = SCRIPT_PATH.read_text()

    assert "run_local_workers()" in script
    assert '[[ "${LOCAL_RUN}" == "1" ]] || ! command -v srun >/dev/null 2>&1' in script
    assert 'CUDA_VISIBLE_DEVICES="${gpu_ids[rank]}"' in script
