from pathlib import Path
import sys


def _atom_line(serial, atom, resname, chain, resseq, x, y, z):
    element = atom.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom:^4s} {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2s}\n"
    )


def _write_two_chain_fixture(path: Path) -> None:
    lines = [
        _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
        _atom_line(2, "CA", "ALA", "A", 2, 3.0, 0.0, 0.0),
        _atom_line(3, "CA", "ALA", "A", 3, 30.0, 0.0, 0.0),
        "TER\n",
        _atom_line(4, "CA", "GLY", "B", 1, 0.5, 0.0, 0.0),
        _atom_line(5, "CA", "GLY", "B", 2, 3.5, 0.0, 0.0),
        "TER\n",
        "END\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def test_extract_case_info_uses_shortest_chain_and_contact_hotspots(tmp_path):
    from BondFlow.experiment.model_eval.apcyc.design_cyclic_peptides_apcyc import (
        _extract_case_info,
    )

    pdb_path = tmp_path / "toy_complex.pdb"
    _write_two_chain_fixture(pdb_path)

    case = _extract_case_info(
        pdb_path,
        chain_select_ratio=0.25,
        hotspot_max_total=2,
        contact_dist=1.0,
        min_chain_contacts=1.0,
    )

    assert case.pdb_id == "toy_complex"
    assert case.ligand_chain == "B"
    assert case.ligand_length == 2
    assert case.target_chains == ("A",)
    assert case.hotspot_tokens == ("A1", "A2")


def test_build_apcyc_commands_pin_ligand_length_and_pocket_inputs():
    from BondFlow.experiment.model_eval.apcyc.design_cyclic_peptides_apcyc import (
        CaseInfo,
        _build_apcyc_commands,
    )

    case = CaseInfo(
        pdb_id="toy_complex",
        pdb_path="/bench/toy_complex.pdb",
        ligand_chain="B",
        ligand_length=2,
        target_chains=("A",),
        hotspot_tokens=("A1", "A2"),
    )

    detect_cmd, run_cmd = _build_apcyc_commands(
        case=case,
        apcyc_root=Path("/opt/APCyc"),
        ckpt=Path("/opt/APCyc/checkpoints/codesign.ckpt"),
        pocket_json=Path("/tmp/pocket.json"),
        raw_out_dir=Path("/tmp/raw"),
        gpu_id="2",
        num_designs=8,
        conda_env="APCyc",
    )

    assert detect_cmd[:5] == ["conda", "run", "--no-capture-output", "-n", "APCyc"]
    assert detect_cmd[5:9] == ["python", "-m", "api.detect_pocket", "--pdb"]
    assert "--target_chains" in detect_cmd
    assert detect_cmd[detect_cmd.index("--target_chains") + 1] == "A"
    assert "--ligand_chains" in detect_cmd
    assert detect_cmd[detect_cmd.index("--ligand_chains") + 1] == "B"

    assert run_cmd[:5] == ["conda", "run", "--no-capture-output", "-n", "APCyc"]
    assert run_cmd[5:9] == ["python", "-m", "api.run", "--mode"]
    assert run_cmd[run_cmd.index("--length_min") + 1] == "2"
    assert run_cmd[run_cmd.index("--length_max") + 1] == "3"
    assert run_cmd[run_cmd.index("--n_samples") + 1] == "8"
    assert run_cmd[run_cmd.index("--ckpt") + 1] == "/opt/APCyc/checkpoints/codesign.ckpt"


def test_copy_apcyc_designs_normalizes_output_names(tmp_path):
    from BondFlow.experiment.model_eval.apcyc.design_cyclic_peptides_apcyc import (
        _copy_apcyc_designs,
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "toy_complex_0.pdb").write_text("MODEL 0\nEND\n", encoding="utf-8")
    (raw_dir / "toy_complex_1.pdb").write_text("MODEL 1\nEND\n", encoding="utf-8")
    (raw_dir / "summary.jsonl").write_text("{}\n", encoding="utf-8")

    case_dir = tmp_path / "normalized" / "toy_complex"
    rows = _copy_apcyc_designs(raw_dir, case_dir, seed_base=10, max_designs=2)

    output_names = [Path(row["output_pdb"]).name for row in rows]
    assert output_names == ["design_000_seed10.pdb", "design_001_seed11.pdb"]
    assert (case_dir / "design_000_seed10.pdb").read_text(encoding="utf-8") == "MODEL 0\nEND\n"
    assert (case_dir / "design_001_seed11.pdb").read_text(encoding="utf-8") == "MODEL 1\nEND\n"
    assert all(row["status"] == "ok" for row in rows)


def test_evaluator_accepts_apcyc_eval_source_and_uses_shortest_chain(tmp_path):
    analysis_dir = Path(__file__).resolve().parents[1] / "experiment" / "analysis"
    sys.path.insert(0, str(analysis_dir))
    from BondFlow.experiment.analysis import evaluate_benchmark_designs as eval_mod

    pdb_path = tmp_path / "apcyc_design.pdb"
    _write_two_chain_fixture(pdb_path)

    assert "apcyc" in eval_mod.EVAL_SOURCES
    assert eval_mod._pick_ligand_chain(str(pdb_path), eval_source="apcyc") == ("B", 2)
