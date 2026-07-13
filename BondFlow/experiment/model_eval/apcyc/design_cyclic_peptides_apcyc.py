#!/usr/bin/env python3
"""
APCyc baseline runner for BondFlow cyclic-peptide benchmarks.

For each benchmark complex:
1) infer the shortest peptide/protein chain as the original ligand;
2) select receptor chains and contact hotspots using the same convention as
   the existing benchmark baselines;
3) call APCyc pocket detection and codesign sampling;
4) normalize generated PDBs into <out_root>/<pdb_id>/design_XXX_seedYYY.pdb.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from Bio import PDB
from Bio.PDB import NeighborSearch


@dataclass(frozen=True)
class CaseInfo:
    pdb_id: str
    pdb_path: str
    ligand_chain: str
    ligand_length: int
    target_chains: Tuple[str, ...]
    hotspot_tokens: Tuple[str, ...]


def _is_peptide_residue(residue: PDB.Residue.Residue) -> bool:
    return PDB.is_aa(residue, standard=False)


def _get_chain_residues(chain: PDB.Chain.Chain) -> List[PDB.Residue.Residue]:
    return [res for res in chain.get_residues() if _is_peptide_residue(res)]


def _heavy_atoms(residue: PDB.Residue.Residue) -> List[PDB.Atom.Atom]:
    atoms: List[PDB.Atom.Atom] = []
    for atom in residue.get_atoms():
        element = (atom.element or "").upper().strip()
        if element == "H":
            continue
        name = atom.get_name().upper()
        if name.startswith("H"):
            continue
        atoms.append(atom)
    return atoms


def _load_structure(pdb_path: Path) -> PDB.Structure.Structure:
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure(pdb_path.stem, str(pdb_path))


def _choose_ligand_chain(model: PDB.Model.Model) -> Tuple[str, Dict[str, int]]:
    chain_lengths: Dict[str, int] = {}
    for chain in model.get_chains():
        n_res = len(_get_chain_residues(chain))
        if n_res > 0:
            chain_lengths[chain.id] = n_res
    if len(chain_lengths) < 2:
        raise ValueError("Need at least 2 peptide/protein chains for APCyc benchmark design.")
    ligand_chain = min(chain_lengths.items(), key=lambda x: x[1])[0]
    return ligand_chain, chain_lengths


def _score_target_chains(
    model: PDB.Model.Model,
    ligand_chain_id: str,
    contact_dist: float,
) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]]]:
    chain_map = {chain.id: chain for chain in model.get_chains()}
    ligand_chain = chain_map[ligand_chain_id]
    ligand_atoms: List[PDB.Atom.Atom] = []
    for res in _get_chain_residues(ligand_chain):
        ligand_atoms.extend(_heavy_atoms(res))
    if not ligand_atoms:
        raise ValueError(f"Ligand chain {ligand_chain_id} has no heavy atoms.")
    ns = NeighborSearch(ligand_atoms)

    chain_scores: Dict[str, float] = {}
    residue_scores: Dict[str, Dict[int, float]] = {}
    for chain_id, chain in chain_map.items():
        if chain_id == ligand_chain_id:
            continue
        per_res_score: Dict[int, float] = {}
        total = 0.0
        for res in _get_chain_residues(chain):
            rid = int(res.id[1])
            score = 0.0
            for atom in _heavy_atoms(res):
                neighbors = ns.search(atom.coord, contact_dist, level="A")
                score += float(len(neighbors))
            if score > 0:
                per_res_score[rid] = per_res_score.get(rid, 0.0) + score
                total += score
        chain_scores[chain_id] = total
        residue_scores[chain_id] = per_res_score
    return chain_scores, residue_scores


def _select_target_chains(
    chain_scores: Dict[str, float],
    chain_lengths: Dict[str, int],
    ligand_chain: str,
    ratio: float,
    min_contacts: float,
) -> List[str]:
    non_ligand = [chain for chain in chain_scores if chain != ligand_chain]
    if not non_ligand:
        raise ValueError("No receptor chains available.")

    max_score = max(chain_scores[chain] for chain in non_ligand)
    selected = [
        chain
        for chain in non_ligand
        if chain_scores[chain] >= (max_score * ratio) and chain_scores[chain] >= min_contacts
    ]
    if selected:
        return sorted(selected)

    if max_score > 0:
        return [max(non_ligand, key=lambda chain: chain_scores[chain])]
    return [max(non_ligand, key=lambda chain: chain_lengths.get(chain, 0))]


def _pick_hotspots(
    residue_scores: Dict[str, Dict[int, float]],
    target_chains: Sequence[str],
    max_total: int,
) -> List[str]:
    scored: List[Tuple[str, int, float]] = []
    for chain_id in target_chains:
        for rid, score in residue_scores.get(chain_id, {}).items():
            scored.append((chain_id, int(rid), float(score)))
    scored.sort(key=lambda item: item[2], reverse=True)

    hotspots: List[str] = []
    seen = set()
    for chain_id, rid, _ in scored:
        token = f"{chain_id}{rid}"
        if token in seen:
            continue
        seen.add(token)
        hotspots.append(token)
        if len(hotspots) >= max_total:
            break
    return hotspots


def _collect_input_pdbs(input_path: Path, recursive: bool = False) -> List[Path]:
    resolved = input_path.resolve()
    if resolved.is_file():
        if resolved.suffix.lower() != ".pdb":
            raise ValueError(f"Input file must be a .pdb file: {resolved}")
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")
    return sorted(resolved.rglob("*.pdb") if recursive else resolved.glob("*.pdb"))


def _extract_case_info(
    pdb_path: Path,
    chain_select_ratio: float,
    hotspot_max_total: int,
    contact_dist: float,
    min_chain_contacts: float,
) -> CaseInfo:
    structure = _load_structure(pdb_path)
    model = structure[0]
    ligand_chain, chain_lengths = _choose_ligand_chain(model)
    chain_scores, residue_scores = _score_target_chains(
        model=model,
        ligand_chain_id=ligand_chain,
        contact_dist=contact_dist,
    )
    target_chains = _select_target_chains(
        chain_scores=chain_scores,
        chain_lengths=chain_lengths,
        ligand_chain=ligand_chain,
        ratio=chain_select_ratio,
        min_contacts=min_chain_contacts,
    )
    hotspots = _pick_hotspots(
        residue_scores=residue_scores,
        target_chains=target_chains,
        max_total=hotspot_max_total,
    )
    return CaseInfo(
        pdb_id=pdb_path.stem,
        pdb_path=str(pdb_path.resolve()),
        ligand_chain=ligand_chain,
        ligand_length=int(chain_lengths[ligand_chain]),
        target_chains=tuple(target_chains),
        hotspot_tokens=tuple(hotspots),
    )


def _make_cases(args: argparse.Namespace) -> List[CaseInfo]:
    pdb_files = _collect_input_pdbs(Path(args.input_dir), recursive=bool(args.recursive_input))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found under: {args.input_dir}")
    return [
        _extract_case_info(
            pdb_path=pdb_path,
            chain_select_ratio=float(args.chain_select_ratio),
            hotspot_max_total=int(args.hotspot_max_total),
            contact_dist=float(args.contact_dist),
            min_chain_contacts=float(args.min_chain_contacts),
        )
        for pdb_path in pdb_files
    ]


def _parse_gpu_ids(gpus: str) -> List[str]:
    out: List[str] = []
    for item in gpus.split(","):
        token = item.strip()
        if not token:
            continue
        if token.startswith("cuda:"):
            token = token.split(":", 1)[1]
        out.append(token)
    if not out:
        raise ValueError("No valid GPUs provided. Example: --gpus 0,1,2,3")
    return out


def _command_prefix(conda_env: str | None) -> List[str]:
    if conda_env and conda_env.lower() not in {"none", "null", "system"}:
        return ["conda", "run", "--no-capture-output", "-n", conda_env]
    return []


def _build_apcyc_commands(
    case: CaseInfo,
    apcyc_root: Path,
    ckpt: Path,
    pocket_json: Path,
    raw_out_dir: Path,
    gpu_id: str,
    num_designs: int,
    conda_env: str | None,
) -> Tuple[List[str], List[str]]:
    prefix = _command_prefix(conda_env)
    detect_cmd = [
        *prefix,
        "python",
        "-m",
        "api.detect_pocket",
        "--pdb",
        case.pdb_path,
        "--target_chains",
        *case.target_chains,
        "--ligand_chains",
        case.ligand_chain,
        "--out",
        str(pocket_json),
    ]
    run_cmd = [
        *prefix,
        "python",
        "-m",
        "api.run",
        "--mode",
        "codesign",
        "--pdb",
        case.pdb_path,
        "--pocket",
        str(pocket_json),
        "--ckpt",
        str(ckpt),
        "--out_dir",
        str(raw_out_dir),
        "--length_min",
        str(int(case.ligand_length)),
        "--length_max",
        str(int(case.ligand_length) + 1),
        "--n_samples",
        str(int(num_designs)),
        "--gpu",
        "0",
    ]
    return detect_cmd, run_cmd


def _run_cmd(cmd: Sequence[str], cwd: Path, env: Dict[str, str], log_prefix: str) -> None:
    print(f"{log_prefix} CMD: {' '.join(cmd)}", flush=True)
    subprocess.run(list(cmd), cwd=str(cwd), env=env, check=True)


def _copy_apcyc_designs(
    raw_out_dir: Path,
    case_dir: Path,
    seed_base: int,
    max_designs: int,
) -> List[Dict[str, str]]:
    raw_pdbs = sorted(
        path
        for path in raw_out_dir.rglob("*.pdb")
        if path.is_file() and not path.name.startswith(".")
    )
    if not raw_pdbs:
        raise RuntimeError(f"No APCyc PDB outputs found under {raw_out_dir}")

    case_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    for design_index, src in enumerate(raw_pdbs[: int(max_designs)]):
        seed = int(seed_base) + design_index
        dst = case_dir / f"design_{design_index:03d}_seed{seed}.pdb"
        shutil.copy2(src, dst)
        rows.append(
            {
                "design_index": str(design_index),
                "seed": str(seed),
                "status": "ok",
                "output_pdb": str(dst),
                "raw_output_pdb": str(src),
                "error": "",
            }
        )
    return rows


def _run_single_case(
    case: CaseInfo,
    case_index: int,
    gpu_id: str,
    cfg: Dict[str, str],
) -> List[Dict[str, str]]:
    out_root = Path(cfg["out_root"]).resolve()
    apcyc_root = Path(cfg["apcyc_root"]).resolve()
    ckpt = Path(cfg["ckpt"]).resolve()
    num_designs = int(cfg["num_designs_per_target"])
    seed_base = int(cfg["seed_base"]) + case_index * num_designs
    dry_run = cfg["dry_run"] == "true"

    case_dir = out_root / case.pdb_id
    raw_out_dir = out_root / "_apcyc_raw" / case.pdb_id
    pocket_json = out_root / "_apcyc_pockets" / f"{case.pdb_id}_pocket.json"
    commands_json = case_dir / "apcyc_commands.json"
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    pocket_json.parent.mkdir(parents=True, exist_ok=True)

    detect_cmd, run_cmd = _build_apcyc_commands(
        case=case,
        apcyc_root=apcyc_root,
        ckpt=ckpt,
        pocket_json=pocket_json,
        raw_out_dir=raw_out_dir,
        gpu_id=gpu_id,
        num_designs=num_designs,
        conda_env=cfg["conda_env"],
    )
    commands_json.write_text(
        json.dumps({"detect_pocket": detect_cmd, "run_codesign": run_cmd}, indent=2) + "\n",
        encoding="utf-8",
    )

    row_base = {
        "worker_gpu_id": str(gpu_id),
        "pdb_id": case.pdb_id,
        "pdb_path": case.pdb_path,
        "ligand_chain": case.ligand_chain,
        "ligand_length": str(case.ligand_length),
        "target_chains": ",".join(case.target_chains),
        "hotspots": ",".join(case.hotspot_tokens),
        "target_binder_length": str(case.ligand_length),
        "pocket_json": str(pocket_json),
        "commands_json": str(commands_json),
    }

    if dry_run:
        return [
            {
                **row_base,
                "design_index": "",
                "seed": str(seed_base),
                "status": "dry_run",
                "output_pdb": "",
                "raw_output_pdb": "",
                "error": "",
            }
        ]

    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        prefix = f"[APCyc:{case.pdb_id}:gpu{gpu_id}]"
        _run_cmd(detect_cmd, cwd=apcyc_root, env=env, log_prefix=f"{prefix}[pocket]")
        _run_cmd(run_cmd, cwd=apcyc_root, env=env, log_prefix=f"{prefix}[sample]")
        design_rows = _copy_apcyc_designs(
            raw_out_dir=raw_out_dir,
            case_dir=case_dir,
            seed_base=seed_base,
            max_designs=num_designs,
        )
        return [{**row_base, **row} for row in design_rows]
    except Exception as exc:  # pylint: disable=broad-except
        err_file = case_dir / f"{case.pdb_id}.error.txt"
        err_file.write_text(traceback.format_exc(), encoding="utf-8")
        return [
            {
                **row_base,
                "design_index": "",
                "seed": str(seed_base),
                "status": "failed",
                "output_pdb": "",
                "raw_output_pdb": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        ]


def _write_case_summary(out_root: Path, cases: Sequence[CaseInfo]) -> None:
    case_csv = out_root / "_case_summary.csv"
    with open(case_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pdb_id",
                "pdb_path",
                "ligand_chain",
                "ligand_length",
                "target_chains",
                "hotspots",
            ],
        )
        writer.writeheader()
        for case in cases:
            data = asdict(case)
            writer.writerow(
                {
                    "pdb_id": data["pdb_id"],
                    "pdb_path": data["pdb_path"],
                    "ligand_chain": data["ligand_chain"],
                    "ligand_length": data["ligand_length"],
                    "target_chains": ",".join(data["target_chains"]),
                    "hotspots": ",".join(data["hotspot_tokens"]),
                }
            )


def _write_design_summary(out_root: Path, rows: Sequence[Dict[str, str]]) -> None:
    design_csv = out_root / "_design_summary.csv"
    fieldnames = [
        "worker_gpu_id",
        "pdb_id",
        "pdb_path",
        "design_index",
        "seed",
        "ligand_chain",
        "ligand_length",
        "target_chains",
        "hotspots",
        "target_binder_length",
        "status",
        "output_pdb",
        "raw_output_pdb",
        "pocket_json",
        "commands_json",
        "error",
    ]
    with open(design_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run APCyc baseline for cyclic-peptide benchmark complexes.")
    parser.add_argument("--input_dir", type=str, required=True, help="Benchmark PDB directory or single PDB file.")
    parser.add_argument("--out_root", type=str, required=True, help="Output root for normalized APCyc designs.")
    parser.add_argument("--apcyc_root", type=str, required=True, help="APCyc repository root.")
    parser.add_argument("--ckpt", type=str, default=None, help="APCyc codesign checkpoint path.")
    parser.add_argument("--conda_env", type=str, default="APCyc", help="Conda env for APCyc; use 'none' for current env.")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3.")
    parser.add_argument("--num_designs_per_target", type=int, default=8, help="Designs per benchmark target.")
    parser.add_argument("--seed_base", type=int, default=0, help="Base seed used in normalized output filenames.")
    parser.add_argument("--recursive_input", action="store_true", help="Recursively find PDB files under input_dir.")
    parser.add_argument("--chain_select_ratio", type=float, default=0.25, help="Receptor chain selection ratio.")
    parser.add_argument("--min_chain_contacts", type=float, default=1.0, help="Minimum receptor chain contact score.")
    parser.add_argument("--contact_dist", type=float, default=6.0, help="Heavy-atom contact cutoff in Angstroms.")
    parser.add_argument("--hotspot_max_total", type=int, default=6, help="Maximum hotspot residues recorded per case.")
    parser.add_argument("--max_parallel_cases", type=int, default=None, help="Max parallel cases; defaults to GPU count.")
    parser.add_argument("--dry_run", action="store_true", help="Write inferred cases and APCyc commands without running APCyc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root).resolve()
    apcyc_root = Path(args.apcyc_root).resolve()
    ckpt = Path(args.ckpt).resolve() if args.ckpt else apcyc_root / "checkpoints" / "codesign.ckpt"

    if not apcyc_root.is_dir() and not args.dry_run:
        raise NotADirectoryError(f"APCyc root is not a directory: {apcyc_root}")
    if not ckpt.is_file() and not args.dry_run:
        raise FileNotFoundError(f"APCyc checkpoint not found: {ckpt}")

    out_root.mkdir(parents=True, exist_ok=True)
    cases = _make_cases(args)
    _write_case_summary(out_root, cases)

    gpu_ids = _parse_gpu_ids(args.gpus)
    max_parallel = args.max_parallel_cases if args.max_parallel_cases is not None else len(gpu_ids)
    max_parallel = max(1, min(int(max_parallel), len(cases), len(gpu_ids)))
    cfg = {
        "out_root": str(out_root),
        "apcyc_root": str(apcyc_root),
        "ckpt": str(ckpt),
        "conda_env": str(args.conda_env),
        "num_designs_per_target": str(args.num_designs_per_target),
        "seed_base": str(args.seed_base),
        "dry_run": "true" if args.dry_run else "false",
    }

    all_rows: List[Dict[str, str]] = []
    if max_parallel == 1:
        for case_index, case in enumerate(cases):
            gpu_id = gpu_ids[case_index % len(gpu_ids)]
            all_rows.extend(_run_single_case(case, case_index, gpu_id, cfg))
    else:
        future_to_case: Dict = {}
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            for case_index, case in enumerate(cases):
                gpu_id = gpu_ids[case_index % len(gpu_ids)]
                future = executor.submit(_run_single_case, case, case_index, gpu_id, cfg)
                future_to_case[future] = case.pdb_id
            for future in as_completed(future_to_case):
                all_rows.extend(future.result())

    all_rows.sort(key=lambda row: (row.get("pdb_id", ""), row.get("design_index", ""), row.get("seed", "")))
    _write_design_summary(out_root, all_rows)
    n_ok = sum(1 for row in all_rows if row.get("status") == "ok")
    n_failed = sum(1 for row in all_rows if row.get("status") == "failed")
    n_dry = sum(1 for row in all_rows if row.get("status") == "dry_run")
    print(f"Cases: {len(cases)}")
    print(f"Rows: {len(all_rows)}")
    print(f"Success: {n_ok}, Failed: {n_failed}, Dry-run: {n_dry}")
    print(f"Case summary: {out_root / '_case_summary.csv'}")
    print(f"Design summary: {out_root / '_design_summary.csv'}")


if __name__ == "__main__":
    main()
