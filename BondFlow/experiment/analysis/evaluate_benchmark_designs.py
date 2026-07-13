#!/usr/bin/env python3
import argparse
import itertools
import logging
import math
import os
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import PDB
from Bio.PDB import DSSP, MMCIFParser, PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import PPBuilder, is_aa, protein_letters_3to1

from TMalign import run_usalign

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


AA20 = list("ACDEFGHIKLMNPQRSTVWY")
SS3 = ["H", "E", "C"]
STRUCT_EXTS = {".pdb", ".cif", ".mmcif"}
BOND_TYPES = ["disulfide", "isopeptide", "lactone", "head_tail", "other"]
EVAL_SOURCES = {"bondflow", "funcbind", "afcycdesign", "apcyc"}


def _compact_error_message(msg: object) -> str:
    text = "" if msg is None else str(msg)
    text = text.replace("\r", " ").replace("\n", " | ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text


class _OneChainSelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id


def _get_parser_by_ext(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def _collect_structure_files(folder: str) -> List[str]:
    out = []
    root = Path(folder)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in STRUCT_EXTS:
            out.append(str(p))
    return sorted(out)


def _filter_design_files_by_subdir(design_files: List[str], design_root: str, subdir_keyword: str) -> List[str]:
    """
    Keep only design files whose relative path contains the target subdir keyword,
    e.g. subdir_keyword='post_refine' keeps paths like */post_refine/*.pdb
    """
    key = (subdir_keyword or "").strip().strip("/\\")
    if not key:
        return design_files

    out = []
    for fp in design_files:
        rel = os.path.relpath(fp, design_root)
        parts = rel.replace("\\", "/").split("/")
        if key in parts:
            out.append(fp)
    return sorted(out)


def _parse_csv_list_arg(text: Optional[str]) -> set:
    if not text:
        return set()
    return {x.strip() for x in str(text).split(",") if x.strip()}


def _parse_task_types_arg(text: Optional[str]) -> set:
    if not text:
        return {"all"}
    parsed = {x.strip().lower() for x in str(text).split(",") if x.strip()}
    if not parsed:
        return {"all"}
    valid = {"all", "tm", "bond", "energy", "composition"}
    invalid = parsed - valid
    if invalid:
        raise ValueError(f"Invalid task types: {sorted(invalid)}; valid={sorted(valid)}")
    if "all" in parsed and len(parsed) > 1:
        return {"all"}
    return parsed


def _task_key_from_task(task: Dict) -> Tuple[str, str]:
    return str(task["benchmark_name"]), str(task["design_path"])


def _task_key_from_row(row: pd.Series) -> Tuple[str, str]:
    return str(row.get("Benchmark", "")), str(row.get("Design_Path", ""))


def _load_existing_design_metrics(design_csv: Path, logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], set]:
    if not design_csv.exists():
        return None, set()
    try:
        existing_df = pd.read_csv(design_csv)
    except Exception as e:
        logger.warning("Failed to load existing design metrics: %s", e)
        return None, set()

    if "Benchmark" not in existing_df.columns or "Design_Path" not in existing_df.columns:
        logger.warning("Existing design metrics missing Benchmark/Design_Path; ignore reuse.")
        return None, set()

    keys = set(zip(existing_df["Benchmark"].astype(str), existing_df["Design_Path"].astype(str)))
    logger.info("Loaded existing design metrics rows=%d", len(existing_df))
    return existing_df, keys


def _select_tasks_to_run(
    tasks: List[Dict],
    reuse_existing: bool,
    existing_keys: set,
    rerun_benchmarks: set,
    rerun_task_types: set,
) -> List[Dict]:
    selected = []
    rerun_all_existing_by_task_type = (rerun_task_types != {"all"}) and (not rerun_benchmarks)
    for t in tasks:
        key = _task_key_from_task(t)
        bname = str(t["benchmark_name"])
        must_rerun = False
        if rerun_benchmarks:
            must_rerun = (bname in rerun_benchmarks)
        elif rerun_all_existing_by_task_type:
            # If user requests partial task-type rerun (e.g., energy only) without specifying
            # benchmarks, default to rerun those task types for all mapped benchmarks.
            must_rerun = True
        if must_rerun:
            selected.append(t)
            continue
        if reuse_existing:
            if key not in existing_keys:
                selected.append(t)
        else:
            selected.append(t)
    return selected


def _merge_results_with_existing(
    existing_df: Optional[pd.DataFrame],
    run_df: pd.DataFrame,
    all_tasks: List[Dict],
) -> pd.DataFrame:
    key_cols = ["Benchmark", "Design_Path"]

    if existing_df is None or existing_df.empty:
        out = run_df.copy()
    else:
        existing = existing_df.copy()
        if run_df is not None and not run_df.empty:
            run_idx = run_df.set_index(key_cols)
            existing_idx = existing.set_index(key_cols)

            # Add any new columns from run output first.
            for c in run_idx.columns:
                if c not in existing_idx.columns:
                    existing_idx[c] = np.nan

            # Overwrite rerun rows column-by-column (including NaN when rerun produced NaN).
            shared_idx = run_idx.index.intersection(existing_idx.index)
            for c in run_idx.columns:
                # Ensure destination dtype can hold source values (avoid pandas FutureWarning on mixed dtypes).
                src_dtype = run_idx[c].dtype
                dst_dtype = existing_idx[c].dtype
                if src_dtype == object and dst_dtype != object:
                    existing_idx[c] = existing_idx[c].astype(object)
                existing_idx.loc[shared_idx, c] = run_idx.loc[shared_idx, c]

            # Append brand new rows.
            new_idx = run_idx.index.difference(existing_idx.index)
            if len(new_idx) > 0:
                existing_idx = pd.concat([existing_idx, run_idx.loc[new_idx]], axis=0, sort=False)

            out = existing_idx.reset_index()
        else:
            out = existing

    # Keep only rows that belong to current mapping.
    task_keys = {_task_key_from_task(t) for t in all_tasks}
    if not out.empty and "Benchmark" in out.columns and "Design_Path" in out.columns:
        out = out[out.apply(lambda r: _task_key_from_row(r) in task_keys, axis=1)].reset_index(drop=True)
    return out


def _find_bonds_txt_for_structure(struct_path: str) -> Optional[str]:
    d = os.path.dirname(struct_path)
    base = os.path.splitext(os.path.basename(struct_path))[0]
    candidates = [
        os.path.join(d, f"{base}.txt"),
        os.path.join(d, f"{base}_bonds.txt"),
        os.path.join(d, f"bonds_{base}.txt"),
    ]
    if base.startswith("partial_"):
        clean = base.replace("partial_", "")
        candidates.append(os.path.join(d, f"bonds_{clean}.txt"))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _safe_name(path: str) -> str:
    return path.replace("/", "__").replace("\\", "__")


def _pick_shortest_chain(struct_path: str) -> Tuple[Optional[str], int]:
    parser = _get_parser_by_ext(struct_path)
    structure = parser.get_structure("struct", struct_path)
    model = structure[0]
    best_chain = None
    best_len = None
    for chain in model:
        aa_len = sum(1 for r in chain if is_aa(r, standard=True))
        if aa_len <= 0:
            continue
        if best_len is None or aa_len < best_len:
            best_chain = chain.id
            best_len = aa_len
    if best_chain is None:
        return None, 0
    return str(best_chain), int(best_len)


def _pick_unl_chain(struct_path: str) -> Tuple[Optional[str], int]:
    parser = _get_parser_by_ext(struct_path)
    structure = parser.get_structure("struct", struct_path)
    model = structure[0]
    best = None
    for chain in model:
        unl_count = sum(1 for r in chain if str(r.get_resname()).upper() == "UNL")
        if unl_count <= 0:
            continue
        aa_len = sum(1 for r in chain if is_aa(r, standard=True))
        candidate = (-int(unl_count), int(aa_len), str(chain.id))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None, 0
    return best[2], int(-best[0])


def _pick_ligand_chain(struct_path: str, eval_source: str) -> Tuple[Optional[str], int]:
    source = (eval_source or "bondflow").strip().lower()
    if source == "funcbind":
        chain, length = _pick_unl_chain(struct_path)
        if chain is not None:
            return chain, length
    # bondflow/afcycdesign/apcyc and funcbind fallback all use original shortest-AA behavior.
    return _pick_shortest_chain(struct_path)


def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _prepare_funcbind_unl_params(
    benchmark_name: str,
    funcbind_sdf_dir: str,
    params_output_dir: Path,
    molfile_to_params_script: Optional[str],
    paramsgen_env: str,
    logger: logging.Logger,
) -> Optional[str]:
    def _sanitize_unl_params_remove_ring_decls(params_in: Path, params_out: Path) -> bool:
        """
        Precisely "sanitize" a UNL.params file via text processing:

        Goal: keep the kinematic-tree topology information (CUT_BOND/VIRTUAL_SHADOW/ICOOR_INTERNAL),
        but remove the explicit ring-conformer sampling semantics so PyRosetta doesn't need
        ring_conformer_sets for very large rings.

        Specifically:
        - delete entire `ADD_RING ...` lines
        - delete `CYCLIC` tag in `PROPERTIES ...` (or drop the line if empty)
        - delete `NU ...` lines
        - delete `LOW_RING_CONFORMERS ...` and `LOWEST_RING_CONFORMER ...` lines

        Keep:
        - all `CUT_BOND ...` lines
        - all `VIRTUAL_SHADOW ...` lines
        - all `ICOOR_INTERNAL ...` lines (and anything else not matched above)

        This is intentionally aggressive and assumes our evaluation only needs physical connectivity
        (not ring pucker conformer enumeration).
        """
        removed_any = False

        in_lines = params_in.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        out_lines: List[str] = []
        for raw in in_lines:
            s = raw.strip()
            if not s:
                out_lines.append(raw)
                continue

            u = s.upper()
            if u.startswith("ADD_RING"):
                removed_any = True
                continue
            if u.startswith("NU " ) or u == "NU":
                removed_any = True
                continue
            if u.startswith("LOW_RING_CONFORMERS") or u.startswith("LOWEST_RING_CONFORMER"):
                removed_any = True
                continue
            if u.startswith("PROPERTIES"):
                # Remove only the CYCLIC tag; keep other properties.
                # Examples in this repo look like: `PROPERTIES CYCLIC` or `PROPERTIES <tags...>`
                toks = s.split()
                if len(toks) >= 2:
                    filtered = [t for t in toks[1:] if t.upper() != "CYCLIC"]
                    if not filtered:
                        removed_any = True
                        continue
                    new_line = "PROPERTIES " + " ".join(filtered)
                    if raw.endswith("\n"):
                        new_line += "\n"
                    out_lines.append(new_line)
                    removed_any = True
                    continue

            out_lines.append(raw)

        if not out_lines:
            return False

        params_out.write_text("".join(out_lines), encoding="utf-8")
        if removed_any:
            logger.info("funcbind mode: sanitized %s -> %s (ring conformer DB friendly)", params_in, params_out)
        return True

    sdf_dir = Path(funcbind_sdf_dir)
    candidates = [
        sdf_dir / f"{benchmark_name}.sdf",
        sdf_dir / f"{benchmark_name}.mol2",
        sdf_dir / benchmark_name / "ligand.sdf",
        sdf_dir / benchmark_name / "ligand.mol2",
    ]
    # funcbind outputs are often named ligand_001.sdf / ligand_002.sdf ...
    try:
        candidates.extend(sorted(sdf_dir.glob("ligand_*.sdf")))
        candidates.extend(sorted(sdf_dir.glob("ligand_*.mol2")))
    except Exception:
        pass
    sdf_path = _find_first_existing(candidates)
    if sdf_path is None:
        logger.warning("funcbind mode: no ligand sdf/mol2 found for %s under %s", benchmark_name, sdf_dir)
        return None

    params_dir = params_output_dir / benchmark_name
    params_dir.mkdir(parents=True, exist_ok=True)
    params_path = params_dir / "UNL.params"
    params_sanitized_path = params_dir / "UNL_sanitized.params"
    ligand_pdb_path = params_dir / "UNL_ligand.pdb"
    if params_path.exists() and ligand_pdb_path.exists():
        ok = _sanitize_unl_params_remove_ring_decls(params_in=params_path, params_out=params_sanitized_path)
        return str(params_sanitized_path) if ok and params_sanitized_path.exists() else str(params_path)

    def _generate_with_paramsgen(input_path: Path, out_params: Path, out_pdb: Path) -> bool:
        # Use a dedicated conda env (defaults to "paramsgen") to avoid NumPy/RDKit ABI issues
        # in the analysis environment.
        py_code = r"""
import sys
from pathlib import Path
from rdkit import Chem
from rdkit_to_params import Params

inp = Path(sys.argv[1])
outp = Path(sys.argv[2])
outpd = Path(sys.argv[3])
suffix = inp.suffix.lower()
mol = None
if suffix == ".sdf":
    suppl = Chem.SDMolSupplier(str(inp), removeHs=False)
    for m in suppl:
        if m is not None:
            mol = m
            break
elif suffix == ".mol2":
    mol = Chem.MolFromMol2File(str(inp), removeHs=False)
else:
    raise RuntimeError(f"Unsupported ligand format: {inp}")

if mol is None:
    raise RuntimeError(f"Failed to parse ligand file: {inp}")

p = Params.from_mol(mol, name="UNL")
p.dump(str(outp))
p.dump_pdb(str(outpd))
print(f"WROTE {outp} and {outpd}")
"""
        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            paramsgen_env,
            "python",
            "-c",
            py_code,
            str(input_path),
            str(out_params),
            str(out_pdb),
        ]
        proc = subprocess.run(cmd, cwd=str(params_dir), capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning(
                "funcbind mode: paramsgen generation failed for %s (env=%s, code=%s). stderr=%s",
                benchmark_name,
                paramsgen_env,
                proc.returncode,
                proc.stderr.strip()[:500],
            )
            return False
        return out_params.exists()

    script_path = Path(molfile_to_params_script).expanduser() if molfile_to_params_script else None
    if script_path is None or (not script_path.exists()):
        ok = _generate_with_paramsgen(sdf_path, params_path, ligand_pdb_path)
        if ok:
            _sanitize_unl_params_remove_ring_decls(params_in=params_path, params_out=params_sanitized_path)
            return str(params_sanitized_path) if params_sanitized_path.exists() else str(params_path)
        logger.warning(
            "funcbind mode: cannot generate params for %s. molfile_to_params.py missing "
            "and paramsgen fallback failed. ligand=%s",
            benchmark_name,
            sdf_path,
        )
        return None

    cmd = [
        sys.executable,
        str(script_path),
        "-n",
        "UNL",
        "-p",
        "UNL",
        "--clobber",
        "--keep-names",
        str(sdf_path),
    ]
    proc = subprocess.run(cmd, cwd=str(params_dir), capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning(
            "funcbind mode: molfile_to_params failed for %s (code=%s). stderr=%s. "
            "Trying paramsgen fallback...",
            benchmark_name,
            proc.returncode,
            proc.stderr.strip()[:500],
        )
        ok = _generate_with_paramsgen(sdf_path, params_path, ligand_pdb_path)
        if ok:
            _sanitize_unl_params_remove_ring_decls(params_in=params_path, params_out=params_sanitized_path)
            return str(params_sanitized_path) if params_sanitized_path.exists() else str(params_path)
        return None
    if not params_path.exists():
        logger.warning("funcbind mode: params generation finished but UNL.params not found for %s", benchmark_name)
        return None
    if not ligand_pdb_path.exists():
        ok = _generate_with_paramsgen(sdf_path, params_path, ligand_pdb_path)
        if not ok:
            logger.warning("funcbind mode: failed to generate UNL ligand PDB for %s", benchmark_name)
    _sanitize_unl_params_remove_ring_decls(params_in=params_path, params_out=params_sanitized_path)
    return str(params_sanitized_path) if params_sanitized_path.exists() else str(params_path)


def _extract_chain_to_pdb(input_path: str, chain_id: str, output_pdb: str) -> bool:
    parser = _get_parser_by_ext(input_path)
    structure = parser.get_structure("struct", input_path)
    model = structure[0]
    has_chain = any(ch.id == chain_id for ch in model)
    if not has_chain:
        return False
    # Defensive: recreate parent dir in case temporary/output folders were removed mid-run.
    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, select=_OneChainSelect(chain_id))
    return True


def _dssp8_to_ss3(ss8: str) -> str:
    if not ss8 or ss8 == " ":
        return "C"
    if ss8 in {"H", "G", "I"}:
        return "H"
    if ss8 in {"E", "B"}:
        return "E"
    return "C"


def _phi_psi_to_ss3(phi, psi) -> str:
    if phi is None or psi is None:
        return "C"
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)
    if -160.0 <= phi_deg <= -30.0 and -100.0 <= psi_deg <= 20.0:
        return "H"
    if -180.0 <= phi_deg <= -40.0 and (70.0 <= psi_deg <= 180.0 or -180.0 <= psi_deg <= -140.0):
        return "E"
    return "C"


def _sanitize_pdb_for_dssp_if_needed(struct_path: str) -> str:
    """
    Some generated PDBs include non-standard REMARK lines like:
      REMARK AtomGroup Unnamed
    which makes mkdssp fail with "Not a valid integer in PDB record".
    For DSSP only, create a cleaned temporary PDB that drops malformed REMARK lines.
    """
    ext = os.path.splitext(struct_path)[1].lower()
    if ext != ".pdb":
        return struct_path
    try:
        with open(struct_path, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return struct_path

    changed = False
    out_lines: List[str] = []
    for line in lines:
        if line.startswith("REMARK"):
            # Standard PDB REMARK has integer remark number around cols 8-10.
            # Drop malformed lines (e.g., "REMARK AtomGroup Unnamed").
            if len(line) < 10 or (not line[7:10].strip().isdigit()):
                changed = True
                continue
        out_lines.append(line)

    if not changed:
        return struct_path

    tmp_dir = Path(tempfile.gettempdir()) / "bondflow_dssp_clean"
    tmp_path = str(tmp_dir / f"{Path(struct_path).stem}__dssp_clean_{os.getpid()}_{uuid.uuid4().hex}.pdb")
    try:
        Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w") as f:
            f.writelines(out_lines)
        return tmp_path
    except Exception:
        return struct_path


def _extract_chain_aa_ss_counts(struct_path: str, chain_id: str, prefer_dssp: bool = True) -> Tuple[Counter, Counter]:
    parser = _get_parser_by_ext(struct_path)
    structure = parser.get_structure("struct", struct_path)
    model = structure[0]
    if chain_id not in model:
        return Counter(), Counter()
    chain = model[chain_id]

    aa_counter = Counter()
    for residue in chain:
        if not is_aa(residue, standard=True):
            continue
        res_name = residue.get_resname().upper()
        if res_name in protein_letters_3to1:
            aa = protein_letters_3to1[res_name]
            if aa in AA20:
                aa_counter[aa] += 1

    ss_counter = Counter()
    if prefer_dssp:
        dssp_input = struct_path
        try:
            dssp_input = _sanitize_pdb_for_dssp_if_needed(struct_path)
            dssp = DSSP(model, dssp_input)
            for key in dssp.keys():
                if key[0] != chain_id:
                    continue
                ss_counter[_dssp8_to_ss3(dssp[key][2])] += 1
        except Exception:
            pass
        finally:
            if dssp_input != struct_path:
                try:
                    os.remove(dssp_input)
                except Exception:
                    pass

    if sum(ss_counter.values()) == 0:
        print("No DSSP or phi/psi data found, using peptide builder")
        ppb = PPBuilder()
        peptides = ppb.build_peptides(chain)
        if peptides:
            for pep in peptides:
                for phi, psi in pep.get_phi_psi_list():
                    ss_counter[_phi_psi_to_ss3(phi, psi)] += 1
        else:
            for residue in chain:
                if is_aa(residue, standard=True):
                    ss_counter["C"] += 1

    return aa_counter, ss_counter


def _to_freq(counter: Counter, keys: List[str]) -> Dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {k: 0.0 for k in keys}
    return {k: float(counter.get(k, 0)) / total for k in keys}


def _divergence_metrics(freq_design: Dict[str, float], freq_ref: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    p = np.array([freq_design[k] for k in keys], dtype=float)
    q = np.array([freq_ref[k] for k in keys], dtype=float)
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    m = 0.5 * (p_safe + q_safe)
    kl_pq = float(np.sum(p_safe * np.log2(p_safe / q_safe)))
    kl_qp = float(np.sum(q_safe * np.log2(q_safe / p_safe)))
    js = float(0.5 * np.sum(p_safe * np.log2(p_safe / m)) + 0.5 * np.sum(q_safe * np.log2(q_safe / m)))
    l1 = float(np.sum(np.abs(p - q)))
    l2 = float(np.sqrt(np.sum((p - q) ** 2)))
    return {
        "L1_Distance": l1,
        "L2_Distance": l2,
        "KL_Design_to_Ref": kl_pq,
        "KL_Ref_to_Design": kl_qp,
        "JS_Divergence": js,
    }


def _compute_binding_energy_for_ligand_chain(
    struct_path: str,
    ligand_chain_id: str,
    bonds_path: Optional[str],
    relax: bool,
    extra_res_fa_paths: Optional[List[str]] = None,
) -> Tuple[float, str]:
    try:
        from energy import _extra_res_fa_paths, apply_custom_bonds, apply_pdb_links
        import pyrosetta
        from pyrosetta import rosetta
        from pyrosetta import pose_from_file
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

        def _recover_chain_endings_from_pdb_info(pose) -> int:
            """
            Some cleaned PDBs are loaded with Rosetta chain topology that does not
            fully reflect PDB chain IDs (e.g., multiple PDB chain letters merged
            into one Rosetta chain). Recover topology by inserting chain endings
            at PDB chain-letter transitions that are not already Rosetta chain ends.
            """
            if pose.total_residue() < 2:
                return 0
            pdb_info = pose.pdb_info()
            if not pdb_info:
                return 0

            existing_chain_ends = {int(pose.chain_end(c)) for c in range(1, pose.num_chains())}
            cuts = []
            prev_chain = pdb_info.chain(1)
            for i in range(2, pose.total_residue() + 1):
                cur_chain = pdb_info.chain(i)
                cut = i - 1
                if (
                    cur_chain
                    and prev_chain
                    and cur_chain != prev_chain
                    and cut not in existing_chain_ends
                ):
                    cuts.append(cut)
                prev_chain = cur_chain

            if not cuts:
                return 0
            for cut in sorted(set(cuts), reverse=True):
                pose.conformation().insert_chain_ending(int(cut))
            return len(cuts)

        if not rosetta.basic.was_init_called():
            flags = "-ignore_unrecognized_res true -mute all -multithreading:total_threads 1"
            for p in _extra_res_fa_paths():
                if os.path.exists(p):
                    flags += f" -extra_res_fa {p}"
            for p in (extra_res_fa_paths or []):
                if p and os.path.exists(p):
                    flags += f" -extra_res_fa {p}"
            pyrosetta.init(flags)

        # In funcbind mode we generate an additional ligand PDB from the input ligand SDF/mol2.
        # Some complex PDBs may carry atom naming incompatible with the generated UNL.params, so
        # we graft the generated ligand coordinates/naming into the complex before loading.
        struct_path_loaded = struct_path
        ligand_pdb_path = None
        for p in (extra_res_fa_paths or []):
            if not p:
                continue
            cand = Path(p).with_name("UNL_ligand.pdb")
            if cand.exists():
                ligand_pdb_path = cand
                break

        if ligand_pdb_path is not None:
            try:
                lig_resseq_field = "   1"
                lig_serials: List[str] = []
                with open(struct_path, "r", errors="ignore") as f:
                    for line in f:
                        if not line.startswith("HETATM"):
                            continue
                        if line[17:20].strip().upper() != "UNL":
                            continue
                        if line[21].strip() != ligand_chain_id:
                            continue
                        lig_resseq_field = line[22:26]
                        lig_serials.append(line[6:11])
                        # Keep scanning to collect all serials (used by CONECT records, if any).

                grafted_path = os.path.join(
                    os.path.dirname(struct_path),
                    f"{Path(struct_path).stem}__pyrosetta_grafted_UNL_{os.getpid()}.pdb",
                )
                out_lines: List[str] = []
                end_lines: List[str] = []
                with open(struct_path, "r", errors="ignore") as f:
                    for line in f:
                        if line.startswith("HETATM"):
                            if line[17:20].strip().upper() == "UNL" and line[21].strip() == ligand_chain_id:
                                continue
                        if line.startswith(("END", "ENDMDL")):
                            end_lines.append(line)
                            continue
                        out_lines.append(line)

                # Append generated ligand atoms right before the final END record.
                lig_lines_written = 0
                atom_i = 0
                with open(ligand_pdb_path, "r", errors="ignore") as f:
                    for line in f:
                        if not (line.startswith("ATOM") or line.startswith("HETATM")):
                            continue
                        if len(line) < 26:
                            continue
                        line_list = list(line)
                        # chain id (PDB col 22 -> index 21)
                        line_list[21] = ligand_chain_id[:1] if ligand_chain_id else "X"
                        # residue sequence (PDB cols 23-26 -> [22:26])
                        line_list[22:26] = list(lig_resseq_field)
                        # keep original atom serial numbers if we can, to preserve possible CONECT records
                        if atom_i < len(lig_serials) and len(lig_serials[atom_i]) == 5:
                            line_list[6:11] = list(lig_serials[atom_i])
                        out_lines.append("".join(line_list))
                        atom_i += 1
                        lig_lines_written += 1

                if end_lines:
                    out_lines.extend(end_lines)
                else:
                    out_lines.append("END\n")

                Path(grafted_path).parent.mkdir(parents=True, exist_ok=True)
                with open(grafted_path, "w") as f:
                    f.writelines(out_lines)

                struct_path_loaded = grafted_path
            except Exception:
                # If grafting fails for any reason, fall back to the original structure.
                struct_path_loaded = struct_path

        pose = pose_from_file(struct_path_loaded)
        scorefxn = pyrosetta.rosetta.core.scoring.get_score_function()
        apply_pdb_links(pose, struct_path_loaded, strict=True)
        if bonds_path:
            apply_custom_bonds(pose, bonds_path, strict=True)
        _recover_chain_endings_from_pdb_info(pose)

        if relax:
            from pyrosetta.rosetta.protocols.relax import FastRelax

            relaxer = FastRelax()
            relaxer.set_scorefxn(scorefxn)
            relaxer.apply(pose)

        chain_ids = []
        if pose.pdb_info():
            for chain_num in range(1, pose.num_chains() + 1):
                c_start = pose.chain_begin(chain_num)
                c_id = pose.pdb_info().chain(c_start)
                if c_id not in chain_ids:
                    chain_ids.append(c_id)
            # Some cleaned PDBs are represented as a single Rosetta chain while
            # still preserving multiple original PDB chain IDs in pdb_info.
            # Include all unique chain letters from residue-level pdb_info to
            # avoid false "monomer" detection.
            for i in range(1, pose.total_residue() + 1):
                c_id = pose.pdb_info().chain(i)
                if c_id and c_id not in chain_ids:
                    chain_ids.append(c_id)
        if not chain_ids:
            chain_ids = [chr(64 + i) for i in range(1, pose.num_chains() + 1)]
        if len(chain_ids) < 2:
            return float("nan"), "monomer"

        lig = ligand_chain_id if ligand_chain_id in chain_ids else chain_ids[0]
        others = "".join([c for c in chain_ids if c != lig])
        if not others:
            return float("nan"), "no_receptor"

        iam = InterfaceAnalyzerMover(f"{lig}_{others}", False, scorefxn)
        iam.set_pack_rounds(1)
        iam.set_pack_input(True)
        iam.set_compute_packstat(False)
        iam.set_pack_separated(True)
        iam.apply(pose)
        return float(iam.get_interface_dG()), ""
    except Exception as e:
        return float("nan"), _compact_error_message(e)


def _pair_in_bond_scope(i: int, j: int, pdb_idx: List[Tuple[str, str]], ligand_chain: str, bond_scope: str) -> bool:
    if i < 0 or j < 0 or i >= len(pdb_idx) or j >= len(pdb_idx):
        return False
    ci = str(pdb_idx[i][0])
    cj = str(pdb_idx[j][0])
    if bond_scope == "all":
        return True
    if bond_scope == "ligand_internal":
        return ci == ligand_chain and cj == ligand_chain
    # ligand_related: ligand-ligand + ligand-receptor
    return ci == ligand_chain or cj == ligand_chain


def _eval_bonds_scoped(
    struct_path: str,
    bonds_path: str,
    link_csv: str,
    ligand_chain: str,
    bond_scope: str,
) -> Tuple[Dict, Dict[str, Tuple[float, float, float, int, int, int, int]]]:
    from BondFlow.data.link_utils import get_valid_links
    from eval_bonds import (
        _compute_chain_terminals,
        _load_structure_and_feats,
        auc_from_scores,
        compute_prf_per_type,
        compute_scores_for_pairs,
        read_bonds_txt,
    )

    seq, coords, pdb_idx = _load_structure_and_feats(struct_path)
    L = len(seq)
    head_mask = torch.zeros(L, dtype=torch.bool)
    tail_mask = torch.zeros(L, dtype=torch.bool)
    head_mask[0] = True
    tail_mask[-1] = True

    pred_pairs_all = read_bonds_txt(bonds_path, pdb_idx)
    ones = torch.ones((L, L), dtype=torch.float32)
    ones.fill_diagonal_(0.0)
    reports_all = get_valid_links(
        seq,
        coords,
        ones,
        link_csv,
        head_mask=head_mask,
        tail_mask=tail_mask,
        include_invalid=True,
    )

    reports = []
    for rep in reports_all:
        i = int(rep["i"])
        j = int(rep["j"])
        if _pair_in_bond_scope(i, j, pdb_idx, ligand_chain=ligand_chain, bond_scope=bond_scope):
            reports.append(rep)

    pred_pairs = set()
    for i, j in pred_pairs_all:
        if _pair_in_bond_scope(int(i), int(j), pdb_idx, ligand_chain=ligand_chain, bond_scope=bond_scope):
            pred_pairs.add((int(i), int(j)))

    terminals = _compute_chain_terminals(pdb_idx)
    score_map = compute_scores_for_pairs(reports)
    scores_all = [v[0] for v in score_map.values()]
    labels_all = [v[1] for v in score_map.values()]
    auc = auc_from_scores(scores_all, labels_all) if score_map else float("nan")

    prf = compute_prf_per_type(reports, pred_pairs, terminals, pdb_idx)
    prec_all, rec_all, f1_all, tp_all, fp_all, fn_all, tn_all = prf.get(
        "all", (0.0, 0.0, 0.0, 0, 0, 0, 0)
    )
    struct_name = os.path.splitext(os.path.basename(struct_path))[0]
    summary = {
        "Structure": struct_name,
        "TP": tp_all,
        "FP": fp_all,
        "FN": fn_all,
        "TN": tn_all,
        "AUC": auc,
        "Precision_all": prec_all,
        "Recall_all": rec_all,
        "F1_all": f1_all,
    }
    return summary, prf


def _infer_bonds_scoped_from_structure(
    struct_path: str,
    link_csv: str,
    ligand_chain: str,
    bond_scope: str,
) -> Dict[str, Tuple[float, float, float, int, int, int, int]]:
    """
    当设计缺少 bonds*.txt 时，用结构几何 + link.csv 规则推断“结构中是否存在”允许的键类型。

    实现方式：把所有 residue-pair 都当作候选（bond_mat=ones），然后用 get_valid_links 计算每对 (i,j)
    的 is_valid 标签；由于没有 pred_pairs，这里把 pred_pairs 设为空集合，让 compute_prf_per_type 的
    FN 对应于“结构里存在但未被 bonds*.txt 预测出来”的有效键数量。
    """
    from BondFlow.data.link_utils import get_valid_links
    from eval_bonds import (
        _compute_chain_terminals,
        _load_structure_and_feats,
        compute_prf_per_type,
    )

    seq, coords, pdb_idx = _load_structure_and_feats(struct_path)
    L = len(seq)
    head_mask = torch.zeros(L, dtype=torch.bool)
    tail_mask = torch.zeros(L, dtype=torch.bool)
    head_mask[0] = True
    tail_mask[-1] = True

    ones = torch.ones((L, L), dtype=torch.float32)
    ones.fill_diagonal_(0.0)
    reports_all = get_valid_links(
        seq,
        coords,
        ones,
        link_csv,
        head_mask=head_mask,
        tail_mask=tail_mask,
        include_invalid=True,
    )

    reports = []
    for rep in reports_all:
        i = int(rep["i"])
        j = int(rep["j"])
        if _pair_in_bond_scope(i, j, pdb_idx, ligand_chain=ligand_chain, bond_scope=bond_scope):
            reports.append(rep)

    pred_pairs: set[tuple[int, int]] = set()
    terminals = _compute_chain_terminals(pdb_idx)
    prf = compute_prf_per_type(reports, pred_pairs, terminals, pdb_idx)
    return prf


def _match_designs_for_benchmark(
    benchmark_stem: str,
    design_root: str,
    all_design_files: List[str],
) -> List[str]:
    root = Path(design_root)
    subdir = root / benchmark_stem
    subdir_prefix = str(subdir) + os.sep
    from_subdir = [f for f in all_design_files if f.startswith(subdir_prefix)]
    if from_subdir:
        return sorted(from_subdir)

    prefix_hits = [
        f for f in all_design_files if os.path.splitext(os.path.basename(f))[0].startswith(benchmark_stem)
    ]
    if prefix_hits:
        return sorted(prefix_hits)

    contains_hits = []
    for f in all_design_files:
        rel = os.path.relpath(f, design_root)
        stem = os.path.splitext(os.path.basename(f))[0]
        if benchmark_stem in rel or benchmark_stem in stem:
            contains_hits.append(f)
    return sorted(contains_hits)


def _evaluate_design_task(task: Dict) -> Dict:
    benchmark_name = task["benchmark_name"]
    benchmark_ligand_pdb = task["benchmark_ligand_pdb"]
    design_path = task["design_path"]
    tmp_dir = task["tmp_dir"]
    usalign_path = task["usalign_path"]
    link_csv = task["link_csv"]
    prefer_dssp = bool(task["prefer_dssp"])
    compute_energy = bool(task["compute_energy"])
    energy_relax = bool(task["energy_relax"])
    bond_scope = str(task["bond_scope"])
    benchmark_ref_dg = task.get("benchmark_ref_dg", float("nan"))
    benchmark_ref_energy_err = task.get("benchmark_ref_energy_err", "")
    eval_source = str(task.get("eval_source", "bondflow")).strip().lower()
    extra_res_fa_paths = [p for p in task.get("extra_res_fa_paths", []) if isinstance(p, str) and p]
    task_types = set(task.get("task_types", {"all"}))
    do_all = "all" in task_types
    do_tm = do_all or ("tm" in task_types)
    do_bond = do_all or ("bond" in task_types)
    do_energy = (do_all or ("energy" in task_types)) and compute_energy
    do_comp = do_all or ("composition" in task_types)

    design_name = os.path.splitext(os.path.basename(design_path))[0]
    out = {
        "Benchmark": benchmark_name,
        "Design_Path": design_path,
        "Design_Name": design_name,
        "Eval_Source": eval_source,
    }

    try:
        ligand_chain, ligand_len = _pick_ligand_chain(design_path, eval_source=eval_source)
        if ligand_chain is None:
            out["Error"] = "No amino-acid chain found"
            return out
        out["Ligand_Chain"] = ligand_chain
        out["Ligand_Length"] = ligand_len

        ligand_pdb = os.path.join(tmp_dir, f"{_safe_name(design_path)}__ligand.pdb")
        os.makedirs(tmp_dir, exist_ok=True)
        if not _extract_chain_to_pdb(design_path, ligand_chain, ligand_pdb):
            out["Error"] = f"Failed to extract ligand chain {ligand_chain}"
            return out
        out["Ligand_PDB"] = ligand_pdb

        if do_tm:
            _, _, _, tm1, tm2 = run_usalign(ligand_pdb, benchmark_ligand_pdb, usalign_path=usalign_path)
            out["TMscore_DesignNorm"] = tm1
            out["TMscore_RefNorm"] = tm2
            out["TMscore_Mean"] = np.nanmean([tm1, tm2]) if (tm1 is not None or tm2 is not None) else float("nan")

        if do_comp:
            aa_counter, ss_counter = _extract_chain_aa_ss_counts(design_path, ligand_chain, prefer_dssp=prefer_dssp)
            aa_freq = _to_freq(aa_counter, AA20)
            ss_freq = _to_freq(ss_counter, SS3)
            for aa in AA20:
                out[f"AA_Count_{aa}"] = int(aa_counter.get(aa, 0))
                out[f"AA_Freq_{aa}"] = float(aa_freq[aa])
            for s in SS3:
                out[f"SS_Count_{s}"] = int(ss_counter.get(s, 0))
                out[f"SS_Freq_{s}"] = float(ss_freq[s])

        bonds_path = _find_bonds_txt_for_structure(design_path)

        if do_bond:
            out["Bonds_Txt"] = bonds_path if bonds_path else ""
            out["Bond_Scope"] = bond_scope
            if bonds_path:
                try:
                    summary, prf = _eval_bonds_scoped(
                        struct_path=design_path,
                        bonds_path=bonds_path,
                        link_csv=link_csv,
                        ligand_chain=ligand_chain,
                        bond_scope=bond_scope,
                    )
                    out["Bond_TP"] = int(summary.get("TP", 0))
                    out["Bond_FP"] = int(summary.get("FP", 0))
                    out["Bond_FN"] = int(summary.get("FN", 0))
                    out["Bond_TN"] = int(summary.get("TN", 0))
                    out["Bond_AUC"] = float(summary.get("AUC", np.nan))
                    out["Bond_Precision"] = float(summary.get("Precision_all", np.nan))
                    out["Bond_Recall"] = float(summary.get("Recall_all", np.nan))
                    out["Bond_F1"] = float(summary.get("F1_all", np.nan))
                    out["Cyclization_Success"] = 1 if int(summary.get("TP", 0)) > 0 else 0
                    for bt in BOND_TYPES:
                        vals = prf.get(bt, (0.0, 0.0, 0.0, 0, 0, 0, 0))
                        out[f"BondType_{bt}_TP"] = int(vals[3])
                        out[f"BondType_{bt}_F1"] = float(vals[2])
                        out[f"BondType_{bt}_Has"] = 1 if int(vals[3]) > 0 else 0
                except Exception as e:
                    out["Bond_TP"] = np.nan
                    out["Bond_FP"] = np.nan
                    out["Bond_FN"] = np.nan
                    out["Bond_TN"] = np.nan
                    out["Bond_AUC"] = np.nan
                    out["Bond_Precision"] = np.nan
                    out["Bond_Recall"] = np.nan
                    out["Bond_F1"] = np.nan
                    out["Cyclization_Success"] = np.nan
                    out["Bond_Eval_Error"] = _compact_error_message(e)
                    for bt in BOND_TYPES:
                        out[f"BondType_{bt}_TP"] = np.nan
                        out[f"BondType_{bt}_F1"] = np.nan
                        out[f"BondType_{bt}_Has"] = np.nan
            else:
                # No bonds*.txt: infer cyclization presence directly from structure geometry using link.csv rules.
                out["Bond_TP"] = np.nan
                out["Bond_FP"] = np.nan
                out["Bond_FN"] = np.nan
                out["Bond_TN"] = np.nan
                out["Bond_AUC"] = np.nan
                out["Bond_Precision"] = np.nan
                out["Bond_Recall"] = np.nan
                out["Bond_F1"] = np.nan
                out["Bond_Eval_Error"] = "bonds_txt_missing; inferred from structure"
                try:
                    prf = _infer_bonds_scoped_from_structure(
                        struct_path=design_path,
                        link_csv=link_csv,
                        ligand_chain=ligand_chain,
                        bond_scope=bond_scope,
                    )
                    any_has = False
                    for bt in BOND_TYPES:
                        vals = prf.get(bt, (0.0, 0.0, 0.0, 0, 0, 0, 0))
                        # compute_prf_per_type returns (prec, rec, f1, tp, fp, fn, tn); with pred_pairs empty, fn>0 means structure contains that link type.
                        fn_cnt = int(vals[5])
                        has = 1 if fn_cnt > 0 else 0
                        out[f"BondType_{bt}_TP"] = 0
                        out[f"BondType_{bt}_F1"] = np.nan
                        out[f"BondType_{bt}_Has"] = has
                        any_has = any_has or (has == 1)
                    out["Cyclization_Success"] = 1 if any_has else 0
                except Exception as e:
                    out["Cyclization_Success"] = np.nan
                    out["Bond_Eval_Error"] = _compact_error_message(e)
                    for bt in BOND_TYPES:
                        out[f"BondType_{bt}_TP"] = np.nan
                        out[f"BondType_{bt}_F1"] = np.nan
                        out[f"BondType_{bt}_Has"] = np.nan

        if do_energy:
            dg, energy_err = _compute_binding_energy_for_ligand_chain(
                struct_path=design_path,
                ligand_chain_id=ligand_chain,
                bonds_path=bonds_path,
                relax=energy_relax,
                extra_res_fa_paths=extra_res_fa_paths,
            )
            out["Binding_Energy_dG"] = dg
            out["Binding_Energy_Error"] = energy_err
            out["Benchmark_Binding_Energy_dG"] = benchmark_ref_dg
            out["Benchmark_Binding_Energy_Error"] = benchmark_ref_energy_err
            if np.isfinite(dg) and np.isfinite(benchmark_ref_dg):
                out["Binding_Energy_Delta_vs_Ref"] = float(dg - benchmark_ref_dg)
                out["Binding_Energy_Better_Than_Ref"] = 1 if float(dg) < float(benchmark_ref_dg) else 0
            else:
                out["Binding_Energy_Delta_vs_Ref"] = np.nan
                out["Binding_Energy_Better_Than_Ref"] = np.nan
        elif do_all and compute_energy:
            out["Binding_Energy_dG"] = np.nan
            out["Binding_Energy_Error"] = ""
            out["Benchmark_Binding_Energy_dG"] = benchmark_ref_dg
            out["Benchmark_Binding_Energy_Error"] = benchmark_ref_energy_err
            out["Binding_Energy_Delta_vs_Ref"] = np.nan
            out["Binding_Energy_Better_Than_Ref"] = np.nan
        elif do_all and not compute_energy:
            out["Binding_Energy_dG"] = np.nan
            out["Binding_Energy_Error"] = ""
            out["Benchmark_Binding_Energy_dG"] = np.nan
            out["Benchmark_Binding_Energy_Error"] = ""
            out["Binding_Energy_Delta_vs_Ref"] = np.nan
            out["Binding_Energy_Better_Than_Ref"] = np.nan

        return out
    except Exception as e:
        out["Error"] = str(e)
        return out


def _compute_pairwise_tm_for_benchmark(
    benchmark_name: str,
    ligand_paths: List[str],
    usalign_path: str,
    workers: int,
) -> Tuple[pd.DataFrame, float]:
    if len(ligand_paths) < 2:
        return pd.DataFrame(), float("nan")

    pairs = list(itertools.combinations(sorted(ligand_paths), 2))
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(run_usalign, a, b, usalign_path): (a, b) for a, b in pairs}
        for fut in as_completed(futures):
            a, b = futures[fut]
            _, _, _, tm1, tm2 = fut.result()
            tm_mean = np.nanmean([tm1, tm2]) if (tm1 is not None or tm2 is not None) else float("nan")
            rows.append(
                {
                    "Benchmark": benchmark_name,
                    "Ligand_A": a,
                    "Ligand_B": b,
                    "TMscore_A_to_B": tm1,
                    "TMscore_B_to_A": tm2,
                    "TMscore_Mean": tm_mean,
                }
            )

    pair_df = pd.DataFrame(rows)
    avg_pair_tm = float(pair_df["TMscore_Mean"].mean()) if not pair_df.empty else float("nan")
    diversity = avg_pair_tm - 1.0 if not np.isnan(avg_pair_tm) else float("nan")
    return pair_df, diversity


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark complexes vs BondFlow designs at design-level, benchmark-level, and global-level."
    )
    parser.add_argument("--benchmark_dir", required=True, help="Benchmark structures directory")
    parser.add_argument("--design_dir", required=True, help="Design structures root directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--eval_source",
        type=str,
        default="bondflow",
        choices=sorted(EVAL_SOURCES),
        help="Evaluation source mode: bondflow (default), funcbind (UNL ligand + optional params generation), afcycdesign, or apcyc.",
    )
    parser.add_argument(
        "--funcbind_sdf_dir",
        type=str,
        default="/home/xjt/funcbind/exps/benchmark_diverse4_test/runs/1sfi/samples/target_00/ligands_sdf",
        help="funcbind mode: directory containing per-benchmark ligand sdf/mol2 files (e.g. 1sfi.sdf).",
    )
    parser.add_argument(
        "--molfile_to_params_script",
        type=str,
        default=None,
        help="Path to Rosetta molfile_to_params.py (required only if funcbind mode needs UNL.params auto-generation).",
    )
    parser.add_argument(
        "--funcbind_paramsgen_env",
        type=str,
        default="paramsgen",
        help="Conda environment used by funcbind mode to generate UNL.params via rdkit_to_params fallback.",
    )
    parser.add_argument("--usalign_path", default="USalign", help="USalign/TMalign executable path")
    parser.add_argument(
        "--link_csv",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "link.csv")),
        help="BondFlow link.csv path used by eval_bonds",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Process count for design-level evaluation")
    parser.add_argument("--pair_workers", type=int, default=8, help="Process count for pairwise design TMscore")
    parser.add_argument(
        "--design_subdir",
        type=str,
        default=None,
        help="Only evaluate design files under this subdirectory name (e.g., post_refine)",
    )
    parser.add_argument(
        "--bond_scope",
        type=str,
        default="ligand_internal",
        choices=["all", "ligand_internal", "ligand_related"],
        help="Bond analysis scope: all bonds, ligand internal only, or bonds involving ligand chain",
    )
    parser.add_argument(
        "--reuse_existing_design_metrics",
        action="store_true",
        help="Reuse existing design_level_metrics.csv and only recompute selected/missing tasks",
    )
    parser.add_argument(
        "--rerun_benchmarks",
        type=str,
        default=None,
        help="Comma-separated benchmark names to force recomputation (e.g., 4k1e,6n7q)",
    )
    parser.add_argument(
        "--rerun_task_types",
        type=str,
        default="all",
        help="Comma-separated task types to recompute: all,tm,bond,energy,composition",
    )
    parser.add_argument("--prefer_dssp", action="store_true", help="Use DSSP first for secondary structure")
    parser.add_argument("--compute_energy", action="store_true", help="Compute PyRosetta binding dG")
    parser.add_argument(
        "--energy_relax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run FastRelax before energy (default: enabled; use --no-energy_relax to disable)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("benchmark_eval")
    eval_source = str(args.eval_source).strip().lower()
    if eval_source == "afcycdesign":
        logger.warning("eval_source=afcycdesign is currently a placeholder; behavior falls back to bondflow ligand-selection logic.")
    if eval_source == "apcyc":
        logger.info("eval_source=apcyc uses shortest-chain ligand-selection logic.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_ligand_dir = output_dir / "tmp_ligands"
    tmp_ligand_dir.mkdir(parents=True, exist_ok=True)
    funcbind_params_dir = output_dir / "funcbind_params"
    funcbind_params_dir.mkdir(parents=True, exist_ok=True)

    benchmark_files = _collect_structure_files(args.benchmark_dir)
    design_files = _collect_structure_files(args.design_dir)
    if args.design_subdir:
        design_files = _filter_design_files_by_subdir(
            design_files=design_files,
            design_root=args.design_dir,
            subdir_keyword=args.design_subdir,
        )
    if not benchmark_files:
        raise ValueError(f"No structure files found in benchmark_dir: {args.benchmark_dir}")
    if not design_files:
        raise ValueError(
            f"No structure files found in design_dir: {args.design_dir}"
            + (f" (after --design_subdir filter: {args.design_subdir})" if args.design_subdir else "")
        )

    logger.info("Found %d benchmark structures, %d design structures", len(benchmark_files), len(design_files))
    logger.info("Evaluation source mode: %s", eval_source)
    if args.design_subdir:
        logger.info("Design file filter enabled: subdir keyword = %s", args.design_subdir)
    rerun_benchmarks = _parse_csv_list_arg(args.rerun_benchmarks)
    rerun_task_types = _parse_task_types_arg(args.rerun_task_types)
    if "energy" in rerun_task_types and not args.compute_energy:
        logger.warning("--rerun_task_types includes energy but --compute_energy was not set; auto-enabling --compute_energy.")
        args.compute_energy = True
    if rerun_benchmarks:
        logger.info("Force rerun benchmarks: %s", ",".join(sorted(rerun_benchmarks)))
    elif rerun_task_types != {"all"}:
        logger.info("No --rerun_benchmarks specified; rerun_task_types will apply to all mapped benchmarks.")
    logger.info("Rerun task types: %s", ",".join(sorted(rerun_task_types)))

    benchmark_meta = {}
    tasks = []
    mapping_rows = []

    for bpath in benchmark_files:
        bname = os.path.splitext(os.path.basename(bpath))[0]
        try:
            b_chain, b_len = _pick_ligand_chain(bpath, eval_source=eval_source)
            if b_chain is None:
                logger.warning("Skip benchmark %s: no amino-acid chain", bname)
                continue
            extra_res_fa_paths = []
            if eval_source == "funcbind":
                unl_params = _prepare_funcbind_unl_params(
                    benchmark_name=bname,
                    funcbind_sdf_dir=args.funcbind_sdf_dir,
                    params_output_dir=funcbind_params_dir,
                    molfile_to_params_script=args.molfile_to_params_script,
                    paramsgen_env=args.funcbind_paramsgen_env,
                    logger=logger,
                )
                if unl_params:
                    extra_res_fa_paths.append(unl_params)
            b_ligand_pdb = str(tmp_ligand_dir / f"benchmark__{_safe_name(bname)}__ligand.pdb")
            ok = _extract_chain_to_pdb(bpath, b_chain, b_ligand_pdb)
            if not ok:
                logger.warning("Skip benchmark %s: failed to extract chain %s", bname, b_chain)
                continue
            b_aa, b_ss = _extract_chain_aa_ss_counts(bpath, b_chain, prefer_dssp=args.prefer_dssp)
            b_bonds_path = _find_bonds_txt_for_structure(bpath)
            if args.compute_energy:
                b_dg, b_energy_err = _compute_binding_energy_for_ligand_chain(
                    struct_path=bpath,
                    ligand_chain_id=b_chain,
                    bonds_path=b_bonds_path,
                    relax=args.energy_relax,
                    extra_res_fa_paths=extra_res_fa_paths,
                )
            else:
                b_dg, b_energy_err = float("nan"), ""

            benchmark_meta[bname] = {
                "Eval_Source": eval_source,
                "Benchmark_Path": bpath,
                "Benchmark_Ligand_Chain": b_chain,
                "Benchmark_Ligand_Length": b_len,
                "Benchmark_Ligand_PDB": b_ligand_pdb,
                "Benchmark_AA_Counter": b_aa,
                "Benchmark_SS_Counter": b_ss,
                "Benchmark_Bonds_Txt": b_bonds_path if b_bonds_path else "",
                "Benchmark_Binding_Energy_dG": b_dg,
                "Benchmark_Binding_Energy_Error": b_energy_err,
                "Extra_Res_Fa_Paths": extra_res_fa_paths,
            }
        except Exception as e:
            logger.warning("Skip benchmark %s due to setup error: %s", bname, e)
            continue

        matched_designs = _match_designs_for_benchmark(bname, args.design_dir, design_files)
        logger.info("Benchmark %s matched %d design files", bname, len(matched_designs))
        for dpath in matched_designs:
            mapping_rows.append({"Benchmark": bname, "Benchmark_Path": bpath, "Design_Path": dpath})
            tasks.append(
                {
                    "eval_source": eval_source,
                    "benchmark_name": bname,
                    "benchmark_ligand_pdb": b_ligand_pdb,
                    "design_path": dpath,
                    "tmp_dir": str(tmp_ligand_dir / bname),
                    "usalign_path": args.usalign_path,
                    "link_csv": args.link_csv,
                    "prefer_dssp": args.prefer_dssp,
                    "compute_energy": args.compute_energy,
                    "energy_relax": args.energy_relax,
                    "bond_scope": args.bond_scope,
                    "benchmark_ref_dg": benchmark_meta[bname].get("Benchmark_Binding_Energy_dG", float("nan")),
                    "benchmark_ref_energy_err": benchmark_meta[bname].get("Benchmark_Binding_Energy_Error", ""),
                    "extra_res_fa_paths": benchmark_meta[bname].get("Extra_Res_Fa_Paths", []),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapping_rows).to_csv(output_dir / "benchmark_design_mapping.csv", index=False)
    logger.info("Prepared %d benchmark-design evaluation tasks", len(tasks))
    if not tasks:
        logger.warning("No benchmark-design tasks were generated. Please check folder organization.")
        return

    design_csv = output_dir / "design_level_metrics.csv"
    existing_df, existing_keys = (None, set())
    if args.reuse_existing_design_metrics:
        existing_df, existing_keys = _load_existing_design_metrics(design_csv, logger)

    tasks_to_run = _select_tasks_to_run(
        tasks=tasks,
        reuse_existing=args.reuse_existing_design_metrics,
        existing_keys=existing_keys,
        rerun_benchmarks=rerun_benchmarks,
        rerun_task_types=rerun_task_types,
    )
    logger.info("Tasks to execute now: %d", len(tasks_to_run))

    results = []
    if tasks_to_run:
        with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
            futures = []
            for t in tasks_to_run:
                t2 = dict(t)
                key = _task_key_from_task(t2)
                if args.reuse_existing_design_metrics and key in existing_keys:
                    t2["task_types"] = rerun_task_types
                else:
                    # Missing rows need full compute to avoid creating sparse rows.
                    t2["task_types"] = {"all"}
                futures.append(ex.submit(_evaluate_design_task, t2))

            for idx, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                results.append(res)
                if idx % 20 == 0 or idx == len(futures):
                    logger.info("Progress: %d/%d designs evaluated", idx, len(futures))

    run_df = pd.DataFrame(results)
    design_df = _merge_results_with_existing(
        existing_df=existing_df,
        run_df=run_df,
        all_tasks=tasks,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    design_df.to_csv(design_csv, index=False)
    logger.info("Saved design-level metrics: %s", design_csv)

    # Pairwise TM / Ligand_Diversity only depends on TMscore pairwise results.
    # If user reruns only non-TM task types (e.g., bond/energy), reuse the existing
    # Ligand_Diversity_MeanPairTM_minus1 from the previous benchmark summary in output_dir.
    skip_pairwise_tm = ("tm" not in rerun_task_types) and ("all" not in rerun_task_types)
    existing_ligand_div_by_bench: Dict[str, float] = {}
    if skip_pairwise_tm:
        old_bench_csv = output_dir / "benchmark_level_summary.csv"
        if old_bench_csv.exists():
            try:
                old_bench_df = pd.read_csv(old_bench_csv)
                if "Benchmark" in old_bench_df.columns and "Ligand_Diversity_MeanPairTM_minus1" in old_bench_df.columns:
                    existing_ligand_div_by_bench = {
                        str(k): float(v) if pd.notna(v) else float("nan")
                        for k, v in zip(
                            old_bench_df["Benchmark"].astype(str).tolist(),
                            old_bench_df["Ligand_Diversity_MeanPairTM_minus1"].tolist(),
                        )
                    }
                else:
                    logger.warning("Old benchmark summary missing required columns; cannot reuse Ligand diversity.")
            except Exception as e:
                logger.warning("Failed to load old benchmark summary for Ligand diversity reuse: %s", e)

    pair_rows = []
    benchmark_rows = []
    for bname, group in design_df.groupby("Benchmark"):
        meta = benchmark_meta.get(bname, {})
        valid_group = group[group["Error"].isna()] if "Error" in group.columns else group
        n_total = len(group)
        n_valid = len(valid_group)

        ligand_tm_div = float("nan")
        if skip_pairwise_tm:
            ligand_tm_div = existing_ligand_div_by_bench.get(bname, float("nan"))
        else:
            ligand_paths = [
                p
                for p in valid_group.get("Ligand_PDB", pd.Series(dtype=str)).dropna().tolist()
                if os.path.exists(p)
            ]
            pair_df, ligand_tm_div = _compute_pairwise_tm_for_benchmark(
                benchmark_name=bname,
                ligand_paths=ligand_paths,
                usalign_path=args.usalign_path,
                workers=args.pair_workers,
            )
            if not pair_df.empty:
                pair_rows.extend(pair_df.to_dict(orient="records"))

        cycl_success_rate = float(valid_group["Cyclization_Success"].dropna().mean()) if "Cyclization_Success" in valid_group.columns else float("nan")
        bond_type_counts = {}
        for bt in BOND_TYPES:
            col = f"BondType_{bt}_Has"
            bond_type_counts[bt] = int(valid_group[col].fillna(0).sum()) if col in valid_group.columns else 0
        richness = int(sum(1 for v in bond_type_counts.values() if v > 0))
        total_type_hits = float(sum(bond_type_counts.values()))
        if total_type_hits > 0:
            probs = [v / total_type_hits for v in bond_type_counts.values() if v > 0]
            cycl_shannon = float(-sum(p * math.log(p + 1e-12) for p in probs))
        else:
            cycl_shannon = float("nan")

        design_aa_counter = Counter()
        design_ss_counter = Counter()
        for _, row in valid_group.iterrows():
            for aa in AA20:
                design_aa_counter[aa] += int(row.get(f"AA_Count_{aa}", 0) or 0)
            for s in SS3:
                design_ss_counter[s] += int(row.get(f"SS_Count_{s}", 0) or 0)

        ref_aa_counter = meta.get("Benchmark_AA_Counter", Counter())
        ref_ss_counter = meta.get("Benchmark_SS_Counter", Counter())
        design_aa_freq = _to_freq(design_aa_counter, AA20)
        ref_aa_freq = _to_freq(ref_aa_counter, AA20)
        design_ss_freq = _to_freq(design_ss_counter, SS3)
        ref_ss_freq = _to_freq(ref_ss_counter, SS3)
        aa_div = _divergence_metrics(design_aa_freq, ref_aa_freq, AA20)
        ss_div = _divergence_metrics(design_ss_freq, ref_ss_freq, SS3)

        row = {
            "Benchmark": bname,
            "Benchmark_Path": meta.get("Benchmark_Path", ""),
            "Benchmark_Ligand_Chain": meta.get("Benchmark_Ligand_Chain", ""),
            "Benchmark_Ligand_Length": meta.get("Benchmark_Ligand_Length", np.nan),
            "Benchmark_Binding_Energy_dG": meta.get("Benchmark_Binding_Energy_dG", np.nan),
            "Benchmark_Binding_Energy_Error": meta.get("Benchmark_Binding_Energy_Error", ""),
            "Num_Designs_Total": n_total,
            "Num_Designs_Valid": n_valid,
            "Cyclization_Success_Rate": cycl_success_rate,
            "Cyclization_Type_Richness": richness,
            "Cyclization_Type_Shannon": cycl_shannon,
            "Binding_Energy_Mean": float(valid_group["Binding_Energy_dG"].dropna().mean()) if "Binding_Energy_dG" in valid_group.columns else float("nan"),
            "Binding_Energy_Median": float(valid_group["Binding_Energy_dG"].dropna().median()) if "Binding_Energy_dG" in valid_group.columns else float("nan"),
            "Binding_Energy_Delta_vs_Ref_Mean": float(valid_group["Binding_Energy_Delta_vs_Ref"].dropna().mean()) if "Binding_Energy_Delta_vs_Ref" in valid_group.columns else float("nan"),
            "Binding_Energy_Delta_vs_Ref_Median": float(valid_group["Binding_Energy_Delta_vs_Ref"].dropna().median()) if "Binding_Energy_Delta_vs_Ref" in valid_group.columns else float("nan"),
            "Binding_Energy_Better_Than_Ref_Rate": float(valid_group["Binding_Energy_Better_Than_Ref"].dropna().mean()) if "Binding_Energy_Better_Than_Ref" in valid_group.columns else float("nan"),
            "TMscore_to_Ref_Mean": float(valid_group["TMscore_Mean"].dropna().mean()) if "TMscore_Mean" in valid_group.columns else float("nan"),
            "TMscore_to_Ref_Median": float(valid_group["TMscore_Mean"].dropna().median()) if "TMscore_Mean" in valid_group.columns else float("nan"),
            "Ligand_Diversity_MeanPairTM_minus1": ligand_tm_div,
            "Bond_AUC_Mean": float(valid_group["Bond_AUC"].dropna().mean()) if "Bond_AUC" in valid_group.columns else float("nan"),
            "Bond_F1_Mean": float(valid_group["Bond_F1"].dropna().mean()) if "Bond_F1" in valid_group.columns else float("nan"),
        }

        for bt in BOND_TYPES:
            row[f"Cyclization_Type_Count_{bt}"] = bond_type_counts[bt]
        for s in SS3:
            row[f"Design_SS_Freq_{s}"] = design_ss_freq[s]
            row[f"Ref_SS_Freq_{s}"] = ref_ss_freq[s]
            row[f"SS_Freq_Diff_{s}"] = design_ss_freq[s] - ref_ss_freq[s]
        for aa in AA20:
            row[f"Design_AA_Freq_{aa}"] = design_aa_freq[aa]
            row[f"Ref_AA_Freq_{aa}"] = ref_aa_freq[aa]
            row[f"AA_Freq_Diff_{aa}"] = design_aa_freq[aa] - ref_aa_freq[aa]
        for k, v in ss_div.items():
            row[f"SS_{k}"] = v
        for k, v in aa_div.items():
            row[f"AA_{k}"] = v

        benchmark_rows.append(row)

    benchmark_df = pd.DataFrame(benchmark_rows).sort_values("Benchmark")
    benchmark_csv = output_dir / "benchmark_level_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(benchmark_csv, index=False)
    logger.info("Saved benchmark-level summary: %s", benchmark_csv)

    if pair_rows:
        pair_csv = output_dir / "design_ligand_pairwise_tm.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pair_rows).to_csv(pair_csv, index=False)
        logger.info("Saved pairwise design-ligand TM table: %s", pair_csv)

    global_row = {
        "Num_Benchmarks": int(benchmark_df["Benchmark"].nunique()) if not benchmark_df.empty else 0,
        "Num_Designs_Total": int(len(design_df)),
        "Num_Designs_Valid": int(design_df["Error"].isna().sum()) if "Error" in design_df.columns else int(len(design_df)),
        "Global_Cyclization_Success_Rate": float(design_df["Cyclization_Success"].dropna().mean()) if "Cyclization_Success" in design_df.columns else float("nan"),
        "Global_Binding_Energy_Mean": float(design_df["Binding_Energy_dG"].dropna().mean()) if "Binding_Energy_dG" in design_df.columns else float("nan"),
        "Global_Binding_Energy_Delta_vs_Ref_Mean": float(design_df["Binding_Energy_Delta_vs_Ref"].dropna().mean()) if "Binding_Energy_Delta_vs_Ref" in design_df.columns else float("nan"),
        "Global_Binding_Energy_Better_Than_Ref_Rate": float(design_df["Binding_Energy_Better_Than_Ref"].dropna().mean()) if "Binding_Energy_Better_Than_Ref" in design_df.columns else float("nan"),
        "Global_TMscore_to_Ref_Mean": float(design_df["TMscore_Mean"].dropna().mean()) if "TMscore_Mean" in design_df.columns else float("nan"),
        "Global_Bond_AUC_Mean": float(design_df["Bond_AUC"].dropna().mean()) if "Bond_AUC" in design_df.columns else float("nan"),
        "Global_Bond_F1_Mean": float(design_df["Bond_F1"].dropna().mean()) if "Bond_F1" in design_df.columns else float("nan"),
        "Benchmark_Average_of_Ligand_Diversity": float(benchmark_df["Ligand_Diversity_MeanPairTM_minus1"].dropna().mean()) if not benchmark_df.empty else float("nan"),
    }
    global_csv = output_dir / "global_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([global_row]).to_csv(global_csv, index=False)
    logger.info("Saved global summary: %s", global_csv)
    logger.info("All done.")


if __name__ == "__main__":
    main()
