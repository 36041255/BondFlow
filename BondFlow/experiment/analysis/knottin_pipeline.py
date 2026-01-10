#!/usr/bin/env python3
"""
Complete pipeline for generating and evaluating knottin binders with disulfide bonds.

Pipeline steps:
1. Generate knottin binders using Sampler (conda apm_env)
2. Evaluate with PyRosetta (energy + SAP, conda analysis)
3. Extract target chain, run HighFold prediction, calculate scRMSD and PLDDT
4. Aggregate all metrics to CSV
5. Filter by thresholds and copy passed structures
"""
import sys
# 强制重定向 stdout 到无缓存模式
sys.stdout.reconfigure(line_buffering=True)
import os
import sys
import glob
import argparse
import subprocess
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import PDB
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Configure Python to use unbuffered output for real-time printing
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
sys.path.insert(0, PROJECT_ROOT)

# Import evaluation modules
from BondFlow.experiment.analysis.energy import batch_energy, _collect_pdb_like_files
from BondFlow.experiment.analysis.calc_sap import (
    calculate_sap_metrics,
    extract_chain,
    detect_disulfides_safe,
    worker_init,
)
from BondFlow.experiment.analysis.evaluate_knottins import (
    get_structure_info,
    get_best_plddt_and_structure,
)

# HighFold paths
COLABFOLD_ENV_PATH = os.path.join(PROJECT_ROOT, "HighFold2/localcolabfold/colabfold-conda")
COLABFOLD_BIN = os.path.join(COLABFOLD_ENV_PATH, "bin/colabfold_batch")
# Try to determine conda environment name
# Option 1: If COLABFOLD_ENV_PATH is a conda env, try to get its name
# Option 2: Use conda run -p <path> to use the environment by path
# Option 3: Use the environment's python directly
COLABFOLD_ENV_NAME = os.path.basename(COLABFOLD_ENV_PATH)  # e.g., "colabfold-conda"
# Alternative: use conda run -p for path-based activation
USE_CONDA_RUN = True  # Set to False to use direct path execution


def _get_parser(file_path):
    """Get appropriate parser (PDB or MMCIF) based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        return MMCIFParser(QUIET=True)
    else:
        return PDB.PDBParser(QUIET=True)


def extract_chain_a_pdb(input_pdb, output_pdb, chain_id="A"):
    """Extract specified chain from PDB/CIF file and save to new PDB file.
    
    For CIF files, uses auth_asym_id (what users expect) instead of label_asym_id
    (what BioPython uses by default).
    """
    ext = os.path.splitext(input_pdb)[1].lower()
    target_chain = None
    
    # For CIF files, we need to handle auth_asym_id vs label_asym_id
    if ext in [".cif", ".mmcif"]:
        try:
            # Parse CIF and get the mmcif_dict to access auth_asym_id
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('struct', input_pdb)
            mmcif_dict = parser._mmcif_dict
            model = structure[0]
            
            # Get auth_asym_id mapping (this is what users expect)
            if '_atom_site.auth_asym_id' in mmcif_dict:
                auth_asym_ids = mmcif_dict['_atom_site.auth_asym_id']
                label_asym_ids = mmcif_dict.get('_atom_site.label_asym_id', [])
                
                # Find which label_asym_id corresponds to the requested auth_asym_id
                # Create mapping: find atoms that belong to the requested auth_asym_id
                atom_indices = []
                for i, auth_id in enumerate(auth_asym_ids):
                    if str(auth_id).strip() == str(chain_id).strip():
                        atom_indices.append(i)
                
                if not atom_indices:
                    print(f"Warning: Chain {chain_id} (auth_asym_id) not found in {input_pdb}")
                    return False
                
                # Get the label_asym_id for the first matching atom
                first_idx = atom_indices[0]
                label_chain_id = label_asym_ids[first_idx] if first_idx < len(label_asym_ids) else None
                
                # Find chain by label_asym_id in structure
                if label_chain_id:
                    for chain in model:
                        if chain.id == label_chain_id:
                            target_chain = chain
                            break
                
                if target_chain is None:
                    print(f"Warning: Chain {chain_id} (auth_asym_id) found but corresponding label_asym_id chain not found in structure")
                    return False
                
                # Rename chain to use auth_asym_id for consistency
                target_chain.id = chain_id
            else:
                # Fallback: use standard parsing
                for chain in model:
                    if chain.id == chain_id:
                        target_chain = chain
                        break
                
                if target_chain is None:
                    print(f"Warning: Chain {chain_id} not found in {input_pdb}")
                    return False
        except Exception as e:
            print(f"Error extracting chain {chain_id} from CIF {input_pdb}: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        # For PDB files, use standard parsing
        parser = PDB.PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('struct', input_pdb)
            model = structure[0]
            
            # Find chain
            for chain in model:
                if chain.id == chain_id:
                    target_chain = chain
                    break
            
            if target_chain is None:
                print(f"Warning: Chain {chain_id} not found in {input_pdb}")
                return False
        except Exception as e:
            print(f"Error extracting chain {chain_id} from {input_pdb}: {e}")
            return False
    
    # Create new structure with only the target chain
    io = PDB.PDBIO()
    try:
        new_structure = PDB.Structure.Structure('knottin')
        new_model = PDB.Model.Model(0)
        new_model.add(target_chain)
        new_structure.add(new_model)
        
        # Save
        io.set_structure(new_structure)
        io.save(output_pdb)
        return True
    except Exception as e:
        print(f"Error saving extracted chain {chain_id} to {output_pdb}: {e}")
        return False


def get_sequence_from_pdb(pdb_path):
    """Extract amino acid sequence from PDB/CIF file."""
    parser = _get_parser(pdb_path)
    try:
        structure = parser.get_structure('struct', pdb_path)
        model = structure[0]
        chain = list(model)[0]  # Get first chain
        
        seq = ""
        for residue in chain.get_residues():
            if PDB.is_aa(residue, standard=True):
                seq += seq1(residue.get_resname())
        return seq
    except Exception as e:
        print(f"Error extracting sequence from {pdb_path}: {e}")
        return None


def calculate_sc_rmsd(ref_pdb, pred_pdb):
    """Calculate sidechain RMSD between reference and predicted structures."""
    ref_parser = _get_parser(ref_pdb)
    pred_parser = _get_parser(pred_pdb)
    
    try:
        ref_structure = ref_parser.get_structure('ref', ref_pdb)
        pred_structure = pred_parser.get_structure('pred', pred_pdb)
        
        ref_model = ref_structure[0]
        pred_model = pred_structure[0]
        
        ref_chain = list(ref_model)[0]
        pred_chain = list(pred_model)[0]
        
        ref_residues = [r for r in ref_chain.get_residues() if PDB.is_aa(r, standard=True)]
        pred_residues = [r for r in pred_chain.get_residues() if PDB.is_aa(r, standard=True)]
        
        if len(ref_residues) != len(pred_residues):
            print(f"Warning: Residue count mismatch {len(ref_residues)} vs {len(pred_residues)}")
            return None
        
        # Align by CA atoms first
        ref_ca = []
        pred_ca = []
        for r1, r2 in zip(ref_residues, pred_residues):
            if 'CA' in r1 and 'CA' in r2:
                ref_ca.append(r1['CA'])
                pred_ca.append(r2['CA'])
        
        if len(ref_ca) < 3:
            return None
        
        superimposer = PDB.Superimposer()
        superimposer.set_atoms(ref_ca, pred_ca)
        superimposer.apply(pred_model.get_atoms())
        
        # Calculate sidechain RMSD (all non-backbone atoms)
        sc_atoms_ref = []
        sc_atoms_pred = []
        backbone_atoms = {'N', 'CA', 'C', 'O', 'H', 'HA'}
        
        for r1, r2 in zip(ref_residues, pred_residues):
            for atom1 in r1:
                if atom1.name not in backbone_atoms:
                    if atom1.name in r2:
                        sc_atoms_ref.append(atom1)
                        sc_atoms_pred.append(r2[atom1.name])
        
        if len(sc_atoms_ref) == 0:
            return None
        
        # Calculate RMSD
        coords_ref = np.array([atom.coord for atom in sc_atoms_ref])
        coords_pred = np.array([atom.coord for atom in sc_atoms_pred])
        
        rmsd = np.sqrt(np.mean(np.sum((coords_ref - coords_pred)**2, axis=1)))
        return float(rmsd)
        
    except Exception as e:
        print(f"Error calculating scRMSD: {e}")
        return None


def run_highfold_for_sequence(seq, ss_pairs, is_cyclic, output_dir, gpu_id=0, msa_mode="single_sequence", msa_threads=None, max_msa=None):
    """
    Run HighFold prediction for a sequence using conda environment.
    
    Args:
        seq: Amino acid sequence
        ss_pairs: Disulfide bond pairs (1-based indices)
        is_cyclic: Whether the sequence is cyclic
        output_dir: Output directory for results
        gpu_id: GPU ID to use
        msa_mode: MSA mode for colabfold (default: "single_sequence")
        msa_threads: Number of threads for MSA search (MMseqs2) per job
        max_msa: MSA depth in format "max-seq:max-extra-seq" (e.g., "512:5120")
    """
    import uuid
    import subprocess
    
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(output_dir, f"job_{job_id}")
    os.makedirs(job_dir, exist_ok=True)
    
    fasta_path = os.path.join(job_dir, "input.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">target\n{seq}\n")
    
    # Build command with conda environment activation
    # Use conda run with environment path (-p) which is more reliable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Add the conda environment's bin to PATH as backup
    env["PATH"] = f"{COLABFOLD_ENV_PATH}/bin:{env.get('PATH', '')}"
    
    # Use conda run -p to activate environment by path (most reliable)
    cmd = [
        "conda", "run", "-p", COLABFOLD_ENV_PATH,
        "--no-capture-output",
        COLABFOLD_BIN,
        "--msa-mode", msa_mode,
    ]
    
    # Add MSA depth if specified
    if max_msa is not None and msa_mode != "single_sequence":
        cmd.extend(["--max-msa", max_msa])
    
    # Force local MSA search to avoid timeout with remote server
    if msa_mode != "single_sequence":
        # Disable remote MSA server, force local MMseqs2
        env["COLABFOLD_MSA_SERVER"] = ""
        env["MMSEQS_SERVER"] = ""
        # Ensure local MMseqs2 is in PATH
        mmseqs_dir = os.path.join(PROJECT_ROOT, "HighFold2/localcolabfold/colabfold-conda/bin")
        if os.path.exists(mmseqs_dir):
            env["PATH"] = mmseqs_dir + ":" + env.get("PATH", "")
        # Disable server fallback (force local only)
        env["COLABFOLD_NO_SERVER"] = "1"
    
    # Set MSA search threads if specified
    if msa_threads is not None and msa_mode != "single_sequence":
        # MMseqs2 uses OMP_NUM_THREADS for parallelization
        env["OMP_NUM_THREADS"] = str(msa_threads)
        # Some versions also use MMSEQS_NUM_THREADS
        env["MMSEQS_NUM_THREADS"] = str(msa_threads)
    
    if ss_pairs:
        cmd.append("--disulfide-bond-pairs")
        cmd.extend([str(x) for x in ss_pairs])
    
    if is_cyclic:
        cmd.append("--head-to-tail")
    else:
        cmd.append("--no-head-to-tail")
    
    cmd.append(fasta_path)
    cmd.append(job_dir)
    
    try:
        # Use Popen for real-time output streaming
        print(f"  [GPU {gpu_id}] Starting HighFold prediction for {job_id}...", flush=True)
        process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # Merge stderr to stdout
            universal_newlines=True,
            bufsize=1,  # Line buffered
        )
        
        # Real-time output streaming
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            print(f"  [GPU {gpu_id}] {line}", flush=True)
            output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            error_msg = "\n".join(output_lines[-20:])  # Last 20 lines for error context
            print(f"  [GPU {gpu_id}] Error running HighFold for job {job_id} (return code {return_code})", flush=True)
            print(f"  [GPU {gpu_id}] Last output: {error_msg}", flush=True)
            return None, None
            
    except subprocess.TimeoutExpired:
        print(f"  [GPU {gpu_id}] HighFold prediction timeout for job {job_id}", flush=True)
        return None, None
    except Exception as e:
        print(f"  [GPU {gpu_id}] Exception running HighFold for job {job_id}: {e}", flush=True)
        return None, None
    
    plddt, best_pdb = get_best_plddt_and_structure(job_dir)
    return plddt, best_pdb


def evaluate_single_structure_worker(args):
    """Worker function for parallel HighFold evaluation."""
    # Ensure unbuffered output in worker process
    import sys
    sys.stdout = sys.__stdout__  # Use original stdout
    sys.stderr = sys.__stderr__  # Use original stderr
    
    relaxed_pdb, output_dir, gpu_id, chain_id, msa_mode, msa_threads, max_msa = args
    pdb_name = os.path.basename(relaxed_pdb).replace("_relaxed.pdb", "").replace(".pdb", "")
    print(f"[GPU {gpu_id}] Processing {pdb_name}...", flush=True)
    try:
        result = evaluate_single_structure(
            pdb_path=relaxed_pdb,
            relaxed_pdb_path=relaxed_pdb,
            output_dir=output_dir,
            gpu_id=gpu_id,
            chain_id=chain_id,
            msa_mode=msa_mode,
            msa_threads=msa_threads,
            max_msa=max_msa,
        )
        print(f"[GPU {gpu_id}] Completed {pdb_name}", flush=True)
        return result
    except Exception as e:
        print(f"[GPU {gpu_id}] Error processing {pdb_name}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"PDB": pdb_name, "Error": str(e)}


def run_highfold_parallel(
    relaxed_pdb_files,
    output_dir,
    gpu_ids,
    jobs_per_gpu=1,
    chain_id="A",
    msa_mode="single_sequence",
    msa_threads=None,
    max_msa=None
):
    """Run HighFold predictions in parallel across multiple GPUs."""
    # Setup GPU queue
    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()
    for gid in gpu_ids:
        for _ in range(jobs_per_gpu):
            gpu_queue.put(int(gid.strip()))
    
    # Prepare tasks with GPU assignment
    tasks = []
    for relaxed_pdb in relaxed_pdb_files:
        gpu_id = gpu_queue.get()
        tasks.append((relaxed_pdb, output_dir, gpu_id, chain_id, msa_mode, msa_threads, max_msa))
        gpu_queue.put(gpu_id)  # Return GPU to queue for reuse
    
    # Actually, we'll use a simpler approach: round-robin GPU assignment
    num_workers = len(gpu_ids) * jobs_per_gpu
    tasks_with_gpu = []
    for i, relaxed_pdb in enumerate(relaxed_pdb_files):
        gpu_idx = i % len(gpu_ids)
        gpu_id = int(gpu_ids[gpu_idx])
        tasks_with_gpu.append((relaxed_pdb, output_dir, gpu_id, chain_id, msa_mode, msa_threads, max_msa))
    
    print(f"Distributing {len(tasks_with_gpu)} tasks across {len(gpu_ids)} GPUs...")
    
    # Run in parallel
    results = []
    print(f"Starting {len(tasks_with_gpu)} HighFold predictions with {num_workers} workers...", flush=True)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(evaluate_single_structure_worker, task): task[0]
            for task in tasks_with_gpu
        }
        
        completed = 0
        for future in as_completed(futures):
            pdb_path = futures[future]
            completed += 1
            try:
                result = future.result()
                if result:
                    results.append(result)
                    pdb_name = os.path.basename(pdb_path).replace("_relaxed.pdb", "").replace(".pdb", "")
                    plddt = result.get('PLDDT', 'N/A')
                    sc_rmsd = result.get('scRMSD', 'N/A')
                    print(f"[{completed}/{len(tasks_with_gpu)}] ✓ {pdb_name}: PLDDT={plddt}, scRMSD={sc_rmsd}", flush=True)
                else:
                    pdb_name = os.path.basename(pdb_path).replace("_relaxed.pdb", "").replace(".pdb", "")
                    print(f"[{completed}/{len(tasks_with_gpu)}] ✗ {pdb_name}: Failed", flush=True)
            except Exception as e:
                pdb_name = os.path.basename(pdb_path).replace("_relaxed.pdb", "").replace(".pdb", "")
                print(f"[{completed}/{len(tasks_with_gpu)}] ✗ {pdb_name}: Exception - {e}", flush=True)
                import traceback
                traceback.print_exc()
    
    print(f"HighFold predictions complete: {len(results)}/{len(tasks_with_gpu)} successful", flush=True)
    return results


def evaluate_single_structure(
    pdb_path,
    relaxed_pdb_path,
    output_dir,
    gpu_id=0,
    chain_id="A",
    msa_mode="single_sequence",
    msa_threads=None,
    max_msa=None
):
    """Evaluate a single structure: HighFold prediction and scRMSD."""
    pdb_name = os.path.basename(pdb_path).replace("_relaxed.pdb", "").replace(".pdb", "")
    results = {"PDB": pdb_name}
    
    # 1. Extract target chain
    chain_dir = os.path.join(output_dir, f"chain{chain_id}_extracted")
    os.makedirs(chain_dir, exist_ok=True)
    chain_pdb = os.path.join(chain_dir, f"{pdb_name}_chain{chain_id}.pdb")
    if not extract_chain_a_pdb(relaxed_pdb_path, chain_pdb, chain_id):
        results["Error"] = f"Failed to extract chain {chain_id}"
        return results
    
    # 2. Get sequence and structure info
    seq = get_sequence_from_pdb(chain_pdb)
    if seq is None:
        results["Error"] = "Failed to extract sequence"
        return results
    results["Sequence"] = seq
    results["Seq_Length"] = len(seq)
    
    # Get structure info (disulfide bonds, cyclization)
    info = get_structure_info(chain_pdb)
    if info is None:
        results["Error"] = "Failed to get structure info"
        return results
    
    seq_info, ss_pairs, is_cyclic, ref_ca_atoms = info
    results["IsCyclic"] = is_cyclic
    results["NumSS"] = len(ss_pairs) // 2
    
    # 3. Run HighFold prediction (check if results already exist)
    highfold_dir = os.path.join(output_dir, "highfold", pdb_name)
    os.makedirs(highfold_dir, exist_ok=True)
    
    # Check if HighFold results already exist
    plddt, pred_pdb = get_best_plddt_and_structure(highfold_dir)
    
    if plddt is None or pred_pdb is None:
        # Results don't exist, run HighFold prediction
        plddt, pred_pdb = run_highfold_for_sequence(
            seq, ss_pairs, is_cyclic, highfold_dir, gpu_id, msa_mode, msa_threads, max_msa
        )
    
    if plddt is None or pred_pdb is None:
        results["PLDDT"] = None
        results["scRMSD"] = None
        results["HighFold_Error"] = "Prediction failed"
    else:
        results["PLDDT"] = float(plddt)
        
        # Calculate scRMSD (check if chain_pdb and pred_pdb exist)
        if os.path.exists(chain_pdb) and os.path.exists(pred_pdb):
            sc_rmsd = calculate_sc_rmsd(chain_pdb, pred_pdb)
            results["scRMSD"] = sc_rmsd if sc_rmsd is not None else None
        else:
            results["scRMSD"] = None
    
    return results


def main():
    print("enter main function")
    parser = argparse.ArgumentParser(
        description="Complete pipeline for knottin binder generation and evaluation"
    )
    
    # Generation (Step 1)
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate structures using Sampler (requires conda apm_env)"
    )
    parser.add_argument(
        "--gen_config", type=str,
        help="YAML config file for generation"
    )
    parser.add_argument(
        "--gen_output_dir", type=str,
        help="Output directory for generated structures"
    )
    
    # Evaluation (Steps 2-3)
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory containing PDB files (generated or existing)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--chain", default="A",
        help="Target chain ID (default: A)"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU ID for HighFold prediction (deprecated, use --gpus instead)"
    )
    parser.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated list of GPU IDs for parallel HighFold prediction (e.g. '0,1,2,3')"
    )
    parser.add_argument(
        "--jobs_per_gpu", type=int, default=1,
        help="Number of concurrent HighFold jobs per GPU (default: 1)"
    )
    parser.add_argument(
        "--n_cores", type=int, default=4,
        help="Number of CPU cores for parallel processing"
    )
    
    # HighFold MSA options
    parser.add_argument(
        "--msa-mode", default="single_sequence",
        choices=["single_sequence", "mmseqs2_uniref_env", "mmseqs2_uniref"],
        help="MSA mode: 'single_sequence' (no MSA, fast), 'mmseqs2_uniref_env' (with MSA, slower but more accurate), 'mmseqs2_uniref' (UniRef only)"
    )
    parser.add_argument(
        "--msa-threads", type=int, default=None,
        help="Number of threads for MSA search (MMseqs2) per job. If not set, uses all available CPU cores. Only applies when using MSA modes."
    )
    parser.add_argument(
        "--max-msa", type=str, default=None,
        help="MSA depth in format 'max-seq:max-extra-seq' (e.g., '512:5120'). Default is ColabFold default (typically '512:5120' for AlphaFold2). max-seq: sequence clusters, max-extra-seq: extra sequences. Only applies when using MSA modes."
    )
    
    # Energy calculation options
    parser.add_argument(
        "--relax", action="store_true", default=True,
        help="Run FastRelax before energy calculation"
    )
    parser.add_argument(
        "--link_config", type=str,
        help="Path to link.csv config file"
    )
    parser.add_argument(
        "--extract_dslf_fa13", action="store_true", default=False,
        help="Extract disulfide bond score (dslf_fa13) from Rosetta energy calculation"
    )
    
    # Filtering (Step 4-5)
    parser.add_argument(
        "--thresholds", type=str,
        help="JSON string with threshold values, e.g. '{\"Total_Energy\": 100, \"SAP_total\": 50, \"PLDDT\": 70, \"scRMSD\": 2.0}'"
    )
    parser.add_argument(
        "--passed_dir", type=str,
        help="Directory to copy passed structures (original)"
    )
    parser.add_argument(
        "--passed_relax_dir", type=str,
        help="Directory to copy passed relaxed structures"
    )
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    energy_dir = os.path.join(args.output_dir, "energy_results")
    relaxed_dir = os.path.join(energy_dir, "relaxed_structures")
    highfold_dir = os.path.join(args.output_dir, "highfold")
    chain_dir = os.path.join(args.output_dir, f"chain{args.chain}_extracted")
    os.makedirs(relaxed_dir, exist_ok=True)
    os.makedirs(highfold_dir, exist_ok=True)
    os.makedirs(chain_dir, exist_ok=True)
    
    # Step 1: Generate structures (if requested)
    print("decide to generate structures")
    if args.generate:
        if not args.gen_config or not args.gen_output_dir:
            print("Error: --gen_config and --gen_output_dir required for generation")
            sys.exit(1)
        
        print("=" * 80)
        print("Step 1: Generating knottin binders...")
        print("=" * 80)
        
        # Activate conda environment and run generation
        gen_script = os.path.join(PROJECT_ROOT, "BondFlow", "sample.py")
        cmd = [
            "conda", "run", "-n", "apm_env",
            "python", gen_script,
            "--cfg", args.gen_config,
        ]
        
        # Set output directory in config or via environment
        env = os.environ.copy()
        if args.gen_output_dir:
            env["OUTPUT_DIR"] = args.gen_output_dir
        
        print(f"Running: {' '.join(cmd)}", flush=True)
        try:
            # Run generation with real-time output
            process = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(f"  [GEN] {line.rstrip()}", flush=True)
            
            return_code = process.wait()
            if return_code != 0:
                print(f"Error during generation (return code {return_code})", flush=True)
                sys.exit(1)
            
            print(f"Generation complete. Structures saved to {args.gen_output_dir}", flush=True)
            # Update input_dir to generated structures
            args.input_dir = args.gen_output_dir
        except subprocess.CalledProcessError as e:
            print(f"Error during generation: {e}", flush=True)
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error during generation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Step 2: Energy calculation and SAP (using energy.py)
    print("\n" + "=" * 80)
    print("Step 2: Calculating energy and SAP...")
    print("=" * 80)
    
    # Find structure files (PDB/CIF) - use the same function as energy.py
    # This function handles both PDB and CIF formats, converting CIF to PDB if needed
    pdb_files = _collect_pdb_like_files(args.input_dir, args.output_dir)
    if not pdb_files:
        print(f"Error: No structure files (PDB/CIF) found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(pdb_files)} structure files (PDB/CIF)", flush=True)
    
    # Auto-detect head-to-tail cyclization and check for LINK records
    print(f"\nDetecting head-to-tail cyclization for chain {args.chain}...", flush=True)
    has_cyclic_without_link = False
    
    for pdb_file in pdb_files:
        # Check if structure is cyclic using get_structure_info
        try:
            # Extract target chain for detection (same as in evaluate_single_structure)
            chain_dir = os.path.join(args.output_dir, f"chain{args.chain}_extracted")
            os.makedirs(chain_dir, exist_ok=True)
            pdb_name = os.path.basename(pdb_file).replace(".pdb", "")
            chain_pdb = os.path.join(chain_dir, f"{pdb_name}_chain{args.chain}_temp.pdb")
            
            # Extract target chain
            if extract_chain_a_pdb(pdb_file, chain_pdb, args.chain):
                # Check if cyclic
                info = get_structure_info(chain_pdb)
                if info is not None:
                    seq_info, ss_pairs, is_cyclic, ref_ca_atoms = info
                    if is_cyclic:
                        # Check if PDB file has LINK record for head-to-tail
                        has_link = False
                        try:
                            with open(pdb_file, 'r') as f:
                                for line in f:
                                    if line.startswith("LINK"):
                                        # Check if it's a C-N or N-C link
                                        a1 = line[12:16].strip().upper()
                                        a2 = line[42:46].strip().upper()
                                        # Also check chain ID in LINK record
                                        c1 = line[21].strip() if line[21].strip() else line[20].strip()
                                        c2 = line[51].strip() if line[51].strip() else line[50].strip()
                                        if (a1, a2) in [("C", "N"), ("N", "C")] and (c1 == args.chain or c2 == args.chain):
                                            has_link = True
                                            break
                        except Exception:
                            pass
                        
                        if not has_link:
                            print(f"  Detected cyclic structure without LINK record: {pdb_name} (chain {args.chain})", flush=True)
                            has_cyclic_without_link = True
                            break  # Found at least one, enable the options
                # Clean up temporary file
                try:
                    if os.path.exists(chain_pdb):
                        os.remove(chain_pdb)
                except Exception:
                    pass
        except Exception as e:
            # Skip if detection fails
            continue
    
    # Set parameters based on detection
    auto_head_tail_bond = has_cyclic_without_link
    link_constraints = has_cyclic_without_link
    
    if has_cyclic_without_link:
        print("  Enabling auto_head_tail_bond=True and link_constraints=True for cyclic structures", flush=True)
    else:
        print("  No cyclic structures without LINK records detected, using default settings", flush=True)
    
    # Check if energy calculation can be skipped
    energy_csv = os.path.join(energy_dir, "Energy_results.csv")
    skip_energy = False
    
    if os.path.exists(energy_csv):
        # Check if relaxed structures exist and match the count
        existing_relaxed = sorted(glob.glob(os.path.join(relaxed_dir, "*_relaxed.pdb")))
        if existing_relaxed:
            # Load existing results to check
            try:
                existing_energy_df = pd.read_csv(energy_csv)
                # Check if we have results for all input files
                pdb_basenames = {os.path.splitext(os.path.basename(f))[0] for f in pdb_files}
                energy_pdb_names = set(existing_energy_df["PDB"].astype(str).str.replace("_relaxed", "").str.replace(".pdb", ""))
                
                # Check if all input files have energy results
                if pdb_basenames.issubset(energy_pdb_names) and len(existing_relaxed) >= len(pdb_files):
                    skip_energy = True
                    print(f"\n✓ Energy calculation results already exist: {energy_csv}", flush=True)
                    print(f"  Found {len(existing_relaxed)} relaxed structures, {len(existing_energy_df)} energy results", flush=True)
                    print(f"  Skipping energy calculation step...", flush=True)
            except Exception as e:
                print(f"Warning: Could not verify existing energy results: {e}", flush=True)
                skip_energy = False
    
    if not skip_energy:
        # Run batch energy calculation in analysis conda environment
        # Note: This needs to be run in conda analysis environment
        print("\nNote: Energy calculation requires conda 'analysis' environment", flush=True)
        print("Running energy calculation...", flush=True)
        
        try:
            # Import and run in current environment (should be analysis)
            energy_df = batch_energy(
                pdb_folder=args.input_dir,
                output_dir=args.output_dir,
                num_workers=args.n_cores,
                relax=args.relax,
                save_results=True,
                compute_pnear=False,  # Don't compute PNear as requested
                save_relaxed_pdb=True,
                relaxed_pdb_dir=relaxed_dir,
                link_constraints=link_constraints,
                auto_head_tail_bond=auto_head_tail_bond,
                link_csv_path=args.link_config,
                extract_dslf_fa13=args.extract_dslf_fa13,
            )
            print(f"Energy calculation complete. Results saved to {energy_dir}", flush=True)
            
            # Load energy results from CSV if available
            if os.path.exists(energy_csv):
                energy_df = pd.read_csv(energy_csv)
                print(f"Loaded energy results from {energy_csv}", flush=True)
        except Exception as e:
            print(f"Error during energy calculation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Try to load existing results if available
            if os.path.exists(energy_csv):
                print(f"Loading existing energy results from {energy_csv}", flush=True)
                energy_df = pd.read_csv(energy_csv)
            else:
                print("No existing energy results found. Continuing without energy data.", flush=True)
                energy_df = pd.DataFrame()
    else:
        # Load existing energy results
        energy_df = pd.read_csv(energy_csv)
        print(f"Loaded existing energy results from {energy_csv}", flush=True)
    
    # Step 3: HighFold evaluation for each structure
    print("\n" + "=" * 80, flush=True)
    print("Step 3: Running HighFold predictions and calculating scRMSD...", flush=True)
    print("=" * 80, flush=True)
    
    # Find relaxed PDB files
    relaxed_pdb_files = sorted(glob.glob(os.path.join(relaxed_dir, "*_relaxed.pdb")))
    if not relaxed_pdb_files:
        # Fallback to original PDB files if no relaxed ones
        print("Warning: No relaxed PDB files found, using original files", flush=True)
        relaxed_pdb_files = pdb_files
    
    print(f"Evaluating {len(relaxed_pdb_files)} structures...", flush=True)
    
    # Check if SAP calculation can be skipped (check if SAP results exist in energy CSV or all_metrics CSV)
    skip_sap = False
    all_metrics_csv = os.path.join(args.output_dir, "all_metrics.csv")
    
    if os.path.exists(all_metrics_csv):
        try:
            existing_metrics_df = pd.read_csv(all_metrics_csv)
            if "SAP_total" in existing_metrics_df.columns and "SAP_mean" in existing_metrics_df.columns:
                # Check if we have SAP results for all relaxed structures
                relaxed_basenames = {os.path.basename(f).replace("_relaxed.pdb", "").replace(".pdb", "") for f in relaxed_pdb_files}
                metrics_pdb_names = set(existing_metrics_df["PDB"].astype(str).str.replace("_relaxed", "").str.replace(".pdb", ""))
                
                # Check if all relaxed files have SAP results
                if relaxed_basenames.issubset(metrics_pdb_names):
                    # Check if SAP values are not all NaN
                    sap_valid = existing_metrics_df["SAP_total"].notna().sum()
                    if sap_valid > 0:
                        skip_sap = True
                        print(f"\n✓ SAP results already exist in {all_metrics_csv}", flush=True)
                        print(f"  Found SAP results for {sap_valid} structures", flush=True)
                        print(f"  Skipping SAP calculation step...", flush=True)
        except Exception as e:
            print(f"Warning: Could not verify existing SAP results: {e}", flush=True)
            skip_sap = False
    
    # Also check energy CSV for SAP results
    if not skip_sap and os.path.exists(energy_csv):
        try:
            existing_energy_df = pd.read_csv(energy_csv)
            if "SAP_total" in existing_energy_df.columns and "SAP_mean" in existing_energy_df.columns:
                relaxed_basenames = {os.path.basename(f).replace("_relaxed.pdb", "").replace(".pdb", "") for f in relaxed_pdb_files}
                energy_pdb_names = set(existing_energy_df["PDB"].astype(str).str.replace("_relaxed", "").str.replace(".pdb", ""))
                
                if relaxed_basenames.issubset(energy_pdb_names):
                    sap_valid = existing_energy_df["SAP_total"].notna().sum()
                    if sap_valid > 0:
                        skip_sap = True
                        print(f"\n✓ SAP results already exist in {energy_csv}", flush=True)
                        print(f"  Found SAP results for {sap_valid} structures", flush=True)
                        print(f"  Skipping SAP calculation step...", flush=True)
        except Exception as e:
            pass
    
    if not skip_sap:
        # Also calculate SAP for all relaxed structures (target chain only)
        print(f"\nCalculating SAP for all structures (chain {args.chain})...", flush=True)
        sap_results = []
        chain_dir = os.path.join(args.output_dir, f"chain{args.chain}_extracted")
        for i, relaxed_pdb in enumerate(relaxed_pdb_files):
            pdb_name = os.path.basename(relaxed_pdb).replace("_relaxed.pdb", "").replace(".pdb", "")
            print(f"  [{i+1}/{len(relaxed_pdb_files)}] Calculating SAP for {pdb_name}...", flush=True)
            try:
                import pyrosetta
                from pyrosetta import rosetta
                
                if not rosetta.basic.was_init_called():
                    worker_init()
                
                # Extract target chain for SAP calculation
                chain_pdb = os.path.join(chain_dir, f"{pdb_name}_chain{args.chain}.pdb")
                if not os.path.exists(chain_pdb):
                    # Extract if not exists
                    extract_chain_a_pdb(relaxed_pdb, chain_pdb, args.chain)
                
                if os.path.exists(chain_pdb):
                    pose = pyrosetta.pose_from_pdb(chain_pdb)
                    detect_disulfides_safe(pose)
                    sap_total, sap_mean, nres = calculate_sap_metrics(pose)
                    sap_results.append({
                        "PDB": pdb_name,
                        "SAP_total": float(sap_total),
                        "SAP_mean": float(sap_mean),
                    })
                else:
                    sap_results.append({
                        "PDB": pdb_name,
                        "SAP_total": None,
                        "SAP_mean": None,
                        "SAP_Error": f"Failed to extract chain {args.chain}",
                    })
            except Exception as e:
                print(f"    Error: {e}", flush=True)
                sap_results.append({
                    "PDB": pdb_name,
                    "SAP_total": None,
                    "SAP_mean": None,
                    "SAP_Error": str(e),
                })
        
        sap_df = pd.DataFrame(sap_results)
        print(f"SAP calculation complete for {len(sap_df)} structures", flush=True)
    else:
        # Extract SAP results from existing CSV
        if os.path.exists(all_metrics_csv):
            existing_metrics_df = pd.read_csv(all_metrics_csv)
            sap_df = existing_metrics_df[["PDB", "SAP_total", "SAP_mean"]].copy()
            sap_df["PDB"] = sap_df["PDB"].astype(str).str.replace("_relaxed", "").str.replace(".pdb", "")
        elif os.path.exists(energy_csv):
            existing_energy_df = pd.read_csv(energy_csv)
            sap_df = existing_energy_df[["PDB", "SAP_total", "SAP_mean"]].copy()
            sap_df["PDB"] = sap_df["PDB"].astype(str).str.replace("_relaxed", "").str.replace(".pdb", "")
        else:
            sap_df = pd.DataFrame(columns=["PDB", "SAP_total", "SAP_mean"])
    
    # Check which HighFold results already exist
    highfold_dir = os.path.join(args.output_dir, "highfold")
    existing_highfold_results = {}
    
    # Check existing all_metrics.csv for HighFold results
    if os.path.exists(all_metrics_csv):
        try:
            existing_metrics_df = pd.read_csv(all_metrics_csv)
            if "PLDDT" in existing_metrics_df.columns and "scRMSD" in existing_metrics_df.columns:
                for _, row in existing_metrics_df.iterrows():
                    pdb_name = str(row.get("PDB", "")).replace("_relaxed", "").replace(".pdb", "")
                    plddt = row.get("PLDDT")
                    sc_rmsd = row.get("scRMSD")
                    if pd.notna(plddt) and pd.notna(sc_rmsd):
                        existing_highfold_results[pdb_name] = {
                            "PLDDT": float(plddt),
                            "scRMSD": float(sc_rmsd) if pd.notna(sc_rmsd) else None,
                        }
        except Exception as e:
            print(f"Warning: Could not read existing HighFold results: {e}", flush=True)
    
    # Filter out structures that already have HighFold results
    relaxed_pdb_files_to_process = []
    highfold_results_from_cache = []
    
    for relaxed_pdb in relaxed_pdb_files:
        pdb_name = os.path.basename(relaxed_pdb).replace("_relaxed.pdb", "").replace(".pdb", "")
        if pdb_name in existing_highfold_results:
            # Use existing result
            cached_result = existing_highfold_results[pdb_name].copy()
            cached_result["PDB"] = pdb_name
            highfold_results_from_cache.append(cached_result)
            print(f"  ✓ Skipping {pdb_name}: PLDDT={cached_result['PLDDT']}, scRMSD={cached_result['scRMSD']} (already computed)", flush=True)
        else:
            # Need to process
            relaxed_pdb_files_to_process.append(relaxed_pdb)
    
    if highfold_results_from_cache:
        print(f"\n✓ Found {len(highfold_results_from_cache)} existing HighFold results, skipping...", flush=True)
    
    # HighFold evaluation (with multi-GPU support)
    if relaxed_pdb_files_to_process:
        print(f"\nRunning HighFold predictions for {len(relaxed_pdb_files_to_process)} structures...", flush=True)
        
        # Determine GPU configuration
        if args.gpus:
            gpu_ids = [gid.strip() for gid in args.gpus.split(',')]
            use_multi_gpu = len(gpu_ids) > 1 or args.jobs_per_gpu > 1
        else:
            gpu_ids = [str(args.gpu_id)]
            use_multi_gpu = False
        
        if use_multi_gpu:
            # Multi-GPU parallel processing
            print(f"Using multi-GPU parallel processing: GPUs={args.gpus}, jobs_per_gpu={args.jobs_per_gpu}", flush=True)
            new_highfold_results = run_highfold_parallel(
                relaxed_pdb_files=relaxed_pdb_files_to_process,
                output_dir=args.output_dir,
                gpu_ids=gpu_ids,
                jobs_per_gpu=args.jobs_per_gpu,
                chain_id=args.chain,
                msa_mode=args.msa_mode,
                msa_threads=args.msa_threads,
                max_msa=args.max_msa,
            )
        else:
            # Single GPU sequential processing
            print(f"Using single GPU: {gpu_ids[0]}", flush=True)
            new_highfold_results = []
            for i, relaxed_pdb in enumerate(relaxed_pdb_files_to_process):
                print(f"\n[{i+1}/{len(relaxed_pdb_files_to_process)}] Processing {os.path.basename(relaxed_pdb)}", flush=True)
                result = evaluate_single_structure(
                    pdb_path=relaxed_pdb,
                    relaxed_pdb_path=relaxed_pdb,
                    output_dir=args.output_dir,
                    gpu_id=int(gpu_ids[0]),
                    chain_id=args.chain,
                    msa_mode=args.msa_mode,
                    msa_threads=args.msa_threads,
                    max_msa=args.max_msa,
                )
                new_highfold_results.append(result)
                pdb_name = os.path.basename(relaxed_pdb).replace("_relaxed.pdb", "").replace(".pdb", "")
                plddt = result.get('PLDDT', 'N/A')
                sc_rmsd = result.get('scRMSD', 'N/A')
                print(f"  ✓ {pdb_name}: PLDDT={plddt}, scRMSD={sc_rmsd}", flush=True)
        
        # Combine cached and new results
        highfold_results = highfold_results_from_cache + new_highfold_results
    else:
        print(f"\n✓ All HighFold results already exist, skipping HighFold prediction step...", flush=True)
        highfold_results = highfold_results_from_cache
    
    highfold_df = pd.DataFrame(highfold_results)
    
    # Step 4: Merge all results
    print("\n" + "=" * 80, flush=True)
    print("Step 4: Aggregating all metrics...", flush=True)
    print("=" * 80, flush=True)
    
    # Prepare base names for merging
    def get_base_name(pdb_name):
        return str(pdb_name).replace("_relaxed", "").replace(".pdb", "").strip()
    
    # Merge all dataframes
    merged_df = None
    
    # Start with energy results
    if not energy_df.empty and "PDB" in energy_df.columns:
        energy_df["PDB_base"] = energy_df["PDB"].apply(get_base_name)
        merged_df = energy_df.copy()
    
    # Merge SAP results
    if not sap_df.empty:
        sap_df["PDB_base"] = sap_df["PDB"].apply(get_base_name)
        if merged_df is not None:
            merged_df = pd.merge(
                merged_df, sap_df[["PDB_base", "SAP_total", "SAP_mean"]],
                on="PDB_base", how="outer"
            )
        else:
            merged_df = sap_df.copy()
    
    # Merge HighFold results
    if not highfold_df.empty:
        highfold_df["PDB_base"] = highfold_df["PDB"].apply(get_base_name)
        if merged_df is not None:
            merged_df = pd.merge(
                merged_df, highfold_df,
                on="PDB_base", how="outer"
            )
        else:
            merged_df = highfold_df.copy()
    
    if merged_df is None:
        print("Warning: No results to merge")
        merged_df = pd.DataFrame()
    
    # Save aggregated results
    results_csv = os.path.join(args.output_dir, "all_metrics.csv")
    if not merged_df.empty:
        merged_df.to_csv(results_csv, index=False)
        print(f"All metrics saved to {results_csv}", flush=True)
        print(f"Total structures evaluated: {len(merged_df)}", flush=True)
        
        # Print summary statistics
        print("\nSummary Statistics:", flush=True)
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        for col in ["Total_Energy", "Binding_Energy", "SAP_total", "SAP_mean", "PLDDT", "scRMSD", "dslf_fa13"]:
            if col in merged_df.columns:
                valid = merged_df[col].dropna()
                if len(valid) > 0:
                    print(f"  {col}: mean={valid.mean():.2f}, min={valid.min():.2f}, max={valid.max():.2f}", flush=True)
    else:
        print("Warning: No results to save", flush=True)
    
    # Step 5: Filter and copy passed structures
    if args.thresholds:
        print("\n" + "=" * 80, flush=True)
        print("Step 5: Filtering by thresholds and copying passed structures...", flush=True)
        print("=" * 80, flush=True)
        
        try:
            thresholds = json.loads(args.thresholds)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in --thresholds")
            sys.exit(1)
        
        # Warn if dslf_fa13 is in thresholds but not extracted
        if "dslf_fa13" in thresholds and not args.extract_dslf_fa13:
            print("Warning: dslf_fa13 is in thresholds but --extract_dslf_fa13 is not enabled.", flush=True)
            print("  Structures without dslf_fa13 values will be excluded from filtering.", flush=True)
        
        # Create filter conditions
        filter_conditions = []
        for key, value in thresholds.items():
            if key in merged_df.columns:
                # Handle NaN values: exclude them from filtering (they don't pass)
                col_data = merged_df[key]
                if key in ["scRMSD", "SAP_total", "SAP_mean", "dslf_fa13"]:
                    # Lower is better - exclude NaN values
                    condition = (col_data.notna()) & (col_data <= value)
                    filter_conditions.append(condition)
                elif key in ["Binding_Energy"]:
                    # Binding energy: lower is better (usually negative, so <= threshold means more negative)
                    # If threshold is -10, we want Binding_Energy <= -10 (i.e., more negative/better)
                    condition = (col_data.notna()) & (col_data <= value)
                    filter_conditions.append(condition)
                else:
                    # Higher is better (PLDDT, etc.) - exclude NaN values
                    condition = (col_data.notna()) & (col_data >= value)
                    filter_conditions.append(condition)
        
        if filter_conditions:
            passed_mask = pd.concat(filter_conditions, axis=1).all(axis=1)
            passed_df = merged_df[passed_mask]
            
            print(f"Found {len(passed_df)} structures passing all thresholds", flush=True)
            print(f"Thresholds: {thresholds}", flush=True)
            
            if args.passed_dir or args.passed_relax_dir:
                os.makedirs(args.passed_dir, exist_ok=True) if args.passed_dir else None
                os.makedirs(args.passed_relax_dir, exist_ok=True) if args.passed_relax_dir else None
                
                for _, row in passed_df.iterrows():
                    pdb_base = row.get("PDB_base", row.get("PDB", ""))
                    if not pdb_base:
                        continue
                    
                    # Find original PDB
                    original_pdb = None
                    for pdb_file in pdb_files:
                        if pdb_base in os.path.basename(pdb_file):
                            original_pdb = pdb_file
                            break
                    
                    # Find relaxed PDB
                    relaxed_pdb = None
                    for rpdb in relaxed_pdb_files:
                        if pdb_base in os.path.basename(rpdb):
                            relaxed_pdb = rpdb
                            break
                    
                    # Copy files
                    if args.passed_dir and original_pdb:
                        dest = os.path.join(args.passed_dir, os.path.basename(original_pdb))
                        shutil.copy2(original_pdb, dest)
                        print(f"  Copied original: {os.path.basename(original_pdb)}", flush=True)
                    
                    if args.passed_relax_dir and relaxed_pdb:
                        dest = os.path.join(args.passed_relax_dir, os.path.basename(relaxed_pdb))
                        shutil.copy2(relaxed_pdb, dest)
                        print(f"  Copied relaxed: {os.path.basename(relaxed_pdb)}", flush=True)
            
            # Save passed results
            passed_csv = os.path.join(args.output_dir, "passed_structures.csv")
            passed_df.to_csv(passed_csv, index=False)
            print(f"Passed structures saved to {passed_csv}", flush=True)
        else:
            print("Warning: No valid threshold columns found in results", flush=True)
    
    print("\n" + "=" * 80, flush=True)
    print("Pipeline complete!", flush=True)
    print("=" * 80, flush=True)
    print(f"Results directory: {args.output_dir}", flush=True)
    print(f"All metrics CSV: {results_csv}", flush=True)


if __name__ == "__main__":
    main()

