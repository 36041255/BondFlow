#!/usr/bin/env python3
"""
Complete pipeline for generating and evaluating knottin binders with disulfide bonds.

Pipeline steps:
1. Generate knottin binders using Sampler (conda apm_env)
2. Evaluate with PyRosetta (energy + SAP, conda analysis)
3. Extract chain A, run HighFold prediction, calculate scRMSD and PLDDT
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
from BondFlow.experiment.analysis.energy import batch_energy
from BondFlow.experiment.analysis.calc_sap import (
    calculate_sap_metrics,
    extract_chain,
    detect_disulfides_safe,
    worker_init,
)
from BondFlow.experiment.analysis.evaluate_knottins import (
    get_structure_info,
    run_highfold,
    calculate_rmsd,
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


def extract_chain_a_pdb(input_pdb, output_pdb, chain_id="A"):
    """Extract chain A from PDB file and save to new file."""
    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()
    
    try:
        structure = parser.get_structure('struct', input_pdb)
        model = structure[0]
        
        # Find chain
        target_chain = None
        for chain in model:
            if chain.id == chain_id:
                target_chain = chain
                break
        
        if target_chain is None:
            print(f"Warning: Chain {chain_id} not found in {input_pdb}")
            return False
        
        # Create new structure with only chain A
        new_structure = PDB.Structure.Structure('knottin')
        new_model = PDB.Model.Model(0)
        new_model.add(target_chain)
        new_structure.add(new_model)
        
        # Save
        io.set_structure(new_structure)
        io.save(output_pdb)
        return True
    except Exception as e:
        print(f"Error extracting chain {chain_id} from {input_pdb}: {e}")
        return False


def get_sequence_from_pdb(pdb_path):
    """Extract amino acid sequence from PDB file."""
    parser = PDB.PDBParser(QUIET=True)
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
    parser = PDB.PDBParser(QUIET=True)
    
    try:
        ref_structure = parser.get_structure('ref', ref_pdb)
        pred_structure = parser.get_structure('pred', pred_pdb)
        
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


def run_highfold_for_sequence(seq, ss_pairs, is_cyclic, output_dir, gpu_id=0):
    """Run HighFold prediction for a sequence using conda environment."""
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
        "--msa-mode", "single_sequence",
    ]
    
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
    
    relaxed_pdb, output_dir, gpu_id, chain_id = args
    pdb_name = os.path.basename(relaxed_pdb).replace("_relaxed.pdb", "").replace(".pdb", "")
    print(f"[GPU {gpu_id}] Processing {pdb_name}...", flush=True)
    try:
        result = evaluate_single_structure(
            pdb_path=relaxed_pdb,
            relaxed_pdb_path=relaxed_pdb,
            output_dir=output_dir,
            gpu_id=gpu_id,
            chain_id=chain_id,
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
    chain_id="A"
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
        tasks.append((relaxed_pdb, output_dir, gpu_id, chain_id))
        gpu_queue.put(gpu_id)  # Return GPU to queue for reuse
    
    # Actually, we'll use a simpler approach: round-robin GPU assignment
    num_workers = len(gpu_ids) * jobs_per_gpu
    tasks_with_gpu = []
    for i, relaxed_pdb in enumerate(relaxed_pdb_files):
        gpu_idx = i % len(gpu_ids)
        gpu_id = int(gpu_ids[gpu_idx])
        tasks_with_gpu.append((relaxed_pdb, output_dir, gpu_id, chain_id))
    
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
    chain_id="A"
):
    """Evaluate a single structure: HighFold prediction and scRMSD."""
    pdb_name = os.path.basename(pdb_path).replace("_relaxed.pdb", "").replace(".pdb", "")
    results = {"PDB": pdb_name}
    
    # 1. Extract chain A
    chain_dir = os.path.join(output_dir, "chainA_extracted")
    os.makedirs(chain_dir, exist_ok=True)
    chain_a_pdb = os.path.join(chain_dir, f"{pdb_name}_chainA.pdb")
    if not extract_chain_a_pdb(relaxed_pdb_path, chain_a_pdb, chain_id):
        results["Error"] = "Failed to extract chain A"
        return results
    
    # 2. Get sequence and structure info
    seq = get_sequence_from_pdb(chain_a_pdb)
    if seq is None:
        results["Error"] = "Failed to extract sequence"
        return results
    results["Sequence"] = seq
    results["Seq_Length"] = len(seq)
    
    # Get structure info (disulfide bonds, cyclization)
    info = get_structure_info(chain_a_pdb)
    if info is None:
        results["Error"] = "Failed to get structure info"
        return results
    
    seq_info, ss_pairs, is_cyclic, ref_ca_atoms = info
    results["IsCyclic"] = is_cyclic
    results["NumSS"] = len(ss_pairs) // 2
    
    # 3. Run HighFold prediction
    highfold_dir = os.path.join(output_dir, "highfold", pdb_name)
    os.makedirs(highfold_dir, exist_ok=True)
    
    plddt, pred_pdb = run_highfold_for_sequence(
        seq, ss_pairs, is_cyclic, highfold_dir, gpu_id
    )
    
    if plddt is None or pred_pdb is None:
        results["PLDDT"] = None
        results["scRMSD"] = None
        results["HighFold_Error"] = "Prediction failed"
    else:
        results["PLDDT"] = float(plddt)
        
        # Calculate scRMSD
        sc_rmsd = calculate_sc_rmsd(chain_a_pdb, pred_pdb)
        results["scRMSD"] = sc_rmsd if sc_rmsd is not None else None
    
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
    
    # Energy calculation options
    parser.add_argument(
        "--relax", action="store_true", default=True,
        help="Run FastRelax before energy calculation"
    )
    parser.add_argument(
        "--link_config", type=str,
        help="Path to link.csv config file"
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
    chain_dir = os.path.join(args.output_dir, "chainA_extracted")
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
    
    # Find PDB files
    pdb_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pdb")))
    if not pdb_files:
        print(f"Error: No PDB files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(pdb_files)} PDB files", flush=True)
    
    # Run batch energy calculation in analysis conda environment
    # Note: This needs to be run in conda analysis environment
    print("Note: Energy calculation requires conda 'analysis' environment", flush=True)
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
            link_constraints=False,  # Can be enabled if needed
            link_csv_path=args.link_config,
        )
        print(f"Energy calculation complete. Results saved to {energy_dir}", flush=True)
        
        # Load energy results from CSV if available
        energy_csv = os.path.join(energy_dir, "Energy_results.csv")
        if os.path.exists(energy_csv):
            energy_df = pd.read_csv(energy_csv)
            print(f"Loaded energy results from {energy_csv}", flush=True)
    except Exception as e:
        print(f"Error during energy calculation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Try to load existing results if available
        energy_csv = os.path.join(energy_dir, "Energy_results.csv")
        if os.path.exists(energy_csv):
            print(f"Loading existing energy results from {energy_csv}", flush=True)
            energy_df = pd.read_csv(energy_csv)
        else:
            print("No existing energy results found. Continuing without energy data.", flush=True)
            energy_df = pd.DataFrame()
    
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
    
    # Also calculate SAP for all relaxed structures (chain A only)
    print("\nCalculating SAP for all structures (chain A)...", flush=True)
    sap_results = []
    for i, relaxed_pdb in enumerate(relaxed_pdb_files):
        pdb_name = os.path.basename(relaxed_pdb).replace("_relaxed.pdb", "").replace(".pdb", "")
        print(f"  [{i+1}/{len(relaxed_pdb_files)}] Calculating SAP for {pdb_name}...", flush=True)
        try:
            import pyrosetta
            from pyrosetta import rosetta
            
            if not rosetta.basic.was_init_called():
                worker_init()
            
            # Extract chain A for SAP calculation
            chain_a_pdb = os.path.join(chain_dir, f"{pdb_name}_chainA.pdb")
            if not os.path.exists(chain_a_pdb):
                # Extract if not exists
                extract_chain_a_pdb(relaxed_pdb, chain_a_pdb, args.chain)
            
            if os.path.exists(chain_a_pdb):
                pose = pyrosetta.pose_from_pdb(chain_a_pdb)
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
                    "SAP_Error": "Failed to extract chain A",
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
    
    # HighFold evaluation (with multi-GPU support)
    print("\nRunning HighFold predictions...")
    
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
        highfold_results = run_highfold_parallel(
            relaxed_pdb_files=relaxed_pdb_files,
            output_dir=args.output_dir,
            gpu_ids=gpu_ids,
            jobs_per_gpu=args.jobs_per_gpu,
            chain_id=args.chain,
        )
    else:
        # Single GPU sequential processing
        print(f"Using single GPU: {gpu_ids[0]}", flush=True)
        highfold_results = []
        for i, relaxed_pdb in enumerate(relaxed_pdb_files):
            print(f"\n[{i+1}/{len(relaxed_pdb_files)}] Processing {os.path.basename(relaxed_pdb)}", flush=True)
            result = evaluate_single_structure(
                pdb_path=relaxed_pdb,
                relaxed_pdb_path=relaxed_pdb,
                output_dir=args.output_dir,
                gpu_id=int(gpu_ids[0]),
                chain_id=args.chain,
            )
            highfold_results.append(result)
            pdb_name = os.path.basename(relaxed_pdb).replace("_relaxed.pdb", "").replace(".pdb", "")
            plddt = result.get('PLDDT', 'N/A')
            sc_rmsd = result.get('scRMSD', 'N/A')
            print(f"  ✓ {pdb_name}: PLDDT={plddt}, scRMSD={sc_rmsd}", flush=True)
    
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
        for col in ["Total_Energy", "Binding_Energy", "SAP_total", "SAP_mean", "PLDDT", "scRMSD"]:
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
        
        # Create filter conditions
        filter_conditions = []
        for key, value in thresholds.items():
            if key in merged_df.columns:
                if key in ["scRMSD"]:
                    # Lower is better
                    filter_conditions.append(merged_df[key] <= value)
                elif key in ["Binding_Energy"]:
                    # Binding energy: lower is better (usually negative, so <= threshold means more negative)
                    # If threshold is -10, we want Binding_Energy <= -10 (i.e., more negative/better)
                    filter_conditions.append(merged_df[key] <= value)
                else:
                    # Higher is better (PLDDT, etc.)
                    filter_conditions.append(merged_df[key] >= value)
        
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

