import os
import sys
import glob
import argparse
import subprocess
import shutil
import uuid
import json
import time
import multiprocessing
import numpy as np
import matplotlib
# Set non-interactive backend for headless environments (e.g., SLURM)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio import PDB
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

# Configuration
# Script is in BondFlow/BondFlow/experiment/analysis/
# HighFold2 is in BondFlow/HighFold2/ (sibling of BondFlow/BondFlow/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels: analysis -> experiment -> BondFlow -> BondFlow root
# Then go to parent to get BondFlow/, where HighFold2 is located
BOND_FLOW_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
COLABFOLD_BIN = os.path.join(BOND_FLOW_ROOT, "HighFold2/localcolabfold/colabfold-conda/bin/colabfold_batch")
# Resolve any symlinks to get the actual path
COLABFOLD_BIN = os.path.realpath(COLABFOLD_BIN)

def get_structure_info(pdb_path):
    """
    Parses PDB/CIF to get sequence, disulfide bonds, and cyclization status.
    Returns:
        seq (str): Amino acid sequence
        ss_pairs (list): List of 1-based indices [i1, j1, i2, j2, ...]
        is_cyclic (bool): True if N-term and C-term are connected
        ca_coords (list): List of CA coordinates for RMSD calc
    """
    # Support both PDB and CIF formats
    ext = os.path.splitext(pdb_path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('struct', pdb_path)
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return None
        
    model = structure[0]
    # Assume first chain is the target
    chain = list(model)[0]
    
    std_residues = []
    seq = ""
    non_standard_residues = []  # Track non-standard amino acids
    
    for r in chain.get_residues():
        # Check if it's a standard amino acid
        if PDB.is_aa(r, standard=True):
            std_residues.append(r)
            # bio.sequtils.seq1 converts 'ALA' to 'A'
            seq += seq1(r.get_resname())
        elif PDB.is_aa(r, standard=False):
            # It's an amino acid but not standard (e.g., modified residues like SEP, PTR, MSE)
            # Skip the entire structure if any non-standard amino acid is found
            non_standard_residues.append((r.get_resname(), r.id[1] if hasattr(r, 'id') else 'unknown'))
        # else: ignore non-amino acid residues (waters, ions, etc.)
    
    if not seq:
        return None
    
    # Skip entire structure if non-standard amino acids were found
    if non_standard_residues:
        pdb_name = os.path.basename(pdb_path)
        print(f"Skipping [{pdb_name}]: Contains {len(non_standard_residues)} non-standard amino acid(s): {non_standard_residues[:5]}{'...' if len(non_standard_residues) > 5 else ''}")
        return None

    # Disulfide detection (SG-SG < 3.0 A)
    # Use greedy algorithm to ensure each CYS participates in only one disulfide bond
    ss_pairs = []
    cys_indices = [i for i, r in enumerate(std_residues) if r.get_resname() == 'CYS']
    
    # Collect all possible CYS pairs with distances
    candidate_pairs = []
    for i in range(len(cys_indices)):
        for j in range(i+1, len(cys_indices)):
            idx1 = cys_indices[i]
            idx2 = cys_indices[j]
            res1 = std_residues[idx1]
            res2 = std_residues[idx2]
            
            # Skip adjacent residues (they rarely form disulfide bonds)
            if abs(idx1 - idx2) == 1:
                continue
            
            if 'SG' in res1 and 'SG' in res2:
                d = res1['SG'] - res2['SG']
                if d < 3.0: 
                    # Store as (distance, idx1, idx2) for sorting
                    candidate_pairs.append((d, idx1, idx2))
    
    # Sort by distance (closest first) and greedily assign pairs
    # Ensure each CYS residue participates in only one disulfide bond
    candidate_pairs.sort(key=lambda x: x[0])  # Sort by distance
    used_indices = set()
    for d, idx1, idx2 in candidate_pairs:
        # Only add if neither residue is already paired
        if idx1 not in used_indices and idx2 not in used_indices:
            # 1-based indices
            ss_pairs.extend([idx1 + 1, idx2 + 1])
            used_indices.add(idx1)
            used_indices.add(idx2)
                    
    # Head-to-tail detection (N(first) - C(last) < 2.5 A)
    is_cyclic = False
    if len(std_residues) > 1:
        n_term = std_residues[0]
        c_term = std_residues[-1]
        if 'N' in n_term and 'C' in c_term:
            d = n_term['N'] - c_term['C']
            if d < 2.5: # Peptide bond length is ~1.33A
                is_cyclic = True

    # Extract CA atoms for RMSD
    ca_atoms = []
    for r in std_residues:
        if 'CA' in r:
            ca_atoms.append(r['CA'])
        else:
            # Fallback if CA is missing? Unlikely for standard PDBs
            pass
            
    return seq, ss_pairs, is_cyclic, ca_atoms

def run_highfold(seq, ss_pairs, is_cyclic, output_dir, gpu_id, msa_mode="single_sequence", msa_threads=None, max_msa=None):
    """
    Runs colabfold_batch for a single sequence.
    
    Args:
        msa_mode: MSA mode for colabfold. Options:
            - "single_sequence": No MSA (fastest, lower accuracy)
            - "mmseqs2_uniref_env": Use MMseqs2 to search UniRef and environmental sequences (slower, higher accuracy)
            - "mmseqs2_uniref": Use MMseqs2 to search UniRef only
        msa_threads: Number of threads for MSA search (MMseqs2). If None, uses all available CPU cores.
                     This is per-job (each job uses this many threads for its MSA search).
        max_msa: MSA depth in format "max-seq:max-extra-seq" (e.g., "512:5120").
                 If None, uses ColabFold default (typically "512:5120" for AlphaFold2).
                 max-seq: number of sequence clusters to use
                 max-extra-seq: number of extra sequences to use
    """
    # Create a unique job ID/dir
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(output_dir, f"job_{job_id}")
    os.makedirs(job_dir, exist_ok=True)
    
    fasta_path = os.path.join(job_dir, "input.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">target\n{seq}\n")
        
    cmd = [
        COLABFOLD_BIN,
        "--msa-mode", msa_mode,
        # "--model-type", "alphafold2_multimer_v3" if len(seq) > 0 else "alphafold2", 
        # "--amber"
    ]
    
    # Add MSA depth if specified
    if max_msa is not None and msa_mode != "single_sequence":
        cmd.extend(["--max-msa", max_msa])
    
    # Add disulfide bonds
    if ss_pairs:
        cmd.append("--disulfide-bond-pairs")
        cmd.extend([str(x) for x in ss_pairs])
        
    # Add cyclic constraint
    if is_cyclic:
        cmd.append("--head-to-tail")
    else:
        cmd.append("--no-head-to-tail")
        
    cmd.append(fasta_path)
    cmd.append(job_dir)
        
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Force local MSA search to avoid timeout with remote server
    if msa_mode != "single_sequence":
        # Disable remote MSA server, force local MMseqs2
        env["COLABFOLD_MSA_SERVER"] = ""
        env["MMSEQS_SERVER"] = ""
        # Ensure local MMseqs2 is in PATH
        mmseqs_dir = os.path.join(BOND_FLOW_ROOT, "HighFold2/localcolabfold/colabfold-conda/bin")
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
    
    # Debug: Print MSA parameters
    if msa_mode != "single_sequence":
        print(f"[MSA Config] Mode: {msa_mode}")
        if msa_threads is not None:
            print(f"[MSA Config] Threads: {msa_threads} (per job)")
        else:
            print(f"[MSA Config] Threads: default (all available CPU cores, per job)")
        if max_msa is not None:
            print(f"[MSA Config] Depth: {max_msa} (max-seq:max-extra-seq)")
        else:
            print(f"[MSA Config] Depth: default (typically 512:5120)")
    
    print(f"Running command: {' '.join(cmd)}")
    # Run command
    try:
        # Capture output to avoid spamming console
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error running HighFold for job {job_id}: {e.stderr.decode()}")
        return None
        
    return job_dir

def calculate_rmsd(ref_ca_atoms, pred_pdb_path):
    """
    Aligns predicted structure to reference CA atoms and calculates RMSD.
    """
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('pred', pred_pdb_path)
    except:
        return None
        
    model = structure[0]
    chain = list(model)[0]
    
    pred_ca_atoms = []
    for r in chain.get_residues():
        if PDB.is_aa(r, standard=True) and 'CA' in r:
            pred_ca_atoms.append(r['CA'])
            
    if len(pred_ca_atoms) != len(ref_ca_atoms):
        print(f"Warning: atom count mismatch {len(pred_ca_atoms)} vs {len(ref_ca_atoms)}")
        return None
        
    superimposer = PDB.Superimposer()
    superimposer.set_atoms(ref_ca_atoms, pred_ca_atoms)
    superimposer.apply(model.get_atoms())
    
    return superimposer.rms

def get_best_plddt_and_structure(job_dir):
    """
    Finds the best model based on PLDDT (or rank_1).
    Colabfold outputs *_rank_1_model_*.pdb
    Also *_scores.json or similar.
    We'll look for the rank_1 pdb file.
    
    Supports both flat structure (PDB files directly in job_dir) and nested structure
    (PDB files in job_* subdirectories, e.g., job_dir/job_*/0_unrelaxed_rank_001_*.pdb).
    """
    # Pattern: *_rank_001_*.pdb or *_rank_1_*.pdb depending on version
    # Modern colabfold: input_rank_001_alphafold2_...pdb
    
    # First, try direct search in job_dir (flat structure)
    pdbs = glob.glob(os.path.join(job_dir, "*_rank_001_*.pdb"))
    if not pdbs:
        pdbs = glob.glob(os.path.join(job_dir, "*_rank_1_*.pdb"))
    
    # If not found, try recursive search in subdirectories (nested structure)
    # This handles cases like job_dir/job_*/0_unrelaxed_rank_001_*.pdb
    if not pdbs:
        pdbs = glob.glob(os.path.join(job_dir, "**", "*_rank_001_*.pdb"), recursive=True)
    if not pdbs:
        pdbs = glob.glob(os.path.join(job_dir, "**", "*_rank_1_*.pdb"), recursive=True)
        
    if not pdbs:
        return None, None
    
    # Sort to ensure consistent selection (prefer shorter paths, then alphabetical)
    pdbs = sorted(pdbs)
    best_pdb = pdbs[0] # The rank 1 is the best
    
    # Extract PLDDT from filename or B-factors
    # Filename format often contains plddt, but B-factor is reliable.
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('best', best_pdb)
    
    plddts = []
    for atom in structure.get_atoms():
        if atom.name == 'CA':
            plddts.append(atom.bfactor)
            
    if not plddts:
        return None, best_pdb
        
    avg_plddt = np.mean(plddts)
    return avg_plddt, best_pdb

def process_pdb(args):
    """
    Worker function to process a single PDB.
    """
    pdb_path, output_root, gpu_queue, msa_mode, msa_threads, max_msa = args
    
    # 1. Parse Info
    info = get_structure_info(pdb_path)
    if not info:
        return None
    seq, ss_pairs, is_cyclic, ref_ca_atoms = info
    
    pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
    output_dir = os.path.join(output_root, pdb_name)
    
    # 2. Get GPU
    gpu_id = gpu_queue.get()
    try:
        # 3. Run HighFold
        job_dir = run_highfold(seq, ss_pairs, is_cyclic, output_dir, gpu_id, msa_mode, msa_threads, max_msa)
    finally:
        gpu_queue.put(gpu_id)
        
    if not job_dir:
        return None
        
    # 4. Analyze
    plddt, best_pdb_path = get_best_plddt_and_structure(job_dir)
    if plddt is None or best_pdb_path is None:
        return None
        
    rmsd = calculate_rmsd(ref_ca_atoms, best_pdb_path)
    
    return (pdb_name, plddt, rmsd, is_cyclic, len(ss_pairs)//2)

def extract_results_from_output_dir(output_dir, input_dir=None):
    """
    Extract results from existing prediction output directories.
    Scans output_dir for PDB subdirectories, finds job directories, and extracts results.
    
    Args:
        output_dir: Directory containing prediction results (with subdirs for each PDB)
        input_dir: Optional directory with original PDB files for RMSD calculation.
                   Can be the same as output_dir (will only search for files, not subdirs).
    
    Returns:
        List of tuples: (pdb_name, plddt, rmsd, is_cyclic, num_ss)
    """
    results = []
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        return results
    
    # Get all subdirectories in output_dir (each should be a PDB name)
    subdirs = [d for d in os.listdir(output_dir) 
               if os.path.isdir(os.path.join(output_dir, d))]
    
    print(f"Scanning {len(subdirs)} subdirectories in {output_dir}...")
    
    for pdb_name in subdirs:
        pdb_dir = os.path.join(output_dir, pdb_name)
        
        # Find job directories (job_xxx)
        job_dirs = glob.glob(os.path.join(pdb_dir, "job_*"))
        if not job_dirs:
            print(f"Warning: No job directories found in {pdb_dir}, skipping {pdb_name}")
            continue
        
        # Use the first job directory (or could use the most recent one)
        job_dir = job_dirs[0]
        if len(job_dirs) > 1:
            # Use the most recently modified one
            job_dir = max(job_dirs, key=os.path.getmtime)
        
        # Get prediction results
        plddt, best_pdb_path = get_best_plddt_and_structure(job_dir)
        if plddt is None or best_pdb_path is None:
            print(f"Warning: Could not extract results from {job_dir}, skipping {pdb_name}")
            continue
        
        # Get reference structure info for RMSD calculation
        rmsd = None
        is_cyclic = False
        num_ss = 0
        
        if input_dir:
            # Try to find original PDB file
            # First try with .pdb extension
            ref_pdb_path = os.path.join(input_dir, f"{pdb_name}.pdb")
            if os.path.exists(ref_pdb_path) and os.path.isfile(ref_pdb_path):
                # Found it
                pass
            else:
                # Try without .pdb extension
                ref_pdb_path = os.path.join(input_dir, pdb_name)
                if os.path.exists(ref_pdb_path) and os.path.isfile(ref_pdb_path):
                    # Found it
                    pass
                else:
                    # Try case-insensitive search
                    ref_pdb_path = None
                    if os.path.exists(input_dir):
                        for f in os.listdir(input_dir):
                            f_path = os.path.join(input_dir, f)
                            # Only check files, not directories
                            if os.path.isfile(f_path):
                                f_base = os.path.splitext(f)[0]
                                if f_base.lower() == pdb_name.lower():
                                    ref_pdb_path = f_path
                                    break
        else:
            ref_pdb_path = None
        
        if ref_pdb_path:
            # Get structure info and calculate RMSD
            info = get_structure_info(ref_pdb_path)
            if info:
                seq, ss_pairs, is_cyclic, ref_ca_atoms = info
                num_ss = len(ss_pairs) // 2
                rmsd = calculate_rmsd(ref_ca_atoms, best_pdb_path)
            else:
                print(f"Warning: Could not parse reference PDB {ref_pdb_path}, RMSD will be None")
        else:
            # Try to get structure info from the predicted PDB itself
            # This won't give us RMSD, but we can get cyclic and SS info
            # Actually, we can't get cyclic/SS from predicted structure reliably
            # So we'll just set defaults
            print(f"Warning: Reference PDB not found for {pdb_name}, RMSD will be None")
        
        results.append((pdb_name, plddt, rmsd, is_cyclic, num_ss))
        rmsd_str = f"{rmsd:.2f}" if rmsd is not None else "N/A"
        print(f"Extracted {pdb_name}: PLDDT={plddt:.2f}, RMSD={rmsd_str}, Cyclic={is_cyclic}, SS={num_ss}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate HighFold on Knottin PDBs")
    parser.add_argument("--input_dir", required=False, help="Directory containing input PDB files (not needed in --plot-only mode)")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--gpus", default="0", help="Comma separated list of GPU IDs (e.g. 0,1,2)")
    parser.add_argument("--jobs_per_gpu", type=int, default=1, help="Number of concurrent jobs per GPU")
    parser.add_argument("--msa-mode", default="single_sequence", 
                       choices=["single_sequence", "mmseqs2_uniref_env", "mmseqs2_uniref"],
                       help="MSA mode: 'single_sequence' (no MSA, fast), 'mmseqs2_uniref_env' (with MSA, slower but more accurate), 'mmseqs2_uniref' (UniRef only)")
    parser.add_argument("--msa-threads", type=int, default=None,
                       help="Number of threads for MSA search (MMseqs2) per job. If not set, uses all available CPU cores. Only applies when using MSA modes.")
    parser.add_argument("--max-msa", type=str, default=None,
                       help="MSA depth in format 'max-seq:max-extra-seq' (e.g., '512:5120'). Default is ColabFold default (typically '512:5120' for AlphaFold2). max-seq: sequence clusters, max-extra-seq: extra sequences. Only applies when using MSA modes.")
    parser.add_argument("--plot-only", action="store_true",
                       help="Skip prediction and only generate plots from existing results.csv file")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If plot-only mode, extract results from existing prediction directories
    if args.plot_only:
        # First try to load from existing CSV file
        csv_path = os.path.join(args.output_dir, "results.csv")
        if os.path.exists(csv_path):
            print(f"Found existing CSV file at {csv_path}")
            print("Loading results from CSV file...")
            results = []
            with open(csv_path, "r") as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    # Skip header line
                    for line in lines[1:]:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(',')
                        if len(parts) >= 5:
                            pdb_name = parts[0]
                            try:
                                plddt = float(parts[1])
                                rmsd_str = parts[2].strip()
                                rmsd = float(rmsd_str) if rmsd_str and rmsd_str.lower() != 'none' else None
                                is_cyclic = parts[3].lower() in ('true', '1', 'yes')
                                num_ss = int(parts[4])
                                results.append((pdb_name, plddt, rmsd, is_cyclic, num_ss))
                            except ValueError as e:
                                print(f"Warning: Skipping invalid line: {line} (Error: {e})")
                                continue
            
            if results:
                print(f"Loaded {len(results)} results from CSV file.")
            else:
                print("CSV file exists but contains no valid data. Extracting from output directories...")
                results = extract_results_from_output_dir(args.output_dir, args.input_dir)
        else:
            # No CSV file, extract from output directories
            print(f"No existing CSV file found. Extracting results from output directories...")
            if not args.input_dir:
                print("Warning: --input_dir not provided. RMSD calculation will be skipped.")
            results = extract_results_from_output_dir(args.output_dir, args.input_dir)
        
        if not results:
            print("Error: No results found. Please check the output directory structure.")
            sys.exit(1)
        
        # Save/update CSV file with extracted results
        csv_path = os.path.join(args.output_dir, "results.csv")
        with open(csv_path, "w") as f:
            f.write("PDB,pLDDT,RMSD,IsCyclic,NumSS\n")
            for res in results:
                rmsd_str = f"{res[2]:.4f}" if res[2] is not None else "None"
                f.write(f"{res[0]},{res[1]:.2f},{rmsd_str},{res[3]},{res[4]}\n")
        
        print(f"Results saved to {csv_path}")
    else:
        # Normal prediction mode
        if not args.input_dir:
            print("Error: --input_dir is required when not using --plot-only mode")
            sys.exit(1)
        
        if not os.path.exists(COLABFOLD_BIN):
            print(f"Error: ColabFold binary not found at {COLABFOLD_BIN}")
            sys.exit(1)
        
        pdb_files = glob.glob(os.path.join(args.input_dir, "*.pdb"))
        if not pdb_files:
            print(f"No PDB files found in {args.input_dir}")
            sys.exit(0)
            
        print(f"Found {len(pdb_files)} PDBs.")
        
        # Setup GPU queue
        gpu_ids = args.gpus.split(',')
        manager = multiprocessing.Manager()
        gpu_queue = manager.Queue()
        for gid in gpu_ids:
            for _ in range(args.jobs_per_gpu):
                gpu_queue.put(gid.strip())
            
        # Prepare tasks
        tasks = [(f, args.output_dir, gpu_queue, args.msa_mode, args.msa_threads, args.max_msa) for f in pdb_files]
        
        # Run parallel
        # Limit processes to number of GPUs to avoid contention, or use more?
        # ColabFold takes significant VRAM, best to limit to 1 job per GPU.
        num_workers = len(gpu_ids) * args.jobs_per_gpu
        
        print(f"Starting processing with {num_workers} workers on GPUs {args.gpus} (jobs per gpu: {args.jobs_per_gpu})...")
        
        results = []
        with multiprocessing.Pool(num_workers) as pool:
            for res in pool.imap_unordered(process_pdb, tasks):
                if res:
                    rmsd_str = f"{res[2]:.2f}" if res[2] is not None else "N/A"
                    print(f"Finished {res[0]}: PLDDT={res[1]:.2f}, RMSD={rmsd_str}, Cyclic={res[3]}, SS={res[4]}")
                    results.append(res)
                else:
                    print("Failed to process a PDB")
        
        # Explicitly close and join the pool to ensure all processes are finished
        print("All tasks completed. Closing pool...")
        
        if not results:
            print("No results generated.")
            return
        
        # Save CSV after prediction
        csv_path = os.path.join(args.output_dir, "results.csv")
        with open(csv_path, "w") as f:
            f.write("PDB,pLDDT,RMSD,IsCyclic,NumSS\n")
            for res in results:
                rmsd_str = f"{res[2]:.4f}" if res[2] is not None else "None"
                f.write(f"{res[0]},{res[1]:.2f},{rmsd_str},{res[3]},{res[4]}\n")
                
        print(f"Results saved to {csv_path}")

    # Plotting
    print(f"Generating plots for {len(results)} results...")
    names, plddts, rmsds, cyclics, ss_counts = zip(*results)
    
    # Filter out None RMSD values for plotting
    valid_indices = [i for i, rmsd in enumerate(rmsds) if rmsd is not None]
    if not valid_indices:
        print("Warning: No valid RMSD values found. Cannot generate scatter plot.")
        print("Results CSV has been saved, but plot generation skipped.")
        return
    
    valid_rmsds = [rmsds[i] for i in valid_indices]
    valid_plddts = [plddts[i] for i in valid_indices]
    
    if len(valid_indices) < len(results):
        print(f"Warning: {len(results) - len(valid_indices)} results have missing RMSD values and will be excluded from plot.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid_rmsds, valid_plddts, c='blue', alpha=0.6)
    ax.set_xlabel("RMSD (Angstrom)")
    ax.set_ylabel("pLDDT")
    ax.set_title("HighFold Prediction Accuracy on Knottins")
    ax.grid(True)
    
    # Annotate points? Maybe too cluttered if many.
    # Optionally color by cyclic or ss_count
    
    plot_path = os.path.join(args.output_dir, "plddt_vs_rmsd.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Explicitly close figure to release resources
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()

