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
import matplotlib.pyplot as plt
from Bio import PDB
from Bio.SeqUtils import seq1

# Configuration
# Assuming the script is in BondFlow/experiment/
# HighFold2 is in ../../HighFold2/ relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
COLABFOLD_BIN = os.path.join(PROJECT_ROOT, "HighFold2/localcolabfold/colabfold-conda/bin/colabfold_batch")

def get_structure_info(pdb_path):
    """
    Parses PDB to get sequence, disulfide bonds, and cyclization status.
    Returns:
        seq (str): Amino acid sequence
        ss_pairs (list): List of 1-based indices [i1, j1, i2, j2, ...]
        is_cyclic (bool): True if N-term and C-term are connected
        ca_coords (list): List of CA coordinates for RMSD calc
    """
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
    
    for r in chain.get_residues():
        # Only consider standard amino acids (skip hetatms/waters)
        if PDB.is_aa(r, standard=True):
            std_residues.append(r)
            # bio.sequtils.seq1 converts 'ALA' to 'A'
            seq += seq1(r.get_resname())
            
    if not seq:
        return None

    # Disulfide detection (SG-SG < 3.0 A)
    ss_pairs = []
    cys_indices = [i for i, r in enumerate(std_residues) if r.get_resname() == 'CYS']
    
    for i in range(len(cys_indices)):
        for j in range(i+1, len(cys_indices)):
            idx1 = cys_indices[i]
            idx2 = cys_indices[j]
            res1 = std_residues[idx1]
            res2 = std_residues[idx2]
            
            if 'SG' in res1 and 'SG' in res2:
                d = res1['SG'] - res2['SG']
                if d < 3.0: 
                    # 1-based indices
                    ss_pairs.extend([idx1 + 1, idx2 + 1])
                    
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

def run_highfold(seq, ss_pairs, is_cyclic, output_dir, gpu_id):
    """
    Runs colabfold_batch for a single sequence.
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
        "--msa-mode", "single_sequence",
        # "--model-type", "alphafold2_multimer_v3" if len(seq) > 0 else "alphafold2", 
        # "--amber"
    ]
    
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
    """
    # Pattern: *_rank_001_*.pdb or *_rank_1_*.pdb depending on version
    # Modern colabfold: input_rank_001_alphafold2_...pdb
    pdbs = glob.glob(os.path.join(job_dir, "*_rank_001_*.pdb"))
    if not pdbs:
        pdbs = glob.glob(os.path.join(job_dir, "*_rank_1_*.pdb"))
        
    if not pdbs:
        return None, None
        
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
    pdb_path, output_root, gpu_queue = args
    
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
        job_dir = run_highfold(seq, ss_pairs, is_cyclic, output_dir, gpu_id)
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate HighFold on Knottin PDBs")
    parser.add_argument("--input_dir", required=True, help="Directory containing input PDB files")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--gpus", default="0", help="Comma separated list of GPU IDs (e.g. 0,1,2)")
    parser.add_argument("--jobs_per_gpu", type=int, default=1, help="Number of concurrent jobs per GPU")
    
    args = parser.parse_args()
    
    if not os.path.exists(COLABFOLD_BIN):
        print(f"Error: ColabFold binary not found at {COLABFOLD_BIN}")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    tasks = [(f, args.output_dir, gpu_queue) for f in pdb_files]
    
    # Run parallel
    # Limit processes to number of GPUs to avoid contention, or use more?
    # ColabFold takes significant VRAM, best to limit to 1 job per GPU.
    num_workers = len(gpu_ids) * args.jobs_per_gpu
    
    print(f"Starting processing with {num_workers} workers on GPUs {args.gpus} (jobs per gpu: {args.jobs_per_gpu})...")
    
    results = []
    with multiprocessing.Pool(num_workers) as pool:
        for res in pool.imap_unordered(process_pdb, tasks):
            if res:
                print(f"Finished {res[0]}: PLDDT={res[1]:.2f}, RMSD={res[2]:.2f}, Cyclic={res[3]}, SS={res[4]}")
                results.append(res)
            else:
                print("Failed to process a PDB")
                
    if not results:
        print("No results generated.")
        return

    # Plotting
    names, plddts, rmsds, cyclics, ss_counts = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(rmsds, plddts, c='blue', alpha=0.6)
    plt.xlabel("RMSD (Angstrom)")
    plt.ylabel("pLDDT")
    plt.title("HighFold Prediction Accuracy on Knottins")
    plt.grid(True)
    
    # Annotate points? Maybe too cluttered if many.
    # Optionally color by cyclic or ss_count
    
    plot_path = os.path.join(args.output_dir, "plddt_vs_rmsd.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Save CSV
    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w") as f:
        f.write("PDB,pLDDT,RMSD,IsCyclic,NumSS\n")
        for res in results:
            f.write(f"{res[0]},{res[1]:.2f},{res[2]:.4f},{res[3]},{res[4]}\n")
            
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()

