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
# Assuming the script is in BondFlow/experiment/analysis/
# HighFold3 is in ../../HighFold3/ relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT ="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/"
HIGHFOLD3_DIR = os.path.join(PROJECT_ROOT, "HighFold3")
HIGHFOLD3_SCRIPT = os.path.join(HIGHFOLD3_DIR, "run_alphafold.py")

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
    # Use greedy algorithm to ensure each CYS participates in only one disulfide bond
    # IMPORTANT: HighFold3 expects sequence positions (1-based), not PDB residue numbers!
    # The sequence in JSON is built from std_residues in order, so we need positions in that sequence
    ss_pairs = []
    
    # Create mapping from residue object to its position in std_residues (sequence position)
    residue_to_seqpos = {r: idx + 1 for idx, r in enumerate(std_residues)}
    # +1 because HighFold3 uses 1-based indexing for sequence positions
    
    # Find all CYS residues with their sequence positions
    cys_residues = [(r, residue_to_seqpos[r]) for r in std_residues if r.get_resname() == 'CYS']
    
    # Collect all possible CYS pairs with distances
    candidate_pairs = []
    for i in range(len(cys_residues)):
        for j in range(i+1, len(cys_residues)):
            res1, seqpos1 = cys_residues[i]
            res2, seqpos2 = cys_residues[j]
            
            # Skip adjacent residues (seqpos difference = 1) - they rarely form disulfide bonds
            # and might be false positives due to close proximity in 3D space
            if abs(seqpos1 - seqpos2) == 1:
                continue
            
            if 'SG' in res1 and 'SG' in res2:
                d = res1['SG'] - res2['SG']
                if d < 3.0: 
                    # Store as (distance, seqpos1, seqpos2) for sorting
                    # seqpos1 and seqpos2 are 1-based sequence positions
                    candidate_pairs.append((d, seqpos1, seqpos2))
    
    # Sort by distance (closest first) and greedily assign pairs
    # Ensure each CYS residue participates in only one disulfide bond
    candidate_pairs.sort(key=lambda x: x[0])  # Sort by distance
    used_seqpos = set()
    for d, seqpos1, seqpos2 in candidate_pairs:
        # Only add if neither residue is already paired
        if seqpos1 not in used_seqpos and seqpos2 not in used_seqpos:
            # seqpos1 and seqpos2 are 1-based sequence positions (what HighFold3 expects)
            ss_pairs.extend([seqpos1, seqpos2])
            used_seqpos.add(seqpos1)
            used_seqpos.add(seqpos2)
                    
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

def create_json_input(seq, output_path, name="target"):
    """
    Creates a HighFold3 JSON input file.
    """
    json_data = {
        "name": name,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": seq,
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": []
                }
            }
        ],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1
    }
    
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    return output_path

def run_highfold3(seq, ss_pairs, is_cyclic, output_dir, gpu_id, model_dir, 
                  num_recycles=5, num_diffusion_samples=3, run_data_pipeline=False,
                  xla_mem_fraction=None):
    """
    Runs HighFold3 for a single sequence.
    """
    # Create a unique job ID/dir
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(output_dir, f"job_{job_id}")
    os.makedirs(job_dir, exist_ok=True)
    
    # Convert to absolute paths (required because run_alphafold.py runs with cwd=HIGHFOLD3_DIR)
    job_dir = os.path.abspath(job_dir)
    
    # Create JSON input
    json_path = os.path.join(job_dir, "input.json")
    create_json_input(seq, json_path, name=f"target_{job_id}")
    
    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        HIGHFOLD3_SCRIPT,
        "--json_path", json_path,
        "--output_dir", job_dir,
        "--model_dir", model_dir,
        "--gpu_device", str(gpu_id),
        "--num_recycles", str(num_recycles),
        "--num_diffusion_samples", str(num_diffusion_samples),
        "--run_data_pipeline", str(run_data_pipeline).lower(),  # MSA pipeline
        "--run_inference", "True",
    ]
    
    # Add disulfide bonds
    # HighFold3 format: --disulfide_chain_res [[1,3,5,7,9]] for chain 1, 
    # residues 3-5 form one bond, 7-9 form another bond
    if ss_pairs:
        # Convert [i1, j1, i2, j2, ...] to [[1, i1, j1, i2, j2, ...]]
        # All pairs in one list for chain 1
        ss_list = [1] + ss_pairs  # Chain 1 followed by all residue pairs
        if ss_list:
            cmd.append("--disulfide_chain_res")
            cmd.append(str([ss_list]))  # Wrap in outer list
    
    # Add cyclic constraint
    # HighFold3 uses absl flags, boolean flags can be set with --flag or --noflag
    # Or pass as string "true"/"false"
    if is_cyclic:
        cmd.append("--head_to_tail=true")
    else:
        cmd.append("--head_to_tail=false")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Set XLA memory optimization environment variables
    # These are critical for managing GPU memory usage
    if "XLA_FLAGS" not in env:
        env["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
    if "XLA_PYTHON_CLIENT_PREALLOCATE" not in env:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    
    # XLA_CLIENT_MEM_FRACTION: use the provided value (already calculated in main())
    if xla_mem_fraction is not None:
        env["XLA_CLIENT_MEM_FRACTION"] = str(xla_mem_fraction)
    elif "XLA_CLIENT_MEM_FRACTION" not in env:
        # Fallback: should not reach here if auto-calculation works correctly
        env["XLA_CLIENT_MEM_FRACTION"] = "0.124"
    
    print(f"Running command: {' '.join(cmd)}")
    # Run command
    try:
        # Capture output to avoid spamming console
        result = subprocess.run(
            cmd, 
            env=env, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=HIGHFOLD3_DIR  # Run from HighFold3 directory
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running HighFold3 for job {job_id}: {e.stderr.decode()}")
        return None
        
    return job_dir

def calculate_rmsd(ref_ca_atoms, pred_pdb_path):
    """
    Aligns predicted structure to reference CA atoms and calculates RMSD.
    Supports both PDB and mmCIF formats.
    """
    parser = PDB.PDBParser(QUIET=True)
    mmcif_parser = PDB.MMCIFParser(QUIET=True)
    
    try:
        # Try mmCIF first (HighFold3 default), then PDB
        if pred_pdb_path.endswith('.cif'):
            structure = mmcif_parser.get_structure('pred', pred_pdb_path)
        else:
            structure = parser.get_structure('pred', pred_pdb_path)
    except Exception as e:
        print(f"Error parsing {pred_pdb_path}: {e}")
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
    Finds the best model based on ranking score.
    HighFold3 outputs in a subdirectory named {name}_{timestamp}/:
    - ranking_scores.csv with scores for each sample
    - seed-{seed}_sample-{sample}/model.cif for each sample
    - {name}_model.cif in the root (best model)
    """
    # HighFold3 creates a subdirectory with name and timestamp
    # Find the subdirectory
    if not os.path.exists(job_dir):
        return None, None
    
    try:
        subdirs = [d for d in os.listdir(job_dir) 
                   if os.path.isdir(os.path.join(job_dir, d)) and not d.startswith('seed-')]
        if not subdirs:
            # Try looking for files directly in job_dir
            actual_dir = job_dir
        else:
            # Use the first subdirectory (should be only one)
            actual_dir = os.path.join(job_dir, subdirs[0])
    except OSError:
        actual_dir = job_dir
    
    # Check for ranking_scores.csv
    ranking_csv = os.path.join(actual_dir, "ranking_scores.csv")
    best_sample = None
    best_score = -1
    
    if os.path.exists(ranking_csv):
        with open(ranking_csv, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    seed = parts[0]
                    sample = parts[1]
                    score = float(parts[2])
                    if score > best_score:
                        best_score = score
                        best_sample = (seed, sample)
    
    # Try to find the best model file
    best_pdb = None
    if best_sample:
        seed, sample = best_sample
        sample_dir = os.path.join(actual_dir, f"seed-{seed}_sample-{sample}")
        model_cif = os.path.join(sample_dir, "model.cif")
        if os.path.exists(model_cif):
            best_pdb = model_cif
    
    # Fallback: look for root model.cif or any model.cif
    if not best_pdb:
        # Check root directory for {name}_model.cif
        root_models = glob.glob(os.path.join(actual_dir, "*_model.cif"))
        if root_models:
            best_pdb = root_models[0]
        else:
            # Find any model.cif in sample directories
            sample_dirs = glob.glob(os.path.join(actual_dir, "seed-*_sample-*"))
            for sample_dir in sorted(sample_dirs):
                model_cif = os.path.join(sample_dir, "model.cif")
                if os.path.exists(model_cif):
                    best_pdb = model_cif
                    break
    
    if not best_pdb:
        return None, None
    
    # Extract PLDDT from confidences.json
    # Try to find confidences.json in the same directory as model.cif
    plddt = None
    confidences_json = best_pdb.replace("model.cif", "confidences.json")
    if os.path.exists(confidences_json):
        try:
            with open(confidences_json, "r") as f:
                conf_data = json.load(f)
                if "atom_plddts" in conf_data:
                    plddts = conf_data["atom_plddts"]
                    # Filter for CA atoms only (every Nth atom, depends on structure)
                    # For simplicity, take average of all plddts
                    plddt = np.mean(plddts)
        except Exception as e:
            print(f"Warning: Could not read confidences.json: {e}")
    
    # Fallback: try to extract from model.cif B-factors
    if plddt is None:
        try:
            mmcif_parser = PDB.MMCIFParser(QUIET=True)
            structure = mmcif_parser.get_structure('best', best_pdb)
            plddts = []
            for atom in structure.get_atoms():
                if atom.name == 'CA':
                    plddts.append(atom.bfactor)
            if plddts:
                plddt = np.mean(plddts)
        except Exception as e:
            print(f"Warning: Could not extract pLDDT from B-factors: {e}")
    
    return plddt, best_pdb

def process_pdb(args):
    """
    Worker function to process a single PDB.
    """
    pdb_path, output_root, gpu_queue, model_dir, num_recycles, num_diffusion_samples, run_data_pipeline, xla_mem_fraction = args
    
    # 1. Parse Info
    info = get_structure_info(pdb_path)
    if not info:
        return None
    seq, ss_pairs, is_cyclic, ref_ca_atoms = info
    
    pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
    output_dir = os.path.join(output_root, pdb_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Get GPU
    gpu_id = gpu_queue.get()
    try:
        # 3. Run HighFold3
        job_dir = run_highfold3(
            seq, ss_pairs, is_cyclic, output_dir, gpu_id, 
            model_dir, num_recycles, num_diffusion_samples, run_data_pipeline, xla_mem_fraction
        )
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
    parser = argparse.ArgumentParser(description="Evaluate HighFold3 on Knottin PDBs")
    parser.add_argument("--input_dir", required=True, help="Directory containing input PDB files")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--model_dir", required=True, help="Path to HighFold3 model directory")
    parser.add_argument("--gpus", default="0", help="Comma separated list of GPU IDs (e.g. 0,1,2)")
    parser.add_argument("--jobs_per_gpu", type=int, default=1, help="Number of concurrent jobs per GPU")
    parser.add_argument("--num_recycles", type=int, default=5, help="Number of recycles for inference")
    parser.add_argument("--num_diffusion_samples", type=int, default=3, help="Number of diffusion samples")
    parser.add_argument("--run_data_pipeline", action="store_true", help="Run MSA data pipeline (slower but more accurate)")
    parser.add_argument("--xla_mem_fraction", type=float, default=None, 
                       help="XLA memory fraction (0.0-1.0). If not specified, automatically calculated "
                            "based on --jobs_per_gpu (0.75/jobs_per_gpu). Can be overridden by environment variable.")
    
    args = parser.parse_args()
    
    # Auto-calculate XLA memory fraction based on jobs_per_gpu if not specified
    # Formula: 0.75 / jobs_per_gpu, with a minimum of 0.1 and maximum of 0.75
    if args.xla_mem_fraction is None:
        if "XLA_CLIENT_MEM_FRACTION" not in os.environ:
            # Auto-calculate: use 0.75 for single job, divide by jobs_per_gpu for multiple jobs
            auto_mem_fraction = min(0.75, max(0.1, 0.75 / args.jobs_per_gpu))
            args.xla_mem_fraction = auto_mem_fraction
            print(f"Auto-setting XLA_CLIENT_MEM_FRACTION={auto_mem_fraction:.3f} based on --jobs_per_gpu={args.jobs_per_gpu}")
        else:
            # Use environment variable value
            env_value = float(os.environ["XLA_CLIENT_MEM_FRACTION"])
            args.xla_mem_fraction = env_value
            print(f"Using XLA_CLIENT_MEM_FRACTION={env_value:.3f} from environment variable")
    else:
        print(f"Using XLA_CLIENT_MEM_FRACTION={args.xla_mem_fraction:.3f} from command line argument")
    
    if not os.path.exists(HIGHFOLD3_SCRIPT):
        print(f"Error: HighFold3 script not found at {HIGHFOLD3_SCRIPT}")
        sys.exit(1)
        
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found at {args.model_dir}")
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
    tasks = [
        (f, args.output_dir, gpu_queue, args.model_dir, 
         args.num_recycles, args.num_diffusion_samples, args.run_data_pipeline, args.xla_mem_fraction) 
        for f in pdb_files
    ]
    
    # Run parallel
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
    plt.title("HighFold3 Prediction Accuracy on Knottins")
    plt.grid(True)
    
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

