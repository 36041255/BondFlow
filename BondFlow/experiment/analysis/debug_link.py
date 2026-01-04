
import os
import pyrosetta
from pyrosetta import *

pyrosetta.init("-ignore_unrecognized_res true -mute all")

pdb_path = "BondFlow/experiment/analysis/test_case/MDM2_head_tail_disuful.pdb"
pose = pose_from_file(pdb_path)

print(f"Pose loaded. Residues: {pose.total_residue()}")
print(f"Info for res 1: {pose.pdb_info().chain(1)} {pose.pdb_info().number(1)}")

def find_residue_index(pose, chain, res_seq):
    if chain:
        p = pose.pdb_info().pdb2pose(chain, res_seq)
        if p != 0: return p
    return 0

with open(pdb_path, 'r') as f:
    for line in f:
        if line.startswith("LINK"):
            print(f"Found LINK: {line.strip()}")
            a1 = line[12:16].strip()
            c1 = line[21].strip() if line[21].strip() else line[20].strip()
            r1 = int(line[22:26].strip())
            
            a2 = line[42:46].strip()
            c2 = line[51].strip() if line[51].strip() else line[50].strip()
            r2 = int(line[52:56].strip())
            
            print(f"Parsed: {c1}.{r1}.{a1} - {c2}.{r2}.{a2}")
            
            p1 = find_residue_index(pose, c1, r1)
            p2 = find_residue_index(pose, c2, r2)
            print(f"Mapped to Pose indices: {p1} - {p2}")
            
            if p1 and p2:
                res1 = pose.residue(p1)
                res2 = pose.residue(p2)
                bonded = res1.is_bonded(res2)
                print(f"Is bonded currently? {bonded}")


