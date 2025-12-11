import numpy as np
import os
import re
from omegaconf import DictConfig
import torch
import torch.nn.functional as nn
import networkx as nx
from Bio.PDB import MMCIFParser, MMCIF2Dict
from scipy.spatial.transform import Rotation as scipy_R
from rfdiff.util import rigid_from_3_points
from rfdiff import util
from rfdiff import chemical as che
import random
import logging

# openfold_get_torsions 依赖于额外的 OpenFold/APM/mdtraj 等包。
# 为了让诸如 parse_cif_structure 这类轻量函数在没有这些依赖时也能正常使用，
# 这里使用 try/except 延迟失败：只有真正需要扭转角时才报错。
try:
    from BondFlow.models.allatom_wrapper import openfold_get_torsions
except Exception:
    openfold_get_torsions = None

from rfdiff.chemical import one_aa2num,aa2num, num2aa, aa_321, aa_123, aabonds, aa2long
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
import pandas as pd
from .link_utils import load_allowed_bonds_from_csv
import networkit as nk
import BondFlow.data.SM_utlis as smu
import math

###########################################################
#### Functions which can be called outside of Denoiser ####
###########################################################
TOR_INDICES  = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES   = util.reference_angles
STANDARD_AMINO_ACIDS = num2aa
MAIN_CHAIN_ATOMS = {'N', 'CA', 'C', 'O', 'OXT'}
BOND_LENGTH_THRESHOLD = 5.0

# --- Constants for atom names ---
ATOM_CA = "CA"
ATOM_C = "C"
ATOM_N = "N"
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


# @torch.no_grad()
# def diffusion_distance_tensor(
#     A_adj_batch: torch.Tensor,            # (B, L, L)
#     node_mask: torch.Tensor,              # (B, L)
#     times: Sequence[int],                 # e.g., (0,1,4)
#     k: int = 256,                          # if -1: use all components except the first when skip_top=True
#     skip_top: bool = True,
#     eps: float = 1e-12,
#     rbf_num: int = 0,                     # if >0, return RBF-embedded distances with this feature dimension
#     rbf_gamma: Optional[float] = None,    # if None, auto-set based on spacing
#     energy_thresh: Optional[float] = None,# per-sample adaptive k based on spectral energy if set
#     k_ratio: Optional[float] = None,      # per-sample adaptive k based on a ratio of available components
#     t_ref: Optional[int] = None,          # reference time for energy (default=max(times))
# ) -> torch.Tensor:
#     """Compute diffusion distances at multiple times.

#     Args:
#       k: number of eigencomponents to use; if -1, use all available (except the top one when skip_top=True).
#       rbf_num: number of RBF centers in [0,1]; if 0, return raw distances per t.
#       rbf_gamma: Gaussian width; default auto = 1/(2*delta^2) with delta=1/(rbf_num-1).
#       energy_thresh: pick smallest k per sample to reach this spectral energy (|lambda|^(2*t_ref)).
#       k_ratio: pick smallest k per sample based on a ratio of available components. Overrides energy_thresh.
#       t_ref: reference time for energy threshold; default max(times) if None.

#     Returns:
#       If rbf_num == 0: (B, L, L, T) each slice is D_t in [0,1] (normalized per-batch).
#       If rbf_num > 0: (B, L, L, T*rbf_num) RBF features per time.
#     """
#     from BondFlow.data.utils import diffusion_map_pair_features

#     assert A_adj_batch.dim() == 3 and A_adj_batch.shape[-1] == A_adj_batch.shape[-2]
#     device = A_adj_batch.device
#     dtype = A_adj_batch.dtype
#     B, L, _ = A_adj_batch.shape
#     # Ensure node_mask on same device
#     node_mask = node_mask.to(torch.bool).to(device)

#     # Determine effective k for the batch
#     if int(k) == -1:
#         start = 1 if skip_top else 0
#         n_subs = node_mask.sum(dim=1)                     # (B,)
#         k_eff = int(n_subs.max().item()) - start         # max valid per-batch minus skipped top
#         k_eff = max(k_eff, 1)
#     else:
#         k_eff = int(k)

#     # Reuse eigensolver to get (lam, U) with correct masking/sorting
#     _, (lam_all, U_all) = diffusion_map_pair_features(
#         A_adj_batch, times=times, k=k_eff, skip_top=skip_top, node_mask=node_mask
#     )
#     # lam_all: (B, k_eff), U_all: (B, L, k_eff)

#     mask_2d = (node_mask[:, :, None] & node_mask[:, None, :])  # (B,L,L)
    
#     # Per-sample adaptive k selection
#     if k_ratio is not None:
#         # Select k based on a fixed ratio of available components
#         if energy_thresh is not None:
#             print("Warning: Both k_ratio and energy_thresh are provided. k_ratio will take precedence.")
        
#         total_avail = (node_mask.sum(dim=1) - (1 if skip_top else 0)).clamp(min=1)
#         # Use ceil to ensure at least 1 component is selected for ratio > 0
#         k_sel = torch.ceil(total_avail.float() * float(k_ratio)).long().clamp(min=1, max=lam_all.size(1))

#         # print(f"\n--- Adaptive k selection based on ratio: {k_ratio:.2f} ---")
#         # for i in range(A_adj_batch.shape[0])[:5]:
#         #     L_i = int(node_mask[i].sum().item())
#         #     total_i = int(total_avail[i].item())
#         #     k_sel_i = int(k_sel[i].item())
#         #     ratio_i = k_sel_i / total_i if total_i > 0 else 0.0
#         #     print(f"Sample {i:2d} (L={L_i:3d}): Selected k={k_sel_i:3d}, Total available={total_i:3d}, Ratio={ratio_i:.4f}")
#         # print("--- End of adaptive k report ---\n")

#         idx_range = torch.arange(lam_all.size(1), device=device).unsqueeze(0)  # (1,k)
#         comp_mask = (idx_range < k_sel.unsqueeze(1)).to(lam_all.dtype)        # (B,k)

#         lam_all = lam_all * comp_mask
#         U_all = U_all * comp_mask.unsqueeze(1)

#     # Per-sample adaptive k via spectral energy threshold (optional)
#     elif energy_thresh is not None:
#         # Special case for energy_thresh=1.0: select all available components
#         # if abs(float(energy_thresh) - 1.0) < 1e-6:
#         #     total_avail = (node_mask.sum(dim=1) - (1 if skip_top else 0)).clamp(min=1)
#         #     k_sel = total_avail  # Select all available components

#         #     # ratio_vs_total = (k_sel.float() / total_avail.float())
#         #     # print("\n--- Adaptive k selection (energy_thresh=1.0) ---")
#         #     # print("Selecting all available components for each sample.")
#         #     # for i in range(A_adj_batch.shape[0]):
#         #     #     L_i = int(node_mask[i].sum().item())
#         #     #     total_i = int(total_avail[i].item())
#         #     #     k_sel_i = int(k_sel[i].item())
#         #     #     ratio_i = ratio_vs_total[i].item()
#         #     #     print(f"Sample {i:2d} (L={L_i:3d}): Selected k={k_sel_i:3d}, Total available={total_i:3d}, Ratio={ratio_i:.4f}")
#         #     # print("--- End of adaptive k report ---\n")

#         # else:
#         tref = int(max(times) if t_ref is None else t_ref)
#         w = lam_all.abs().pow(2 * tref)                        # (B,k_eff)
        
#         # For padded components, lambda can be zero. Mask them out from energy calculation.
#         idx_range_w = torch.arange(w.size(1), device=device).unsqueeze(0) # (1, k_eff)
#         valid_comps = (node_mask.sum(dim=1, keepdim=True) - (1 if skip_top else 0)) # (B, 1)
#         w_mask = (idx_range_w < valid_comps).to(w.dtype)
#         w = w * w_mask

#         denom = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
#         ratio = torch.cumsum(w, dim=1) / denom                 # (B,k_eff)
        
#         effective_thresh = float(energy_thresh)

#         hits = ratio >= effective_thresh
#         # default to all components if never reaches threshold
#         first_idx = torch.where(hits.any(dim=1), hits.float().argmax(dim=1), torch.full((B,), lam_all.size(1) - 1, device=device, dtype=torch.long))
#         k_sel = torch.clamp(first_idx + 1, min=1, max=lam_all.size(1))  # (B,)
        
#         total_avail = (node_mask.sum(dim=1) - (1 if skip_top else 0)).clamp(min=1)        # (B,)
#         ratio_vs_total = (k_sel.float() / total_avail.float())                # (B,)

#         # print("\n--- Adaptive k selection based on energy threshold ---")
#         # print(f"Energy threshold: {energy_thresh}, effective: {effective_thresh:.6f}")
#         # print(f"Reference time t_ref: {tref}")
#         # for i in range(A_adj_batch.shape[0])[:5]:
#         #     L_i = int(node_mask[i].sum().item())
#         #     total_i = int(total_avail[i].item())
#         #     k_sel_i = int(k_sel[i].item())
#         #     ratio_i = ratio_vs_total[i].item()
#         #     print(f"Sample {i:2d} (L={L_i:3d}): Selected k={k_sel_i:3d}, Total available={total_i:3d}, Ratio={ratio_i:.4f}")
#         # print("--- End of adaptive k report ---\n")

#         idx_range = torch.arange(lam_all.size(1), device=device).unsqueeze(0)  # (1,k)
#         comp_mask = (idx_range < k_sel.unsqueeze(1)).to(lam_all.dtype)        # (B,k)

#         lam_all = lam_all * comp_mask
#         U_all = U_all * comp_mask.unsqueeze(1)

#     dists_per_t = []
#     for t in times:
#         lam_pow = lam_all.pow(int(t))              # (B,k_eff)
#         emb = U_all * lam_pow[:, None, :]          # (B, L, k_eff)
#         # Pairwise squared distances via Gram trick
#         sq = (emb.pow(2).sum(dim=-1, keepdim=True))         # (B, L, 1)
#         gram = torch.matmul(emb, emb.transpose(1, 2))       # (B, L, L)
#         dist2 = sq + sq.transpose(1, 2) - 2.0 * gram        # (B, L, L)
#         dist2 = torch.clamp(dist2, min=0.0)
#         dist = torch.sqrt(dist2 + eps)

#         # Zero-out invalid pairs and enforce zero diagonal
#         dist = dist * mask_2d.float()
#         eye = torch.eye(L, dtype=dist.dtype, device=dist.device).unsqueeze(0)
#         dist = dist * (1.0 - eye)
#         dists_per_t.append(dist)

#     dist_all_t = torch.stack(dists_per_t, dim=-1)  # (B, L, L, T)

#     # Normalize distances to [0,1] per-batch for stable RBFs (exclude diagonal)
#     dist_valid = dist_all_t.clone()
#     mask3 = mask_2d.unsqueeze(-1).expand_as(dist_valid)
#     dist_valid = dist_valid * mask3
#     B_, L_, _, T_ = dist_valid.shape
#     eye3 = torch.eye(L_, device=device, dtype=dtype).view(1, L_, L_, 1)
#     dist_valid = dist_valid * (1.0 - eye3)

#     # Per-batch max (over i,j,t) for normalization; avoid div by 0
#     max_per_b = dist_valid.view(B_, -1).amax(dim=1).clamp_min(1e-6).view(B_, 1, 1, 1)
#     dist_norm = (dist_all_t / max_per_b).clamp(0.0, 1.0)

#     if int(rbf_num) <= 0:
#         return dist_norm.to(device=device, dtype=dtype)

#     # Build RBF centers uniformly in [0,1]
#     if rbf_num == 1:
#         centers = torch.tensor([0.5], device=device, dtype=dtype)
#         gamma = torch.tensor(1.0, device=device, dtype=dtype) if rbf_gamma is None else torch.tensor(float(rbf_gamma), device=device, dtype=dtype)
#     else:
#         centers = torch.linspace(0.0, 1.0, steps=int(rbf_num), device=device, dtype=dtype)
#         delta = (1.0 / float(rbf_num - 1))
#         gamma = (1.0 / (2.0 * (delta ** 2))) if rbf_gamma is None else float(rbf_gamma)
#         gamma = torch.tensor(gamma, device=device, dtype=dtype)

#     # Expand to features: exp(-gamma * (d - c)^2) for each center c
#     # dist_norm: (B,L,L,T) -> (B,L,L,T,C)
#     diff = dist_norm.unsqueeze(-1) - centers.view(1, 1, 1, 1, -1)
#     feats_rbf = torch.exp(-gamma * (diff ** 2))
#     # Mask invalid pairs
#     feats_rbf = feats_rbf * mask3.unsqueeze(-1)

#     # Flatten time and centers into feature dim: (B,L,L,T*C)
#     B_, L_, _, T_, C_ = feats_rbf.shape
#     feats_flat = feats_rbf.view(B_, L_, L_, T_ * C_)
#     return feats_flat.to(device=device, dtype=dtype)


# def diffusion_map_pair_features(
#     A_batch: torch.Tensor,
#     times: Sequence[int],
#     k: int,
#     skip_top: bool = True,
#     node_mask: Optional[torch.Tensor] = None,
# ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#     """
#     Batched diffusion-map pair features with node_mask (B, L) indicating valid nodes.
#     Robust assignment using integer indices (index_copy_) to avoid boolean-index shape issues.
#     """
#     assert A_batch.dim() == 3 and A_batch.shape[-1] == A_batch.shape[-2], "A_batch must be (B,L,L)"
#     B, L, _ = A_batch.shape
#     assert k >= 1, "k must be >= 1"
#     assert len(times) >= 1, "times must be non-empty"

#     device = A_batch.device
#     dtype = A_batch.dtype

#     if node_mask is None:
#         node_mask = torch.ones((B, L), dtype=torch.bool, device=device)
#     else:
#         assert node_mask.shape == (B, L), "node_mask must have shape (B, L)"
#         node_mask = node_mask.to(torch.bool).to(device)

#     start = 1 if skip_top else 0


#     # Batched construction of normalized adjacency on full LxL with masking
#     lam_all = torch.zeros((B, k), dtype=dtype, device=device)
#     U_all = torch.zeros((B, L, k), dtype=dtype, device=device)

#     valid_pair = (node_mask[:, :, None] & node_mask[:, None, :])  # (B,L,L)
#     A_masked = A_batch * valid_pair.to(dtype)

#     d = A_masked.sum(dim=-1)  # (B,L)
#     inv_sqrt_d = torch.where(
#         node_mask,
#         d.clamp_min(1e-15).pow(-0.5),
#         torch.zeros_like(d)
#     )
#     S = inv_sqrt_d[:, :, None] * A_masked * inv_sqrt_d[:, None, :]

#     # Enforce symmetry and add small jitter on the diagonal
#     S = 0.5 * (S + S.transpose(-1, -2))
#     jitter = 1e-7
#     I = torch.eye(L, dtype=dtype, device=device).unsqueeze(0)
#     S = S + jitter * I

#     # Batched eigendecomposition with robust CPU float64 fallback
#     try:
#         evals, evecs = torch.linalg.eigh(S)  # (B,L), (B,L,L), ascending
#     except Exception:
#         try:
#             print("linalg.eigh failed, using float64 fallback")
#             S64 = S.to(torch.float64).cpu()
#             S64 = 0.5 * (S64 + S64.transpose(-1, -2))
#             S64 = S64 + (jitter * torch.eye(L, dtype=torch.float64).unsqueeze(0))
#             evals64, evecs64 = torch.linalg.eigh(S64)
#             evals = evals64.to(dtype=dtype, device=device)
#             evecs = evecs64.to(dtype=dtype, device=device)
#         except Exception:
#             print("float64 fallback failed, returning zeros features and eigen data")
#             # Total failure: return zeros features and eigen data
#             features_t = []
#             for t in times:
#                 feats_t = torch.zeros((B, L, L, 2 * k), dtype=dtype, device=device)
#                 features_t.append(feats_t)
#             features = torch.cat(features_t, dim=-1)
#             return features, (lam_all, U_all)

#     # Sort descending per batch and select per-sample k_eff = min(k, n_valid - start)
#     sort_idx = torch.argsort(evals, descending=True, dim=-1)  # (B,L)
#     # Reorder eigenvalues and eigenvectors
#     evals_sorted = torch.gather(evals, -1, sort_idx)
#     sort_idx_exp = sort_idx.unsqueeze(-2).expand(B, L, L)
#     evecs_sorted = torch.gather(evecs, -1, sort_idx_exp)

#     # Per-sample write-back honoring varying number of valid nodes
#     # Vectorized selection and padding
#     n_subs = node_mask.sum(dim=1)  # (B,)
#     k_cap = min(k, max(0, L - start))
#     if k_cap > 0:
#         k_effs = torch.clamp(n_subs - start, min=0, max=k_cap)  # (B,)
#         idx_range = torch.arange(k_cap, device=device).unsqueeze(0)  # (1,k_cap)
#         mask_k = idx_range < k_effs.unsqueeze(-1)  # (B,k_cap)

#         sel_evals = evals_sorted[:, start:start + k_cap]  # (B,k_cap)
#         lam_all[:, :k_cap] = torch.where(mask_k, sel_evals, torch.zeros_like(sel_evals))

#         sel_evecs = evecs_sorted[:, :, start:start + k_cap]  # (B,L,k_cap)
#         U_mask = mask_k.unsqueeze(1).expand(-1, L, -1)  # (B,L,k_cap)
#         U_all[:, :, :k_cap] = torch.where(U_mask, sel_evecs, torch.zeros_like(sel_evecs))

#     # Zero-out invalid-node rows for all k
#     U_all = U_all * node_mask.unsqueeze(-1)

#     # Build pairwise features and zero-out pairs where either node invalid
#     features_t = []
#     for t in times:
#         assert int(t) >= 0
#         lam_pow = lam_all.pow(int(t))                 # (B, k)
#         emb_t = U_all * lam_pow[:, None, :]           # (B, L, k)

#         emb_i = emb_t[:, :, None, :].expand(-1, L, L, -1)
#         emb_j = emb_t[:, None, :, :].expand(-1, L, L, -1)
#         feats_t = torch.cat([emb_i, emb_j], dim=-1)   # (B, L, L, 2k)

#         valid_pair = (node_mask[:, :, None] & node_mask[:, None, :]).unsqueeze(-1)
#         feats_t = feats_t * valid_pair

#         features_t.append(feats_t)

#     features = torch.cat(features_t, dim=-1)
#     return features, (lam_all, U_all)


# def parse_connections(config_path):
#     """
#     Parses the connection configuration from a CSV file.
#     Handles multiple connection types for a single residue pair.
#     """
#     connections = {}
    
#     try:
#         with open(config_path, 'r') as f:
#             next(f)  # Skip header
#             for line in f:
#                 try:
#                     parts = line.strip().split(',')
#                     if len(parts) < 4: continue
#                     res1, res2, atom1, atom2 = [p.strip() for p in parts[:4]]
                    
#                     if res1 not in aa2num or res2 not in aa2num:
#                         continue

#                     res1_num, res2_num = aa2num[res1], aa2num[res2]
                    
#                     # Classify connection type
#                     is_backbone_conn = atom1 in BACKBONE_ATOMS or atom2 in BACKBONE_ATOMS
#                     conn_type = 'backbone' if is_backbone_conn else 'sidechain'
                    
#                     conn_tuple = (atom1, atom2, conn_type)

#                     # Initialize dict entry if not present
#                     if (res1_num, res2_num) not in connections:
#                         connections[(res1_num, res2_num)] = []
#                     if (res2_num, res1_num) not in connections:
#                         connections[(res2_num, res1_num)] = []

#                     connections[(res1_num, res2_num)].append(conn_tuple)
#                     connections[(res2_num, res1_num)].append((atom2, atom1, conn_type))

#                 except (ValueError, IndexError):
#                     # Silently skip malformed lines
#                     continue
#     except FileNotFoundError:
#         # This is a valid case if no special connections are needed.
#         print(f"Info: Connection config file not found at {config_path}. Continuing without special connections.")
#         return {}
            
#     return connections

# def get_CA_dist_matrix(adj, seq, connections, pdb_idx, rf_index, dmax=128, 
#                        N_connect_idx=None, C_connect_idx=None,
#                        head_mask: torch.Tensor = None, tail_mask: torch.Tensor = None):
#     """
#     Computes the CA distance matrix based on covalent bonds using Scipy for performance.
#     """
#     B, L = adj.shape[:2]
#     dist_matrix = torch.full((B, L, L), float(dmax), dtype=torch.float32)
    
#     GLY_IDX = aa2num['GLY']
#     MASK_IDX = 20

#     def is_chain_terminus(b, i, rf_index_b_set):
#         res_idx = rf_index[b, i].item()
#         is_n_term = (res_idx - 1) not in rf_index_b_set
#         is_c_term = (res_idx + 1) not in rf_index_b_set
#         if is_n_term or is_c_term:
#             return True
#         else:
#             is_interchain_break = pdb_idx[b][i] != pdb_idx[b][i+1] or pdb_idx[b][i] != pdb_idx[b][i-1]
#         return is_interchain_break

#     for b in range(B):
#         seq_b = seq[b].clone()
#         seq_b[seq_b == MASK_IDX] = GLY_IDX
        
#         rf_index_b_set = set(rf_index[b].tolist())

#         atom_to_node = {}
#         node_counter = 0
        
#         # 1. Map all heavy atoms to integer node indices
#         for i in range(L):
#             res_type = seq_b[i].item()
#             atoms_in_res = {atom.strip() for bond in aabonds[res_type] for atom in bond}
#             for atom_name in atoms_in_res:
#                 if (i, atom_name) not in atom_to_node:
#                     atom_to_node[(i, atom_name)] = node_counter
#                     node_counter += 1
        
#         num_nodes = node_counter
#         if num_nodes == 0: continue

#         row, col = [], []

#         # Helper to add edges to the adjacency list
#         def add_edge(u, v):
#             row.extend([u, v])
#             col.extend([v, u])

#         # 2. Add intra-residue and peptide bonds
#         for i in range(L):
#             # Intra-residue
#             res_type = seq_b[i].item()
#             for atom1, atom2 in aabonds[res_type]:
#                 u = atom_to_node.get((i, atom1.strip()))
#                 v = atom_to_node.get((i, atom2.strip()))
#                 if u is not None and v is not None:
#                     add_edge(u, v)
#             # Peptide
#             skip_peptide = False
#             if head_mask is not None:
#                 if bool(head_mask[b, i]) or (i < L - 1 and bool(head_mask[b, i+1])):
#                     skip_peptide = True
#             if tail_mask is not None:
#                 if bool(tail_mask[b, i]) or (i < L - 1 and bool(tail_mask[b, i+1])):
#                     skip_peptide = True
#             if (not skip_peptide) and i < L - 1 and rf_index[b, i+1] == rf_index[b, i] + 1:
#                 u = atom_to_node.get((i, ATOM_C))
#                 v = atom_to_node.get((i + 1, ATOM_N))
#                 if u is not None and v is not None:
#                     add_edge(u, v)

#         # 3. Add special connections from the adjacency matrix
#         for i_idx, j_idx in torch.argwhere(adj[b] == 1):
#             i, j = i_idx.item(), j_idx.item()
#             if i >= j: continue

#             res_i_type, res_j_type = seq_b[i].item(), seq_b[j].item()
#             atom_i_name, atom_j_name = None, None

#             is_i_N_term = N_connect_idx is not None and i == N_connect_idx[b]
#             is_i_C_term = C_connect_idx is not None and i == C_connect_idx[b]
#             is_j_N_term = N_connect_idx is not None and j == N_connect_idx[b]
#             is_j_C_term = C_connect_idx is not None and j == C_connect_idx[b]

#             if (is_i_C_term and is_j_N_term) or (is_i_N_term and is_j_C_term):
#                 atom_i_name = ATOM_C if is_i_C_term else ATOM_N
#                 atom_j_name = ATOM_N if is_j_N_term else ATOM_C
#             else:
#                 preferred_conn_type = 'backbone' if (is_i_N_term or is_i_C_term or is_j_N_term or is_j_C_term) else 'sidechain'
#                 possible_conns = connections.get((res_i_type, res_j_type), [])
#                 if not possible_conns: continue
                
#                 filtered_conns = [c for c in possible_conns if c[2] == preferred_conn_type]
#                 conn_to_use = filtered_conns[0] if filtered_conns else possible_conns[0]
#                 atom_i_name, atom_j_name, _ = conn_to_use

#             if atom_i_name and atom_j_name:
#                 if (atom_i_name in BACKBONE_ATOMS and not is_chain_terminus(b, i, rf_index_b_set)) or \
#                    (atom_j_name in BACKBONE_ATOMS and not is_chain_terminus(b, j, rf_index_b_set)):
#                     continue
#                 u = atom_to_node.get((i, atom_i_name))
#                 v = atom_to_node.get((j, atom_j_name))
#                 if u is not None and v is not None:
#                     add_edge(u, v)

#         # 4. Build sparse matrix and calculate all-pairs shortest paths
#         adj_matrix_sparse = csr_matrix((([1]*len(row)), (row, col)), shape=(num_nodes, num_nodes))
        
#         atom_dist_matrix = shortest_path(csgraph=adj_matrix_sparse, directed=False, unweighted=True)

#         # 5. Populate the final distance matrix for CA atoms
#         for i in range(L):
#             dist_matrix[b, i, i] = 0
#             node_i = atom_to_node.get((i, ATOM_CA))
#             if node_i is None: continue
#             for j in range(i + 1, L):
#                 node_j = atom_to_node.get((j, ATOM_CA))
#                 if node_j is None: continue
                
#                 dist = atom_dist_matrix[node_i, node_j]
#                 final_dist = min(dist, dmax)
#                 dist_matrix[b, i, j] = final_dist
#                 dist_matrix[b, j, i] = final_dist
                
#     return dist_matrix

# def get_residue_dist_matrix(adj, rf_index, dmax=32):
#     """
#     Computes the residue-level shortest path distance matrix.

#     The graph is constructed with residues as nodes. Edges exist between
#     adjacent residues (peptide bonds) and between residues with special
#     connections specified in the `adj` matrix. The distance is the
#     minimum number of residues in the path.

#     Args:
#         adj (torch.Tensor): The adjacency matrix indicating special connections
#                             (e.g., disulfide bonds). Shape: (B, L, L).
#         rf_index (torch.Tensor): Residue indices from the PDB, used to identify
#                                  sequential residues for peptide bonds.
#                                  Shape: (B, L).
#         dmax (int): The maximum distance to consider. Paths longer than this
#                     will be capped at this value.

#     Returns:
#         torch.Tensor: A tensor of shape (B, L, L) containing the shortest
#                       path distances between each pair of residues.
#     """
#     B, L = adj.shape[:2]
#     # Initialize the final distance matrix with the maximum value
#     dist_matrix = torch.full((B, L, L), float(dmax), dtype=torch.float32, device=adj.device)

#     for b in range(B):
#         # The nodes of our graph are the residues, so there are L nodes.
#         if L == 0:
#             continue

#         row, col = [], []

#         # Helper to add a bi-directional edge to the graph
#         def add_edge(u, v):
#             row.extend([u, v])
#             col.extend([v, u])

#         # 1. Add edges for peptide bonds (adjacent residues)
#         for i in range(L - 1):
#             # Check if residue i and i+1 are sequential in the chain
#             if  rf_index[b, i+1] == rf_index[b, i] + 1:
#                 add_edge(i, i + 1)

#         # 2. Add edges for special connections from the input adjacency matrix
#         # These could be disulfide bonds, cyclic connections, etc.
#         special_connections = torch.argwhere(adj[b] == 1)
#         for conn in special_connections:
#             i, j = conn[0].item(), conn[1].item()
#             # Avoid duplicate edges and self-loops
#             if i < j:
#                 add_edge(i, j)
        
#         # If there are no connections at all, fill with dmax and continue
#         if not row:
#             dist_matrix[b].fill_(dmax)
#             dist_matrix[b].fill_diagonal_(0)
#             continue

#         # 3. Build a sparse matrix and calculate all-pairs shortest paths
#         # The graph is unweighted, so each edge has a weight of 1.
#         adj_matrix_sparse = csr_matrix(([1] * len(row), (row, col)), shape=(L, L))
        
#         # Calculate shortest paths. Unreachable nodes will have a distance of 'inf'.
#         residue_dist = shortest_path(csgraph=adj_matrix_sparse, directed=False, unweighted=True)
        
#         # 4. Populate the final distance matrix for this batch item
#         residue_dist_tensor = torch.from_numpy(residue_dist).to(dtype=torch.float32, device=adj.device)
        
#         # Clamp the infinite distances (unreachable pairs) to dmax
#         dist_matrix[b] = torch.clamp(residue_dist_tensor, max=dmax)

#     return dist_matrix

# def make_sub_doubly2doubly_stochastic(sub_ds_matrix: torch.Tensor) -> torch.Tensor:
#     """
#     将一个亚双随机对称矩阵通过在对角线添加差额来转换为双随机对称矩阵。
#     参数:
#     sub_ds_matrix (torch.Tensor): 输入的张量，形状可以是 (N, N) 或 (B, N, N)，
#                                   其中 B 是批处理大小，N 是矩阵维度。
#     返回:
#     torch.Tensor: 转换后的双随机对称矩阵。
#     """
#     # 确保输入至少是二维的
#     dtype = sub_ds_matrix.dtype
#     if sub_ds_matrix.dim() < 2:
#         raise ValueError("输入张量至少需要是二维 (N, N)。")
    
#     # --- （可选）健壮性检查 ---
#     # 检查对称性 (允许有微小的浮点误差)
#     assert torch.allclose(sub_ds_matrix, sub_ds_matrix.transpose(-2, -1)), "输入矩阵必须是对称的"
#     # 检查非负性
#     assert torch.all(sub_ds_matrix >= 0), "输入矩阵元素必须非负"
    
#     # 1. 计算每行的和。对于(B, N, N)的张量，我们对最后一个维度(dim=-1)求和。
#     row_sums = torch.sum(sub_ds_matrix, dim=-1)
    
#     # 检查行和是否小于等于1 (允许有微小的浮点误差)
#     assert torch.all(row_sums <= 1.0 + 1e-6), "输入矩阵的行和必须小于等于1"

#     # 结果的形状是 (N,) 或 (B, N)
#     deficits = 1.0 - row_sums

#     # torch.diag_embed 会将一个 (B, N) 的向量转换为一个 (B, N, N) 的对角矩阵
#     diagonal_additions = torch.diag_embed(deficits)

#     doubly_stochastic_matrix = sub_ds_matrix + diagonal_additions

#     return doubly_stochastic_matrix.to(dtype)


# def sample_symmetric_permutation_by_pairs(A: torch.Tensor, mask: torch.Tensor, mode: str = "random") -> torch.Tensor:
#     """
#     通过迭代采样"节点对"来生成一个对称置换矩阵，同时考虑一个掩码。

#     参数:
#         A (torch.Tensor): 对称双随机矩阵，形状 (L, L)。
#         mask (torch.Tensor): 0-1 或布尔掩码，形状 (L, L)，1/True 表示允许配对。
#         mode (str): "random"（按权重随机采样，默认）、"greedy"（每步取当前最大权重）或 "opt"（全局最优，Blossom 最大权匹配）。
#     返回:
#         torch.Tensor: 对称 0-1 置换矩阵，形状 (L, L)。
#     """
#     assert mode in ("random", "greedy", "opt"), "mode must be 'random', 'greedy' or 'opt'"

#     L = A.shape[0]
#     device = A.device

#     # 规范 mask：保留原有行为（如果传入 float 也可）
#     mask = mask.to(device=device)
#     mask_bool = mask.to(dtype=torch.bool, device=device)

#     # OPT 模式：调用 Blossom（networkx）
#     if mode == "opt":
#         if nx is None:
#             raise RuntimeError("mode='opt' requires networkx to be installed (pip install networkx).")

#         # 防止 log(0) 并改为 NumPy 计算以减少张量<->NumPy开销
#         eps = 1e-12
#         A_np = A.detach().to('cpu').numpy()
#         mask_np = mask_bool.detach().to('cpu').numpy()
#         logA = np.log(np.clip(A_np, eps, None))

#         # delta_ij = 2*logA_ij - logA_ii - logA_jj
#         diag = np.diag(logA)  # (L,)
#         delta = (2.0 * logA) - diag[None, :] - diag[:, None]

#         # 掩码与上三角过滤，只保留正权边（负权边不可能出现在最优匹配中）
#         tri_i, tri_j = np.triu_indices(L, k=1)
#         valid_mask = mask_np[tri_i, tri_j]
#         w = delta[tri_i, tri_j]
#         finite_pos = np.isfinite(w) & (w > 0.0) & valid_mask
#         ei = tri_i[finite_pos]
#         ej = tri_j[finite_pos]
#         ew = w[finite_pos].astype(float)

#         # 优先尝试更快的后端（NetworKit：近似最大权匹配），失败则回退到 networkx 精确 Blossom
#         matching_pairs = None
#         try:
#             if ei.size > 0:
#                 G_nk = nk.Graph(L, weighted=True, directed=False)
#                 for u, v, ww in zip(ei.tolist(), ej.tolist(), ew.tolist()):
#                     G_nk.addEdge(int(u), int(v), float(ww))
#                 matcher = nk.matching.LocalMaxMatcher(G_nk)
#                 matcher.run()
#                 M = matcher.getMatching()
#                 # 提取匹配对
#                 pairs = []
#                 for u in range(L):
#                     if M.isMatched(u):
#                         v = M.mate(u)
#                         if v != -1 and u < v:
#                             pairs.append((u, v))
#                 matching_pairs = pairs
#         except Exception:
#             matching_pairs = None

#         if matching_pairs is None:
#             # 回退到 networkx：构图并批量添加边
#             G = nx.Graph()
#             G.add_nodes_from(range(L))
#             if ei.size > 0:
#                 edges = list(zip(ei.tolist(), ej.tolist(), ew.tolist()))
#                 G.add_weighted_edges_from(edges)
#             matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)
#             matching_pairs = list(matching)

#         # construct permutation matrix
#         P = torch.zeros((L, L), dtype=A.dtype, device=device)
#         matched_nodes = set()
#         for (i, j) in matching_pairs:
#             P[i, j] = 1.0
#             P[j, i] = 1.0
#             matched_nodes.add(i)
#             matched_nodes.add(j)
#         for i in range(L):
#             if i not in matched_nodes:
#                 P[i, i] = 1.0
#         return P

#     # RANDOM / GREEDY 模式：逐步构造（保留原意）
#     permutation_matrix = torch.zeros(L, L, device=device, dtype=A.dtype)
#     nodes_available_mask = torch.ones(L, dtype=torch.bool, device=device)
#     eps_sum = 1e-9

#     # Precompute diag for speed
#     diag = A.diag()
#     mask_diag = mask.diag()

#     while torch.any(nodes_available_mask):
#         available_indices = torch.where(nodes_available_mask)[0]
#         n_avail = available_indices.shape[0]

#         # self-loop weights (for available nodes)
#         self_loop_weights = diag[available_indices] * mask_diag[available_indices]

#         # swap (2-cycle) weights
#         if n_avail >= 2:
#             # torch.combinations returns pairs of actual node indices
#             swap_pairs_indices = torch.combinations(available_indices, r=2)  # (n_combs,2)
#             swap_weights = A[swap_pairs_indices[:, 0], swap_pairs_indices[:, 1]] * \
#                            mask[swap_pairs_indices[:, 0], swap_pairs_indices[:, 1]]
#         else:
#             swap_pairs_indices = torch.empty(0, 2, dtype=torch.long, device=device)
#             swap_weights = torch.empty(0, device=device, dtype=A.dtype)

#         # merge: note 原实现中 swap 权重乘以2（代表两个 off-diagonal entries）
#         all_weights = torch.cat([self_loop_weights, swap_weights * 2.0])

#         # 如果所有权重都近似为0 -> 把剩余节点都设为自环
#         if all_weights.sum() < eps_sum:
#             permutation_matrix[available_indices, available_indices] = 1.0
#             break

#         # 选择 index（随机或贪婪）
#         if mode == "random":
#             # multinomial 需要浮点 non-negative 和和>0
#             # torch.multinomial 在 CPU/GPU 都可用
#             chosen_flat_idx = int(torch.multinomial(all_weights, 1).item())
#         else:  # greedy
#             chosen_flat_idx = int(torch.argmax(all_weights).item())

#         # apply chosen
#         n_self = self_loop_weights.shape[0]
#         if chosen_flat_idx < n_self:
#             # self-loop chosen
#             node_idx = int(available_indices[chosen_flat_idx].item())
#             permutation_matrix[node_idx, node_idx] = 1.0
#             nodes_available_mask[node_idx] = False
#         else:
#             swap_idx = chosen_flat_idx - n_self
#             node1_idx = int(swap_pairs_indices[swap_idx, 0].item())
#             node2_idx = int(swap_pairs_indices[swap_idx, 1].item())
#             permutation_matrix[node1_idx, node2_idx] = 1.0
#             permutation_matrix[node2_idx, node1_idx] = 1.0
#             nodes_available_mask[node1_idx] = False
#             nodes_available_mask[node2_idx] = False

#     return permutation_matrix


# def sample_permutation(A_batch: torch.Tensor, mask_2d: torch.Tensor = None, mode: str = "opt") -> torch.Tensor:
#     """
#     批量版本：对 A_batch 中每个矩阵分别调用 sample_symmetric_permutation_by_pairs。
#     参数:
#         A_batch (B, L, L)
#         mask_2d (B, L, L) or None
#         mode: "random", "greedy", or "opt"
#     返回:
#         (B, L, L) 的 0-1 对称置换矩阵批量
#     """
#     if mask_2d is None:
#         mask_2d = torch.ones_like(A_batch, dtype=torch.bool, device=A_batch.device)

#     perm_matrices = [
#         sample_symmetric_permutation_by_pairs(A_batch[i], mask_2d[i], mode=mode)
#         for i in range(A_batch.shape[0])
#     ]
#     return torch.stack(perm_matrices, dim=0)


def get_R_from_xyz(xyz):
    """
    Get rotation matrix from xyz coordinates
    Args:
        xyz: coordinates of shape [B, L, 3, 3]
    Returns:
        R: rotation matrix of shape [B, L, 3, 3]
    """
    B, L = xyz.shape[:2]
    N_0, Ca_0, C_0 = xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :]

    # Build local right-handed frames via Gram–Schmidt (fully differentiable, no SVD/NumPy)
    eps = 1e-8

    v1 = C_0 - Ca_0  # primary axis (C -> CA)
    v2 = N_0 - Ca_0  # helper axis (N -> CA)

    def _normalize(vec):
        n = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(eps)
        return vec / n

    e1 = _normalize(v1)
    # remove component of v2 along e1
    v2_ortho = v2 - (e1 * (e1 * v2).sum(dim=-1, keepdim=True))
    e2 = _normalize(v2_ortho)
    # right-handed third axis
    e3 = torch.cross(e1, e2, dim=-1)
    e3 = _normalize(e3)
    # improve orthogonality of e2
    e2 = torch.cross(e3, e1, dim=-1)

    R = torch.stack([e1, e2, e3], dim=-1)  # [B, L, 3, 3]

    # Fallback to identity on non-finite
    invalid = ~torch.isfinite(R).all(dim=(-1, -2))
    if invalid.any():
        print("warning: get_R_from_xyz: invalid R")
        I = torch.eye(3, device=xyz.device, dtype=R.dtype).view(1, 1, 3, 3).expand(B, L, 3, 3)
        R = torch.where(invalid.unsqueeze(-1).unsqueeze(-1), I, R)

    return R.to(device=xyz.device, dtype=xyz.dtype)

def get_xyz_from_RT(R, T):
    """
    Get backbone xyz coordinates from rotation matrix R and translation T.
    Args:
        R: rotation matrix of shape [B, L, 3, 3]
        T: translation vector of shape [B, L, 3] (Ca coordinates)
    Returns:
        xyz: backbone coordinates of shape [B, L, 3, 3] (N, Ca, C)
    """
    # Ideal local coordinates of N and C relative to Ca
    # These values are from rfdiffusion.util
    N_ideal = che.init_N.to(device=R.device).float()
    C_ideal = che.init_C.to(device=R.device).float()
    R = R.float()
    # Transform local coordinates to global
    # R is [B, L, 3, 3], N_ideal is [3] -> N_global is [B, L, 3]

    N_global = torch.einsum('blij,j->bli', R, N_ideal).float() + T
    C_global = torch.einsum('blij,j->bli', R, C_ideal).float() + T
    Ca_global = T

    # Stack to get backbone coordinates
    # Unsqueeze to add atom dimension for stacking
    xyz = torch.stack([N_global, Ca_global, C_global], dim=-2)
    
    return xyz

def parse_cif(filename, **kwargs):
    """
    Extract xyz coords for all heavy atoms from a cif file using BioPython.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("cif_structure", filename)
    cif_dict = MMCIF2Dict.MMCIF2Dict(filename)
    return parse_cif_structure(structure, cif_dict, **kwargs)


def parse_cif_structure(structure, cif_dict, parse_hetatom=False, ignore_het_h=True, 
                        parse_link=True, link_csv_path=None):
    """
    Extracts information from a BioPython structure object, mirroring parse_pdb_lines.
    """
    res, pdb_idx, rf_idx = [], [], []
    rf_index = 0
    last_index = None
    last_xyz = np.array([0, 0, 0], dtype=np.float32)
    
    residues_to_process = []
    
    # First pass: identify valid residues
    #for model in structure[0]:
    model = structure[0]
    for chain in model:
        for residue in chain:
            hetflag, resseq, icode = residue.get_id()
            if hetflag.strip() != '':  # Skip HETATM residues in this pass
                continue

            resname = residue.get_resname()
            if resname not in util.aa2num:
                continue
            
            # Skip glycine without sidechain
            if resname != 'GLY' and 'CB' not in residue.child_dict:
                continue
            
            # Check for complete backbone
            if 'N' in residue and 'CA' in residue and 'C' in residue:
                residues_to_process.append(residue)

    # Sort residues like in a PDB file (by chain then by residue number)
    residues_to_process.sort(key=lambda r: (r.get_parent().id, r.get_id()[1], r.get_id()[2]))
    for residue in residues_to_process:
        resname = residue.get_resname()
        chain_id = residue.get_parent().id
        _, resseq, icode = residue.get_id()
        
        resseq_str = str(resseq) + (icode if icode.strip() else "")
        current_index = (chain_id, resseq_str)
        current_xyz = residue['CA'].get_coord()

        if last_index is not None and last_index[0] == current_index[0]:
            try:
                # Attempt to parse residue numbers as integers for gap calculation
                last_res_num = int(re.match(r'^-?\d+', last_index[1]).group(0))
                current_res_num = int(re.match(r'^-?\d+', current_index[1]).group(0))
                distance = np.linalg.norm(current_xyz - last_xyz)
                
                res_diff = current_res_num - last_res_num
                if 1 < res_diff < 200 and distance > BOND_LENGTH_THRESHOLD:
                    rf_index += res_diff
                else:
                    rf_index += 1
            except (ValueError, TypeError):
                # Fallback if residue numbers are not simple integers
                rf_index += 1
        else:
            rf_index += 1
        res.append((resseq_str, resname))
        pdb_idx.append(current_index)
        rf_idx.append(rf_index)

        last_xyz = current_xyz
        last_index = current_index

    seq = [util.aa2num.get(r[1], 20) for r in res]
    
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    
    pdb_idx_to_res_idx = {pdb: i for i, pdb in enumerate(pdb_idx)}

    for residue in residues_to_process:
        chain_id = residue.get_parent().id
        _, resseq, icode = residue.get_id()
        resseq_str = str(resseq) + (icode if icode.strip() else "")
        current_pdb_idx = (chain_id, resseq_str)
        
        i = pdb_idx_to_res_idx.get(current_pdb_idx)
        if i is None:
            continue

        resname = residue.get_resname()
        if resname in util.aa2num:
            res_map_idx = util.aa2num[resname]
            for atom in residue:
                atom_name = atom.get_name()
                # Find atom index in the 14-atom representation
                try:
                    # aa2long is a list of lists, so we access the one for the current residue
                    atom_idx_in_14 = util.aa2long[res_map_idx].index(" "+atom_name.ljust(3))
                    xyz[i, atom_idx_in_14, :] = atom.get_coord()
                except (ValueError, IndexError):
                    continue # Atom not in the 14-atom list for this residue

    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz)] = 0.0

    # This part for removing duplicates is kept from parse_pdb, though less likely with BioPython
    new_idx, i_unique, new_rf_idx = [], [], []
    for i, idx_tuple in enumerate(pdb_idx):
        if idx_tuple not in new_idx:
            new_idx.append(idx_tuple)
            new_rf_idx.append(rf_idx[i])
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]
    seq = np.array(seq)[i_unique]

    out = {
        "xyz": xyz,
        "mask": mask,
        "idx": new_rf_idx,
        "seq": np.array(seq),
        "pdb_idx": pdb_idx,
    }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        model = structure[0]
        #for model in structure[:1]:
        for chain in model:
            for residue in chain:
                hetflag, _, _ = residue.get_id()
                if hetflag.strip() != '' and hetflag.strip() != 'W':
                    for atom in residue:
                        if ignore_het_h and atom.element == 'H':
                            continue
                        info_het.append(
                            dict(
                                idx=atom.get_serial_number(),
                                atom_id=atom.get_name(),
                                atom_type=atom.element,
                                name=residue.get_resname(),
                            )
                        )
                        xyz_het.append(atom.get_coord())

        out["xyz_het"] = np.array(xyz_het, dtype=np.float32)
        out["info_het"] = info_het
        
        
    if parse_link:
        allowed_link_types = load_allowed_bonds_from_csv(link_csv_path)
        links = []
        # Build a quick lookup for atom objects from the parsed structure
        atom_lookup = {}
        for r in residues_to_process:
            chain_id = r.get_parent().id
            _, resseq, icode = r.get_id()
            resseq_str = str(resseq) + (icode if icode.strip() else "")
            pdb_tuple = (chain_id, resseq_str)
            if pdb_tuple in pdb_idx:
                for atom in r:
                    atom_lookup[(chain_id, resseq_str, atom.get_name())] = atom

        # 1. 只检查最核心的、一定存在的列来确定循环次数
        #    几乎所有的 _struct_conn 记录都至少有 conn_type_id
        core_key = '_struct_conn.conn_type_id'
        if core_key in cif_dict:
            num_conns = len(cif_dict[core_key])
            
            # 2. 安全地获取每个数据列，如果列不存在，则提供一个默认的空列表
            conn_types = cif_dict.get('_struct_conn.conn_type_id', [])
            
            p1_chains = cif_dict.get('_struct_conn.ptnr1_auth_asym_id', []) or cif_dict.get('_struct_conn.ptnr1_label_asym_id', [])
            p1_res_names = cif_dict.get('_struct_conn.ptnr1_auth_comp_id', []) or cif_dict.get('_struct_conn.ptnr1_label_comp_id', [])
            p1_res_nums = cif_dict.get('_struct_conn.ptnr1_auth_seq_id', []) or cif_dict.get('_struct_conn.ptnr1_label_seq_id', [])
            p1_atom_names = cif_dict.get('_struct_conn.ptnr1_label_atom_id', [])
            # 对于可选的插入码列，如果不存在，我们让它返回一个空列表
            p1_ins_codes = cif_dict.get('_struct_conn.pdbx_ptnr1_PDB_ins_code', [])

            p2_chains = cif_dict.get('_struct_conn.ptnr2_auth_asym_id', []) or cif_dict.get('_struct_conn.ptnr2_label_asym_id', [])
            p2_res_names = cif_dict.get('_struct_conn.ptnr2_auth_comp_id', []) or cif_dict.get('_struct_conn.ptnr2_label_comp_id', [])
            p2_res_nums = cif_dict.get('_struct_conn.ptnr2_auth_seq_id', []) or cif_dict.get('_struct_conn.ptnr2_label_seq_id', [])
            p2_atom_names = cif_dict.get('_struct_conn.ptnr2_label_atom_id', [])
            p2_ins_codes = cif_dict.get('_struct_conn.pdbx_ptnr2_PDB_ins_code', [])
            # print(f"Processing {num_conns} connections")
            for i in range(num_conns):
                try:
                    conn_type = conn_types[i]
                    if conn_type not in ['disulf', 'covale']: # 增加对 isopeptide 的支持
                        continue

                    # 3. 在循环内部安全地获取每个值
                    #    如果列表为空（因为列不存在），则提供默认值 '?'
                    p1_chain = p1_chains[i]
                    p1_res_name = p1_res_names[i]
                    p1_res_num = p1_res_nums[i]
                    p1_atom_name = p1_atom_names[i]
                    p1_ins_code = p1_ins_codes[i] if i < len(p1_ins_codes) else '?'
                    
                    p2_chain = p2_chains[i]
                    p2_res_name = p2_res_names[i]
                    p2_res_num = p2_res_nums[i]
                    p2_atom_name = p2_atom_names[i]
                    p2_ins_code = p2_ins_codes[i] if i < len(p2_ins_codes) else '?'
                    
                    # Build residue identifiers robustly: keep insertion codes and avoid int() casts.
                    # Skip connections with missing residue numbers ('.' or '?').
                    p1_res_num_str = (p1_res_num or '').strip()
                    p1_ins_code_clean = (p1_ins_code or '').strip()
                    if p1_res_num_str in ('.', '?', ''):
                        continue
                    if p1_ins_code_clean in ('.', '?'):
                        p1_ins_code_clean = ''
                    p1_res_num_str = p1_res_num_str + p1_ins_code_clean
                    idx1 = (p1_chain, p1_res_num_str)

                    p2_res_num_str = (p2_res_num or '').strip()
                    p2_ins_code_clean = (p2_ins_code or '').strip()
                    if p2_res_num_str in ('.', '?', ''):
                        continue
                    if p2_ins_code_clean in ('.', '?'):
                        p2_ins_code_clean = ''
                    p2_res_num_str = p2_res_num_str + p2_ins_code_clean
                    idx2 = (p2_chain, p2_res_num_str)
                    # (后续的逻辑保持不变)
                    if idx1 == idx2: continue
                    if not (idx1 in pdb_idx and idx2 in pdb_idx): continue

                    atom1_obj = atom_lookup.get((p1_chain, p1_res_num_str, p1_atom_name))
                    atom2_obj = atom_lookup.get((p2_chain, p2_res_num_str, p2_atom_name))
                    if atom1_obj is None or atom2_obj is None: continue
                    
                    distance = np.linalg.norm(atom1_obj.get_coord() - atom2_obj.get_coord())
                    if distance > BOND_LENGTH_THRESHOLD: continue

                    # 仅当提供的 link_csv_path 指定了允许的键时才进行过滤
                    if allowed_link_types:
                        res1_up = (p1_res_name or '').upper()
                        res2_up = (p2_res_name or '').upper()
                        atom1_up = (p1_atom_name or '').upper()
                        atom2_up = (p2_atom_name or '').upper()
                        if (res1_up, res2_up, atom1_up, atom2_up) not in allowed_link_types:
                            # 不在允许列表中，跳过
                            continue
                    # Check if this link already exists in the links list
                    link_exists = False
                    for existing_link in links:
                        if ((existing_link["idx1"] == idx1 and existing_link["idx2"] == idx2 and
                             existing_link["res1"] == p1_res_name and existing_link["res2"] == p2_res_name) or
                            (existing_link["idx1"] == idx2 and existing_link["idx2"] == idx1 and
                             existing_link["res1"] == p2_res_name and existing_link["res2"] == p1_res_name)):
                            link_exists = True
                            break
                    
                    if link_exists:
                        continue
                    links.append({
                        "res1": p1_res_name, "idx1": idx1, "atom1": p1_atom_name,
                        "res2": p2_res_name, "idx2": idx2, "atom2": p2_atom_name,
                        "distance": distance
                    })

                except (ValueError, KeyError, IndexError) as e:
                    # INSERT_YOUR_CODE
                    # 打印出错的cif id（文件名），如果有
                    if 'cif_dict' in locals() and '_entry.id' in cif_dict:
                        print(f"Error in CIF id: {cif_dict['_entry.id'][0]}")
                    elif 'structure' in locals():
                        try:
                            print(f"Error in CIF structure id: {structure.id}")
                        except Exception:
                            pass
                    print(f"Could not parse a LINK/SSBOND record from CIF at index {i}: {e}")
                    continue
                    
        out["links"] = links

    return out


    

def process_target(pdb_path, parse_hetatom=False, center=True,parse_link=True, parse_alpha = True, link_csv_path=None):
    
    target_struct = parse_cif(pdb_path, parse_hetatom=parse_hetatom, parse_link=parse_link, link_csv_path=link_csv_path)

    # Zero-center positions
    ca_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True)
    if not center:
        ca_center = 0
    xyz = torch.from_numpy(target_struct["xyz"] - ca_center)
    seq_orig = torch.from_numpy(target_struct["seq"])
    atom_mask = torch.from_numpy(target_struct["mask"])

    out = {
        "xyz_14": xyz,
        "mask_14": atom_mask,
        "seq": seq_orig,
        "pdb_idx": target_struct["pdb_idx"], #[('A', '100'), (A', '101'), ...]
        "idx": target_struct["idx"], #[0,1,2,3,4,...]
        "chains": list(set([i[0] for i in target_struct["pdb_idx"]])),
    }
    if parse_hetatom:
        out["xyz_het"] = target_struct["xyz_het"]
        out["info_het"] = target_struct["info_het"]
    if parse_link:
        out["links"] = target_struct["links"]
    if parse_alpha:
        if openfold_get_torsions is None:
            raise ImportError(
                "openfold_get_torsions is not available. "
                "Torsion-angle features require extra dependencies (OpenFold/APM/mdtraj)."
            )
        L = xyz.shape[0]
        xyz_27 = torch.cat((xyz, torch.full((L,13,3), float('nan'))), dim=1)
        #alpha, alpha_alt, alpha_mask, _ = util.get_torsions(xyz_27.unsqueeze(0), seq_orig.unsqueeze(0), TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)

        alpha, alpha_alt, alpha_mask, _ = openfold_get_torsions(
            seq_orig.unsqueeze(0), xyz_27.unsqueeze(0)
        )
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha_alt[torch.isnan(alpha_alt)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_alt = alpha_alt.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        out['alpha'] = alpha.squeeze()
        out['alpha_tor_mask'] = alpha_mask.squeeze()
        out['alpha_alt'] = alpha_alt.squeeze()

    out['pdb_id'] = os.path.basename(pdb_path).split('.')[0]  # pdb_id is the name of the pdb file without extension

    return out


class Target:
    """
    Class to handle targets (parsed chains).

    """

    def __init__(self, conf: DictConfig, pdb_parsed, N_C_add=True):
        self.design_conf = conf
        self.N_C_add = N_C_add
        if self.design_conf.contigs is None:
            raise ValueError(
                "No design configuration provided"
            )
        self.chain_order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.pdb = pdb_parsed
        # if self.design_conf.input_pdb:      
        #     self.pdb = process_target(self.design_conf.input_pdb,
        #                             parse_hetatom=False, 
        #                             center=False,
        #                             parse_link=True)
        (
            self.full_seq,
            self.full_xyz,
            self.full_rf_idx,
            self.full_mask_str,  # False is fixed, True is changable
            self.full_mask_seq,  # False is fixed, True is changable
            self.full_pdb_idx,
            self.full_origin_pdb_idx,
            self.full_alpha,
            self.full_alpha_alt,
            self.full_alpha_tor_mask, # False is no sidechain  
        ) = self.parse_contigs(
                            self.design_conf.contigs, 
                            self.design_conf.length
                            )

        self.full_bond_matrix, self.full_bond_mask = self.parse_bond_condition()
        self.full_bond_matrix = smu.make_sub_doubly2doubly_stochastic(self.full_bond_matrix)

    def parse_bond_condition(self):

        # default perturbate all bond including in the res that contigs specify
        # the all pdb index below are remnamed, depend on the contigs, default chain index is ABCDEF... 
        # fix bond in chain A, B, A-B: ['A|A','B|B','A|B']
        # fix bond with in A100-200,B100-200 provied by PDB: ['A100-A200|B100-B200']
        # fix self-defined bond, 0 or 1: ['A100-A100|A200-A200:1:FIX'], mask is False, means the str is fixed
        # PNA(Partial Noise Addition) self-defined bond, 0 or 1: ['A100-A100|A200-A200:0:PNA'] mask is True, means the str is not fixed
        # fix start or end part: ['Astart-Astart|Aend-Aend:1:FIX'] or ['Astart-A2|A10-A10:1:FIX','B200-Bend|A100-A100:0:FIX']
        
        L = len(self.full_seq)
        bond_matrix = torch.zeros((L, L), dtype=torch.long)
        bond_mask = torch.ones((L, L), dtype=torch.bool) # True means changeable
        #bond_mask = bond_mask ^ torch.eye(L, dtype=torch.bool) # diagonal is fixed, means no change

        # If pdb input, check if links from pdb are preserved
        if self.pdb and 'links' in self.pdb and self.pdb['links']:
            N_C_add_enabled = self.N_C_add

            def resolve_endpoint(pdb_origin, atom_name):
                atom_up = (atom_name or '').upper()
                if not N_C_add_enabled:
                    # legacy behavior
                    if pdb_origin in self.full_origin_pdb_idx:
                        return self.full_origin_pdb_idx.index(pdb_origin)
                    return None

                # Prefer terminal node for first/last origins if backbone atom
                if hasattr(self, '_origin_to_nter_idx') and pdb_origin in getattr(self, '_origin_to_nter_idx') and atom_up in BACKBONE_ATOMS:
                    return self._origin_to_nter_idx[pdb_origin]
                if hasattr(self, '_origin_to_cter_idx') and pdb_origin in getattr(self, '_origin_to_cter_idx') and atom_up in BACKBONE_ATOMS:
                    return self._origin_to_cter_idx[pdb_origin]
                # Otherwise map to body residue index
                if hasattr(self, '_origin_to_body_idx') and pdb_origin in getattr(self, '_origin_to_body_idx'):
                    return self._origin_to_body_idx[pdb_origin]
                # Fallback to first occurrence
                if pdb_origin in self.full_origin_pdb_idx:
                    return self.full_origin_pdb_idx.index(pdb_origin)
                return None

            for link in self.pdb['links']:
                res1_pdb_idx = link['idx1']
                res2_pdb_idx = link['idx2']
                atom1 = link.get('atom1', None)
                atom2 = link.get('atom2', None)

                idx1 = resolve_endpoint(res1_pdb_idx, atom1)
                idx2 = resolve_endpoint(res2_pdb_idx, atom2)
                if idx1 is not None and idx2 is not None:
                    bond_matrix[idx1, idx2] = 1
                    bond_matrix[idx2, idx1] = 1

        if self.design_conf.bond_condition is None:
            return bond_matrix, bond_mask

        for bond_contig in self.design_conf.bond_condition:
            parts = bond_contig.split(':')
            res_parts = parts[0].split('|')
            
            res1_spec, res2_spec = res_parts[0], res_parts[1]

            def get_indices(spec):
                if len(spec) == 1: # Chain, e.g. 'A'
                    return [i for i, p_idx in enumerate(self.full_pdb_idx) if p_idx[0] == spec]
                
                range_parts = spec.split('-')

                def get_res_idx(res_spec):
                    # Support new style "A/2" as well as legacy "A2"
                    if '/' in res_spec:
                        chain_id, token = res_spec.split('/', 1)
                    else:
                        chain_id, token = res_spec[0], res_spec[1:]

                    if token == 'start':
                        chain_indices = [i for i, p_idx in enumerate(self.full_pdb_idx) if p_idx[0] == chain_id]
                        return min(chain_indices) if chain_indices else -1
                    elif token == 'end':
                        chain_indices = [i for i, p_idx in enumerate(self.full_pdb_idx) if p_idx[0] == chain_id]
                        return max(chain_indices) if chain_indices else -1
                    else:
                        res_num = int(token)
                        try:
                            return self.full_pdb_idx.index((chain_id, res_num))
                        except ValueError:
                            return -1

                start_idx = get_res_idx(range_parts[0])
                end_idx = get_res_idx(range_parts[1])
                
                if start_idx == -1 or end_idx == -1:
                    raise ValueError(f"Invalid residue specification for bond: {spec}")

                return list(range(start_idx, end_idx + 1))

            indices1 = get_indices(res1_spec)
            indices2 = get_indices(res2_spec)

            value = None # Default to None, meaning no specific value set
            mask_value = False # Default to FIX
                
            if len(parts) > 1:
                value = int(parts[1])
            if len(parts) > 2:
                if parts[2] == 'PNA' and self.design_conf.partial_t is None:
                    raise ValueError("Partial Noise Addition (PNA) requires partial_t to be set.")
                mask_value = (parts[2] == 'PNA')
            _bond_mask = bond_mask.clone()
            for i in indices1:
                for j in indices2:
                    # if i == j: continue
                    if value:
                        bond_matrix[i, j] = value
                        bond_matrix[j, i] = value
                    _bond_mask[i, j] = mask_value
                    _bond_mask[j, i] = mask_value

        # If any fixed entry equals 1 in a row/column, expand the fixed mask to the
        # whole row and column to respect the doubly-stochastic constraint.
        # Only entries explicitly fixed (mask False) and equal to 1 trigger expansion.
        fixed_one = (bond_matrix == 1) & (~_bond_mask)
        if fixed_one.any():
            rows_to_fix = fixed_one.any(dim=1)
            cols_to_fix = fixed_one.any(dim=0)
            if rows_to_fix.any():
                bond_mask[rows_to_fix, :] = False
            if cols_to_fix.any():
                bond_mask[:, cols_to_fix] = False
        
        return bond_matrix, bond_mask

    def parse_contigs(self, contigs, length=None, chain_offset=200):
        """
        Parse a contig from the pdb file.

        Args:
            - contigs: list of contigs, each contig is a dict contain the following keys
                - seq: sequence of the contig
                - xyz: coordinates of the contig
                - rf_index: residue index of the contig
            - bond_mask: not used
            - length: specify the length of each chain
            - chain_offset: offset for the next chain's residue index

        Outputs:
            - maskseq and maskstr: mask for the contig
            - new pdb index
        """
        self.chain_list = []
        for chain in contigs:
            contig_list = []
            for i, contig in enumerate(chain):
                contig_list.append(self.single_parse_contig(contig))
            self.chain_list.append(contig_list)
        
        # specify the length of the contig
        # example [10,30,-1], while -1 represent for free length
        if length is None:
            length = [-1 for chain in self.chain_list]
 
        full_seq = []
        full_xyz = []
        full_mask_str = []
        full_mask_seq = []
        full_origin_pdb_idx = [] # genetated pdb index is ('?','-1')
        full_pdb_idx = []
        full_rf_idx = []
        full_alpha = []
        full_alpha_alt = []
        full_alpha_tor_mask = []
        full_head_mask = []  # True at NTER positions
        full_tail_mask = []  # True at CTER positions

        # Prepare terminal/body origin mappings
        self.nter_indices = []
        self.cter_indices = []
        self._origin_to_body_idx = {}
        self._origin_to_nter_idx = {}
        self._origin_to_cter_idx = {}
        N_C_add_enabled = self.N_C_add
        
        current_offset = 0
        for i, chain in enumerate(self.chain_list):
            part_lengths = sample_parts([ran['length_range'] for ran in chain], length[i])
            if part_lengths is None:
                raise ValueError(f"Cannot satisfy length constraints for chain {i}")

            chain_seq = []
            chain_xyz = []
            chain_alpha = []
            chain_alpha_alt = []
            chain_alpha_tor_mask = []
            chain_rf_idx = []
            chain_origin_list = []
            chain_mask_str_list = []
            chain_mask_seq_list = []
            chain_start_idx = None

            for j, contig in enumerate(chain):
                part_len = part_lengths[j]
                if len(contig['seq']) > 0:
                    chain_seq.append(contig['seq'].clone())
                else:
                    chain_seq.append(torch.full((part_len,), 20, dtype=torch.long))

                if len(contig['xyz']) > 0:
                    chain_xyz.append(contig['xyz'].clone())
                elif contig.get('is_new', False):
                    # For newly created segments (New_), initialize ideal backbone heavy atoms (N, CA, C)
                    init_block = torch.full((part_len, 14, 3), float('nan'), dtype=torch.float32)
                    # N, CA, C from INIT_CRDS (indices 0,1,2)
                    init_block[:, 0, :] = che.INIT_CRDS[0]
                    init_block[:, 1, :] = che.INIT_CRDS[1]
                    init_block[:, 2, :] = che.INIT_CRDS[2]
                    chain_xyz.append(init_block)
                else:
                    chain_xyz.append(torch.full((part_len, 14, 3), np.nan, dtype=torch.float32))
                    
                if len(contig['alpha']) > 0:
                    chain_alpha.append(contig['alpha'].clone())
                else:
                    chain_alpha.append(torch.full((part_len,10,2), 0.0, dtype=torch.float32))

                if len(contig['alpha_alt']) > 0:
                    chain_alpha_alt.append(contig['alpha_alt'].clone())
                else:
                    chain_alpha_alt.append(torch.full((part_len,10,2), 0.0, dtype=torch.float32))

                if len(contig['alpha_tor_mask']) > 0:
                    chain_alpha_tor_mask.append(contig['alpha_tor_mask'].float().clone())
                else:
                    chain_alpha_tor_mask.append(torch.full((part_len,10), 0.0, dtype=torch.float32))

                if len(contig['origin_pdb_idx']) > 0:
                    chain_origin_list += contig['origin_pdb_idx']
                else:
                    chain_origin_list += [('?', '-1')] * part_len # new generated part in old pdb index is ('?','-1')
                
                if len(contig['rf_index']) > 0:
                    if chain_start_idx is None:
                        chain_start_idx = contig['rf_index'][0]
                    chain_rf_idx.append(contig['rf_index'] + current_offset - chain_start_idx)
                else:
                    chain_rf_idx.append(torch.arange(current_offset, current_offset + part_len))
                                                
                chain_mask_str_list += [contig['mask_str_value']] * part_len   # True means changeable
                chain_mask_seq_list += [contig['mask_seq_value']] * part_len

            chain_seq = torch.cat(chain_seq)
            chain_xyz = torch.cat(chain_xyz)
            chain_alpha = torch.cat(chain_alpha)
            chain_alpha_alt = torch.cat(chain_alpha_alt)
            chain_alpha_tor_mask = torch.cat(chain_alpha_tor_mask)
            chain_rf_idx = torch.cat(chain_rf_idx)

            # Preserve original contig order, optionally adding N/C terminal clones
            chain_id = self.chain_order[i]
            is_body_mask = [origin != ('?', '-1') for origin in chain_origin_list]
            body_indices = [k for k, v in enumerate(is_body_mask) if v]

            add_terminals = N_C_add_enabled and len(body_indices) > 0

            # Start with original order
            chain_seq_new = chain_seq
            chain_xyz_new = chain_xyz
            chain_alpha_new = chain_alpha
            chain_alpha_alt_new = chain_alpha_alt
            chain_alpha_mask_new = chain_alpha_tor_mask
            chain_rf_idx_new = chain_rf_idx
            chain_origin_new = list(chain_origin_list)
            chain_mask_str_new = list(chain_mask_str_list)
            chain_mask_seq_new = list(chain_mask_seq_list)

            nter_local_idx = None
            cter_local_idx = None

            if add_terminals:
                first_idx = body_indices[0]
                last_idx = body_indices[-1]
                # NTER clone from first body residue (prepend)
                chain_seq_new = torch.cat([chain_seq[first_idx:first_idx+1], chain_seq_new])
                chain_xyz_new = torch.cat([chain_xyz[first_idx:first_idx+1], chain_xyz_new])
                chain_alpha_new = torch.cat([chain_alpha[first_idx:first_idx+1], chain_alpha_new])
                chain_alpha_alt_new = torch.cat([chain_alpha_alt[first_idx:first_idx+1], chain_alpha_alt_new])
                chain_alpha_mask_new = torch.cat([chain_alpha_tor_mask[first_idx:first_idx+1], chain_alpha_mask_new])
                chain_rf_idx_new = torch.cat([chain_rf_idx[first_idx:first_idx+1], chain_rf_idx_new])
                chain_origin_new = [chain_origin_list[first_idx]] + chain_origin_new
                chain_mask_str_new = [chain_mask_str_list[first_idx]] + chain_mask_str_new
                chain_mask_seq_new = [chain_mask_seq_list[first_idx]] + chain_mask_seq_new
                nter_local_idx = 0
                # CTER clone from last body residue (append)
                # chain_seq_new = torch.cat([chain_seq_new, chain_seq[last_idx:last_idx+1]])
                # chain_xyz_new = torch.cat([chain_xyz_new, chain_xyz[last_idx:last_idx+1]])
                # chain_alpha_new = torch.cat([chain_alpha_new, chain_alpha[last_idx:last_idx+1]])
                # chain_alpha_alt_new = torch.cat([chain_alpha_alt_new, chain_alpha_alt[last_idx:last_idx+1]])
                # chain_alpha_mask_new = torch.cat([chain_alpha_mask_new, chain_alpha_tor_mask[last_idx:last_idx+1]])
                # chain_rf_idx_new = torch.cat([chain_rf_idx_new, chain_rf_idx[last_idx:last_idx+1]])
                # chain_origin_new = chain_origin_new + [chain_origin_list[last_idx]]
                # chain_mask_str_new = chain_mask_str_new + [chain_mask_str_list[last_idx]]
                # chain_mask_seq_new = chain_mask_seq_new + [chain_mask_seq_list[last_idx]]
                # cter_local_idx = chain_seq_new.shape[0] - 1
                
                # CTER clone from last body residue (insert after body residues, before padding)
                # After adding NTER, chain_seq_new = [NTER] + [body_residues] + [padding]
                # last_idx is the index of last body residue in original chain_seq
                # In chain_seq_new, last body residue is at position (last_idx + 1)
                # Insert CTER right after the last body residue, before padding
                cter_insert_pos = last_idx + 2  # +1 for NTER prepended, +1 to insert after last body residue
                chain_seq_new = torch.cat([
                    chain_seq_new[:cter_insert_pos],
                    chain_seq[last_idx:last_idx+1],
                    chain_seq_new[cter_insert_pos:]
                ])
                chain_xyz_new = torch.cat([
                    chain_xyz_new[:cter_insert_pos],
                    chain_xyz[last_idx:last_idx+1],
                    chain_xyz_new[cter_insert_pos:]
                ])
                chain_alpha_new = torch.cat([
                    chain_alpha_new[:cter_insert_pos],
                    chain_alpha[last_idx:last_idx+1],
                    chain_alpha_new[cter_insert_pos:]
                ])
                chain_alpha_alt_new = torch.cat([
                    chain_alpha_alt_new[:cter_insert_pos],
                    chain_alpha_alt[last_idx:last_idx+1],
                    chain_alpha_alt_new[cter_insert_pos:]
                ])
                chain_alpha_mask_new = torch.cat([
                    chain_alpha_mask_new[:cter_insert_pos],
                    chain_alpha_tor_mask[last_idx:last_idx+1],
                    chain_alpha_mask_new[cter_insert_pos:]
                ])
                chain_rf_idx_new = torch.cat([
                    chain_rf_idx_new[:cter_insert_pos],
                    chain_rf_idx[last_idx:last_idx+1],
                    chain_rf_idx_new[cter_insert_pos:]
                ])
                chain_origin_new = (
                    chain_origin_new[:cter_insert_pos] +
                    [chain_origin_list[last_idx]] +
                    chain_origin_new[cter_insert_pos:]
                )
                chain_mask_str_new = (
                    chain_mask_str_new[:cter_insert_pos] +
                    [chain_mask_str_list[last_idx]] +
                    chain_mask_str_new[cter_insert_pos:]
                )
                chain_mask_seq_new = (
                    chain_mask_seq_new[:cter_insert_pos] +
                    [chain_mask_seq_list[last_idx]] +
                    chain_mask_seq_new[cter_insert_pos:]
                )
                cter_local_idx = cter_insert_pos

            # Record global indices and mappings
            chain_global_start = sum(len(t) for t in full_seq)
            if add_terminals:
                nter_global = chain_global_start + (nter_local_idx or 0)
                cter_global = chain_global_start + (cter_local_idx or 0)
                self.nter_indices.append(nter_global)
                self.cter_indices.append(cter_global)
                # Map terminal origins to terminal indices
                self._origin_to_nter_idx[chain_origin_new[nter_local_idx]] = nter_global
                self._origin_to_cter_idx[chain_origin_new[cter_local_idx]] = cter_global

            # Map body origins to body indices (first occurrence only)
            # Body indices in new chain start at (1 if add_terminals else 0)
            body_start_local = 1 if add_terminals else 0
            for j, origin in enumerate(chain_origin_new):
                if origin == ('?', '-1'):
                    continue
                # Prefer mapping to body positions, not terminals
                if add_terminals and (j == nter_local_idx or j == cter_local_idx):
                    continue
                global_j = chain_global_start + j
                if origin not in self._origin_to_body_idx:
                    self._origin_to_body_idx[origin] = global_j

            # Center NEW segments to FIX centroid using CA atoms
            try:
                new_mask_bool = torch.tensor([orig == ('?', '-1') for orig in chain_origin_new], dtype=torch.bool)
                fix_mask_bool = torch.tensor([not m for m in chain_mask_str_new], dtype=torch.bool)
                if fix_mask_bool.any() and new_mask_bool.any():
                    fix_ca = chain_xyz_new[fix_mask_bool, 1, :]
                    valid_fix = ~torch.isnan(fix_ca).any(dim=-1)
                    if valid_fix.any():
                        fix_center = fix_ca[valid_fix].mean(dim=0)
                        new_ca = chain_xyz_new[new_mask_bool, 1, :]
                        valid_new = ~torch.isnan(new_ca).any(dim=-1)
                        if valid_new.any():
                            new_center = new_ca[valid_new].mean(dim=0)
                            delta = fix_center - new_center
                            new_xyz = chain_xyz_new[new_mask_bool]
                            valid_xyz = ~torch.isnan(new_xyz)
                            chain_xyz_new[new_mask_bool] = torch.where(valid_xyz, new_xyz + delta, new_xyz)
            except Exception:
                pass

            # Build per-chain head/tail masks
            chain_head_mask = torch.zeros(chain_seq_new.shape[0], dtype=torch.bool)
            chain_tail_mask = torch.zeros(chain_seq_new.shape[0], dtype=torch.bool)
            if add_terminals:
                chain_head_mask[nter_local_idx] = True
                chain_tail_mask[cter_local_idx] = True
            full_head_mask.append(chain_head_mask)
            full_tail_mask.append(chain_tail_mask)

            # Append to full arrays
            full_seq.append(chain_seq_new)
            full_xyz.append(chain_xyz_new)
            full_alpha.append(chain_alpha_new)
            full_alpha_alt.append(chain_alpha_alt_new)
            full_alpha_tor_mask.append(chain_alpha_mask_new)
            full_rf_idx.append(chain_rf_idx_new)
            full_origin_pdb_idx += chain_origin_new
            full_mask_str += chain_mask_str_new
            full_mask_seq += chain_mask_seq_new

            # Renumber designed pdb_idx per chain after reordering
            pdb_idx = [(chain_id, k + 1) for k in range(len(chain_seq_new))]
            full_pdb_idx.append(pdb_idx)
            current_offset += chain_offset
            # keep legacy offset progression
            current_offset += chain_rf_idx_new[-1].item() 
        
        full_rf_idx = torch.cat(full_rf_idx)
        full_rf_idx = full_rf_idx - full_rf_idx.min()  # make sure rf_idx starts from 0
        # Expose head/tail masks (1D over full length)
        self.full_head_mask = torch.cat(full_head_mask) if len(full_head_mask) > 0 else torch.zeros(0, dtype=torch.bool)
        self.full_tail_mask = torch.cat(full_tail_mask) if len(full_tail_mask) > 0 else torch.zeros(0, dtype=torch.bool)
        return (
            torch.cat(full_seq),
            torch.cat(full_xyz).nan_to_num(0.0),
            full_rf_idx,
            torch.tensor(full_mask_str),
            torch.tensor(full_mask_seq),
            [item for sublist in full_pdb_idx for item in sublist],
            full_origin_pdb_idx,
            torch.cat(full_alpha),
            torch.cat(full_alpha_alt),  # [L,10,2] for alpha torsions, [L,10,2] for alpha torsion alt
            torch.cat(full_alpha_tor_mask),  # [L,10,2] for alpha torsions, [L,10,1] for alpha torsion mask
        )

    def single_parse_contig(self, contig, inference=False):
        """
        Outputs:
            - length range: [min,max]
            - seq: sequence of the contig
            - xyz: xyz coordinates of the contig
            - maskseq and maskstr: mask for the contig
        """

        # select_range:fix_type:fix_type
        # fix_type: str_FIX, seq_FIX, seq_PNA, str_PNA, seq_DNV, str_DNV
        # FIX, DNV(De NoVo), PNA(Partial Noise Addition), PNA need partial_t, DNV and PNA cannot coexist
        # range chain: ['Chain_A:seq_FIX:str_FIX'','Chain_B:seq_FIX:str_DNV']
        # range res: 'A100-A200:seq_FIX:str_DNV','B100-B200:seq_PNA:str_PNA'
        # range 10-20, default and only seq_DNV, str_DNV
        # insert self-defined 'AGGGKI:seq_FIX:str_DNV', self-defined seq can only select str_DNV
        # example: [['Chain_A:seq_FIX:str_FIX'],
        #           ['New_30-40','B100-B120:seq_FIX:str_DNV','New_10-20',
        #           'B160-B200:seq_DNV:str_FIX','10-20','AGISHK:seq_FIX:str_DNV']]
        # example: [['Chain_A:seq_FIX:str_FIX'],
        #           ['B100-B120:seq_FIX:str_PNA'
        #          ]

        seq = []
        xyz = []
        alpha = []
        alpha_alt = []
        alpha_tor_mask = []
        mask_str_value = True  # default is True, means the str is not fixed
        mask_seq_value = True # default is True, means the seq is not fixed
        length_range = [0, 0] # min length, max length
        rf_index = []
        origin_pdb_idx = []
        index = [] # the index in the pdb_parsed list
        # the part need to be fixed
        if ":" in contig:         
            if self.pdb is None:
                raise ValueError(f"No pdb file provided, cannot parse contig {contig}")
        contig = contig.split(":")

        ##########################################
        # parse the contig[0], namely select range
        ##########################################
        # if it is a chain, example: 'Chain_A'
        if "Chain_" in contig[0]:
            chain_id = contig[0].split("_")[1]  # e.g. 'A'
            if chain_id not in self.pdb["chains"]:
                raise ValueError(f"Chain {chain_id} not found in the pdb file")
            index = [i for i in range(len(self.pdb["pdb_idx"])) if self.pdb["pdb_idx"][i][0] == chain_id]
            length_range[0] = len(index)
            length_range[1] = len(index)
        # New_10-20
        elif "New_" in contig[0]:
            if self.design_conf.partial_t is not None and inference:
                raise ValueError(f"Partial t is not supported for contig {contig[0]}")
            if (len(contig) > 1 and contig[1].split("_")[1] != "DNV") or (len(contig) > 2 and contig[2].split("_")[1] != "DNV"):
                raise ValueError(f"Only DNV is supported for contig {contig[0]}")
            parts = contig[0].strip('New_').split("-")
            length_range[0] = int(parts[0])
            length_range[1] = int(parts[1])
            # mark this contig as newly generated for downstream ideal coord filling
            is_new = True
        # if it is a range residue, example: 'A100-A200' or '7100-7200' while '7' is chain id or 'A-10-B-20' -10 to -20
        elif "-" in contig[0]: #and contig[0][0] in self.pdb["chains"] and contig[0].split("-")[1][0] in self.pdb["chains"]:        
            # if it is a range, example: 'A100-A200'
            parts = self.parse_residue_range(contig[0])
            part1 = (parts['start_chain'], parts['start_res'])  # e.g. ('A', '100')
            part2 = (parts['end_chain'], parts['end_res'])
            if part1[0] not in self.pdb["chains"] or part2[0] not in self.pdb["chains"]:
                raise ValueError(f"Chain {part1[0]} or {part2[0]} not found in the {self.pdb['pdb_id']} pdb file")
            # check if the residue is in the pdb file
            if part1 not in self.pdb["pdb_idx"] or part2 not in self.pdb["pdb_idx"]:
                raise ValueError(f"Residue {part1} or {part2} not found in the {self.pdb['pdb_id']} pdb file")
            index1 = self.pdb["pdb_idx"].index(part1)
            index2 = self.pdb["pdb_idx"].index(part2)
            if index1 <= index2:                   
                index = [i for i in range(index1, index2 + 1)]
            else:
                raise ValueError(f"Residue {part1} is after {part2} in the pdb file")
            length_range[0] = len(index)
            length_range[1] = len(index)
        # e.x. AGISHK
        elif contig[0].isalpha():
            if self.design_conf.partial_t is not None:
                raise ValueError(f"Partial t is not supported for contig {contig[0]}")
            if contig[1] != "str_DNV" and contig[2] != "str_DNV":
                raise ValueError(f"Only str_DNV is supported for contig {contig[0]}")
            seq = [one_aa2num[c] for c in contig[0]] # interge seq
            seq = torch.tensor(seq)
            length_range[0] = len(seq)
            length_range[1] = len(seq)
        else:
            raise ValueError(f"Invalid contig format for range {contig[0]}")
        
        ##########################################
        # parse the contig[1] and contig[2], namely parse the type of fixed contig
        ########################################## 
        mask_type = {"PNA":True, "FIX":False, "DNV":True} 
        # parse_alpha = 0
        if len(contig) > 2:
            for con in [contig[1],contig[2]]:
                data_type = con.split("_")[0]
                process_type = con.split("_")[1]
                if process_type == "PNA" and self.design_conf.partial_t is None:
                    raise ValueError(f"Partial t is null , and not supported for contig {con}")
                # if process_type == "FIX":
                #     parse_alpha += 1
                if process_type == "FIX" or process_type == "PNA":
                    # if it is a fixed type, example: 'Chain_A:seq_FIX:str_FIX'
                    if data_type == "seq":
                        seq = self.pdb["seq"][index] if len(index) > 0 else seq
                        mask_seq_value = mask_type[process_type]  # True is changable, False is fixed
                    elif data_type == "str":
                        xyz = self.pdb["xyz_14"][index] if len(index) > 0 else xyz
                        mask_str_value = mask_type[process_type]
        # if parse_alpha >= 2:
        alpha = self.pdb["alpha"][index]
        alpha_tor_mask = self.pdb['alpha_tor_mask'][index]
        alpha_alt = self.pdb["alpha_alt"][index]
        if len(index) > 0:    
            #start_rf_index = self.pdb["idx"][index[0]]  # residue 0-based index
            rf_index = [self.pdb["idx"][i] for i in index]  # residue 0-based index
            origin_pdb_idx = [self.pdb["pdb_idx"][i] for i in index]  # pdb index
        # default to False if not set
        if 'is_new' not in locals():
            is_new = False
        return {"seq": seq, "xyz": xyz, "length_range": length_range, 
                "rf_index": torch.tensor(rf_index), "origin_pdb_idx":origin_pdb_idx,
                "mask_seq_value":mask_seq_value, "mask_str_value":mask_str_value,
                "alpha": alpha,"alpha_alt":alpha_alt, "alpha_tor_mask": alpha_tor_mask,
                "is_new": is_new}
    
    def parse_residue_range(self,range_str):
        """
        健壮地解析使用 '/' 分隔符的残基范围字符串，支持负数残基。

        Args:
            range_str (str): 表示残基范围的字符串。
                            支持格式: 'A/100-A/200', 'A/100-200', 'B/-8-B/117'.

        Returns:
            dict: 包含起始和结束链ID及残基标识符的字典。
        """
        # Regex to handle negative residue numbers correctly.
        # It captures start_chain, start_res, optional end_chain, and end_res.
        pattern = re.compile(
            r"^(?P<start_chain>[^/]+)/(?P<start_res>-?\d+[A-Za-z]*)"  # Start: C/R
            r"-"                                                      # Separator
            r"(?:(?P<end_chain>[^/]+)/)?"                             # Optional End Chain: C/
            r"(?P<end_res>-?\d+[A-Za-z]*)$"                           # End Residue: R
        )
        match = pattern.match(range_str.strip())

        if not match:
            raise ValueError(f"Invalid range format: {range_str}. Expected 'chain/res-res' or 'chain/res-chain/res'.")

        parts = match.groupdict()
        start_chain = parts['start_chain']
        end_chain = parts['end_chain'] if parts['end_chain'] else start_chain

        return {
            'start_chain': start_chain,
            'start_res': parts['start_res'],
            'end_chain': end_chain,
            'end_res': parts['end_res']
        }

def sample_parts(intervals, total_length):
    """
    采样每个部分的值，使得总和等于 total_length，且每个值在对应区间内。
    如果 total_length 为 -1，则在每个区间内随机采样一个值，不考虑总和。
    
    参数:
        intervals: 列表的列表，每个子列表 [min, max] 表示一个部分的区间。
        total_length: 整数，目标总长度。如果为-1，则随机采样。
        
    返回:
        一个列表，包含每个部分采样后的值（整数），或 None（如果无解）。
    """
    # 验证输入区间
    min_vals = []
    max_vals = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("每个区间必须包含两个元素 [min, max]")
        min_val, max_val = interval
        if min_val > max_val:
            raise ValueError(f"区间 {interval} 无效：min 不能大于 max")
        min_vals.append(min_val)
        max_vals.append(max_val)

    # 如果 total_length 是 -1，则随机采样
    if total_length == -1:
        return [random.randint(min_v, max_v) for min_v, max_v in zip(min_vals, max_vals)]
    
    # 计算最小和、最大和
    min_sum = sum(min_vals)
    max_sum = sum(max_vals)
    
    # 可行性检查
    if total_length < min_sum or total_length > max_sum:
        return None
    
    n = len(intervals)
    slacks = [max_vals[i] - min_vals[i] for i in range(n)];  # 每个部分的松弛空间
    remaining = total_length - min_sum  # 剩余需要分配的长度
    
    # 初始化每个部分为最小值
    output = min_vals.copy()
    
    # 准备可增加部分的索引列表（只包含松弛空间大于 0 的部分）
    valid_indices = [i for i in range(n) if slacks[i] > 0]
    
    # 分配剩余长度：每次随机选择一个可增加部分，增加 1，直到剩余为 0
    for _ in range(remaining):
        if not valid_indices:
            break
        # 随机选择一个可增加的部分
        idx = random.choice(valid_indices)
        output[idx] += 1
        # 如果该部分达到最大值，从有效索引中移除
        if output[idx] == max_vals[idx]:
            valid_indices.remove(idx)
    
    return output


def _find_closest_chain_and_interface(pdb_parsed, target_chain_id, cut_off=8.0):
    """
    Finds the chain closest to the target chain and returns their interface residues.
    Uses cKDTree for efficient spatial searching.
    """
    target_chain_mask = (pdb_parsed['pdb_idx_to_chain_id'] == target_chain_id)
    target_indices = np.where(target_chain_mask)[0]
    target_coords = pdb_parsed['xyz'][target_chain_mask][:, 1]

    if target_coords.size == 0:
        return None, np.array([], dtype=int), np.array([], dtype=int)

    max_num_contacts = 0
    closest_chain_id = None
    best_interface1 = np.array([], dtype=int)
    best_interface2 = np.array([], dtype=int)

    chains = np.unique(pdb_parsed['pdb_idx_to_chain_id'])
    for chain_id in chains:
        if chain_id == target_chain_id:
            continue

        chain_mask = (pdb_parsed['pdb_idx_to_chain_id'] == chain_id)
        chain_indices = np.where(chain_mask)[0]
        chain_coords = pdb_parsed['xyz'][chain_mask][:, 1]

        if chain_coords.size == 0:
            continue

        # Use cKDTree for efficient distance calculation
        tree = cKDTree(chain_coords)
        contact_indices_list = tree.query_ball_point(target_coords, r=cut_off)
        
        num_contacts = sum(len(indices) for indices in contact_indices_list)

        if num_contacts > max_num_contacts:
            max_num_contacts = num_contacts
            closest_chain_id = chain_id
            
            # Determine interface residues based on contacts
            interface1_mask = np.array([len(indices) > 0 for indices in contact_indices_list], dtype=bool)
            best_interface1 = target_indices[interface1_mask]
            
            contact_indices_flat = np.unique([idx for sublist in contact_indices_list for idx in sublist])
            best_interface2 = chain_indices[contact_indices_flat]

    return closest_chain_id, best_interface1, best_interface2

def _generate_random_fixed_contigs(chain_id, res_indices, pdb_idx_map, proportion, segments):
    """
    Splits a designable contig into smaller designable and fixed contigs randomly.
    """
    num_residues = len(res_indices)
    if num_residues == 0:
        return []
    if proportion == 0 or segments == 0:
        start_res, end_res = pdb_idx_map[res_indices[0]][1], pdb_idx_map[res_indices[-1]][1]
        return [f"{chain_id}/{start_res}-{chain_id}/{end_res}:seq_PNA:str_PNA"]

    num_to_fix = int(num_residues * proportion)
    if num_to_fix == 0: 
        start_res, end_res = pdb_idx_map[res_indices[0]][1], pdb_idx_map[res_indices[-1]][1]
        return [f"{chain_id}/{start_res}-{chain_id}/{end_res}:seq_PNA:str_PNA"]

    # Generate random segment lengths
    fix_lengths = np.random.multinomial(num_to_fix, np.ones(segments)/segments)
    fix_lengths = fix_lengths[fix_lengths > 0]
    segments = len(fix_lengths)

    design_len = num_residues - num_to_fix
    # Allocate designable residues to gaps between fixed segments
    gaps = segments + 1
    design_lengths = np.random.multinomial(design_len, np.ones(gaps)/gaps)

    contigs = []
    current_idx = 0
    for i in range(segments):
        # Add design segment
        if design_lengths[i] > 0:
            start_res_idx = res_indices[current_idx]
            end_res_idx = res_indices[current_idx + design_lengths[i] - 1]
            start_pdb, end_pdb = pdb_idx_map[start_res_idx][1], pdb_idx_map[end_res_idx][1]
            contigs.append(f"{chain_id}/{start_pdb}-{chain_id}/{end_pdb}:seq_PNA:str_PNA")
            current_idx += design_lengths[i]

        # Add fixed segment
        if fix_lengths[i] > 0:
            start_res_idx = res_indices[current_idx]
            end_res_idx = res_indices[current_idx + fix_lengths[i] - 1]
            start_pdb, end_pdb = pdb_idx_map[start_res_idx][1], pdb_idx_map[end_res_idx][1]
            contigs.append(f"{chain_id}/{start_pdb}-{chain_id}/{end_pdb}:seq_FIX:str_FIX")
            current_idx += fix_lengths[i]
        
    # Add trailing design segment
    if design_lengths[gaps-1] > 0:
        start_res_idx = res_indices[current_idx]
        end_res_idx = res_indices[current_idx + design_lengths[gaps-1] - 1]
        start_pdb, end_pdb = pdb_idx_map[start_res_idx][1], pdb_idx_map[end_res_idx][1]
        contigs.append(f"{chain_id}/{start_pdb}-{chain_id}/{end_pdb}:seq_PNA:str_PNA")

    return contigs

def _split_indices_into_contiguous_blocks(indices):
    """Splits a list of indices into sub-lists of contiguous indices."""
    if len(indices) == 0:
        return []
    blocks = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    return [b.tolist() for b in blocks]

def generate_crop_contigs(
    pdb_parsed,
    target_chain_id,
    mode: str = 'monomer',
    crop_length: int = 100,
    fixed_res=None,
    expand_preference: str = "auto",
    target_expand_bias: float = 1.0,
    target_len_ratio: float = None,
):
    """
    Generates contigs for cropping a protein from a parsed PDB file.

    Args:
        pdb_parsed (dict): Parsed PDB data from process_target.
        target_chain_id (str): The chain ID to crop from.
        mode (str): 'monomer' or 'complex'.
        crop_length (int): The desired length of the cropped segment.
        fixed_res (dict): Dictionary to specify random fixing, e.g. {'proportion': 0.2, 'segments': 2}.
        expand_preference (str): How to bias interface expansion in complex mode.
            - 'auto' (default): use room-based heuristic with a soft bias controlled by target_expand_bias.
            - 'target': always expand target chain when it still has room; fall back to neighbor only when needed.
            - 'neighbor': symmetric to 'target', but favor the partner chain.
        target_expand_bias (float): Only used when expand_preference == 'auto'.
            - > 1.0: softly prefer expanding the target chain when both chains have room.
            - = 1.0: no additional bias (legacy behaviour).
            - 0.0–1.0: softly prefer expanding the neighbour chain (0.0 ~ only neighbour, if possible).
        target_len_ratio (float, optional): Desired final fraction of residues on the target chain
            in complex mode, used only at the final cropping step when current_len > crop_length.
            - If in (0,1), we try to set len(target) ≈ crop_length * target_len_ratio (with clamping
              so that neither chain asks for more residues than it actually has).
            - If None, fall back to using the natural ratio len1 / (len1 + len2).

    Returns:
        tuple: A tuple containing:
            - contigs (list): The generated contigs.
            - res_mask (torch.Tensor): A mask indicating non-padded residues.
    """
    # Add pdb_idx_to_chain_id to pdb_parsed if it doesn't exist
    if 'pdb_idx_to_chain_id' not in pdb_parsed:
        pdb_parsed['pdb_idx_to_chain_id'] = np.array([i[0] for i in pdb_parsed['pdb_idx']])
    
    # Use the correct key for coordinates
    if 'xyz_14' in pdb_parsed and 'xyz' not in pdb_parsed:
        pdb_parsed['xyz'] = pdb_parsed['xyz_14']

    contigs = []
    final_indices = []

    if mode == 'monomer':
        chain_indices = np.where(pdb_parsed['pdb_idx_to_chain_id'] == target_chain_id)[0]
        if len(chain_indices) == 0:
            raise ValueError(f"Chain {target_chain_id} not found in PDB.")

        if len(chain_indices) > crop_length:
            start = np.random.randint(0, len(chain_indices) - crop_length + 1)
        else:
            start = 0
        end = start + crop_length
        
        cropped_indices = chain_indices[max(0, start):min(len(chain_indices), end)]
        final_indices = cropped_indices.tolist()
        
        if fixed_res:
            contigs.append(_generate_random_fixed_contigs(
                target_chain_id, final_indices, pdb_parsed['pdb_idx'], 
                fixed_res.get('proportion', 0), fixed_res.get('segments', 0)
            ))
        else:
            if final_indices:
                res_start = pdb_parsed['pdb_idx'][final_indices[0]][1]
                res_end = pdb_parsed['pdb_idx'][final_indices[-1]][1]
                contigs.append([f"{target_chain_id}/{res_start}-{target_chain_id}/{res_end}:seq_PNA:str_PNA"])

    elif mode == 'complex':
        neighbor_chain_id, interface1, interface2 = _find_closest_chain_and_interface(pdb_parsed, target_chain_id)
        if neighbor_chain_id is None:
            raise ValueError(f"No neighboring chain found for chain {target_chain_id}.")
        
        # Expand if necessary
        current_len = len(interface1) + len(interface2)
        if current_len < crop_length:
            needed = crop_length - current_len
            
            chain1_indices = np.where(pdb_parsed['pdb_idx_to_chain_id'] == target_chain_id)[0]
            chain2_indices = np.where(pdb_parsed['pdb_idx_to_chain_id'] == neighbor_chain_id)[0]

            # Find interface boundaries within the full chain
            if len(interface1) > 0:
                min_idx1, max_idx1 = np.where(np.isin(chain1_indices, interface1))[0][[0, -1]]
            else:  # Handle case where one interface is empty
                min_idx1, max_idx1 = len(chain1_indices) // 2, len(chain1_indices) // 2

            if len(interface2) > 0:
                min_idx2, max_idx2 = np.where(np.isin(chain2_indices, interface2))[0][[0, -1]]
            else:
                min_idx2, max_idx2 = len(chain2_indices) // 2, len(chain2_indices) // 2

            # Expand outwards, preferring the chain that still has free residues.
            # 这里修正了原先 room 计算可能为负、导致提前停止扩展的问题，
            # 确保在有真实残基可以利用时尽量不用 New_ padding。
            expand1, expand2 = 0, 0
            len1 = len(chain1_indices)
            len2 = len(chain2_indices)

            # Sanitize bias
            if target_expand_bias is None or target_expand_bias < 0:
                target_expand_bias = 1.0

            for _ in range(needed):
                # 剩余可扩展空间（分别统计 N 端和 C 端，避免出现负数）
                left1 = max(0, min_idx1 - expand1)
                right1 = max(0, (len1 - 1) - (max_idx1 + expand1))
                room1 = left1 + right1

                left2 = max(0, min_idx2 - expand2)
                right2 = max(0, (len2 - 1) - (max_idx2 + expand2))
                room2 = left2 + right2

                # 两条链都没有空间了，提前停止
                if room1 <= 0 and room2 <= 0:
                    break

                # 根据策略和偏好选择扩展哪条链
                if expand_preference == "target":
                    # 尽量优先扩展目标链
                    if room1 > 0:
                        expand1 += 1
                    elif room2 > 0:
                        expand2 += 1
                elif expand_preference == "neighbor":
                    # 尽量优先扩展邻接链
                    if room2 > 0:
                        expand2 += 1
                    elif room1 > 0:
                        expand1 += 1
                else:
                    # 'auto'：按剩余 room，再叠加 target_expand_bias 作为软偏好
                    weighted_room1 = room1 * target_expand_bias
                    weighted_room2 = room2

                    # 优先扩展仍有空间且“加权后更宽松”的那条链；
                    # 如果 neighbor 链已经没有空间了，就只扩 target 链。
                    if weighted_room1 >= weighted_room2 and room1 > 0:
                        expand1 += 1
                    elif room2 > 0:
                        expand2 += 1

            interface1 = chain1_indices[
                max(0, min_idx1 - expand1) : min(len1, max_idx1 + expand1 + 1)
            ]
            interface2 = chain2_indices[
                max(0, min_idx2 - expand2) : min(len2, max_idx2 + expand2 + 1)
            ]

        # Crop if necessary
        current_len = len(interface1) + len(interface2)
        if current_len > crop_length:
            len1, len2 = len(interface1), len(interface2)

            # ---- Step 1: decide target / neighbour lengths ----
            if target_len_ratio is not None and 0.0 < float(target_len_ratio) < 1.0:
                # User-specified ratio: target ~ crop_length * r, neighbour ~ crop_length * (1-r)
                desired_t = int(round(crop_length * float(target_len_ratio)))
                desired_t = max(1, min(desired_t, crop_length - 1))

                # Clamp by what each chain actually has
                new_len1 = min(desired_t, len1)
                new_len2 = crop_length - new_len1
                if new_len2 > len2:
                    new_len2 = len2
                    new_len1 = crop_length - new_len2
                # Final safety clamp (in pathological cases fall back to natural ratio)
                if new_len1 <= 0 or new_len2 <= 0:
                    new_len1 = round(len1 / current_len * crop_length)
                    new_len2 = crop_length - new_len1
            else:
                # Legacy behaviour: use natural ratio len1 : len2
                new_len1 = round(len1 / current_len * crop_length)
                new_len2 = crop_length - new_len1

            # ---- Step 2: randomly crop contiguous windows on each chain ----
            max_s1 = len1 - new_len1
            max_s2 = len2 - new_len2
            s1 = np.random.randint(0, max_s1 + 1) if max_s1 > 0 else 0
            s2 = np.random.randint(0, max_s2 + 1) if max_s2 > 0 else 0
            interface1, interface2 = interface1[s1:s1+new_len1], interface2[s2:s2+new_len2]

        final_indices = np.concatenate([interface1, interface2]).tolist()

        # Generate contigs for each chain
        target_blocks = _split_indices_into_contiguous_blocks(interface1)
        neighbor_blocks = _split_indices_into_contiguous_blocks(interface2)
        target_contigs_list = []
        if fixed_res:
             # Apply random fixing only on the designable chain (target)
            # target_contigs_list.extend(_generate_random_fixed_contigs(
            #     target_chain_id, interface1, pdb_parsed['pdb_idx'],
            #     fixed_res.get('proportion', 0), fixed_res.get('segments', 0)
            # ))
            for block in target_blocks:
                target_contigs_list.extend(_generate_random_fixed_contigs(
                    target_chain_id, block, pdb_parsed['pdb_idx'],
                    fixed_res['proportion'], fixed_res['segments']
                ))
        else:
            for block in target_blocks:
                start_res, end_res = pdb_parsed['pdb_idx'][block[0]][1], pdb_parsed['pdb_idx'][block[-1]][1]
                target_contigs_list.append(f"{target_chain_id}/{start_res}-{target_chain_id}/{end_res}:seq_PNA:str_PNA")
        contigs.append(target_contigs_list)

        neighbor_contigs_list = []
        for block in neighbor_blocks:
            start_res, end_res = pdb_parsed['pdb_idx'][block[0]][1], pdb_parsed['pdb_idx'][block[-1]][1]
            neighbor_contigs_list.append(f"{neighbor_chain_id}/{start_res}-{neighbor_chain_id}/{end_res}:seq_FIX:str_FIX")
        contigs.append(neighbor_contigs_list)

    else:
        raise ValueError("Mode must be 'monomer' or 'complex'.")

    total_len = len(final_indices)
    # Padding and res_mask
    padding = crop_length - total_len
    if padding > 0:
        # 为了不浪费设计位点：
        # - monomer: 仍然加在唯一一条链（最后一条链）的最后
        # - complex: 把 New 残基补在“设计链”（第一条链, contigs[0]）上
        if not contigs:
            contigs.append([])
        if mode == 'complex':
            # 设计链在 generate_crop_contigs 中总是作为第一条链加入 contigs
            contigs[0].append(f"New_{padding}-{padding}")
        else:
            contigs[-1].append(f"New_{padding}-{padding}")

        res_mask = torch.ones(crop_length)
        if total_len > 0:
            res_mask[total_len:] = 0  # Mark padding as 0
    else:
        res_mask = torch.ones(total_len)

    return contigs, res_mask

import numpy as np
import matplotlib.pyplot as plt
def plot_tensor_heatmap(tensor, title="Tensor Heatmap", cmap="viridis"):
    """
    绘制L×L tensor矩阵的热图
    
    参数:
    tensor -- 输入的L×L矩阵
    title -- 热图标题(可选)
    cmap -- 颜色映射(可选)
    """
    plt.figure(figsize=(8, 6))
    
    # 显示热图
    im = plt.imshow(tensor, cmap=cmap)
    
    # 添加颜色条
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # 设置标题和坐标轴
    plt.title(title)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    
    # 显示网格线(可选)
    plt.grid(False)
    
    plt.show()

def randomly_fix_bonds(target, fixed_bond_config=None):
    """
    Randomly fixes a fraction of existing bonds in the target object.

    This function identifies non-diagonal, non-padded bonds with a value of 1
    in the bond matrix and sets their corresponding mask value to False,
    effectively "fixing" them.

    Args:
        target (Target): The Target object to modify in-place.
        fixed_bond_config (float or dict, optional): Configuration for fixing.
            - If float: The exact fraction of bonds to fix.
            - If dict: Specifies a range {'ratio_min': float, 'ratio_max': float}
                        from which a random fraction is drawn.
            If None, no operation is performed.
    """
    if fixed_bond_config is None:
        return

    # Determine the ratio of bonds to fix
    if isinstance(fixed_bond_config, float):
        ratio = fixed_bond_config
    elif isinstance(fixed_bond_config, dict):
        min_r = fixed_bond_config.get('ratio_min', 0.0)
        max_r = fixed_bond_config.get('ratio_max', 1.0)
        ratio = random.uniform(min_r, max_r)
    else:
        return  # Invalid config

    if not 0 < ratio <= 1.0:
        return

    L = len(target.full_seq)
    device = target.full_bond_matrix.device

    # 1. Find candidate indices to fix (upper triangle only)
    # Candidates are: non-diagonal, bond_matrix==1, non-padded, and currently mutable.
    triu_indices = torch.triu_indices(L, L, offset=1, device=device)
    i_upper, j_upper = triu_indices[0], triu_indices[1]

    is_one = target.full_bond_matrix[i_upper, j_upper] >= 0.9999
    is_not_padded = (target.res_mask[i_upper] == 1) & (target.res_mask[j_upper] == 1)
    is_mutable = target.full_bond_mask[i_upper, j_upper]

    valid_mask = is_one & is_not_padded & is_mutable
    
    candidate_indices = torch.where(valid_mask)[0]
    
    num_candidates = len(candidate_indices)
    if num_candidates == 0:
        return

    # 2. Determine how many bonds to fix
    num_to_fix = math.ceil(num_candidates * ratio)
    
    # 3. Randomly select indices and apply the fix
    perm = torch.randperm(num_candidates, device=device)
    selected_indices_in_candidates = candidate_indices[perm[:num_to_fix]]

    fix_i = i_upper[selected_indices_in_candidates]
    fix_j = j_upper[selected_indices_in_candidates]

    # 4. Update the bond_mask in place, masking full rows and columns for all involved residues
    if fix_i.numel() > 0:
        # Collect all unique residue indices involved in the fixed bonds
        indices_to_fix = torch.unique(torch.cat([fix_i, fix_j]))
        
        # Mask the corresponding rows and columns
        target.full_bond_mask[indices_to_fix, :] = False
        target.full_bond_mask[:, indices_to_fix] = False