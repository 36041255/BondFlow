import numpy as np
import torch
import torch.nn.functional as nn
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import networkit as nk
import torch
from typing import Sequence, Optional, Tuple
import torch
from typing import Sequence, Optional
import warnings

@torch.no_grad()
def diffusion_distance_tensor(
    A_adj_batch: torch.Tensor,            # (B, L, L)
    node_mask: torch.Tensor,              # (B, L)
    times: Sequence[int],                 # e.g., (0,1,4)
    k: int = 256,                          # if -1: use all components except the first when skip_top=True
    skip_top: bool = True,
    eps: float = 1e-12,
    rbf_num: int = 0,                     # if >0, return RBF-embedded distances with this feature dimension
    rbf_gamma: Optional[float] = None,    # if None, auto-set based on spacing
    energy_thresh: Optional[float] = None,# per-sample adaptive k based on spectral energy if set
    k_ratio: Optional[float] = None,      # per-sample adaptive k based on a ratio of available components
    t_ref: Optional[int] = None,          # reference time for energy (default=max(times))
) -> torch.Tensor:
    """Compute diffusion distances at multiple times.

    Args:
      k: number of eigencomponents to use; if -1, use all available (except the top one when skip_top=True).
      rbf_num: number of RBF centers in [0,1]; if 0, return raw distances per t.
      rbf_gamma: Gaussian width; default auto = 1/(2*delta^2) with delta=1/(rbf_num-1).
      energy_thresh: pick smallest k per sample to reach this spectral energy (|lambda|^(2*t_ref)).
      k_ratio: pick smallest k per sample based on a ratio of available components. Overrides energy_thresh.
      t_ref: reference time for energy threshold; default max(times) if None.

    Returns:
      If rbf_num == 0: (B, L, L, T) each slice is D_t in [0,1] (normalized per-batch).
      If rbf_num > 0: (B, L, L, T*rbf_num) RBF features per time.
    """
    # Use local diffusion_map_pair_features defined in this module

    assert A_adj_batch.dim() == 3 and A_adj_batch.shape[-1] == A_adj_batch.shape[-2]
    device = A_adj_batch.device
    dtype = A_adj_batch.dtype
    B, L, _ = A_adj_batch.shape
    # Ensure node_mask on same device
    node_mask = node_mask.to(torch.bool).to(device)

    # Determine effective k for the batch
    if int(k) == -1:
        start = 1 if skip_top else 0
        n_subs = node_mask.sum(dim=1)                     # (B,)
        k_eff = int(n_subs.max().item()) - start         # max valid per-batch minus skipped top
        k_eff = max(k_eff, 1)
    else:
        k_eff = int(k)

    # Reuse eigensolver to get (lam, U) with correct masking/sorting
    _, (lam_all, U_all) = diffusion_map_pair_features(
        A_adj_batch, times=times, k=k_eff, skip_top=skip_top, node_mask=node_mask, compute_features=False
    )
    # lam_all: (B, k_eff), U_all: (B, L, k_eff)
    mask_2d = (node_mask[:, :, None] & node_mask[:, None, :])  # (B,L,L)
    
    # Per-sample adaptive k selection
    if k_ratio is not None:
        # Select k based on a fixed ratio of available components
        if energy_thresh is not None:
            print("Warning: Both k_ratio and energy_thresh are provided. k_ratio will take precedence.")
        
        total_avail = (node_mask.sum(dim=1) - (1 if skip_top else 0)).clamp(min=1)
        # Use ceil to ensure at least 1 component is selected for ratio > 0
        k_sel = torch.ceil(total_avail.float() * float(k_ratio)).long().clamp(min=1, max=lam_all.size(1))

        # print(f"\n--- Adaptive k selection based on ratio: {k_ratio:.2f} ---")
        # for i in range(A_adj_batch.shape[0])[:5]:
        #     L_i = int(node_mask[i].sum().item())
        #     total_i = int(total_avail[i].item())
        #     k_sel_i = int(k_sel[i].item())
        #     ratio_i = k_sel_i / total_i if total_i > 0 else 0.0
        #     print(f"Sample {i:2d} (L={L_i:3d}): Selected k={k_sel_i:3d}, Total available={total_i:3d}, Ratio={ratio_i:.4f}")
        # print("--- End of adaptive k report ---\n")

        idx_range = torch.arange(lam_all.size(1), device=device).unsqueeze(0)  # (1,k)
        comp_mask = (idx_range < k_sel.unsqueeze(1)).to(lam_all.dtype)        # (B,k)

        lam_all = lam_all * comp_mask
        U_all = U_all * comp_mask.unsqueeze(1)

    # Per-sample adaptive k via spectral energy threshold (optional)
    elif energy_thresh is not None:
        # Special case for energy_thresh=1.0: select all available components
        # if abs(float(energy_thresh) - 1.0) < 1e-6:
        #     total_avail = (node_mask.sum(dim=1) - (1 if skip_top else 0)).clamp(min=1)
        #     k_sel = total_avail  # Select all available components

        #     # ratio_vs_total = (k_sel.float() / total_avail.float())
        #     # print("\n--- Adaptive k selection (energy_thresh=1.0) ---")
        #     # print("Selecting all available components for each sample.")
        #     # for i in range(A_adj_batch.shape[0]):
        #     #     L_i = int(node_mask[i].sum().item())
        #     #     total_i = int(total_avail[i].item())
        #     #     k_sel_i = int(k_sel[i].item())
        #     #     ratio_i = ratio_vs_total[i].item()
        #     #     print(f"Sample {i:2d} (L={L_i:3d}): Selected k={k_sel_i:3d}, Total available={total_i:3d}, Ratio={ratio_i:.4f}")
        #     # print("--- End of adaptive k report ---\n")

        # else:
        tref = int(max(times) if t_ref is None else t_ref)
        w = lam_all.abs().pow(2 * tref)                        # (B,k_eff)
        
        # For padded components, lambda can be zero. Mask them out from energy calculation.
        idx_range_w = torch.arange(w.size(1), device=device).unsqueeze(0) # (1, k_eff)
        valid_comps = (node_mask.sum(dim=1, keepdim=True) - (1 if skip_top else 0)) # (B, 1)
        w_mask = (idx_range_w < valid_comps).to(w.dtype)
        w = w * w_mask

        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        ratio = torch.cumsum(w, dim=1) / denom                 # (B,k_eff)
        
        effective_thresh = float(energy_thresh)

        hits = ratio >= effective_thresh
        # default to all components if never reaches threshold
        first_idx = torch.where(hits.any(dim=1), hits.float().argmax(dim=1), torch.full((B,), lam_all.size(1) - 1, device=device, dtype=torch.long))
        k_sel = torch.clamp(first_idx + 1, min=1, max=lam_all.size(1))  # (B,)
        
        total_avail = (node_mask.sum(dim=1) - (1 if skip_top else 0)).clamp(min=1)        # (B,)
        ratio_vs_total = (k_sel.float() / total_avail.float())                # (B,)

        # print("\n--- Adaptive k selection based on energy threshold ---")
        # print(f"Energy threshold: {energy_thresh}, effective: {effective_thresh:.6f}")
        # print(f"Reference time t_ref: {tref}")
        # for i in range(A_adj_batch.shape[0])[:5]:
        #     L_i = int(node_mask[i].sum().item())
        #     total_i = int(total_avail[i].item())
        #     k_sel_i = int(k_sel[i].item())
        #     ratio_i = ratio_vs_total[i].item()
        #     print(f"Sample {i:2d} (L={L_i:3d}): Selected k={k_sel_i:3d}, Total available={total_i:3d}, Ratio={ratio_i:.4f}")
        # print("--- End of adaptive k report ---\n")

        idx_range = torch.arange(lam_all.size(1), device=device).unsqueeze(0)  # (1,k)
        comp_mask = (idx_range < k_sel.unsqueeze(1)).to(lam_all.dtype)        # (B,k)

        lam_all = lam_all * comp_mask
        U_all = U_all * comp_mask.unsqueeze(1)

    dists_per_t = []
    for t in times:
        lam_pow = lam_all.pow(int(t))              # (B,k_eff)
        emb = U_all * lam_pow[:, None, :]          # (B, L, k_eff)
        # Pairwise squared distances via Gram trick
        sq = (emb.pow(2).sum(dim=-1, keepdim=True))         # (B, L, 1)
        gram = torch.matmul(emb, emb.transpose(1, 2))       # (B, L, L)
        dist2 = sq + sq.transpose(1, 2) - 2.0 * gram        # (B, L, L)
        dist2 = torch.clamp(dist2, min=0.0)
        dist = torch.sqrt(dist2 + eps)

        # Zero-out invalid pairs and enforce zero diagonal
        dist = dist * mask_2d.float()
        eye = torch.eye(L, dtype=dist.dtype, device=dist.device).unsqueeze(0)
        dist = dist * (1.0 - eye)
        dists_per_t.append(dist)

    dist_all_t = torch.stack(dists_per_t, dim=-1)  # (B, L, L, T)

    # Normalize distances to [0,1] per-batch for stable RBFs (exclude diagonal)
    dist_valid = dist_all_t.clone()
    mask3 = mask_2d.unsqueeze(-1).expand_as(dist_valid)
    dist_valid = dist_valid * mask3
    B_, L_, _, T_ = dist_valid.shape
    eye3 = torch.eye(L_, device=device, dtype=dtype).view(1, L_, L_, 1)
    dist_valid = dist_valid * (1.0 - eye3)

    # Per-batch max (over i,j,t) for normalization; avoid div by 0
    max_per_b = dist_valid.view(B_, -1).amax(dim=1).clamp_min(1e-6).view(B_, 1, 1, 1)
    dist_norm = (dist_all_t / max_per_b).clamp(0.0, 1.0)

    if int(rbf_num) <= 0:
        return dist_norm.to(device=device, dtype=dtype)

    # Build RBF centers uniformly in [0,1]
    if rbf_num == 1:
        centers = torch.tensor([0.5], device=device, dtype=dtype)
        gamma = torch.tensor(1.0, device=device, dtype=dtype) if rbf_gamma is None else torch.tensor(float(rbf_gamma), device=device, dtype=dtype)
    else:
        centers = torch.linspace(0.0, 1.0, steps=int(rbf_num), device=device, dtype=dtype)
        delta = (1.0 / float(rbf_num - 1))
        gamma = (1.0 / (2.0 * (delta ** 2))) if rbf_gamma is None else float(rbf_gamma)
        gamma = torch.tensor(gamma, device=device, dtype=dtype)

    # Expand to features: exp(-gamma * (d - c)^2) for each center c
    # dist_norm: (B,L,L,T) -> (B,L,L,T,C)
    diff = dist_norm.unsqueeze(-1) - centers.view(1, 1, 1, 1, -1)
    feats_rbf = torch.exp(-gamma * (diff ** 2))
    # Mask invalid pairs
    feats_rbf = feats_rbf * mask3.unsqueeze(-1)

    # Flatten time and centers into feature dim: (B,L,L,T*C)
    B_, L_, _, T_, C_ = feats_rbf.shape
    feats_flat = feats_rbf.view(B_, L_, L_, T_ * C_)
    return feats_flat.to(device=device, dtype=dtype)

def diffusion_map_pair_features(
    A_batch: torch.Tensor,
    times: Sequence[int],
    k: int,
    skip_top: bool = True,
    node_mask: Optional[torch.Tensor] = None,
    compute_features: bool = True,
) -> Tuple[Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Batched diffusion-map pair features with optimized memory usage.
    
    Optimization: Concatenates features at the node level first (B, L, k*T),
    then broadcasts to pair features (B, L, L, 2*k*T) only once at the end.
    """
    assert A_batch.dim() == 3 and A_batch.shape[-1] == A_batch.shape[-2], "A_batch must be (B,L,L)"
    B, L, _ = A_batch.shape
    assert k >= 1, "k must be >= 1"
    assert len(times) >= 1, "times must be non-empty"

    device = A_batch.device
    dtype = A_batch.dtype

    if node_mask is None:
        node_mask = torch.ones((B, L), dtype=torch.bool, device=device)
    else:
        assert node_mask.shape == (B, L), "node_mask must have shape (B, L)"
        node_mask = node_mask.to(torch.bool).to(device)

    start = 1 if skip_top else 0
    total_dim = 2 * k * len(times) # Final feature dimension

    # 1. Batched construction of normalized adjacency
    # -----------------------------------------------------------
    lam_all = torch.zeros((B, k), dtype=dtype, device=device)
    U_all = torch.zeros((B, L, k), dtype=dtype, device=device)

    valid_pair = (node_mask[:, :, None] & node_mask[:, None, :])  # (B,L,L)
    A_masked = A_batch * valid_pair.to(dtype)

    d = A_masked.sum(dim=-1)  # (B,L)
    # Safe inverse sqrt degree
    inv_sqrt_d = torch.where(
        node_mask,
        d.clamp_min(1e-15).pow(-0.5),
        torch.zeros_like(d)
    )
    S = inv_sqrt_d[:, :, None] * A_masked * inv_sqrt_d[:, None, :]
    # Enforce symmetry and add small jitter
    S = 0.5 * (S + S.transpose(-1, -2))
    jitter = 1e-7
    I = torch.eye(L, dtype=dtype, device=device).unsqueeze(0)
    S = S + jitter * I

    # 2. Batched eigendecomposition (Robust Fallback)
    # -----------------------------------------------------------
    try:
        evals, evecs = torch.linalg.eigh(S)  # (B,L), (B,L,L), ascending
    except Exception:
        try:
            print("linalg.eigh failed, using float64 fallback")
            S64 = S.to(torch.float64).cpu()
            S64 = 0.5 * (S64 + S64.transpose(-1, -2))
            S64 = S64 + (jitter * torch.eye(L, dtype=torch.float64).unsqueeze(0))
            evals64, evecs64 = torch.linalg.eigh(S64)
            evals = evals64.to(dtype=dtype, device=device)
            evecs = evecs64.to(dtype=dtype, device=device)
            del S64, evals64, evecs64 # Free CPU memory immediately
        except Exception:
            print("float64 fallback failed, returning zeros features")
            # Return zeros if everything fails
            features = torch.zeros((B, L, L, total_dim), dtype=dtype, device=device)
            return features, (lam_all, U_all)

    # 3. Sort and Select Top-k Eigenpairs
    # -----------------------------------------------------------
    sort_idx = torch.argsort(evals, descending=True, dim=-1)  # (B,L)
    
    # Gather eigenvalues
    evals_sorted = torch.gather(evals, -1, sort_idx)
    
    # Gather eigenvectors (expanding index for gather)
    sort_idx_exp = sort_idx.unsqueeze(-2).expand(B, L, L)
    evecs_sorted = torch.gather(evecs, -1, sort_idx_exp)

    # Clean up large full-size tensors to save memory
    del evals, evecs, S, sort_idx, sort_idx_exp, A_masked

    # 4. Fill output containers based on valid node counts
    # -----------------------------------------------------------
    n_subs = node_mask.sum(dim=1)  # (B,)
    k_cap = min(k, max(0, L - start))
    
    if k_cap > 0:
        k_effs = torch.clamp(n_subs - start, min=0, max=k_cap)  # (B,)
        idx_range = torch.arange(k_cap, device=device).unsqueeze(0)
        mask_k = idx_range < k_effs.unsqueeze(-1)  # (B,k_cap)

        sel_evals = evals_sorted[:, start:start + k_cap]  # (B,k_cap)
        # Fill only valid positions, rest remain 0
        lam_all[:, :k_cap] = torch.where(mask_k, sel_evals, torch.zeros_like(sel_evals))

        sel_evecs = evecs_sorted[:, :, start:start + k_cap]  # (B,L,k_cap)
        U_mask = mask_k.unsqueeze(1).expand(-1, L, -1)  # (B,L,k_cap)
        U_all[:, :, :k_cap] = torch.where(U_mask, sel_evecs, torch.zeros_like(sel_evecs))
    
    # Clean up sorted tensors
    del evals_sorted, evecs_sorted

    # Zero-out invalid-node rows in U_all explicitly
    U_all = U_all * node_mask.unsqueeze(-1)

    if not compute_features:
        return None, (lam_all, U_all)

    # 5. Build Features (MEMORY OPTIMIZED SECTION)
    # -----------------------------------------------------------
    # Instead of creating (B, L, L) tensors for each time t,
    # we first concatenate node embeddings: (B, L, k * len(times))
    
    node_feat_list = []
    
    for t in times:
        assert int(t) >= 0
        lam_pow = lam_all.pow(int(t))       # (B, k)
        # Broadcast multiply: (B, L, k) * (B, 1, k) -> (B, L, k)
        emb_t = U_all * lam_pow.unsqueeze(1)
        node_feat_list.append(emb_t)
        
    # Concatenate all time steps at node level
    # shape: (B, L, k * n_times)
    flat_node_feats = torch.cat(node_feat_list, dim=-1)
    
    # 6. Final Pairwise Broadcasting (Modified for Strict Order Equivalence)
    # -----------------------------------------------------------
    # 目标顺序: [i_t1, j_t1, i_t2, j_t2, ...]
    
    T = len(times)
    
    # 1. 拆分时间维度和特征维度
    # Shape: (B, L, T, k)
    node_feats_reshaped = flat_node_feats.view(B, L, T, k)
    
    # 2. 构建 i 和 j 的 View (零内存拷贝)
    # Shape: (B, L, 1, T, k)
    emb_i = node_feats_reshaped.unsqueeze(2)
    # Shape: (B, 1, L, T, k)
    emb_j = node_feats_reshaped.unsqueeze(1)
    
    # 3. 利用广播和 Stack 交叉排列
    # 我们希望最后两维是 (2, k) 或者是 (i/j, feat)，这样 flatten 后就是 i_feat, j_feat
    # stack dim=-2 会产生形状: (B, L, L, T, 2, k)
    # 这一步会分配最终显存，显存占用与原函数一致，但没有中间峰值
    features_stacked = torch.stack([
        emb_i.expand(-1, -1, L, -1, -1),
        emb_j.expand(-1, L, -1, -1, -1)
    ], dim=-2)
    
    # 4. 展平为最终特征
    # Flatten T, 2, k -> 2*T*k
    # 顺序变为: t1_i, t1_j, t2_i, t2_j ... (与原函数一致)
    features = features_stacked.flatten(start_dim=-3)

    # Apply Pair Mask
    valid_pair_mask = (node_mask[:, :, None] & node_mask[:, None, :]).unsqueeze(-1)
    features = features * valid_pair_mask.to(dtype)

    return features, (lam_all, U_all)
    
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

def get_residue_dist_matrix(adj, rf_index, dmax=32):
    """
    Computes the residue-level shortest path distance matrix.

    The graph is constructed with residues as nodes. Edges exist between
    adjacent residues (peptide bonds) and between residues with special
    connections specified in the `adj` matrix. The distance is the
    minimum number of residues in the path.

    Args:
        adj (torch.Tensor): The adjacency matrix indicating special connections
                            (e.g., disulfide bonds). Shape: (B, L, L).
        rf_index (torch.Tensor): Residue indices from the PDB, used to identify
                                 sequential residues for peptide bonds.
                                 Shape: (B, L).
        dmax (int): The maximum distance to consider. Paths longer than this
                    will be capped at this value.

    Returns:
        torch.Tensor: A tensor of shape (B, L, L) containing the shortest
                      path distances between each pair of residues.
    """
    B, L = adj.shape[:2]
    # Initialize the final distance matrix with the maximum value
    dist_matrix = torch.full((B, L, L), float(dmax), dtype=torch.float32, device=adj.device)

    for b in range(B):
        # The nodes of our graph are the residues, so there are L nodes.
        if L == 0:
            continue

        row, col = [], []

        # Helper to add a bi-directional edge to the graph
        def add_edge(u, v):
            row.extend([u, v])
            col.extend([v, u])

        # 1. Add edges for peptide bonds (adjacent residues)
        for i in range(L - 1):
            # Check if residue i and i+1 are sequential in the chain
            if  rf_index[b, i+1] == rf_index[b, i] + 1:
                add_edge(i, i + 1)

        # 2. Add edges for special connections from the input adjacency matrix
        # These could be disulfide bonds, cyclic connections, etc.
        special_connections = torch.argwhere(adj[b] == 1)
        for conn in special_connections:
            i, j = conn[0].item(), conn[1].item()
            # Avoid duplicate edges and self-loops
            if i < j:
                add_edge(i, j)
        
        # If there are no connections at all, fill with dmax and continue
        if not row:
            dist_matrix[b].fill_(dmax)
            dist_matrix[b].fill_diagonal_(0)
            continue

        # 3. Build a sparse matrix and calculate all-pairs shortest paths
        # The graph is unweighted, so each edge has a weight of 1.
        adj_matrix_sparse = csr_matrix(([1] * len(row), (row, col)), shape=(L, L))
        
        # Calculate shortest paths. Unreachable nodes will have a distance of 'inf'.
        residue_dist = shortest_path(csgraph=adj_matrix_sparse, directed=False, unweighted=True)
        
        # 4. Populate the final distance matrix for this batch item
        residue_dist_tensor = torch.from_numpy(residue_dist).to(dtype=torch.float32, device=adj.device)
        
        # Clamp the infinite distances (unreachable pairs) to dmax
        dist_matrix[b] = torch.clamp(residue_dist_tensor, max=dmax)

    return dist_matrix

def make_sub_doubly2doubly_stochastic(sub_ds_matrix: torch.Tensor) -> torch.Tensor:
    """
    将一个亚双随机对称矩阵通过在对角线添加差额来转换为双随机对称矩阵。
    参数:
    sub_ds_matrix (torch.Tensor): 输入的张量，形状可以是 (N, N) 或 (B, N, N)，
                                  其中 B 是批处理大小，N 是矩阵维度。
    返回:
    torch.Tensor: 转换后的双随机对称矩阵。
    """
    # 确保输入至少是二维的
    dtype = sub_ds_matrix.dtype
    if sub_ds_matrix.dim() < 2:
        raise ValueError("输入张量至少需要是二维 (N, N)。")
    
    # --- （可选）健壮性检查 ---
    # 检查对称性 (允许有微小的浮点误差)
    assert torch.allclose(sub_ds_matrix, sub_ds_matrix.transpose(-2, -1)), "输入矩阵必须是对称的"
    # 检查非负性
    assert torch.all(sub_ds_matrix >= 0), "输入矩阵元素必须非负"
    
    # 1. 计算每行的和。对于(B, N, N)的张量，我们对最后一个维度(dim=-1)求和。
    row_sums = torch.sum(sub_ds_matrix, dim=-1)
    
    # 若存在行和>1，则发出警告并按比例缩放使其<=1（保持对称与非负）
    tol = 1e-6
    if torch.any(row_sums > 1.0 + tol):
        warnings.warn("make_sub_doubly2doubly_stochastic: 检测到行和>1，已按比例缩放至<=1。")
        print(("make_sub_doubly2doubly_stochastic: 检测到行和>1，已按比例缩放至<=1。"))
        print("大于1.0 + tol的行和:",row_sums[row_sums> 1.0 + tol])
        if sub_ds_matrix.dim() == 2:
            # 仅缩放行和>1的行，同时为保持对称，按相同行列因子进行双边缩放
            scales = torch.where(
                row_sums > 1.0 + tol,
                (1.0 / row_sums.clamp_min(1e-12)),
                torch.ones_like(row_sums)
            ).to(dtype=sub_ds_matrix.dtype, device=sub_ds_matrix.device)
            sub_ds_matrix = sub_ds_matrix * scales.view(-1, 1)
            sub_ds_matrix = sub_ds_matrix * scales.view(1, -1)
        else:
            # batched：逐样本对行和>1的行进行双边缩放，保持对称
            scales = torch.where(
                row_sums > 1.0 + tol,
                (1.0 / row_sums.clamp_min(1e-12)),
                torch.ones_like(row_sums)
            ).to(dtype=sub_ds_matrix.dtype, device=sub_ds_matrix.device)  # (B,N)
            sub_ds_matrix = sub_ds_matrix * scales.view(-1, sub_ds_matrix.shape[-1], 1)
            sub_ds_matrix = sub_ds_matrix * scales.view(-1, 1, sub_ds_matrix.shape[-1])
        # 重新计算行和
        row_sums = torch.sum(sub_ds_matrix, dim=-1)

    # 结果的形状是 (N,) 或 (B, N)
    deficits = 1.0 - row_sums

    # torch.diag_embed 会将一个 (B, N) 的向量转换为一个 (B, N, N) 的对角矩阵
    diagonal_additions = torch.diag_embed(deficits)

    doubly_stochastic_matrix = sub_ds_matrix + diagonal_additions

    return doubly_stochastic_matrix.to(dtype)


def sample_symmetric_permutation_by_pairs(A: torch.Tensor, mask: torch.Tensor, mode: str = "random") -> torch.Tensor:
    """
    通过迭代采样"节点对"来生成一个对称置换矩阵，同时考虑一个掩码。

    参数:
        A (torch.Tensor): 对称双随机矩阵，形状 (L, L)。
        mask (torch.Tensor): 0-1 或布尔掩码，形状 (L, L)，1/True 表示允许配对。
        mode (str): "random"（按权重随机采样，默认）、"greedy"（每步取当前最大权重）或 "opt"（全局最优，Blossom 最大权匹配）。
    返回:
        torch.Tensor: 对称 0-1 置换矩阵，形状 (L, L)。
    """
    assert mode in ("random", "greedy", "opt"), "mode must be 'random', 'greedy' or 'opt'"

    L = A.shape[0]
    device = A.device

    # 规范 mask：保留原有行为（如果传入 float 也可）
    mask = mask.to(device=device)
    mask_bool = mask.to(dtype=torch.bool, device=device)

    # OPT 模式：调用 Blossom（networkx）
    if mode == "opt":
        if nx is None:
            raise RuntimeError("mode='opt' requires networkx to be installed (pip install networkx).")

        # 防止 log(0) 并改为 NumPy 计算以减少张量<->NumPy开销
        eps = 1e-12
        A_np = A.detach().to('cpu').numpy()
        mask_np = mask_bool.detach().to('cpu').numpy()
        logA = np.log(np.clip(A_np, eps, None))

        # delta_ij = 2*logA_ij - logA_ii - logA_jj
        diag = np.diag(logA)  # (L,)
        delta = (2.0 * logA) - diag[None, :] - diag[:, None]

        # 掩码与上三角过滤，只保留正权边（负权边不可能出现在最优匹配中）
        tri_i, tri_j = np.triu_indices(L, k=1)
        valid_mask = mask_np[tri_i, tri_j]
        w = delta[tri_i, tri_j]
        finite_pos = np.isfinite(w) & (w > 0.0) & valid_mask
        ei = tri_i[finite_pos]
        ej = tri_j[finite_pos]
        ew = w[finite_pos].astype(float)

        # 优先尝试更快的后端（NetworKit：近似最大权匹配），失败则回退到 networkx 精确 Blossom
        matching_pairs = None
        try:
            if ei.size > 0:
                G_nk = nk.Graph(L, weighted=True, directed=False)
                for u, v, ww in zip(ei.tolist(), ej.tolist(), ew.tolist()):
                    G_nk.addEdge(int(u), int(v), float(ww))
                matcher = nk.matching.LocalMaxMatcher(G_nk)
                matcher.run()
                M = matcher.getMatching()
                # 提取匹配对
                pairs = []
                for u in range(L):
                    if M.isMatched(u):
                        v = M.mate(u)
                        if v != -1 and u < v:
                            pairs.append((u, v))
                matching_pairs = pairs
        except Exception:
            matching_pairs = None

        if matching_pairs is None:
            # 回退到 networkx：构图并批量添加边
            G = nx.Graph()
            G.add_nodes_from(range(L))
            if ei.size > 0:
                edges = list(zip(ei.tolist(), ej.tolist(), ew.tolist()))
                G.add_weighted_edges_from(edges)
            matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)
            matching_pairs = list(matching)

        # construct permutation matrix
        P = torch.zeros((L, L), dtype=A.dtype, device=device)
        matched_nodes = set()
        for (i, j) in matching_pairs:
            P[i, j] = 1.0
            P[j, i] = 1.0
            matched_nodes.add(i)
            matched_nodes.add(j)
        for i in range(L):
            if i not in matched_nodes:
                P[i, i] = 1.0
        return P

    # RANDOM / GREEDY 模式：逐步构造（保留原意）
    permutation_matrix = torch.zeros(L, L, device=device, dtype=A.dtype)
    nodes_available_mask = torch.ones(L, dtype=torch.bool, device=device)
    eps_sum = 1e-9

    # Precompute diag for speed
    diag = A.diag()
    mask_diag = mask.diag()

    while torch.any(nodes_available_mask):
        available_indices = torch.where(nodes_available_mask)[0]
        n_avail = available_indices.shape[0]

        # self-loop weights (for available nodes)
        self_loop_weights = diag[available_indices] * mask_diag[available_indices]

        # swap (2-cycle) weights
        if n_avail >= 2:
            # torch.combinations returns pairs of actual node indices
            swap_pairs_indices = torch.combinations(available_indices, r=2)  # (n_combs,2)
            swap_weights = A[swap_pairs_indices[:, 0], swap_pairs_indices[:, 1]] * \
                           mask[swap_pairs_indices[:, 0], swap_pairs_indices[:, 1]]
        else:
            swap_pairs_indices = torch.empty(0, 2, dtype=torch.long, device=device)
            swap_weights = torch.empty(0, device=device, dtype=A.dtype)

        # merge: note 原实现中 swap 权重乘以2（代表两个 off-diagonal entries）
        all_weights = torch.cat([self_loop_weights, swap_weights * 2.0])

        # 如果所有权重都近似为0 -> 把剩余节点都设为自环
        if all_weights.sum() < eps_sum:
            permutation_matrix[available_indices, available_indices] = 1.0
            break

        # 选择 index（随机或贪婪）
        if mode == "random":
            # multinomial 需要浮点 non-negative 和和>0
            # torch.multinomial 在 CPU/GPU 都可用
            chosen_flat_idx = int(torch.multinomial(all_weights, 1).item())
        else:  # greedy
            chosen_flat_idx = int(torch.argmax(all_weights).item())

        # apply chosen
        n_self = self_loop_weights.shape[0]
        if chosen_flat_idx < n_self:
            # self-loop chosen
            node_idx = int(available_indices[chosen_flat_idx].item())
            permutation_matrix[node_idx, node_idx] = 1.0
            nodes_available_mask[node_idx] = False
        else:
            swap_idx = chosen_flat_idx - n_self
            node1_idx = int(swap_pairs_indices[swap_idx, 0].item())
            node2_idx = int(swap_pairs_indices[swap_idx, 1].item())
            permutation_matrix[node1_idx, node2_idx] = 1.0
            permutation_matrix[node2_idx, node1_idx] = 1.0
            nodes_available_mask[node1_idx] = False
            nodes_available_mask[node2_idx] = False

    return permutation_matrix


def sample_permutation(A_batch: torch.Tensor, mask_2d: torch.Tensor = None, mode: str = "opt") -> torch.Tensor:
    """
    批量版本：对 A_batch 中每个矩阵分别调用 sample_symmetric_permutation_by_pairs。
    参数:
        A_batch (B, L, L)
        mask_2d (B, L, L) or None
        mode: "random", "greedy", or "opt"
    返回:
        (B, L, L) 的 0-1 对称置换矩阵批量
    """
    if mask_2d is None:
        mask_2d = torch.ones_like(A_batch, dtype=torch.bool, device=A_batch.device)

    perm_matrices = [
        sample_symmetric_permutation_by_pairs(A_batch[i], mask_2d[i], mode=mode)
        for i in range(A_batch.shape[0])
    ]
    return torch.stack(perm_matrices, dim=0)