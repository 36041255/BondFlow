import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np
import BondFlow.data.utils as iu
import BondFlow.data.SM_utlis as smu
from rfdiff.kinematics import xyz_to_t2d
from BondFlow.models.layers import TimeEmbedding
# MODIFIED: Import all necessary APM models for the multi-stage wrapper
from apm.apm.models.flow_model import BackboneModel
from apm.apm.models.side_chain_model import SideChainModel, AngleResnet
from apm.apm.models.refine_model import RefineModel
from BondFlow.models.layers import BondingNetwork
from apm.apm.models.utils import get_time_embedding
from multiflow_data import so3_utils as su
import time
import warnings
def compute_rbf_and_project_chunked(
    projection_layer: nn.Linear,
    dist_patches: torch.Tensor, # (B, L1, L2, 9, T)
    mask_patches: torch.Tensor, # (B, L1, L2, 9)
    rbf_num: int,
    rbf_gamma: Optional[float] = None,
    chunk_size: int = 16,
) -> torch.Tensor:
    """
    Computes RBF expansion and projection in chunks to save memory.
    Input:
        dist_patches: (B, L1, L2, 9, T)
    Output:
        X_proj: (B, L1, L2, 9, d_hidden)
    """
    device = dist_patches.device
    dtype = dist_patches.dtype
    B, L1, L2, N_tokens, T = dist_patches.shape
    
    # 1. Setup RBF parameters
    if rbf_num <= 0:
        # Fallback or error; though current usage ensures rbf_num > 0 for this step
        return projection_layer(dist_patches.flatten(start_dim=-2))

    if rbf_num == 1:
        centers = torch.tensor([0.5], device=device, dtype=dtype)
        gamma = torch.tensor(1.0, device=device, dtype=dtype) if rbf_gamma is None else torch.tensor(float(rbf_gamma), device=device, dtype=dtype)
    else:
        centers = torch.linspace(0.0, 1.0, steps=int(rbf_num), device=device, dtype=dtype)
        delta = (1.0 / float(rbf_num - 1))
        gamma = (1.0 / (2.0 * (delta ** 2))) if rbf_gamma is None else float(rbf_gamma)
        gamma = torch.tensor(gamma, device=device, dtype=dtype)

    # 2. Chunked processing
    out_list = []
    
    start_time = time.time()
    for i in range(0, L1, chunk_size):
        # chunk: (B, chunk, L2, 9, T)
        chunk_dist = dist_patches[:, i:i+chunk_size, ...]
        chunk_mask = mask_patches[:, i:i+chunk_size].unsqueeze(-1).unsqueeze(-1) # (B, chunk, L2, 9, 1, 1)

        # RBF Expansion: (..., T) -> (..., T, C)
        diff = chunk_dist.unsqueeze(-1) - centers
        feats_rbf = torch.exp(-gamma * (diff ** 2))
        
        # Apply mask (important for 0-dist padding)
        feats_rbf = feats_rbf * chunk_mask

        # Flatten last two dims (T, C) -> D1
        feats_flat = feats_rbf.flatten(start_dim=-2)
        
        # Project
        chunk_out = projection_layer(feats_flat)
        out_list.append(chunk_out)
        
    end_time = time.time()
    print(f"compute_rbf_and_project_chunked: processed {L1} rows in {end_time - start_time:.4f} seconds (chunk_size={chunk_size})")

    return torch.cat(out_list, dim=1)

class BaseDesignModel(nn.Module):
    """Unified interface for design models (RF/APM).

    Forward signature is aligned with the training loop to minimize changes.
    """

    def forward(
        self,
        *,
        seq_noised: torch.Tensor,
        xyz_noised: torch.Tensor,
        bond_noised: torch.Tensor,
        rf_idx: torch.Tensor,
        pdb_idx: torch.Tensor,
        res_dist_matrix: torch.Tensor,
        alpha_target: torch.Tensor,
        alpha_tor_mask: torch.Tensor,
        partial_T: torch.Tensor,
        str_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        bond_mask: Optional[torch.Tensor] = None,
        res_mask: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
        batch_mask: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError


class RoseTTAFoldWrapper(BaseDesignModel):
    """Thin wrapper around RoseTTAFoldModule to match BaseDesignModel interface."""

    def __init__(self, conf, device: str, d_t1d: int, d_t2d: int):
        super().__init__()
        # Lazy import to avoid circular imports

        from BondFlow.models.RoseTTAFoldModel import RoseTTAFoldModule

        mconf = conf.model
        self.rf = RoseTTAFoldModule(
            n_main_block=mconf.n_main_block,
            n_ref_block=mconf.n_ref_block,
            n_temp_block=mconf.n_temp_block,
            d_msa=mconf.d_msa,
            d_pair=mconf.d_pair,
            d_templ=mconf.d_templ,
            d_condition=mconf.d_condition,
            n_head_msa=mconf.n_head_msa,
            n_head_pair=mconf.n_head_pair,
            n_head_templ=mconf.n_head_templ,
            d_hidden=mconf.d_hidden,
            d_hidden_templ=mconf.d_hidden_templ,
            p_drop=mconf.p_drop,
            d_t1d=d_t1d,
            d_t2d=d_t2d,
            SE3_param_full=dict(mconf.SE3_param_full),
            SE3_param_topk=dict(mconf.SE3_param_topk),
            input_seq_onehot=False,
        ).to(device)
        
        # Local device handle and simple preprocess config defaults
        self.device = device
        try:
            from types import SimpleNamespace
            self.preprocess_conf = getattr(conf, 'preprocess_conf', SimpleNamespace(sidechain_input=False, link_config=None))
        except Exception:
            self.preprocess_conf = type('obj', (), {'sidechain_input': False, 'link_config': None})()

        # Time embedding consistent with mymodel: TimeEmbedding(d_embed=self.d_time)
        self.d_time = getattr(conf.preprocess, 'd_time', 16)
        self.time_embedding = TimeEmbedding(d_embed=self.d_time).to(self.device)
        
    def _preprocess_batch(self, seq, xyz_t, bond_mat,rf_idx,pdb_idx,alpha,alpha_tor_mask,
                            t,str_mask=None,seq_mask=None, bond_mask=None,res_mask=None):
        
        """
        Function to prepare inputs to diffusion model
            t (B,)
            seq (B,L) one-hot sequence 
            bond_mat(B,L,L) doubly_stochastic  matrix for bonds
            msa_full (B,1,L,21)
        
            xyz_t (B,L,14,3) template crds (diffused) 

            t1d (B,L,21 + 16 + 16 ) 
            t2d (B, L, L, 47 + 1 + 16)
            alpha_t (B, L, 30) torsion angles and mask

        """
        
        B,L = seq.shape[:2]
        seq_onehot = torch.nn.functional.one_hot(seq.to(torch.long), num_classes=21).float()  # [B,L,21]
        if str_mask is None:
            str_mask = torch.ones((B,L), dtype=torch.bool, device=self.device)
        if seq_mask is None:
            seq_mask = torch.ones((B,L), dtype=torch.bool, device=self.device)
        if bond_mask is None:
            bond_mask = torch.ones((B,L,L), dtype=torch.bool, device=self.device)
        if res_mask is None:    
            res_mask = torch.ones((B,L), dtype=torch.bool, device=self.device)

        ################
        ### msa_full ###
        ################
        msa_full = torch.zeros((B,1,L,21),device=self.device)
        msa_full[:,:,:,:21] = seq_onehot.unsqueeze(1)

        ###########
        ### t1d ###
        ########### 

        # (B,1,L,0) is str, (B,1,L,1) is seq_onehot
        time_seq = torch.where(seq_mask.bool(), t.unsqueeze(1), 1.0)
        time_seq = self.time_embedding(time_seq)
        time_str = torch.where(str_mask.bool(), t.unsqueeze(1), 1.0)
        time_str = self.time_embedding(time_str)
        t1d = torch.cat((seq_onehot, time_seq, time_str), dim=-1)  # (B,L,21+16+16)
        t1d = t1d # (B,L,21+16+16)

        ###########
        ### t2d ###
        ###########
        t2d = xyz_to_t2d(xyz_t.unsqueeze(1)).squeeze(1) # (B,L,L,37+7)
        
        t2d = torch.cat((t2d,bond_mat.unsqueeze(-1)),dim=-1)  # (B,L,L,44+1)
        time_bond = torch.where(bond_mask.bool(), t[:,None,None], 1.0)
        time_2d = self.time_embedding(time_bond)
        t2d = torch.cat((t2d,time_2d),dim=-1) # (B,L,L,44 + 1 + 16)
        t2d = t2d
        #############
        ### xyz_t ###
        #############
        xyz_t = xyz_t # (B,L,14,3)
        # if self.preprocess_conf.sidechain_input:
        #     xyz_t[torch.where(seq_onehot == 21, True, False),3:,:] = float('nan')
        # else:
        #     xyz_t[~self.mask_str.squeeze(),3:,:] = float('nan')

        ###########      
        ### idx ###
        ###########
        # idx = torch.tensor(self.contig_map.rf)[None].repeat(B,1)
        idx = rf_idx

        ###############
        ### alpha_t ###
        ###############
        if self.preprocess_conf.sidechain_input:
            alpha_mask = (str_mask.bool() | seq_mask.bool())[:,:,None]  # (B,L)
            # seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
            # alpha, _, alpha_mask, _ = util.get_torsions(xyz_t.reshape(-1, L, 27, 3).to('cpu'), seq_tmp.to('cpu'), TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
            # alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
            # alpha[torch.isnan(alpha)] = 0.0
            alpha = alpha.reshape(B,L,10,2)
            alpha_tor_mask = alpha_tor_mask.reshape(B,L,10,1)
            alpha_t = torch.cat((alpha, alpha_tor_mask), dim=-1).reshape(B, L, 30)
            alpha_t[alpha_mask.expand_as(alpha_t)] = 0.0
        else:
            alpha_t = torch.zeros((B,L,30),device=self.device)


        #####################
        ### CA_dist_matrix###
        #####################
        # Parse the connection config
        # _res_mask = res_mask.float()  # (B,L,1)
        # res_mask_2d = _res_mask.unsqueeze(1) * _res_mask.unsqueeze(2)  # (B,L,L)
        # permutation = iu.sample_permutation(bond_mat,res_mask_2d.bool())
        
        # res_dist_matrix = iu.get_residue_dist_matrix(permutation,rf_idx)

        # # 头尾残基选择的 是否用侧链还是头尾
        # connections = iu.parse_connections(self.preprocess_conf.link_config)
        # CA_dist_matrix = iu.get_CA_dist_matrix(permutation, seq.int(), connections, pdb_idx, rf_idx, 
        #                                         N_connect_idx=None, C_connect_idx=None)


        #put tensors on device
        msa_full = msa_full.to(self.device)
        xyz_t = xyz_t.to(self.device)
        idx = idx.to(self.device)
        t1d = t1d.to(self.device)
        t2d = t2d.to(self.device)
        alpha_t = alpha_t.to(self.device)       
        str_mask = str_mask.to(self.device)
        seq_mask = seq_mask.to(self.device)
        bond_mask = bond_mask.to(self.device)
        # res_dist_matrix = res_dist_matrix.to(self.device)

        return  msa_full, xyz_t, alpha_t, idx, t1d, t2d, str_mask, seq_mask, bond_mask

    def forward(
        self,
        *,
        seq_noised: torch.Tensor,
        xyz_noised: torch.Tensor,
        bond_noised: torch.Tensor,
        rf_idx: torch.Tensor,
        pdb_idx: torch.Tensor,
        alpha_target: torch.Tensor,
        alpha_tor_mask: torch.Tensor,
        partial_T: torch.Tensor,
        res_dist_matrix: torch.Tensor,
        str_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        bond_mask: Optional[torch.Tensor] = None,
        res_mask: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
        batch_mask: Optional[torch.Tensor] = None,
        N_C_anchor: Optional[torch.Tensor] = None,
        
    ):
        # Prepare features for RF from BondFlow-style inputs
        (
            msa_full,
            xyz_t,
            alpha_t,
            idx,
            t1d,
            t2d,
            str_mask,
            seq_mask,
            bond_mask,
        ) = self._preprocess_batch(
            seq_noised,
            xyz_noised,
            bond_noised,
            rf_idx,
            pdb_idx,
            alpha_target,
            alpha_tor_mask,
            partial_T,
            str_mask=str_mask,
            seq_mask=seq_mask,
            bond_mask=bond_mask,
            res_mask=res_mask,
        )
        res_dist_matrix = res_dist_matrix.to(self.device)
        return self.rf(
            msa_full,
            seq_noised,
            xyz_noised,
            res_dist_matrix,
            idx,
            t1d=t1d,
            t2d=t2d,
            xyz_t=xyz_t,
            alpha_t=alpha_t,
            return_raw=False,
            return_full=False,
            use_checkpoint=use_checkpoint,
            seq_mask=seq_mask,
            bond_mask=bond_mask,
            str_mask=str_mask,
            batch_mask=batch_mask,
        )


# class APMWrapper(BaseDesignModel):
#     """APM integration wrapper for multi-stage (backbone+sidechain+refine) inference.

#     - Builds APM input_feats from BondFlow-style inputs.
#     - Sequentially calls APM backbone, sidechain, and refine models, gated by time.
#     - Fuses external pair features derived from res_dist_matrix via addition.
#       (Note: APM models must be modified to accept 'external_pair_embed' and add it to their internal pair features).
#     - Maps outputs back to BondFlow format (pred_logits, xyz_pred, alpha_s, bond_matrix).
#     """

#     def __init__(self, conf, device: str, d_t1d: int, d_t2d: int):
         
#         super().__init__()
#         self.device = device
#         self.d_t1d = d_t1d
#         self.d_t2d = d_t2d
#         self._conf = conf
#         print("loading APMBWrapper")
#         # Optional PLM provider
#         _APM_FoldingModel = None
#         try:
#             from apm.apm.models.folding_model import FoldingModel as _APM_FoldingModel
#         except ImportError:
#             print("[APMWrapper] Warning: Could not import APM's FoldingModel. PLM features will be disabled.")

#         # Set up PLM info (default: no PLM)
#         self.PLM_info = (None, None)
#         self._plm_type = None
#         self._folding_model = None
        
#         # Use APM model configuration from config file
#         if not hasattr(conf, 'model'):
#              raise ValueError("APM model configuration not found in config")
#         apm_model_conf = conf.model
#         packing_model_conf = getattr(conf, 'packing_model', apm_model_conf)

#         # If folding config with PLM is provided, initialize PLM provider
#         folding_conf = getattr(conf, 'folding', None)
#         if folding_conf and _APM_FoldingModel:
#             plm_name = getattr(folding_conf, 'PLM', None)
#             if plm_name and plm_name not in ['null', 'None']:
#                 self._folding_model = _APM_FoldingModel(folding_conf)
#                 self._plm_type = plm_name
#                 self.PLM_info = (
#                     getattr(self._folding_model, 'plm_representations_layer', None),
#                     getattr(self._folding_model, 'plm_representations_dim', None),
#                 )

#         # Instantiate all three models
#         self.model = nn.ModuleDict()
#         self.model['backbone'] = BackboneModel(apm_model_conf, self.PLM_info).to(device)
        
#         sidechain_plm_info = self.PLM_info if getattr(packing_model_conf, 'use_plm', False) else (None, None)
#         self.model['sidechain'] = SideChainModel(packing_model_conf, sidechain_plm_info).to(device)

#         self.model['refine'] = RefineModel(apm_model_conf, self.PLM_info).to(device)
        

#         # Head for embedding res_dist_matrix
#         pair_embed_size = getattr(apm_model_conf, 'edge_embed_size', getattr(apm_model_conf.ipa, 'c_z', 128))
#         # self.res_dist_embed = nn.Embedding(33, pair_embed_size).to(device)
#         # self.res_dist_embed.weight.data.zero_()

#         # INSERT_YOUR_CODE
#         # Use nn.Linear with bias=False and zero-initialize weights
#         self.res_dist_embed = nn.Sequential(
#             nn.Linear(3*48*2, pair_embed_size, bias=False),
#             nn.ReLU(),
#             nn.Linear(pair_embed_size, pair_embed_size, bias=False),
#         ).to(device)
#         for m in self.res_dist_embed:
#             if isinstance(m, nn.Linear):
#                 nn.init.zeros_(m.weight)

#         # Bond matrix head
#         node_embed_size = getattr(apm_model_conf, 'node_embed_size', getattr(apm_model_conf.ipa, 'c_s', 256))
#         self.bond_pred = BondingNetwork(
#             d_msa=node_embed_size,
#             d_state=node_embed_size,
#             d_pair=pair_embed_size,
#         ).to(device)
        
#         # Store start times for gating model stages
#         if hasattr(self._conf, 'interpolant'):
#             self.model_start_t = {
#                 'sidechain': self._conf.interpolant.sidechain_start_t,
#                 'refine': self._conf.interpolant.refine_start_t
#             }
#         else: # Fallback for inference
#             self.model_start_t = {'sidechain': 0.0, 'refine': 0.0}

#         self._try_load_multimodel_checkpoint()

#     def _try_load_multimodel_checkpoint(self):
#         import os
#         ckpt_path = getattr(self._conf.model, 'ckpt_path', None)
#         if not ckpt_path or not os.path.exists(ckpt_path):
#             print(f"[APMWrapper] Warning: No checkpoint provided or path invalid: {ckpt_path}. Models are randomly initialized.")
#             return

#         strict_loading = getattr(self._conf.model, 'strict_loading', False)
#         ckpt_obj = torch.load(ckpt_path, map_location=self.device, weights_only=False)

#         # Support multiple checkpoint formats
#         if isinstance(ckpt_obj, dict):
#             if 'model_state_dict' in ckpt_obj:
#                 raw_sd = ckpt_obj['model_state_dict']
#             elif 'state_dict' in ckpt_obj:
#                 raw_sd = ckpt_obj['state_dict']
#             else:
#                 raw_sd = ckpt_obj
#         else:
#             raw_sd = ckpt_obj

#         # First, try to load the entire wrapper (this will also load bond_pred and res_dist_embed if present)
#         load_info_all = self.load_state_dict(raw_sd, strict=False)
#         missing_all = getattr(load_info_all, 'missing_keys', [])
#         unexpected_all = getattr(load_info_all, 'unexpected_keys', [])
#         total_keys = len(raw_sd) if hasattr(raw_sd, 'keys') else 0
#         num_loaded = total_keys - len(unexpected_all)
#         print(
#             f"[APMWrapper] Loaded wrapper from {ckpt_path}: loaded~{num_loaded}/{total_keys}, "
#             f"missing={len(missing_all)}, unexpected={len(unexpected_all)}, strict={strict_loading}"
#         )

#         # If nothing matched (e.g., incompatible naming), fall back to per-submodule loading with prefixes
#         if num_loaded == 0 and isinstance(raw_sd, dict):
#             prefixes = {
#                 'backbone': 'model.backbone.',
#                 'sidechain': 'model.sidechain.',
#                 'refine': 'model.refine.'
#             }

#             for model_name, model in self.model.items():
#                 prefix = prefixes.get(model_name)
#                 if not prefix: continue

#                 extracted_sd = {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}
                
#                 if not extracted_sd:
#                     print(f"[APMWrapper] Warning: No weights found with prefix '{prefix}' for '{model_name}'. Trying to load from raw state_dict.")
#                     load_info = model.load_state_dict(raw_sd, strict=False)
#                 else:
#                     load_info = model.load_state_dict(extracted_sd, strict=strict_loading)
                
#                 missing = getattr(load_info, 'missing_keys', [])
#                 unexpected = getattr(load_info, 'unexpected_keys', [])
#                 print(
#                     f"[APMWrapper] Loaded '{model_name}' weights from {ckpt_path}: "
#                     f"missing={len(missing)}, unexpected={len(unexpected)}"
#                 )
#                 if unexpected:
#                     print(f"   [INFO] Unexpected keys: {unexpected[:3]}")



#     def forward(
#         self,
#         *,
#         seq_noised: torch.Tensor,
#         xyz_noised: torch.Tensor,
#         bond_noised: torch.Tensor, # Consumed by mymodel._preprocess, passed here for consistent signature
#         rf_idx: torch.Tensor,
#         pdb_idx: torch.Tensor,
#         res_dist_matrix: torch.Tensor,
#         alpha_target: torch.Tensor,
#         alpha_tor_mask: torch.Tensor,
#         partial_T: torch.Tensor,
#         str_mask: Optional[torch.Tensor] = None,
#         seq_mask: Optional[torch.Tensor] = None,
#         bond_mask: Optional[torch.Tensor] = None,
#         res_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         tail_mask: Optional[torch.Tensor] = None,
#         use_checkpoint: bool = False,
#         batch_mask: Optional[torch.Tensor] = None,
#         trans_1: Optional[torch.Tensor] = None,
#         rotmats_1: Optional[torch.Tensor] = None,
#         aatypes_1: Optional[torch.Tensor] = None,
#         logits_1: Optional[torch.Tensor] = None,
#         # Add optional self-conditioning tensors
#         trans_sc: Optional[torch.Tensor] = None,
#         aatypes_sc: Optional[torch.Tensor] = None,
#         torsions_sc: Optional[torch.Tensor] = None,
#     ):
#         B, L = seq_noised.shape[:2]

#         # 1. Build input_feats dictionary from BondFlow inputs
#         trans_t = xyz_noised[:, :, 1, :].clone()
#         rotmats_t = iu.get_R_from_xyz(xyz_noised)

#         # Masks
#         str_mask = str_mask if str_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
#         seq_mask = seq_mask if seq_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
#         res_mask = res_mask if res_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
        
#         # Sequence and Self-Conditioning
#         aatypes_t = seq_noised.clone().long()
#         num_tokens = getattr(self.model['backbone']._model_conf, 'aatype_pred_num_tokens', 21)
#         if aatypes_sc is None:
#             aatypes_sc = torch.nn.functional.one_hot(aatypes_t, num_classes=int(num_tokens)).float()
#         if trans_sc is None:
#             trans_sc = torch.zeros(B, L, 3, device=self.device)
#         if torsions_sc is None:
#             torsions_sc = torch.zeros(B, L, 4, device=self.device) # Placeholder for sidechain model

#         # Indices and Time
#         chain_idx = get_chain_idx(pdb_idx, B, L, self.device, rf_idx)
#         res_idx = rf_idx.to(self.device)
#         t_expand = partial_T.reshape(B, 1).to(self.device).expand(B, L)
#         ones_expand = torch.ones_like(t_expand)
#         so3_t = torch.where(str_mask.bool(), t_expand, ones_expand)
#         r3_t = torch.where(str_mask.bool(), t_expand, ones_expand)
#         cat_t = torch.where(seq_mask.bool(), t_expand, ones_expand)
#         tor_t = torch.where(str_mask.bool() & seq_mask.bool(), t_expand, ones_expand) # Add tor_t for sidechain model
        
#         # Create placeholder ground truth tensors for inference compatibility IF NOT PROVIDED
#         if trans_1 is None:
#             trans_1 = torch.zeros_like(trans_t)
#         if rotmats_1 is None:
#             rotmats_1 = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1)
#         if aatypes_1 is None:
#             aatypes_1 = torch.zeros_like(aatypes_t)
#         if logits_1 is None:
#             logits_1 = torch.nn.functional.one_hot(aatypes_1, num_classes=int(num_tokens)).float()

#         # PLM embeddings (if configured)
#         plm_s, PLM_emb_weight = None, None
#         if self.PLM_info[0] is not None and self._folding_model is not None:
#              plm_s, PLM_emb_weight = get_plm_embeddings(
#                  self._folding_model,
#                  self._plm_type,
#                  aatypes_t,
#                  B,
#                  L,
#                  self.device,
#                  head_mask=head_mask,
#                  tail_mask=tail_mask,
#                  chain_idx=chain_idx,
#                  res_mask=res_mask,
#              )
        
#         input_feats = {
#             'res_mask': res_mask.float(), 'diffuse_mask': str_mask.float(),
#             'chain_idx': chain_idx, 'res_idx': res_idx,
#             'so3_t': so3_t, 'r3_t': r3_t, 'cat_t': cat_t, 'tor_t': tor_t,
#             'trans_t': trans_t, 'rotmats_t': rotmats_t, 'aatypes_t': aatypes_t,
#             'trans_sc': trans_sc, 'aatypes_sc': aatypes_sc, 'torsions_sc': torsions_sc,
#             'PLM_emb_weight': PLM_emb_weight,
#             # Add ground truth or placeholders for inference compatibility
#             'trans_1': trans_1,
#             'rotmats_1': rotmats_1,
#             'aatypes_1': aatypes_1,
#             'logits_1': logits_1,
#         }
#         if plm_s is not None:
#             input_feats['PLM_embedding_aatypes_t'] = plm_s

#         # 2. Create and add external_pair_embed from res_dist_matrix
#         res_dist_embed = self.res_dist_embed(res_dist_matrix.to(self.device))
#         input_feats['external_pair_embed'] = res_dist_embed
        
#         # 3. Sequentially run models, updating the input_feats dict at each stage
#         #    to pass outputs from earlier stages to later ones.
        
#         # Run Backbone
#         backbone_output = self.model['backbone'](input_feats)
#         input_feats.update(backbone_output)
#         # Keep backbone predictions attached for supervision. We'll pass detached copies to later stages.
        
#         # Store backbone predictions for supervision access
#         backbone_supervision = {
#             'backbone_supervision_trans': input_feats['backbone_pred_trans'].clone(),
#             'backbone_supervision_rotmats': input_feats['backbone_pred_rotmats'].clone(),
#             'backbone_supervision_logits': input_feats['backbone_pred_logits'].clone(),
#         }

#         # Run Sidechain, gated by time
#         run_sidechain = partial_T.max() >= self.model_start_t['sidechain']
#         if run_sidechain:
#             # Use a detached copy of backbone predictions when feeding sidechain
#             sidechain_input = dict(input_feats)
#             for k in ['backbone_pred_trans', 'backbone_pred_rotmats', 'backbone_pred_logits', 'backbone_pred_aatypes']:
#                 if k in sidechain_input and isinstance(sidechain_input[k], torch.Tensor):
#                     sidechain_input[k] = sidechain_input[k].detach()
#             sidechain_output = self.model['sidechain'](sidechain_input)
#             input_feats.update(sidechain_output)

#         # Run Refine, gated by time
#         run_refine = partial_T.max() >= self.model_start_t['refine']
#         if run_refine:
#             # Prevent refine loss from flowing into sidechain by detaching torsions only for refine
#             refine_input = dict(input_feats)
#             if 'sidechain_pred_torsions' in refine_input and isinstance(refine_input['sidechain_pred_torsions'], torch.Tensor):
#                 refine_input['sidechain_pred_torsions'] = refine_input['sidechain_pred_torsions'].detach()
#             refine_output = self.model['refine'](refine_input)
#             input_feats.update(refine_output)

#         # 4. Combine outputs from the final available stage
#         if run_refine and 'refine_pred_trans' in input_feats:
#             pred_trans = input_feats['refine_pred_trans']
#             pred_rotmats = input_feats['refine_pred_rotmats']
#             pred_logits = input_feats['refine_pred_logits']
#         else:
#             pred_trans = input_feats['backbone_pred_trans']
#             pred_rotmats = input_feats['backbone_pred_rotmats']
#             pred_logits = input_feats['backbone_pred_logits']

#         xyz_pred = iu.get_xyz_from_RT(pred_rotmats, pred_trans).unsqueeze(1)

#         # Torsions (alpha_s)
#         if run_sidechain and 'sidechain_pred_torsions_sincos' in input_feats:
#             sidechain_sincos = input_feats['sidechain_pred_torsions_sincos'] # [B, L, 4, 2]           
#             sidechain_cossin = torch.flip(sidechain_sincos, dims=[-1])
#             alpha_s_chi = torch.zeros(B, L, 10, 2, device=self.device)
#             alpha_s_chi[:, :, 3:7, :] = sidechain_cossin
#             alpha_s = alpha_s_chi
#         else:
#             # Default to ideal torsions for all 10 channels when sidechain isn't run
#             alpha_s = torch.zeros(B, L, 10, 2, device=self.device)


#         # 5. Predict bond matrix using embeddings from the final available stage
#         final_node_embed = input_feats.get('refine_node_embed', input_feats['backbone_node_embed'])
#         final_edge_embed = input_feats.get('refine_edge_embed', input_feats['backbone_edge_embed'])
#         res_mask_bool = (res_mask > 0).to(torch.bool)
#         mask_2d = res_mask_bool.unsqueeze(1) & res_mask_bool.unsqueeze(2)
#         if bond_mask is not None:
#             mask_2d = mask_2d & bond_mask.bool()
#         bond_matrix = self.bond_pred(
#             msa=final_node_embed,
#             state=final_node_embed,
#             pair=final_edge_embed,
#             mask_2d=mask_2d,
#         )

#         # 6. Store backbone supervision info in input_feats for training access
#         input_feats.update(backbone_supervision)
        
#         # Store input_feats for training access (contains all intermediate predictions)
#         self.last_input_feats = input_feats
        
#         # 7. Return 4-tuple
#         return pred_logits, xyz_pred, alpha_s, bond_matrix

class APMBackboneWrapper(BaseDesignModel):
    """APM backbone-only wrapper.

    - Instantiates and runs only the BackboneModel.
    - Builds input_feats from BondFlow inputs, injects external pair embeddings.
    - Maps outputs back to BondFlow format (pred_logits, xyz_pred, alpha_s, bond_matrix).
    - Sidechain torsions are not predicted; returns zero alpha_s placeholders.
    """

    def __init__(self, conf, device: str, d_t1d: int, d_t2d: int):
        super().__init__()
        self.device = device
        self.d_t1d = d_t1d
        self.d_t2d = d_t2d
        self._conf = conf
        print("loading APMBackboneWrapper")
        _APM_FoldingModel = None
        try:
            from apm.apm.models.folding_model import FoldingModel as _APM_FoldingModel
        except ImportError:
            print("[APMBackboneWrapper] Warning: Could not import APM's FoldingModel. PLM features will be disabled.")

        self.PLM_info = (None, None)
        self._plm_type = None
        self._folding_model = None

        if not hasattr(conf, 'model'):
            raise ValueError("APM model configuration not found in config")
        apm_model_conf = conf.model
        self.edge_time_emb_dim = apm_model_conf.edge_features.c_timestep_emb

        folding_conf = getattr(conf, 'folding', None)
        if folding_conf and _APM_FoldingModel:
            plm_name = getattr(folding_conf, 'PLM', None)
            if plm_name and plm_name not in ['null', 'None']:
                self._folding_model = _APM_FoldingModel(folding_conf)
                self._folding_model.init_faESM()
                self._plm_type = plm_name
                self.PLM_info = (
                    getattr(self._folding_model, 'plm_representations_layer', None),
                    getattr(self._folding_model, 'plm_representations_dim', None),
                )
        # Optional maximum chain length for PLM; if not present in config,
        # we do not truncate by default.
        self._plm_max_len = None
        if folding_conf is not None and hasattr(folding_conf, 'plm_max_chain_length'):
            try:
                self._plm_max_len = int(folding_conf.plm_max_chain_length)
            except Exception:
                self._plm_max_len = 1024

        self.backbone = BackboneModel(apm_model_conf, self.PLM_info).to(device)

        # Target pair embedding dimension used by the APM backbone
        pair_embed_size = getattr(apm_model_conf, 'edge_embed_size', getattr(apm_model_conf.ipa, 'c_z', 128))
        self.pair_embed_size = pair_embed_size

        # Dimension of diffusion-distance features from smu.diffusion_distance_tensor

        self._res_dist_dim = len(self._conf.preprocess.diffusion_map_times) * self._conf.preprocess.diffusion_rbf_num

        # Projection + ScoreNet + output head for atomic 3x3 patch features
        d_hidden = self.pair_embed_size
        self.res_atom_proj = nn.Linear(self._res_dist_dim, d_hidden).to(device)
        self.res_atom_score_body = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        ).to(device)
        self.res_atom_score_anchor = nn.Sequential(
            nn.Linear(d_hidden, d_hidden//4),
            nn.ReLU(),
            nn.Linear(d_hidden//4, 1),
        ).to(device)
        # Concatenate time embedding before the final head: [feat, time_emb]
        self.edge_feats_mlp = nn.Sequential(
            nn.Linear(d_hidden + self.edge_time_emb_dim, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, self.pair_embed_size),
        ).to(device)



        node_embed_size = getattr(apm_model_conf, 'node_embed_size', getattr(apm_model_conf.ipa, 'c_s', 256))
        self.bond_pred = BondingNetwork(
            d_msa=node_embed_size,
            d_state=node_embed_size,
            d_pair=pair_embed_size,
        ).to(device)

        # Lightweight torsion head for alpha_s using backbone node embeddings
        self.torsion_head = AngleResnet(
            c_in=node_embed_size,
            c_hidden=node_embed_size,
            no_blocks=getattr(apm_model_conf, 'num_torsion_blocks', 4),
            no_angles=4,
            epsilon=1e-4,
            use_initial=False,
        ).to(device)
        # 2: head_mask and tail_mask, 1: hotspots
        self.node_proj = nn.Linear(2 + 1, node_embed_size).to(device)
        self._try_load_backbone_checkpoint()



    def _try_load_backbone_checkpoint(self):

        def load_state_dict_ignore_mismatch(module, state_dict, strict=False, prefix=""):
            module_sd = module.state_dict()
            filtered_sd = {}
            skipped = []

            for k, v in state_dict.items():
                if k in module_sd:
                    if module_sd[k].shape == v.shape:
                        filtered_sd[k] = v
                    else:
                        skipped.append((k, v.shape, module_sd[k].shape))

            load_info = module.load_state_dict(filtered_sd, strict=strict)

            if skipped:
                print(f"[load_state_dict_ignore_mismatch]{prefix} skipped {len(skipped)} params:")
                for k, s1, s2 in skipped:
                    print(f"  - {k}: ckpt{s1} vs model{s2}")

            return load_info

        import os
        ckpt_path = getattr(self._conf.model, 'ckpt_path', None)
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"[APMBackboneWrapper] Warning: No checkpoint provided or path invalid: {ckpt_path}. Model is randomly initialized.")
            return

        strict_loading = getattr(self._conf.model, 'strict_loading', False)
        ckpt_obj = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if isinstance(ckpt_obj, dict):
            if 'model_state_dict' in ckpt_obj:
                raw_sd = ckpt_obj['model_state_dict']
            elif 'state_dict' in ckpt_obj:
                raw_sd = ckpt_obj['state_dict']
            else:
                raw_sd = ckpt_obj
        else:
            raw_sd = ckpt_obj

        #load_info_all = self.load_state_dict(raw_sd, strict=False)
        load_info_all = load_state_dict_ignore_mismatch(self, raw_sd)
        missing_all = getattr(load_info_all, 'missing_keys', [])
        unexpected_all = getattr(load_info_all, 'unexpected_keys', [])
        total_keys = len(raw_sd) if hasattr(raw_sd, 'keys') else 0
        num_loaded = total_keys - len(unexpected_all)
        print(
            f"[APMBackboneWrapper] Loaded wrapper from {ckpt_path}: loaded~{num_loaded}/{total_keys}, "
            f"missing={len(missing_all)}, unexpected={len(unexpected_all)}, strict={strict_loading}"
        )

        if num_loaded == 0 and isinstance(raw_sd, dict):
            prefix = 'model.backbone.'
            extracted_sd = {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}
            if extracted_sd:
                #load_info = self.backbone.load_state_dict(extracted_sd, strict=strict_loading)
                load_info = load_state_dict_ignore_mismatch(
                                                            self.backbone,
                                                            extracted_sd,
                                                            strict=strict_loading,
                                                            prefix="[backbone]"
                                                        )
                missing = getattr(load_info, 'missing_keys', [])
                unexpected = getattr(load_info, 'unexpected_keys', [])
                print(
                    f"[APMBackboneWrapper] Loaded 'backbone' weights from {ckpt_path}: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )

    def get_time_embedding_nd(self, timesteps, mask):
        orig_shape = timesteps.shape
        t_flat = timesteps.reshape(-1)
        emb_flat = get_time_embedding(t_flat,self.edge_time_emb_dim)
        emb = emb_flat.reshape(*orig_shape, emb_flat.shape[-1])

        # Apply mask if provided. Expect mask shape [B, L]. For pairwise timesteps [B, L, L],
        # construct a pairwise mask [B, L, L] = mask[:, :, None] & mask[:, None, :].
        mask_bool = mask.bool()

        if timesteps.dim() == 3:
            # timesteps: [B, L, L]
            pair_mask = (mask_bool[:, :, None] & mask_bool[:, None, :]).to(emb.dtype)
            emb = emb * pair_mask[..., None]
        elif timesteps.dim() == 2:
            # timesteps: [B, L]
            emb = emb * mask_bool[..., None].to(emb.dtype)
        # If timesteps is [B], mask application is ambiguous; skip.

        return emb


    def bond_mat_2_dist_mat(
        self,
        bond_mat: torch.Tensor,
        rf_idx: torch.Tensor,
        res_mask: torch.Tensor,
        bond_mask: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor] = None,
        tail_mask: Optional[torch.Tensor] = None,
        N_C_anchor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        [Vectorized & Weighted Version]
        Build pairwise diffusion-distance embedding from a bond matrix.
        
        Improvements:
        1. No explicit loops over batch dimension.
        2. Uses actual bond weights for Special Bonds instead of binary 0/1,
           allowing the model to perceive weak connections (e.g., 0.3 vs 0.9).
        """

        B, L_total = bond_mat.shape[:2]

        use_atomic_graph = (
            head_mask is not None
            and tail_mask is not None
            and N_C_anchor is not None
        )

        # -----------------------------
        # Fallback: original behaviour (residue graph)
        # -----------------------------
        if not use_atomic_graph:
            # 此处保留原有的 Fallback 逻辑，为了代码简洁省略具体实现
            # 如果需要完整运行，请确保这里调用了原本的逻辑
            raise ValueError("Atomic graph is not supported")
            # return self._bond_mat_2_dist_mat_fallback(bond_mat, rf_idx, res_mask)

        # ---------------------------------------
        # Upgraded: Atomic Graph (Vectorized & Weighted)
        # ---------------------------------------
        device = bond_mat.device
        dtype = bond_mat.dtype

        head_mask = head_mask.bool()
        tail_mask = tail_mask.bool()
        res_mask_bool = res_mask.bool().squeeze(-1) if res_mask.dim() == 3 else res_mask.bool()

        # 1. Prepare Body Mask & Counts
        body_mask_bool = res_mask_bool & (~head_mask) & (~tail_mask)
        body_counts = body_mask_bool.sum(dim=1)
        
        # Handle degenerate case
        if int(body_counts.max().item()) == 0:
            return self.bond_mat_2_dist_mat(bond_mat, rf_idx, res_mask, bond_mask)

        L_body_max = int(body_counts.max().item())
        N_atom = 3 * L_body_max

        # --- 2. Vectorized Owner Mapping ---
        # Initialize owner as self (0..L-1)
        owner_global = torch.arange(L_total, device=device).view(1, L_total).expand(B, L_total).clone()
        
        # Precompute body local indices [B, L_total] -> 0..Lb or -1
        body_cumsum = body_mask_bool.long().cumsum(dim=1)
        body_local_all = body_cumsum - 1
        body_local_all = body_local_all.masked_fill(~body_mask_bool, -1)

        # Function to update owners based on anchors (Vectorized)
        def _assign_owner_vectorized(layer_idx, func_mask):
            if not func_mask.any(): return
            # anchor_layer: [B, L_func, L_anchor]
            anchor_layer = N_C_anchor[..., layer_idx]
            # Mask out invalid body columns
            valid_anchor_cols = body_mask_bool.unsqueeze(1) 
            masked_anchors = anchor_layer & valid_anchor_cols
            
            anchor_exists = masked_anchors.any(dim=2) 
            anchor_idx = masked_anchors.float().argmax(dim=2) 
            
            update_mask = func_mask & anchor_exists
            owner_global[update_mask] = anchor_idx[update_mask]

        _assign_owner_vectorized(0, head_mask) # N-side
        _assign_owner_vectorized(1, tail_mask) # C-side

        # Atom types: 0->N, 1->CA, 2->C
        center_type = torch.full((B, L_total), 1, device=device, dtype=torch.long)
        center_type[head_mask] = 0
        center_type[tail_mask] = 2

        # --- 3. Build Weighted Atomic Adjacency ---
        A_atom = torch.zeros((B, N_atom, N_atom), device=device, dtype=dtype)
     
        # 3.1 Intra-residue edges (Weight = 1.0)
        # Create indices for valid atoms in batch
        seq_range = torch.arange(L_body_max, device=device).unsqueeze(0)
        len_mask = seq_range < body_counts.unsqueeze(1) # [B, Lb]
        atom_valid_mask = len_mask.repeat_interleave(3, dim=1) # [B, 3Lb]
        
        batch_idx_grid = torch.arange(B, device=device).unsqueeze(1).expand(B, L_body_max)
        b_ids = batch_idx_grid[len_mask]
        l_ids = seq_range.expand(B, L_body_max)[len_mask]
        
        n_ids = 3 * l_ids; ca_ids = n_ids + 1; c_ids = n_ids + 2
        
        b_intra = torch.cat([b_ids, b_ids, b_ids, b_ids])
        u_intra = torch.cat([n_ids, ca_ids, ca_ids, c_ids])
        v_intra = torch.cat([ca_ids, n_ids, c_ids, ca_ids])
        w_intra = torch.ones_like(b_intra, dtype=dtype) # Fixed strong connection

        # 3.2 Peptide bonds (Weight = 1.0)
        diff = rf_idx[:, 1:] - rf_idx[:, :-1]
        is_continuous = (diff == 1) | (diff == 0)  # 允许递增或相等
        
        is_seq = is_continuous & res_mask_bool[:, :-1] & res_mask_bool[:, 1:]
        
        seq_b, seq_i = torch.nonzero(is_seq, as_tuple=True)
        
        if seq_b.numel() > 0:
            owner_i = body_local_all[seq_b, seq_i]
            owner_ip1 = body_local_all[seq_b, seq_i + 1]
            valid_pep = (owner_i >= 0) & (owner_ip1 >= 0)
            
            seq_b = seq_b[valid_pep]
            owner_i = owner_i[valid_pep]
            owner_ip1 = owner_ip1[valid_pep]
            
            c_i = 3 * owner_i + 2
            n_ip1 = 3 * owner_ip1 + 0
            
            b_pep = torch.cat([seq_b, seq_b])
            u_pep = torch.cat([c_i, n_ip1])
            v_pep = torch.cat([n_ip1, c_i])
            w_pep = torch.ones_like(b_pep, dtype=dtype)
        else:
            b_pep = u_pep = v_pep = w_pep = torch.empty(0, device=device, dtype=dtype)

        # 3.3 Special bonds (Weighted from bond_mat)
        # Threshold changed from 0.5 to 0.01 to preserve weak signals
        threshold = 1e-4
        bm_mask = (bond_mat > threshold)
        bm_mask.diagonal(dim1=1, dim2=2).fill_(False)
        
        sp_b, sp_i, sp_j = torch.nonzero(bm_mask, as_tuple=True)
        
        if sp_b.numel() > 0:
            # [KEY CHANGE]: Extract actual weights instead of using 1.0
            edge_weights = bond_mat[sp_b, sp_i, sp_j]

            owner_i_g = owner_global[sp_b, sp_i]
            owner_j_g = owner_global[sp_b, sp_j]
            owner_i_l = body_local_all[sp_b, owner_i_g]
            owner_j_l = body_local_all[sp_b, owner_j_g]
            
            valid_sp = (owner_i_l >= 0) & (owner_j_l >= 0)
            
            sp_b = sp_b[valid_sp]
            edge_weights = edge_weights[valid_sp] # Filter weights too
            owner_i_l = owner_i_l[valid_sp]
            owner_j_l = owner_j_l[valid_sp]
            
            t_i = center_type[sp_b, sp_i[valid_sp]]
            t_j = center_type[sp_b, sp_j[valid_sp]]
            
            atom_i = 3 * owner_i_l + t_i
            atom_j = 3 * owner_j_l + t_j
            
            b_sp = torch.cat([sp_b, sp_b])
            u_sp = torch.cat([atom_i, atom_j])
            v_sp = torch.cat([atom_j, atom_i])
            w_sp = torch.cat([edge_weights, edge_weights]) # Symmetric weights
        else:
            b_sp = u_sp = v_sp = w_sp = torch.empty(0, device=device, dtype=dtype)

        # Combine & Fill
        b_all = torch.cat([b_intra, b_pep, b_sp])
        u_all = torch.cat([u_intra, u_pep, u_sp])
        v_all = torch.cat([v_intra, v_pep, v_sp])
        w_all = torch.cat([w_intra, w_pep, w_sp])

        # -----------------------------------------------------------
        # [NEW] 使用 scatter_reduce_ 实现 Max Aggregation
        # -----------------------------------------------------------
        
        # 1. 计算展平后的 1D 索引 (Flat Indices)
        # A_atom shape: (B, N_atom, N_atom)
        # flat_idx = b * (N * N) + u * N + v
        stride_b = N_atom * N_atom
        stride_u = N_atom
        flat_indices = b_all * stride_b + u_all * stride_u + v_all

        # 2. 展平目标张量
        A_flat = A_atom.view(-1)

        # 3. 执行 Scatter Reduce (取最大值)
        # reduce='amax' 会在重复的索引处取最大值
        # include_self=True 表示与张量中原本的值(0.0)进行比较，确保正数能写进去
        # 由于 A_flat 是 A_atom 的视图 (View)，A_atom 会自动更新
        A_flat.scatter_reduce_(
            dim=0, 
            index=flat_indices.long(), 
            src=w_all, 
            reduce='amax', 
            include_self=True
        )



        # 检查是否有重复的 (b, u, v) 索引
        unique_indices = len(torch.unique(torch.stack([b_all, u_all, v_all], dim=0), dim=1))
        total_indices = len(b_all)
        print(f"Unique indices: {unique_indices}, Total indices: {total_indices}")

        # --- 4. Diffusion & ScoreNet ---
        # Get raw distances (rbf_num=0) to save memory
        res_dist_atom = smu.diffusion_distance_tensor(
            A_adj_batch=A_atom,
            times=self._conf.preprocess.diffusion_map_times,
            k=self._conf.preprocess.diffusion_map_features,
            skip_top=True,
            node_mask=atom_valid_mask.int(),
            rbf_num=0,
            rbf_gamma=None,
            k_ratio=self._conf.preprocess.diffusion_k_ratio,
        )

        print("res_dist_atom nan num", res_dist_atom.isnan().sum().item())
        # res_dist_atom: (B_loc, 3L, 3L, T)
        B_loc, N_loc, _, T = res_dist_atom.shape

        # Prepare mask for patches (B, L, L, 9)
        atom_mask_2d = atom_valid_mask.unsqueeze(2) & atom_valid_mask.unsqueeze(1) # (B, 3L, 3L)
        mask_patches = atom_mask_2d.view(B_loc, L_body_max, 3, L_body_max, 3).permute(0, 1, 3, 2, 4).reshape(B_loc, L_body_max, L_body_max, 9)

        # Reshape to patches (B, L, L, 9, T)
        X = res_dist_atom.view(B_loc, L_body_max, 3, L_body_max, 3, T).permute(0, 1, 3, 2, 4, 5)
        X = X.reshape(B_loc, L_body_max, L_body_max, 9, T)
        
        # Chunked RBF + Projection
        X_proj = compute_rbf_and_project_chunked(
            self.res_atom_proj,
            X,
            mask_patches,
            rbf_num=self._conf.preprocess.diffusion_rbf_num,
            rbf_gamma=None,
            chunk_size=16
        )
 
        # 5.1 Create Index Grid
        # local_map: [B, L_total] containing local body index or -1
        local_map = torch.gather(body_local_all, 1, owner_global)
        valid_map = local_map >= 0
        
        I_map = local_map.unsqueeze(2).expand(B, L_total, L_total)
        J_map = local_map.unsqueeze(1).expand(B, L_total, L_total)
        valid_pair = valid_map.unsqueeze(2) & valid_map.unsqueeze(1)
        
        # Clamp to 0 to prevent gather error (result masked later)
        I_gather = I_map.masked_fill(~valid_pair, 0)
        J_gather = J_map.masked_fill(~valid_pair, 0)
        


        # 5.3 Sparse Aggregation & Fill Helper (New Optimization)
        def _aggregate_and_fill(token_indices, active_mask, score_net,transpose=False):
            """
            只在 active_mask 为 True 的位置计算特征并填充到 feat_full，避免全图计算。
            transpose=True 意味着我们需要取对称位置 (j, i) 的 Body 映射来计算当前 (i, j) 的值。
            """
            if not active_mask.any(): return
            
            # 1. 获取全局坐标 (Global Indices)
            b, i, j = torch.nonzero(active_mask, as_tuple=True)
            
            # 2. 映射到局部 Body 坐标 (Body Indices)
            if transpose:
                i_b = I_gather[b, j, i]
                j_b = J_gather[b, j, i]
            else:
                i_b = I_gather[b, i, j]
                j_b = J_gather[b, i, j]
            
            # 过滤掉无效映射 (理论上 active_mask 已经是 valid_p 的子集，但为了保险)
            # I_gather 之前被 masked_fill 把 -1 变成了 0，但在 valid_p 区域内它是原始有效值
            valid = (i_b >= 0) & (j_b >= 0)
            if not valid.any(): return
            
            b, i, j = b[valid], i[valid], j[valid]
            i_b, j_b = i_b[valid], j_b[valid]

            # 3. 稀疏提取特征 (Sparse Indexing)
            # X_proj: [B, Lb, Lb, 9, D] -> [N_active, 9, D]
            sub_x = X_proj[b, i_b, j_b] 
            
            # 4. 选择 Token 并聚合 (Compute)
            # sub_x: [N_active, 9, D] -> [N_active, K, D]
            sub_x = sub_x[:, token_indices]
            
            scores = score_net(sub_x) # [N_active, K, 1]
            attn = torch.softmax(scores, dim=1) 
            res = (attn * sub_x).sum(dim=1)     # [N_active, D]
            
            # 5. 写入结果 (Scatter)
            feat_full[b, i, j] = res

        # 5.3 Priority Masking (torch.where)
        # Apply Base mask (valid pairs only)
        head_r = head_mask.view(B, L_total, 1)
        tail_r = tail_mask.view(B, L_total, 1)
        head_c = head_mask.view(B, 1, L_total)
        tail_c = tail_mask.view(B, 1, L_total)
        valid_p = valid_pair.squeeze(-1) # [B, L, L]

        # Initialize feat_full with valid body pairs
        # 使用 Sparse Fill 处理 Body 部分，避免大张量中间变量
        # 1. Body (Base) - Sparse-like fill
        d_hidden = X_proj.shape[-1]
        feat_full = torch.zeros((B, L_total, L_total, d_hidden), dtype=X_proj.dtype, device=device)
        idx_all = torch.arange(9, device=device)
        _aggregate_and_fill(idx_all, valid_p, self.res_atom_score_body, transpose=False)

        # 2. Level 1: Functional Rows/Cols (N/C) - Sparse
        idx_N = torch.tensor([0, 1, 2], device=device)
        _aggregate_and_fill(idx_N, (head_r & valid_p), self.res_atom_score_anchor,transpose=False)
        _aggregate_and_fill(idx_N, (head_c & valid_p), self.res_atom_score_anchor,transpose=True)

        idx_C = torch.tensor([6, 7, 8], device=device)
        _aggregate_and_fill(idx_C, (tail_r & valid_p), self.res_atom_score_anchor, transpose=False)
        _aggregate_and_fill(idx_C, (tail_c & valid_p), self.res_atom_score_anchor, transpose=True)

        # 3. Level 2: Specific Functional Pairs (Highest Priority) - Sparse
        # N-N
        idx_NN = torch.tensor([0], device=device)
        _aggregate_and_fill(idx_NN, (head_r & head_c & valid_p), self.res_atom_score_anchor,transpose=False)

        # C-C
        idx_CC = torch.tensor([8], device=device)
        _aggregate_and_fill(idx_CC, (tail_r & tail_c & valid_p), self.res_atom_score_anchor,transpose=False)

        # C-N
        idx_CN = torch.tensor([6], device=device)
        # Tail Row (i is tail), Head Col (j is head) -> Normal CN
        _aggregate_and_fill(idx_CN, (tail_r & head_c & valid_p), self.res_atom_score_anchor,transpose=False)
        # Head Row (i is head), Tail Col (j is tail) -> Transposed CN
        _aggregate_and_fill(idx_CN, (head_r & tail_c & valid_p), self.res_atom_score_anchor,transpose=True)

        # # --- 6. Final Output ---
        # feat_cat = torch.cat([feat_full, time_embedding.to(feat_full.dtype)], dim=-1)
        # res_dist_embed = self.res_atom_out(feat_cat)

        return feat_full

    # def head_tail_interpolate(self, pred_rotmats, pred_trans,head_mask, tail_mask,N_C_anchor, insert_alpha = 1/3):
    #     def apply( mask, anchor_layer):
    #         batch_idx, node_idx = torch.where(mask)
    #         if len(batch_idx) == 0:
    #             return
                
    #         # Get anchor rows: [N_nodes, L]
    #         anchor_rows = N_C_anchor[batch_idx, node_idx, :, anchor_layer]
            
    #         # Find body indices (argmax over L)
    #         # Check if valid anchor exists (row not all zeros)
    #         valid_anchor = anchor_rows.any(dim=-1)
    #         if not valid_anchor.any():
    #             return
                
    #         # Filter to valid updates
    #         batch_idx = batch_idx[valid_anchor]
    #         node_idx = node_idx[valid_anchor]
    #         anchor_rows = anchor_rows[valid_anchor]
            
    #         body_idx = anchor_rows.float().argmax(dim=-1)

    #         return batch_idx, body_idx
        
    #     batch_idx_head, body_idx_head = apply(head_mask, 0)
    #     batch_idx_tail, body_idx_tail = apply(tail_mask, 1)

    #     pred_trans[batch_idx_head, body_idx_head] = (1-insert_alpha) * pred_trans[batch_idx_head, body_idx_head] + insert_alpha * pred_trans[head_mask]
    #     pred_trans[batch_idx_tail, body_idx_tail] = (1-insert_alpha) * pred_trans[batch_idx_tail, body_idx_tail] + insert_alpha * pred_trans[tail_mask]

    #     pred_rotmats[batch_idx_tail, body_idx_tail] = su.geodesic_t(insert_alpha, pred_rotmats[tail_mask],pred_rotmats[batch_idx_tail, body_idx_tail])
    #     pred_rotmats[batch_idx_head, body_idx_head] = su.geodesic_t(insert_alpha, pred_rotmats[head_mask],pred_rotmats[batch_idx_head, body_idx_head])
    #     return pred_trans, pred_rotmats

    # def _get_fullstructure_plm(
    #     self,
    #     pdb_core_id: str,
    #     seq_full,
    #     idx_full,
    # ) -> Optional[torch.Tensor]:
    #     """
    #     Compute (or fetch from cache) PLM embeddings for the *entire* structure
    #     (all chains concatenated) in one call to get_plm_embeddings.

    #     Returns:
    #         Tensor[L_full, num_layers, H] on CPU, or None if PLM is not configured.
    #     """
    #     if self._folding_model is None or self._plm_type is None:
    #         return None

    #     if seq_full is None or idx_full is None:
    #         return None

    #     # Use pdb_core_id as the cache key at structure level.
    #     key = pdb_core_id
    #     if key in self._plm_cache:
    #         return self._plm_cache[key]

    #     # Convert sequence to numpy
    #     if torch.is_tensor(seq_full):
    #         seq_np = seq_full.detach().cpu().numpy().astype(np.int64)
    #     else:
    #         seq_np = np.asarray(seq_full, dtype=np.int64)

    #     L_full = len(seq_np)
    #     if L_full == 0 or L_full != len(idx_full):
    #         return None

    #     device = self.device
    #     aatypes_t = torch.from_numpy(seq_np.astype(np.int64))[None, :].to(device)  # [1, L_full]
    #     B, L = aatypes_t.shape

    #     # Build chain_idx from idx_full: assign integer ids per chain label.
    #     chain_idx = torch.zeros(B, L, dtype=torch.long, device=device)
    #     chain_map = {}
    #     for i, (ch, res) in enumerate(idx_full):
    #         if ch not in chain_map:
    #             chain_map[ch] = len(chain_map) + 1
    #         chain_idx[0, i] = chain_map[ch]

    #     # Build res_mask, optionally enforcing per-chain max length if configured.
    #     res_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    #     if self._plm_max_len is not None:
    #         counts = {}
    #         mask_np = np.ones(L, dtype=bool)
    #         for i, (ch, res) in enumerate(idx_full):
    #             c = counts.get(ch, 0)
    #             if c >= self._plm_max_len:
    #                 mask_np[i] = False
    #             else:
    #                 counts[ch] = c + 1
    #         res_mask = torch.from_numpy(mask_np.reshape(1, L)).to(device)

    #     # Let get_plm_embeddings derive head/tail automatically from chain_idx.
    #     plm_s_full = get_plm_embeddings(
    #         folding_model=self._folding_model,
    #         plm_type=self._plm_type,
    #         aatypes_t=aatypes_t,
    #         B=B,
    #         L=L,
    #         device=device,
    #         head_mask=None,
    #         tail_mask=None,
    #         chain_idx=chain_idx,
    #         res_mask=res_mask,
    #         res_idx=None,
    #     )  # [1, L_full, num_layers, H]

    #     plm_struct = plm_s_full[0].detach().cpu()  # [L_full, num_layers, H]
    #     self._plm_cache[key] = plm_struct
    #     return plm_struct

    # def _compute_plm_for_sample(
    #     self,
    #     pdb_core_id: str,
    #     seq_full,
    #     idx_full,
    #     origins,
    # ) -> Optional[torch.Tensor]:
    #     """
    #     Build per-design-position PLM embeddings for a single sample from full-chain PLMs.

    #     Args:
    #         pdb_core_id: core pdb id without chain suffix.
    #         seq_full: np.ndarray[int] or 1D tensor of length L_full.
    #         idx_full: list of (chain, res) of length L_full.
    #         origins: list of (chain, res) or ('?', '-1') of length L_design.

    #     Returns:
    #         Tensor[L_design, num_layers, H] on self.device, or None if PLM is unavailable.
    #     """
    #     if origins is None:
    #         return None

    #     # Get (or compute) full-structure PLM: [L_full, num_layers, H] on CPU
    #     plm_struct = self._get_fullstructure_plm(pdb_core_id, seq_full, idx_full)
    #     if plm_struct is None:
    #         return None

    #     L_full, num_layers, hid_dim = plm_struct.shape
    #     L_design = len(origins)

    #     # Map origin pdb_idx -> positions in full structure (support duplicates)
    #     origin_to_pos = {}
    #     for i, origin in enumerate(idx_full):
    #         origin_to_pos.setdefault(origin, []).append(i)

    #     # Allocate per-design-position PLM tensor on device
    #     plm_sample = torch.zeros(
    #         L_design,
    #         num_layers,
    #         hid_dim,
    #         dtype=plm_struct.dtype,
    #         device=self.device,
    #     )

    #     for i, origin in enumerate(origins):
    #         if not origin or origin == ('?', '-1'):
    #             continue
    #         pos_list = origin_to_pos.get(origin)
    #         if not pos_list:
    #             continue
    #         pos = pos_list[0]
    #         if 0 <= pos < L_full:
    #             plm_sample[i] = plm_struct[pos].to(self.device)

    #     return plm_sample

    # def _compute_batched_full_plm(
    #     self,
    #     origin_pdb_idx: list,
    #     pdb_seq_full: list,
    #     pdb_idx_full: list,
    #     B: int,
    #     L: int,
    # ) -> Optional[torch.Tensor]:
    #     """
    #     Batched full-structure PLM computation for an entire batch.

    #     Returns:
    #         plm_s: [B, L, num_layers, H] or None if PLM is unavailable.
    #     """
    #     seq_list: list[Optional[np.ndarray]] = []
    #     idx_list: list[Optional[list]] = []
    #     valid_sample = [True] * B
    #     full_lengths: list[int] = []

    #     for b in range(B):
    #         sf = pdb_seq_full[b]
    #         idx_full_b = pdb_idx_full[b]
    #         if sf is None or idx_full_b is None:
    #             seq_list.append(None)
    #             idx_list.append(None)
    #             valid_sample[b] = False
    #             full_lengths.append(0)
    #             continue
    #         if torch.is_tensor(sf):
    #             seq_np = sf.detach().cpu().numpy().astype(np.int64)
    #         else:
    #             seq_np = np.asarray(sf, dtype=np.int64)
    #         if len(seq_np) != len(idx_full_b):
    #             seq_list.append(None)
    #             idx_list.append(None)
    #             valid_sample[b] = False
    #             full_lengths.append(0)
    #             continue
    #         seq_list.append(seq_np)
    #         idx_list.append(idx_full_b)
    #         full_lengths.append(len(seq_np))

    #     if max(full_lengths) <= 0:
    #         return None

    #     L_full_max = max(full_lengths)
    #     device = self.device
    #     MASK_TOKEN = 20

    #     # aatypes_batch: [B, L_full_max], padded with MASK
    #     aatypes_batch = torch.full(
    #         (B, L_full_max),
    #         fill_value=MASK_TOKEN,
    #         dtype=torch.long,
    #         device=device,
    #     )
    #     # res_mask_batch: [B, L_full_max], False for padding and for positions
    #     # truncated by plm_max_len
    #     res_mask_batch = torch.zeros(
    #         B, L_full_max, dtype=torch.bool, device=device
    #     )
    #     # chain_idx_batch: [B, L_full_max], 1-based chain ids
    #     chain_idx_batch = torch.zeros(
    #         B, L_full_max, dtype=torch.long, device=device
    #     )

    #     for b in range(B):
    #         if not valid_sample[b]:
    #             continue
    #         seq_np = seq_list[b]
    #         idx_full_b = idx_list[b]
    #         Lb = full_lengths[b]
    #         # Fill tokens
    #         aatypes_batch[b, :Lb] = torch.from_numpy(seq_np).to(
    #             device=device, dtype=torch.long
    #         )

    #         # Build res_mask with optional per-chain max length
    #         if self._plm_max_len is None:
    #             res_mask_batch[b, :Lb] = True
    #         else:
    #             counts = {}
    #             for i_pos, (ch, res_ch) in enumerate(idx_full_b):
    #                 if i_pos >= L_full_max:
    #                     break
    #                 c = counts.get(ch, 0)
    #                 if c >= self._plm_max_len:
    #                     continue
    #                 counts[ch] = c + 1
    #                 res_mask_batch[b, i_pos] = True

    #         # Build chain_idx for this sample
    #         chain_map = {}
    #         for i_pos, (ch, res_ch) in enumerate(idx_full_b):
    #             if i_pos >= L_full_max:
    #                 break
    #             if ch not in chain_map:
    #                 chain_map[ch] = len(chain_map) + 1
    #             chain_idx_batch[b, i_pos] = chain_map[ch]

    #     plm_batch = get_plm_embeddings(
    #         folding_model=self._folding_model,
    #         plm_type=self._plm_type,
    #         aatypes_t=aatypes_batch,
    #         B=B,
    #         L=L_full_max,
    #         device=device,
    #         head_mask=None,
    #         tail_mask=None,
    #         chain_idx=chain_idx_batch,
    #         res_mask=res_mask_batch,
    #         res_idx=None,
    #     )  # [B, L_full_max, num_layers, H]

    #     _, _, num_layers, hid_dim = plm_batch.shape

    #     # Slice per-sample and map to design positions
    #     plm_list: list[torch.Tensor] = []
    #     for b in range(B):
    #         if not valid_sample[b]:
    #             plm_b = torch.zeros(
    #                 L, num_layers, hid_dim, device=device, dtype=plm_batch.dtype
    #             )
    #             plm_list.append(plm_b)
    #             continue

    #         idx_full_b = idx_list[b]
    #         origins_b = origin_pdb_idx[b]
    #         Lb = full_lengths[b]
    #         plm_full_b = plm_batch[b, :Lb].detach()  # [Lb, num_layers, H]

    #         origin_to_pos = {}
    #         for i_pos, origin in enumerate(idx_full_b):
    #             # Ensure origin keys are hashable and consistently typed (tuple)
    #             if isinstance(origin, list):
    #                 origin_key = tuple(origin)
    #             else:
    #                 origin_key = origin
    #             origin_to_pos.setdefault(origin_key, []).append(i_pos)

    #         plm_b = torch.zeros(
    #             L,
    #             num_layers,
    #             hid_dim,
    #             device=device,
    #             dtype=plm_full_b.dtype,
    #         )
    #         for i_pos, origin in enumerate(origins_b):
    #             if not origin or origin == ('?', '-1'):
    #                 continue
    #             # Match the canonical key type used above
    #             if isinstance(origin, list):
    #                 origin_key = tuple(origin)
    #             else:
    #                 origin_key = origin
    #             pos_list = origin_to_pos.get(origin_key)
    #             if not pos_list:
    #                 continue
    #             pos = pos_list[0]
    #             if 0 <= pos < Lb:
    #                 plm_b[i_pos] = plm_full_b[pos]
    #                 # Debug: for the first sample in the batch, print a few mappings
    #                 # from design index -> full-structure index and corresponding
    #                 # aatype token in aatypes_batch, to verify alignment.
    #                 # if b == 0 and i_pos < 300:
    #                 #     token_val = int(aatypes_batch[0, pos].item())
    #                 #     print(
    #                 #         f"[PLM DEBUG] sample=0, design_idx={i_pos}, full_pos={pos}, "
    #                 #         f"aatype_token={token_val}, origin={origin}"
    #                 #     )

    #         plm_list.append(plm_b)

    #     if not plm_list:
    #         return None
    #     return torch.stack(plm_list, dim=0)  # [B, L, num_layers, H]

    def _compute_plm_cropped_from_noised_seq(
        self,
        seq_noised: torch.Tensor,
        origin_pdb_idx: list,
        pdb_seq_full: list,
        pdb_idx_full: list,
    ) -> Optional[torch.Tensor]:
        """
        Compute PLM embeddings using *noised* design sequences on per-chain
        cropped windows:

        For each sample b and each real chain ch:
          - Find all design positions whose origin is on (ch, res).
          - Map those origins to positions in pdb_idx_full[b] and take
            start = min(pos_list), end = max(pos_list).
          - Build a window [start, end] as PLM input:
                * positions that correspond to at least one design residue
                  use the *noised* token from seq_noised[b];
                * other positions in the window use the original token
                  from pdb_seq_full[b] as structural context (if available),
                  otherwise use MASK (20).
          - Run get_plm_embeddings on all such windows in a single batch.
          - Scatter PLM embeddings back to [B, L, num_layers, H] according
            to origin_pdb_idx.
        """
        device = self.device
        B, L = seq_noised.shape[:2]

        # Collect per-window token sequences and mapping from design positions
        # to (window_index, local_position)
        window_tokens: list[torch.Tensor] = []
        window_meta: list[tuple[int, int, int]] = []  # (b, start_full, end_full)
        design_to_plm: list[dict[int, tuple[int, int]]] = [dict() for _ in range(B)]

        for b in range(B):
            origins_b = origin_pdb_idx[b]
            idx_full_b = pdb_idx_full[b]
            if idx_full_b is None or len(idx_full_b) == 0:
                continue

            # Canonicalize full-structure origins and build origin -> full_pos map
            origin_to_pos: dict[tuple, int] = {}
            for k, origin in enumerate(idx_full_b):
                if isinstance(origin, list):
                    key = tuple(origin)
                else:
                    key = origin
                if key not in origin_to_pos:
                    origin_to_pos[key] = k

            # Build: per-origin -> first design index; and per-chain list of full positions
            origin_to_design_idx: dict[tuple, int] = {}
            chain_to_poslist: dict[str, list[int]] = {}

            for i in range(L):
                origin = origins_b[i]
                if not origin or origin == ('?', '-1'):
                    continue
                if isinstance(origin, list):
                    o_key = tuple(origin)
                else:
                    o_key = origin
                full_pos = origin_to_pos.get(o_key, None)
                if full_pos is None:
                    continue
                # Record first design index for this origin (used for noised token)
                if o_key not in origin_to_design_idx:
                    origin_to_design_idx[o_key] = i
                ch = o_key[0]
                chain_to_poslist.setdefault(ch, []).append(full_pos)

            if not chain_to_poslist:
                continue

            # Convert full seq to numpy for easy indexing, if available
            full_seq_np = None
            if pdb_seq_full[b] is not None:
                sf = pdb_seq_full[b]
                if torch.is_tensor(sf):
                    full_seq_np = sf.detach().cpu().numpy().astype(np.int64)
                else:
                    full_seq_np = np.asarray(sf, dtype=np.int64)

            for ch, pos_list in chain_to_poslist.items():
                if not pos_list:
                    continue
                start = int(min(pos_list))
                end = int(max(pos_list))
                if start < 0 or end >= len(idx_full_b):
                    continue
                length_c = end - start + 1

                tokens_c = torch.full(
                    (length_c,),
                    fill_value=20,
                    dtype=torch.long,
                    device=device,
                )

                for offset in range(length_c):
                    full_pos = start + offset
                    origin_full = idx_full_b[full_pos]
                    if isinstance(origin_full, list):
                        o_key = tuple(origin_full)
                    else:
                        o_key = origin_full

                    if o_key in origin_to_design_idx:
                        i_design = origin_to_design_idx[o_key]
                        tokens_c[offset] = seq_noised[b, i_design].to(device)
                        # Record mapping from this design index to (window_idx, local_pos)
                        # window index will be known once we append tokens_c
                    else:
                        if full_seq_np is not None and 0 <= full_pos < len(full_seq_np):
                            tokens_c[offset] = int(full_seq_np[full_pos])
                        else:
                            tokens_c[offset] = 20  # MASK as fallback

                window_index = len(window_tokens)
                window_tokens.append(tokens_c)
                window_meta.append((b, start, end))

                # Now that window_index is known, fill design_to_plm mapping
                for offset in range(length_c):
                    full_pos = start + offset
                    origin_full = idx_full_b[full_pos]
                    if isinstance(origin_full, list):
                        o_key = tuple(origin_full)
                    else:
                        o_key = origin_full
                    if o_key in origin_to_design_idx:
                        i_design = origin_to_design_idx[o_key]
                        design_to_plm[b][i_design] = (window_index, offset)

        if not window_tokens:
            return None

        # Pack all windows into a single batch for PLM
        B_eff = len(window_tokens)
        max_len = max(int(t.shape[0]) for t in window_tokens)

        aatypes_batch = torch.full(
            (B_eff, max_len),
            fill_value=20,
            dtype=torch.long,
            device=device,
        )
        res_mask_batch = torch.zeros(
            B_eff, max_len, dtype=torch.bool, device=device
        )
        chain_idx_batch = torch.ones(
            B_eff, max_len, dtype=torch.long, device=device
        )
        head_mask_batch = torch.zeros(
            B_eff, max_len, dtype=torch.bool, device=device
        )
        tail_mask_batch = torch.zeros(
            B_eff, max_len, dtype=torch.bool, device=device
        )

        for w_idx, tokens_c in enumerate(window_tokens):
            Lc = int(tokens_c.shape[0])
            aatypes_batch[w_idx, :Lc] = tokens_c

            # Apply optional per-chain PLM max length limit. Only the first
            # `allowed_len` positions of this window are marked as valid in
            # res_mask; tokens beyond this will not be seen by the PLM and
            # their embeddings stay zero.
            if getattr(self, "_plm_max_len", None) is not None:
                allowed_len = min(Lc, int(self._plm_max_len))
            else:
                allowed_len = Lc

            if allowed_len > 0:
                res_mask_batch[w_idx, :allowed_len] = True
                head_mask_batch[w_idx, 0] = True
                tail_mask_batch[w_idx, allowed_len - 1] = True

        plm_eff = get_plm_embeddings(
            folding_model=self._folding_model,
            plm_type=self._plm_type,
            aatypes_t=aatypes_batch,
            B=B_eff,
            L=max_len,
            device=device,
            head_mask=head_mask_batch,
            tail_mask=tail_mask_batch,
            chain_idx=chain_idx_batch,
            res_mask=res_mask_batch,
            res_idx=None,
        )  # [B_eff, max_len, num_layers, H]

        if plm_eff is None:
            return None

        _, _, num_layers, hid_dim = plm_eff.shape
        plm_s = torch.zeros(
            B,
            L,
            num_layers,
            hid_dim,
            dtype=plm_eff.dtype,
            device=device,
        )

        # Scatter back to design positions
        for b in range(B):
            mapping_b = design_to_plm[b]
            if not mapping_b:
                continue
            for i_design, (w_idx, local_pos) in mapping_b.items():
                if (
                    0 <= w_idx < B_eff
                    and 0 <= local_pos < max_len
                    and 0 <= i_design < L
                ):
                    plm_s[b, i_design] = plm_eff[w_idx, local_pos]

        return plm_s

    def forward(
        self,
        *,
        seq_noised: torch.Tensor,
        xyz_noised: torch.Tensor,
        bond_noised: torch.Tensor,
        rf_idx: torch.Tensor,
        pdb_idx: torch.Tensor,
        alpha_target: torch.Tensor,
        alpha_tor_mask: torch.Tensor,
        partial_T: torch.Tensor,
        N_C_anchor,
        str_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        bond_mask: Optional[torch.Tensor] = None,
        res_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        tail_mask: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
        batch_mask: Optional[torch.Tensor] = None,
        trans_1: Optional[torch.Tensor] = None,
        rotmats_1: Optional[torch.Tensor] = None,
        aatypes_1: Optional[torch.Tensor] = None,
        logits_1: Optional[torch.Tensor] = None,
        bond_mat_1: Optional[torch.Tensor] = None,
        trans_sc: Optional[torch.Tensor] = None,
        aatypes_sc: Optional[torch.Tensor] = None,
        torsions_sc: Optional[torch.Tensor] = None,
        chain_ids: Optional[torch.Tensor] = None,
        origin_pdb_idx: Optional[list] = None,
        pdb_seq_full: Optional[list] = None,
        pdb_idx_full: Optional[list] = None,
        pdb_core_id: Optional[list] = None,
        hotspots: Optional[torch.Tensor] = None,
    ):
        B, L = seq_noised.shape[:2]

        trans_t = xyz_noised[:, :, 1, :].clone()
        rotmats_t = iu.get_R_from_xyz(xyz_noised)

        str_mask = str_mask if str_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
        seq_mask = seq_mask if seq_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
        res_mask = res_mask if res_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
        head_mask = head_mask if head_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
        tail_mask = tail_mask if tail_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
        hotspots = hotspots.to(self.device) if hotspots is not None else torch.zeros(B, L, dtype=torch.float32, device=self.device)

        aatypes_t = seq_noised.clone().long()
        num_tokens = getattr(self.backbone._model_conf, 'aatype_pred_num_tokens', 21)
        if aatypes_sc is None:
            aatypes_sc = torch.nn.functional.one_hot(aatypes_t, num_classes=int(num_tokens)).float()
        if trans_sc is None:
            trans_sc = torch.zeros(B, L, 3, device=self.device)
        if torsions_sc is None:
            torsions_sc = torch.zeros(B, L, 4, device=self.device)

        if chain_ids is None:
            raise ValueError("adapter.py: chain_ids is required")
        chain_idx = chain_ids.to(self.device) #get_chain_idx(pdb_idx, B, L, self.device, rf_idx)
        res_idx = rf_idx.to(self.device)
        t_expand = partial_T.reshape(B, 1).to(self.device).expand(B, L)
        ones_expand = torch.ones_like(t_expand)
        so3_t = torch.where(str_mask.bool(), t_expand, ones_expand)
        r3_t = torch.where(str_mask.bool(), t_expand, ones_expand)
        cat_t = torch.where(seq_mask.bool(), t_expand, ones_expand)
        tor_t = torch.where(str_mask.bool() & seq_mask.bool(), t_expand, ones_expand)
        t_expand_bond = partial_T.reshape(B, 1, 1).to(self.device).expand(B, L, L)
        ones_expand_bond = torch.ones_like(t_expand_bond)
        bond_t = torch.where(bond_mask.bool(), t_expand_bond, ones_expand_bond)

        if trans_1 is None:
            trans_1 = torch.zeros_like(trans_t)
        if rotmats_1 is None:
            rotmats_1 = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1)
        if aatypes_1 is None:
            aatypes_1 = torch.zeros_like(aatypes_t)
        if logits_1 is None:
            logits_1 = torch.nn.functional.one_hot(aatypes_1, num_classes=int(num_tokens)).float()

        plm_s, PLM_emb_weight = None, None

        # # 1) Backward-compatible path: externally provided full_plm_emb
        # if full_plm_emb is not None:
        #     # full_plm_emb: [B, L, D_flat] where D_flat = (num_layers) * H
        #     if self.PLM_info[0] is not None:
        #         plm_s = full_plm_emb.to(self.device).reshape(
        #             B, L, int(self.PLM_info[0]) + 1, -1
        #         )
        #     else:
        #         # Fallback: treat as a single-layer embedding
        #         plm_s = full_plm_emb.to(self.device).unsqueeze(2)

        # 2) Preferred path: compute PLM on per-chain cropped windows using
        #    the *noised* design sequence as tokens, with non-design context
        #    positions filled by the original full-chain sequence.

        
        start_time = time.time()
        if (
            origin_pdb_idx is not None
            and pdb_seq_full is not None
            and pdb_idx_full is not None
            and pdb_core_id is not None
            and self._folding_model is not None
            and self._plm_type is not None
        ):
            plm_s = self._compute_plm_cropped_from_noised_seq(
                seq_noised=seq_noised,
                origin_pdb_idx=origin_pdb_idx,
                pdb_seq_full=pdb_seq_full,
                pdb_idx_full=pdb_idx_full,
            )
        else:
            raise ValueError("adapter.py: origin_pdb_idx, pdb_seq_full, pdb_idx_full, pdb_core_id are required")
        end_time = time.time()
        print(f"Time taken to compute PLM: {end_time - start_time} seconds")

        # # 3) Fallback: compute PLM directly on the cropped/design sequence only
        # elif self.PLM_info[0] is not None and self._folding_model is not None:
        #     plm_s = get_plm_embeddings(
        #         self._folding_model,
        #         self._plm_type,
        #         aatypes_t,
        #         B,
        #         L,
        #         self.device,
        #         head_mask=head_mask,
        #         tail_mask=tail_mask,
        #         chain_idx=chain_idx,
        #         res_mask=res_mask,
        #         res_idx=res_idx,
        #     )

        PLM_emb_weight = getattr(self._folding_model, '_plm_emb_weight', None)

        # print("res_mask.shape", res_mask)
        # print("head_mask.shape", head_mask)
        # print("tail_mask.shape", tail_mask)
        # print("chain_idx.shape", chain_idx)
        # print("res_idx.shape", res_idx)
        # print("so3_t.shape", so3_t)
        # print("r3_t.shape", r3_t)
        # print("cat_t.shape", cat_t)
        # print("tor_t.shape", tor_t)
        # print("trans_t.shape", trans_t)
        input_feats = {
            'res_mask': res_mask.float(), 'diffuse_mask': str_mask.float(),
            'chain_idx': chain_idx, 'res_idx': res_idx,
            'so3_t': so3_t, 'r3_t': r3_t, 'cat_t': cat_t, 'tor_t': tor_t,
            'trans_t': trans_t, 'rotmats_t': rotmats_t, 'aatypes_t': aatypes_t,
            'trans_sc': trans_sc, 'aatypes_sc': aatypes_sc, 'torsions_sc': torsions_sc,
            'PLM_emb_weight': PLM_emb_weight,
            'trans_1': trans_1,
            'rotmats_1': rotmats_1,
            'aatypes_1': aatypes_1,
            'logits_1': logits_1,
        }
        if plm_s is not None:
            input_feats['PLM_embedding_aatypes_t'] = plm_s

        # Build pairwise time embeddings for bonds
        time_embedding = self.get_time_embedding_nd(bond_t, res_mask)

        # Build diffusion-distance-based pair embedding directly from the (noised) bond matrix.
        # This avoids pre-computing res_dist_matrix in the sampler and keeps APM-specific
        # logic local to the adapter.
        res_dist_feat = self.bond_mat_2_dist_mat(
            bond_mat=bond_noised,
            rf_idx=rf_idx,
            res_mask=res_mask,
            bond_mask=bond_mask,
            head_mask=head_mask,
            tail_mask=tail_mask,
            N_C_anchor=N_C_anchor,
        )
        edge_feat = torch.cat([res_dist_feat, time_embedding], dim=-1).to(self.device)
        edge_emb = self.edge_feats_mlp(edge_feat)
        input_feats['external_pair_embed'] = edge_emb
        #node_feature = torch.cat([head_mask.unsqueeze(-1), tail_mask.unsqueeze(-1),hotspots.unsqueeze(-1)], dim=-1).float()
        node_feature = torch.cat([head_mask.unsqueeze(-1), tail_mask.unsqueeze(-1), hotspots.unsqueeze(-1)], dim=-1).float()

        head_tail_embed = self.node_proj(node_feature)
        input_feats['external_node_embed'] = head_tail_embed

        backbone_output = self.backbone(input_feats)
        input_feats.update(backbone_output)

        # print(f"Backbone time: {end_time - start_time} seconds")
        pred_trans = input_feats['backbone_pred_trans']
        pred_rotmats = input_feats['backbone_pred_rotmats']
        pred_logits = input_feats['backbone_pred_logits']
        #pred_trans, pred_rotmats = self.head_tail_interpolate(pred_rotmats, pred_trans, head_mask, tail_mask, N_C_anchor)

        xyz_pred = iu.get_xyz_from_RT(pred_rotmats, pred_trans).unsqueeze(1)
        xyz_pred = iu.update_nc_node_coordinates(xyz_pred.squeeze(1), N_C_anchor, head_mask, tail_mask)
        xyz_pred = xyz_pred.unsqueeze(1)
        
        # Predict torsions (chi1-4) from backbone node embeddings via AngleResnet
        # Use backbone initial node embedding as s_initial if available; fallback to node embed itself
        final_node_embed = input_feats['backbone_node_embed']
        unnorm_sc, sc = self.torsion_head(final_node_embed)  # sc: [B, L, 4, 2]
        # # Map 4 chi torsions into 10-slot alpha_s layout; keep others as zeros
        # alpha_s = torch.zeros(B, L, 10, 2, device=self.device)
        # sc_cossin = torch.flip(sc, dims=[-1])  # [B, L, 4, 2], sincos -> cossin
        # alpha_s[:, :, 3:7, :] = sc_cossin
        sc_cossin = torch.flip(sc, dims=[-1])         # [B, L, 4, 2]
        zeros_prefix = torch.zeros(B, L, 3, 2, device=self.device, dtype=sc_cossin.dtype)
        zeros_suffix = torch.zeros(B, L, 3, 2, device=self.device, dtype=sc_cossin.dtype)
        alpha_s = torch.cat([zeros_prefix, sc_cossin, zeros_suffix], dim=2)  # [B, L, 10, 2]

        final_edge_embed = input_feats['backbone_edge_embed']
        res_mask_bool = (res_mask > 0).to(torch.bool)
        mask_2d = res_mask_bool.unsqueeze(1) & res_mask_bool.unsqueeze(2)
        if bond_mask is not None:
            mask_2d = mask_2d & bond_mask.bool()
        bond_matrix = self.bond_pred(
            msa=final_node_embed,
            state=final_node_embed,
            pair=final_edge_embed,
            mask_2d=mask_2d,
            mat_true=bond_mat_1, 
        )

        # Store input_feats for training access (contains backbone predictions)
        self.last_input_feats = input_feats

        return pred_logits, xyz_pred, alpha_s, bond_matrix

# class APMSidechainWrapper(BaseDesignModel):
#     """APM sidechain-only wrapper.

#     - Instantiates and runs only the SideChainModel.
#     - Uses real backbone transforms (trans/rot), sequence, and bond matrix.
#     - Outputs sidechain torsion angles [B, L, 4] (chi1-4, in radians).
#     """

#     def __init__(self, conf, device: str, d_t1d: int, d_t2d: int):
#         super().__init__()
#         self.device = device
#         self._conf = conf
#         print("loading APMSidechainWrapper")

#         _APM_FoldingModel = None
#         try:
#             from apm.apm.models.folding_model import FoldingModel as _APM_FoldingModel
#         except ImportError:
#             print("[APMSidechainWrapper] Warning: Could not import APM's FoldingModel. PLM features will be disabled.")

#         self.PLM_info = (None, None)
#         self._plm_type = None
#         self._folding_model = None
#         # Cache for full-chain PLM embeddings (shared with backbone scheme):
#         # key: (pdb_core_id, chain_id) -> Tensor[L_chain, num_layers, H] on CPU
#         self._plm_cache = {}

#         if not hasattr(conf, 'model'):
#             raise ValueError("APM model configuration not found in config")
#         packing_model_conf = getattr(conf, 'packing_model', conf.model)

#         folding_conf = getattr(conf, 'folding', None)
#         if folding_conf and _APM_FoldingModel:
#             plm_name = getattr(folding_conf, 'PLM', None)
#             if plm_name and plm_name not in ['null', 'None']:
#                 self._folding_model = _APM_FoldingModel(folding_conf)
#                 self._plm_type = plm_name
#                 self.PLM_info = (
#                     getattr(self._folding_model, 'plm_representations_layer', None),
#                     getattr(self._folding_model, 'plm_representations_dim', None),
#                 )
#         # Optional maximum chain length for PLM; if not present in config,
#         # we do not truncate by default.
#         self._plm_max_len = None
#         if folding_conf is not None and hasattr(folding_conf, 'plm_max_chain_length'):
#             try:
#                 self._plm_max_len = int(folding_conf.plm_max_chain_length)
#             except Exception:
#                 self._plm_max_len = None

#         self.sidechain = SideChainModel(packing_model_conf, self.PLM_info).to(device)

#         pair_embed_size = getattr(packing_model_conf, 'edge_embed_size', getattr(packing_model_conf.ipa, 'c_z', 128))
#         self.pair_embed_size = pair_embed_size

#         # Dimension of diffusion-distance features from smu.diffusion_distance_tensor
#         # Previously used as 6*100 in the original self.res_dist_embed head.
#         self._res_dist_dim = 6 * 100

#         d_hidden = self.pair_embed_size
#         self.res_atom_proj = nn.Linear(self._res_dist_dim, d_hidden).to(device)
#         self.res_atom_score = nn.Sequential(
#             nn.Linear(d_hidden, d_hidden),
#             nn.ReLU(),
#             nn.Linear(d_hidden, 1),
#         ).to(device)
#         self.res_atom_out = nn.Sequential(
#             nn.Linear(d_hidden, d_hidden),
#             nn.ReLU(),
#             nn.Linear(d_hidden, self.pair_embed_size),
#         ).to(device)
#         self._try_load_backbone_checkpoint()

#     def _try_load_backbone_checkpoint(self):
#         import os
#         ckpt_path = getattr(self._conf.model, 'ckpt_path', None)
#         if not ckpt_path or not os.path.exists(ckpt_path):
#             print(f"[APMSidechainWrapper] Warning: No checkpoint provided or path invalid: {ckpt_path}. Model is randomly initialized.")
#             return

#         strict_loading = getattr(self._conf.model, 'strict_loading', False)
#         ckpt_obj = torch.load(ckpt_path, map_location=self.device, weights_only=False)

#         if isinstance(ckpt_obj, dict):
#             if 'model_state_dict' in ckpt_obj:
#                 raw_sd = ckpt_obj['model_state_dict']
#             elif 'state_dict' in ckpt_obj:
#                 raw_sd = ckpt_obj['state_dict']
#             else:
#                 raw_sd = ckpt_obj
#         else:
#             raw_sd = ckpt_obj

#         load_info_all = self.load_state_dict(raw_sd, strict=False)
#         missing_all = getattr(load_info_all, 'missing_keys', [])
#         unexpected_all = getattr(load_info_all, 'unexpected_keys', [])
#         total_keys = len(raw_sd) if hasattr(raw_sd, 'keys') else 0
#         num_loaded = total_keys - len(unexpected_all)
#         print(
#             f"[APMSidechainWrapper] Loaded wrapper from {ckpt_path}: loaded~{num_loaded}/{total_keys}, "
#             f"missing={len(missing_all)}, unexpected={len(unexpected_all)}, strict={strict_loading}"
#         )

#         if num_loaded == 0 and isinstance(raw_sd, dict):
#             prefix = 'model.sidechain.'
#             extracted_sd = {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}
#             if extracted_sd:
#                 load_info = self.sidechain.load_state_dict(extracted_sd, strict=strict_loading)
#                 missing = getattr(load_info, 'missing_keys', [])
#                 unexpected = getattr(load_info, 'unexpected_keys', [])
#                 print(
#                     f"[APMSidechainWrapper] Loaded 'Sidechain' weights from {ckpt_path} with prefix '{prefix}': "
#                     f"missing={len(missing)}, unexpected={len(unexpected)}"
#                 )
#                 if missing: print(f"  [INFO] Missing keys: {missing}")
#                 if unexpected: print(f"  [INFO] Unexpected keys: {unexpected}")
#             else:
#                 # Fallback: Try loading raw state dict directly into the sidechain model
#                 print(f"[APMSidechainWrapper] No keys with prefix '{prefix}' found. Attempting to load raw state_dict directly.")
#                 load_info = self.sidechain.load_state_dict(raw_sd, strict=False)
#                 missing = getattr(load_info, 'missing_keys', [])
#                 unexpected = getattr(load_info, 'unexpected_keys', [])
#                 print(
#                     f"[APMSidechainWrapper] Loaded 'Sidechain' weights from raw state_dict: "
#                     f"missing={len(missing)}, unexpected={len(unexpected)}"
#                 )
#                 if missing: print(f"  [INFO] Missing keys: {missing}")
#                 if unexpected: print(f"  [INFO] Unexpected keys: {unexpected}")

#     def bond_mat_2_dist_mat(
#         self,
#         bond_mat: torch.Tensor,
#         rf_idx: torch.Tensor,
#         res_mask: torch.Tensor,
#         bond_mask: Optional[torch.Tensor],
#         head_mask: Optional[torch.Tensor] = None,
#         tail_mask: Optional[torch.Tensor] = None,
#         N_C_anchor: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Same atomic-graph + ScoreNet pipeline as in APMBackboneWrapper, but without
#         time-conditioning (sidechain refinement is time-independent).
#         """
#         device = bond_mat.device
#         B, L = bond_mat.shape[:2]
#         N = 3 * L

#         A_atom = torch.zeros(B, N, N, device=device, dtype=bond_mat.dtype)
#         node_mask_atom = torch.zeros(B, N, dtype=torch.bool, device=device)

#         res_mask_bool = res_mask.bool() if res_mask is not None else torch.ones(B, L, dtype=torch.bool, device=device)
#         head_mask_bool = head_mask.bool() if head_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=device)
#         tail_mask_bool = tail_mask.bool() if tail_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=device)
#         rf_idx_dev = rf_idx.to(device)
#         eye_L = torch.eye(L, device=device, dtype=torch.bool)

#         for b in range(B):
#             valid_res = res_mask_bool[b]
#             if not valid_res.any():
#                 continue

#             valid_idx = torch.nonzero(valid_res, as_tuple=False).view(-1)
#             if valid_idx.numel() == 0:
#                 continue

#             base = 3 * valid_idx.unsqueeze(1)
#             n_idx = base.squeeze(1)
#             ca_idx = n_idx + 1
#             c_idx = n_idx + 2

#             node_mask_atom[b, n_idx] = True
#             node_mask_atom[b, ca_idx] = True
#             node_mask_atom[b, c_idx] = True

#             A_atom[b, n_idx, ca_idx] = 1.0
#             A_atom[b, ca_idx, n_idx] = 1.0
#             A_atom[b, ca_idx, c_idx] = 1.0
#             A_atom[b, c_idx, ca_idx] = 1.0

#             is_sequential = (
#                 (rf_idx_dev[b, 1:] == rf_idx_dev[b, :-1] + 1)
#                 & valid_res[:-1]
#                 & valid_res[1:]
#             )
#             seq_ids = torch.nonzero(is_sequential, as_tuple=False).view(-1)
#             if seq_ids.numel() > 0:
#                 c_i = 3 * seq_ids + 2
#                 n_ip1 = 3 * (seq_ids + 1)
#                 A_atom[b, c_i, n_ip1] = 1.0
#                 A_atom[b, n_ip1, c_i] = 1.0

#             bm_mask_b = bond_mat[b] > 0.5
#             if bond_mask is not None:
#                 bm_mask_b = bm_mask_b & bond_mask[b].bool()
#             bm_mask_b = bm_mask_b & (~eye_L)

#             edges = torch.nonzero(bm_mask_b, as_tuple=False)
#             if edges.numel() > 0:
#                 i_e = edges[:, 0]
#                 j_e = edges[:, 1]
#                 head_b = head_mask_bool[b]
#                 tail_b = tail_mask_bool[b]
#                 body_b = res_mask_bool[b] & (~head_b) & (~tail_b)

#                 for idx in range(i_e.numel()):
#                     i_pos = int(i_e[idx])
#                     j_pos = int(j_e[idx])

#                     def _owner_and_type(pos: int) -> Tuple[int, int]:
#                         owner = pos
#                         a_type = 1
#                         if head_b[pos]:
#                             a_type = 0
#                             if N_C_anchor is not None:
#                                 anchor_row = N_C_anchor[b, pos, :, 0] > 0.5
#                                 anchor_row = anchor_row & body_b
#                                 if anchor_row.any():
#                                     owner = int(torch.nonzero(anchor_row, as_tuple=False)[0])
#                         elif tail_b[pos]:
#                             a_type = 2
#                             if N_C_anchor is not None:
#                                 anchor_row = N_C_anchor[b, pos, :, 1] > 0.5
#                                 anchor_row = anchor_row & body_b
#                                 if anchor_row.any():
#                                     owner = int(torch.nonzero(anchor_row, as_tuple=False)[0])
#                         return owner, a_type

#                     owner_i, type_i = _owner_and_type(i_pos)
#                     owner_j, type_j = _owner_and_type(j_pos)
#                     atom_i = 3 * owner_i + type_i
#                     atom_j = 3 * owner_j + type_j

#                     A_atom[b, atom_i, atom_j] = 1.0
#                     A_atom[b, atom_j, atom_i] = 1.0

#         if not node_mask_atom.any():
#             return torch.zeros(B, L, L, self.pair_embed_size, device=device, dtype=bond_mat.dtype)

#         # Optimized: Get raw distances (rbf_num=0)
#         res_dist_atom = smu.diffusion_distance_tensor(
#             A_adj_batch=A_atom,
#             times=self._conf.preprocess.diffusion_map_times,
#             k=self._conf.preprocess.diffusion_map_features,
#             skip_top=True,
#             node_mask=node_mask_atom.int(),
#             rbf_num=0,
#             rbf_gamma=None,
#             k_ratio=0.6,
#         )
#         B_loc, N_loc, _, T = res_dist_atom.shape
#         assert N_loc == N, f"Unexpected atomic feature shape: N={N_loc}, expected {N}"

#         # Prepare mask for patches
#         atom_mask_2d = node_mask_atom.unsqueeze(2) & node_mask_atom.unsqueeze(1)
#         mask_patches = atom_mask_2d.view(B_loc, L, 3, L, 3).permute(0, 1, 3, 2, 4).reshape(B_loc, L, L, 9)
        
#         X = res_dist_atom.view(B_loc, L, 3, L, 3, T).permute(0, 1, 3, 2, 4, 5)
#         X = X.reshape(B_loc, L, L, 9, T)

#         X_proj = compute_rbf_and_project_chunked(
#             self.res_atom_proj,
#             X,
#             mask_patches,
#             rbf_num=100,
#             rbf_gamma=None,
#             chunk_size=16
#         )
#         scores = self.res_atom_score(X_proj)
#         attn = torch.softmax(scores, dim=3)
#         feat = (attn * X_proj).sum(dim=3)
#         res_dist_embed = self.res_atom_out(feat)

#         return res_dist_embed

#     def forward(
#         self,
#         *,
#         seq_noised: torch.Tensor,
#         xyz_noised: torch.Tensor,
#         bond_noised: torch.Tensor,
#         rf_idx: torch.Tensor,
#         pdb_idx: torch.Tensor,
#         res_dist_matrix: torch.Tensor,
#         alpha_target: torch.Tensor,
#         alpha_tor_mask: torch.Tensor,
#         partial_T: torch.Tensor = None,
#         str_mask: Optional[torch.Tensor] = None,
#         seq_mask: Optional[torch.Tensor] = None,
#         bond_mask: Optional[torch.Tensor] = None,
#         res_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         tail_mask: Optional[torch.Tensor] = None,
#         use_checkpoint: bool = False,
#         batch_mask: Optional[torch.Tensor] = None,
#         trans_sc: Optional[torch.Tensor] = None,
#         aatypes_sc: Optional[torch.Tensor] = None,
#         torsions_sc: Optional[torch.Tensor] = None,
#         N_C_anchor: Optional[torch.Tensor] = None,
#         chain_ids: Optional[torch.Tensor] = None,
#         full_plm_emb: Optional[torch.Tensor] = None,
#         origin_pdb_idx: Optional[list] = None,
#         pdb_seq_full: Optional[list] = None,
#         pdb_idx_full: Optional[list] = None,
#         pdb_core_id: Optional[list] = None,
#     ):
#         B, L = seq_noised.shape[:2]

#         trans_real = xyz_noised[:, :, 1, :].clone()
#         rotmats_real = iu.get_R_from_xyz(xyz_noised[:, :, :3, :].clone())

#         str_mask = str_mask if str_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
#         seq_mask = seq_mask if seq_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
#         res_mask = res_mask if res_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
#         head_mask = head_mask if head_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)
#         tail_mask = tail_mask if tail_mask is not None else torch.ones(B, L, dtype=torch.bool, device=self.device)

#         if chain_ids is None:
#             raise ValueError("adapter.py: chain_ids is required")
#         chain_idx = chain_ids.to(self.device) #get_chain_idx(pdb_idx, B, L, self.device, rf_idx)
#         res_idx = rf_idx.to(self.device)
#         ones_t = torch.ones(B, L, device=self.device)
#         so3_t = ones_t
#         r3_t = ones_t
#         cat_t = ones_t
#         tor_t = ones_t

#         aatypes_real = seq_noised.clone().long()
#         if torsions_sc is None:
#             torsions_sc = torch.zeros(B, L, 4, device=self.device)

#         # Build diffusion-distance-based pair embedding from the (true) bond matrix.
#         res_dist_embed = self.bond_mat_2_dist_mat(
#             bond_mat=bond_noised,
#             rf_idx=rf_idx,
#             res_mask=res_mask,
#             bond_mask=bond_mask,
#             head_mask=head_mask,
#             tail_mask=tail_mask,
#             N_C_anchor=N_C_anchor,
#         )

#         plm_s, PLM_emb_weight = None, None
#         # 1) Backward-compatible path: externally provided full_plm_emb
#         if full_plm_emb is not None:
#             if self.PLM_info[0] is not None:
#                 plm_s = full_plm_emb.to(self.device).reshape(
#                     B, L, int(self.PLM_info[0]) + 1, -1
#                 )
#             else:
#                 plm_s = full_plm_emb.to(self.device).unsqueeze(2)
#         # 2) Preferred path: compute PLM from full-chain sequences using original PDB mapping
#         elif (
#             origin_pdb_idx is not None
#             and pdb_seq_full is not None
#             and pdb_idx_full is not None
#             and pdb_core_id is not None
#             and self._folding_model is not None
#             and self._plm_type is not None
#         ):
#             plm_list: list[torch.Tensor] = []
#             num_layers_default = None
#             hid_dim_default = None
#             if self.PLM_info[0] is not None and self.PLM_info[1] is not None:
#                 num_layers_default = int(self.PLM_info[0]) + 1
#                 hid_dim_default = int(self.PLM_info[1])

#             for b in range(B):
#                 # Reuse the same helper as backbone wrapper via get_plm_embeddings.
#                 # We call the free function directly on full-chain sequences.
#                 if pdb_seq_full[b] is None or pdb_idx_full[b] is None:
#                     plm_b = None
#                 else:
#                     # Convert full-chain seq to numpy
#                     seq_full_b = pdb_seq_full[b]
#                     if torch.is_tensor(seq_full_b):
#                         seq_np = seq_full_b.detach().cpu().numpy().astype(np.int64)
#                     else:
#                         seq_np = np.asarray(seq_full_b, dtype=np.int64)

#                     # Group by chain and restrict to chains used in this design
#                     origins_b = origin_pdb_idx[b]
#                     used_chains = sorted(
#                         {ch for (ch, res) in origins_b if isinstance(ch, str) and ch != '?'}
#                     )
#                     idx_full_b = pdb_idx_full[b]
#                     if len(seq_np) != len(idx_full_b) or not used_chains:
#                         plm_b = None
#                     else:
#                         # For sidechain we only need embeddings at design positions, so we
#                         # mimic the backbone helper logic in a lightweight manner.
#                         chain_to_global = {}
#                         for i_pos, (ch, res) in enumerate(idx_full_b):
#                             if ch in used_chains:
#                                 chain_to_global.setdefault(ch, []).append(i_pos)

#                         chain_to_plm = {}
#                         origin_to_local = {}
#                         num_layers = None
#                         hid_dim = None

#                         for ch in used_chains:
#                             gpos = chain_to_global.get(ch, [])
#                             if not gpos:
#                                 continue
#                             # Respect optional PLM max length per chain
#                             if self._plm_max_len is not None and len(gpos) > self._plm_max_len:
#                                 gpos = gpos[: self._plm_max_len]
#                             chain_seq_np = seq_np[gpos]
#                             # Use a 1-chain call to get_plm_embeddings
#                             aatypes_t_full = torch.from_numpy(chain_seq_np.astype(np.int64))[None, :].to(self.device)
#                             Bb, Ll = aatypes_t_full.shape
#                             res_mask_full = torch.ones(Bb, Ll, dtype=torch.bool, device=self.device)
#                             chain_idx_full = torch.ones(Bb, Ll, dtype=torch.long, device=self.device)
#                             head_mask_full = torch.zeros(Bb, Ll, dtype=torch.bool, device=self.device)
#                             tail_mask_full = torch.zeros(Bb, Ll, dtype=torch.bool, device=self.device)
#                             head_mask_full[:, 0] = True
#                             tail_mask_full[:, Ll - 1] = True
#                             plm_full = get_plm_embeddings(
#                                 folding_model=self._folding_model,
#                                 plm_type=self._plm_type,
#                                 aatypes_t=aatypes_t_full,
#                                 B=Bb,
#                                 L=Ll,
#                                 device=self.device,
#                                 head_mask=head_mask_full,
#                                 tail_mask=tail_mask_full,
#                                 chain_idx=chain_idx_full,
#                                 res_mask=res_mask_full,
#                                 res_idx=None,
#                             )  # [1, Ll, num_layers, H]
#                             plm_chain = plm_full[0].detach()  # [Ll, num_layers, H]

#                             if num_layers is None:
#                                 num_layers = plm_chain.shape[1]
#                                 hid_dim = plm_chain.shape[2]
#                             chain_to_plm[ch] = plm_chain

#                             for local_idx, global_idx in enumerate(gpos):
#                                 origin = idx_full_b[global_idx]
#                                 origin_to_local[origin] = (ch, local_idx)

#                         if not chain_to_plm or num_layers is None or hid_dim is None:
#                             plm_b = None
#                         else:
#                             plm_b = torch.zeros(
#                                 L,
#                                 num_layers,
#                                 hid_dim,
#                                 device=self.device,
#                                 dtype=next(iter(chain_to_plm.values())).dtype,
#                             )
#                             for i_pos, origin in enumerate(origins_b):
#                                 if not origin or origin == ('?', '-1'):
#                                     continue
#                                 entry = origin_to_local.get(origin, None)
#                                 if entry is None:
#                                     continue
#                                 ch, local_idx = entry
#                                 plm_chain = chain_to_plm.get(ch, None)
#                                 if plm_chain is None:
#                                     continue
#                                 if 0 <= local_idx < plm_chain.shape[0]:
#                                     plm_b[i_pos] = plm_chain[local_idx]

#                 if plm_b is None:
#                     if num_layers_default is None or hid_dim_default is None:
#                         continue
#                     plm_b = torch.zeros(
#                         L,
#                         num_layers_default,
#                         hid_dim_default,
#                         device=self.device,
#                         dtype=torch.float32,
#                     )
#                 else:
#                     if num_layers_default is None or hid_dim_default is None:
#                         num_layers_default = plm_b.shape[1]
#                         hid_dim_default = plm_b.shape[2]
#                 plm_list.append(plm_b)

#             if plm_list:
#                 plm_s = torch.stack(plm_list, dim=0)  # [B, L, num_layers, H]

#         # 3) Fallback: compute PLM on cropped/design sequence only
#         elif self.PLM_info[0] is not None and self._folding_model is not None:
#             plm_s = get_plm_embeddings(
#                 self._folding_model,
#                 self._plm_type,
#                 aatypes_real,
#                 B,
#                 L,
#                 self.device,
#                 head_mask=head_mask,
#                 tail_mask=tail_mask,
#                 chain_idx=chain_idx,
#                 res_mask=res_mask,
#                 res_idx=res_idx,
#             )

#         PLM_emb_weight = getattr(self._folding_model, '_plm_emb_weight', None)

#         input_feats = {
#             'res_mask': res_mask.float(), 'diffuse_mask': str_mask.float(),
#             'chain_idx': chain_idx, 'res_idx': res_idx,
#             'so3_t': so3_t, 'r3_t': r3_t, 'cat_t': cat_t, 'tor_t': tor_t,
#             'trans_1': trans_real, 'rotmats_1': rotmats_real, 'aatypes_1': aatypes_real,
#             'backbone_pred_trans': trans_real.detach(),
#             'backbone_pred_rotmats': rotmats_real.detach(),
#             'backbone_pred_aatypes': aatypes_real.detach(),
#             'torsions_sc': torsions_sc,
#             'PLM_emb_weight': PLM_emb_weight,
#             'external_pair_embed': res_dist_embed,
#         }
#         if plm_s is not None:
#             input_feats['PLM_embedding_aatypes_t'] = plm_s

#         sc_out = self.sidechain(input_feats)
#         torsion_angles = sc_out['sidechain_pred_torsions_sincos']

#         sc_cossin = torch.flip(torsion_angles, dims=[-1])         # [B, L, 4, 2]
#         zeros_prefix = torch.zeros(B, L, 3, 2, device=self.device, dtype=sc_cossin.dtype)
#         zeros_suffix = torch.zeros(B, L, 3, 2, device=self.device, dtype=sc_cossin.dtype)
#         alpha_s = torch.cat([zeros_prefix, sc_cossin, zeros_suffix], dim=2)  # [B, L, 10, 2]

#         return alpha_s 



# def get_chain_idx(pdb_idx, B, L, device, rf_idx):
#     """
#     Generates a chain index tensor for APM models from either pdb_idx or rf_idx.
#     APM expects integer chain labels starting from 1.
#     """
#     if isinstance(pdb_idx, (list, tuple)) and len(pdb_idx) > 0 and pdb_idx[0] is not None:
#         chain_idx = torch.zeros(B, L, dtype=torch.long, device=device)
#         is_batched_list = isinstance(pdb_idx[0], (list, tuple)) and len(pdb_idx) == B
        
#         if is_batched_list:
#             # pdb_idx is a list of lists, one for each item in the batch
#             for b in range(B):
#                 # Handle cases where pdb_idx might be shorter than L for a given batch item
#                 len_pdb = len(pdb_idx[b])
#                 if len_pdb == 0: continue

#                 chain_letters = [res[0] for res in pdb_idx[b]]
                
#                 # Find unique chains in order of appearance
#                 unique_chains = []
#                 for chain in chain_letters:
#                     if chain not in unique_chains:
#                         unique_chains.append(chain)
                
#                 chain_map = {chain: i + 1 for i, chain in enumerate(unique_chains)}
                
#                 for i in range(min(L, len_pdb)):
#                     chain_idx[b, i] = chain_map.get(pdb_idx[b][i][0], 0) # Default to 0 if chain not in map
#         else:
#             # pdb_idx is a single list, assumed to be the same for all items in the batch
#             len_pdb = len(pdb_idx)
#             if len_pdb > 0:
#                 chain_letters = [res[0] for res in pdb_idx]
#                 unique_chains = []
#                 for chain in chain_letters:
#                     if chain not in unique_chains:
#                         unique_chains.append(chain)
                
#                 chain_map = {chain: i + 1 for i, chain in enumerate(unique_chains)}
                
#                 for i in range(min(L, len_pdb)):
#                     chain_idx[:, i] = chain_map.get(pdb_idx[i][0], 0)
#     else:
#         # Fallback for unconditional generation: detect new chains by large rf_idx jumps
#         rf_idx_dev = rf_idx.to(device)
#         rf_diff = rf_idx_dev[:, 1:] - rf_idx_dev[:, :-1]
#         chain_breaks = torch.zeros(B, L, dtype=torch.bool, device=device)
#         chain_breaks[:, 1:] = rf_diff > 50  # APM uses large offsets like 200 between chains
#         chain_idx = chain_breaks.long().cumsum(dim=1) + 1  # 1-based chain ids
        
#     return chain_idx

def get_plm_embeddings(
    folding_model,
    plm_type,
    aatypes_t,
    B,
    L,
    device,
    head_mask: Optional[torch.Tensor] = None,
    tail_mask: Optional[torch.Tensor] = None,
    chain_idx: Optional[torch.Tensor] = None,
    res_mask: Optional[torch.Tensor] = None,
    res_idx: Optional[torch.Tensor] = None,
):
    """Computes PLM embeddings using the provided folding model.

    Behavior:
    - faESM-family (faESM2/faESMC): Insert BOS/EOS per-chain (or per contiguous segment)
      only at valid (res_mask=True) ranges, pack segments with cu_seqlens, encode,
      then strip BOS/EOS and scatter back to [B, L, ...].
    - gLM-family: Keep [B, L] shape, set BOS only at valid head positions, set
      invalid tokens (~res_mask) to MASK, and zero-out outputs at invalid positions.
    """
    plm_s = None
    folding_model.eval()
    # Default masks
    if res_mask is None:
        res_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    else:
        res_mask = res_mask.to(torch.bool).to(device)

    # Optional sorting by residue index to restore natural order for PLM, then scatter back
    if res_idx is not None:
        res_idx = res_idx.to(device)
        head_mask_local = head_mask.to(device) if head_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=device)
        tail_mask_local = tail_mask.to(device) if tail_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=device)
        body_mask = (~head_mask_local) & (~tail_mask_local) & res_mask

        # Build a per-batch permutation: only body positions are reordered by res_idx; head/tail stay in place.
        sort_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1).clone()
        for b in range(B):
            body_pos = torch.nonzero(body_mask[b], as_tuple=False).squeeze(-1)
            if body_pos.numel() == 0:
                continue
            body_res = res_idx[b, body_pos]
            order = torch.argsort(body_res)
            sorted_body = body_pos[order]
            # Place sorted indices back into the original body slots
            sort_idx[b, body_pos] = sorted_body

        inv_idx = torch.argsort(sort_idx, dim=1)

        def _sort_tensor(t: Optional[torch.Tensor]):
            if t is None:
                return None
            expand_shape = sort_idx.shape + (1,) * (t.dim() - sort_idx.dim())
            sort_idx_exp = sort_idx.view(expand_shape)
            return torch.gather(t.to(device), 1, sort_idx_exp.expand_as(t))

        aatypes_t = _sort_tensor(aatypes_t)
        res_mask = _sort_tensor(res_mask)
        chain_idx = _sort_tensor(chain_idx)
    else:
        inv_idx = None

    # Derive per-position head/tail masks if not provided
    if chain_idx is not None and (head_mask is None or tail_mask is None):
        # chain_idx: [B, L] with 1-based chain labels
        ci = chain_idx.to(device)
        # Head: start of sequence or chain label change from previous
        ci_pad_left = torch.zeros(B, 1, dtype=ci.dtype, device=device)
        prev_ci = torch.cat([ci_pad_left, ci[:, :-1]], dim=1)
        auto_head = (ci != prev_ci)  # [B, L]
        # Tail: end of sequence or chain label change to next
        ci_pad_right = torch.zeros(B, 1, dtype=ci.dtype, device=device)
        next_ci = torch.cat([ci[:, 1:], ci_pad_right], dim=1)
        auto_tail = (ci != next_ci)
        if head_mask is None:
            head_mask = auto_head
        if tail_mask is None:
            tail_mask = auto_tail

    head_mask = head_mask.to(device) if head_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=device)
    tail_mask = tail_mask.to(device) if tail_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=device)

    # Apply res_mask to head/tail (heads/tails must be valid residues)
    head_mask = head_mask & res_mask
    tail_mask = tail_mask & res_mask

    # faESM-family: pack segments with inserted specials, exclude invalid positions
    if isinstance(plm_type, str) and (plm_type.startswith('faESM2') or plm_type.startswith('faESMC')):
        BOS_TOKEN = 21
        EOS_TOKEN = 22

        # Prefer explicit head/tail masks; fallback to chain-based segmentation
        use_head_tail = head_mask.any() and tail_mask.any()

        packed_tokens_list = []
        seg_valid_pos_list = []  # list of tensors of absolute positions per segment
        cu_list = [torch.tensor([0], device=device, dtype=torch.int32)]
        max_seqlen = 0
        total_len = 0

        for b in range(B):
            if use_head_tail:
                heads = torch.nonzero(head_mask[b], as_tuple=False).squeeze(-1)
                tails = torch.nonzero(tail_mask[b], as_tuple=False).squeeze(-1)
                hi = 0
                ti = 0
                while hi < int(heads.numel()) and ti < int(tails.numel()):
                    h = int(heads[hi].item())
                    # advance tail until >= h
                    while ti < int(tails.numel()) and int(tails[ti].item()) < h:
                        ti += 1
                    if ti >= int(tails.numel()):
                        break
                    t = int(tails[ti].item())
                    if t < h:
                        break
                    # valid positions within [h, t]
                    valid_local = torch.nonzero(res_mask[b, h:t+1], as_tuple=False).squeeze(-1)
                    n_valid = int(valid_local.numel())
                    if n_valid > 0:
                        valid_abs = valid_local + h
                        seg_valid_pos_list.append((b, valid_abs))
                        seg_tokens = aatypes_t[b, valid_abs].to(device).clone()
                        # replace: put BOS at first, EOS at last if length>1
                        if n_valid >= 1:
                            seg_tokens[0] = BOS_TOKEN
                        if n_valid >= 2:
                            seg_tokens[-1] = EOS_TOKEN
                        packed_tokens_list.append(seg_tokens)
                        total_len += seg_tokens.numel()
                        cu_list.append(torch.tensor([total_len], device=device, dtype=torch.int32))
                        if seg_tokens.numel() > max_seqlen:
                            max_seqlen = int(seg_tokens.numel())
                    hi += 1
                    ti += 1
            else:
                # Fallback: contiguous runs by chain label (or single run if chain_idx None)
                if chain_idx is not None:
                    ci = chain_idx.to(device)
                else:
                    ci = torch.ones(B, L, dtype=torch.long, device=device)
                runs = []
                start = 0
                for i in range(1, L):
                    if ci[b, i] != ci[b, i - 1]:
                        runs.append((start, i))
                        start = i
                runs.append((start, L))

                for (s, e) in runs:
                    valid_local = torch.nonzero(res_mask[b, s:e], as_tuple=False).squeeze(-1)
                    n_valid = int(valid_local.numel())
                    if n_valid == 0:
                        continue
                    valid_abs = valid_local + s
                    seg_valid_pos_list.append((b, valid_abs))
                    seg_tokens = aatypes_t[b, valid_abs].to(device).clone()
                    if n_valid >= 1:
                        seg_tokens[0] = BOS_TOKEN
                    if n_valid >= 2:
                        seg_tokens[-1] = EOS_TOKEN

                    packed_tokens_list.append(seg_tokens)
                    total_len += seg_tokens.numel()
                    cu_list.append(torch.tensor([total_len], device=device, dtype=torch.int32))
                    if seg_tokens.numel() > max_seqlen:
                        max_seqlen = int(seg_tokens.numel())

        
        if total_len == 0:
            # No valid residues at all
            plm_s = torch.zeros(B, L, getattr(folding_model, 'plm_representations_layer', 0)+1, getattr(folding_model, 'plm_representations_dim', 1), device=aatypes_t.device)
        else:
            # INSERT_YOUR_CODE
            # 统计packed_tokens_list中 大于22的元素个数
            num_greater_than_22 = sum(int((seg_tokens > 22).sum().item()) for seg_tokens in packed_tokens_list)
            if num_greater_than_22 > 0:
                warnings.warn(f"[WARNING] Number of elements in packed_tokens_list > 22: {num_greater_than_22}")

            packed_tokens = torch.cat(packed_tokens_list, dim=0).unsqueeze(0).long()  # [1, total_len]
            cu_seqlens = torch.cat(cu_list, dim=0)
            
            if plm_type.startswith('faESM2'):
                enc = folding_model.faESM2_encoding(packed_tokens, cu_seqlens, max_seqlen)
            else:
                enc = folding_model.faESMC_encoding(packed_tokens, cu_seqlens, max_seqlen)
            # enc: [total_len, num_layer, H]

            num_layers = enc.shape[-2]
            hid_dim = enc.shape[-1]
            plm_out = torch.zeros(B, L, num_layers, hid_dim, device=aatypes_t.device, dtype=enc.dtype)

            offset = 0
            seg_idx = 0
            for seg_tokens in packed_tokens_list:
                seg_len = int(seg_tokens.numel())
                b, valid_abs = seg_valid_pos_list[seg_idx]
                # Direct 1:1 scatter (no insertion), BOS/EOS embeddings already aligned
                plm_out[b, valid_abs, :, :] = enc[offset: offset + seg_len]
                offset += seg_len
                seg_idx += 1

            plm_s = plm_out
    elif isinstance(plm_type, str) and plm_type.startswith('gLM'):
        # gLM style: BOS only at head positions, keep [B, L]
        tokens = aatypes_t.clone().to(device)
        # Set BOS only at valid heads
        if head_mask.any():
            tokens[head_mask] = 21
        # Mask out invalid residues
        tokens[~res_mask] = 20  # MASK token in openfold space -> maps to <mask>
        enc = folding_model.gLM_encoding(tokens.long())  # [B, L, num_layer, H]
        # Zero-out invalid positions explicitly
        plm_s = enc * res_mask[..., None, None].to(enc.dtype)
    else:
        raise NotImplementedError(f"Unsupported PLM type: {plm_type}")

    # Scatter back to original (possibly permuted) order if sorted earlier
    if inv_idx is not None:
        gather_idx = inv_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, plm_s.shape[-2], plm_s.shape[-1])
        plm_s = torch.gather(plm_s, 1, gather_idx)

    plm_s = plm_s.to(torch.float32).to(aatypes_t.device).detach()


    return plm_s


class ChainPLMEncoder:
    """
    Top-level callable wrapper around PLM encoding so that it can be pickled
    when used with `torch.multiprocessing.spawn` and DataLoader workers.

    The call signature matches the previous inner `plm_encoder` closure:

        encoder(chain_seq_np: np.ndarray[int], chain_id: str, max_len: int|None)
            -> np.ndarray[float32]
    """

    def __init__(self, folding_model, plm_type: str, device: str):
        self.folding_model = folding_model.to(device)
        self.plm_type = plm_type
        self.device = device

    def __call__(self, chain_seq_np: np.ndarray, chain_id: str, max_len = 1024) -> np.ndarray:
        """
        Encode a single chain sequence (integer-coded) into per-residue embeddings.

        Args:
            chain_seq_np: np.ndarray[int] of shape [L_chain]
            chain_id: chain identifier (unused here, kept for API symmetry)
            max_len: optional maximum length; sequences longer than this will be truncated.
        """
        if chain_seq_np.size == 0:
            return np.zeros(
                (0, getattr(self.folding_model, 'plm_representations_dim', 1)),
                dtype=np.float32,
            )

        device = self.device
        folding_model = self.folding_model
        plm_type = self.plm_type

        seq_tensor = torch.from_numpy(chain_seq_np.astype(np.int64))[None, :]  # [1, L]
        if max_len is not None and seq_tensor.shape[1] > max_len:
            seq_tensor = seq_tensor[:, :max_len]

        seq_tensor = seq_tensor.to(device)
        B = seq_tensor.shape[0]
        seq_tensor = torch.cat(
            [
                torch.ones(B, 1, dtype=torch.long, device=device) * 21,
                seq_tensor,
                torch.ones(B, 1, dtype=torch.long, device=device) * 21,
            ],
            dim=1,
        )
        B, L = seq_tensor.shape

        # Simple masks: single valid chain, all residues valid
        res_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        chain_idx = torch.ones(B, L, dtype=torch.long, device=device)  # single chain id = 1
        head_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        tail_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        head_mask[:, 0] = True
        tail_mask[:, L - 1] = True

        with torch.no_grad():
            plm_s = get_plm_embeddings(
                folding_model=folding_model,
                plm_type=plm_type,
                aatypes_t=seq_tensor,
                B=B,
                L=L,
                device=device,
                head_mask=head_mask,
                tail_mask=tail_mask,
                chain_idx=chain_idx,
                res_mask=res_mask,
                res_idx=None,
            )

        # plm_s: [B, L, num_layers, H]; use the last layer as embedding
        if plm_s.ndim != 4:
            raise ValueError(
                f"Unexpected PLM embedding shape {plm_s.shape}, expected [B, L, num_layers, H]."
            )
        emb = (
            plm_s[0, 1:-1, :, :]
            .reshape(L - 2, -1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return emb


def build_plm_encoder(model, plm_type, device: str):
    """
    Build a simple per-chain PLM encoder callable from the model config.

    This is intended to be passed into `process_target` via dataloader, so that
    ESM2 / gLM embeddings are computed once per *full chain* and then later
    sliced by cropping logic (e.g., complex_space) instead of feeding broken
    fragments to the PLM.

    Returns:
        encoder(chain_seq_np: np.ndarray[int], chain_id: str, max_len: int|None)
            -> np.ndarray[float32]
        or None if no PLM is configured.
    """
    folding_model = model
    return ChainPLMEncoder(folding_model=folding_model, plm_type=plm_type, device=device)


def build_design_model(model_type, device: str, d_t1d: int, d_t2d: int) -> BaseDesignModel:
    """Factory to build a design model based on configuration.

    Args:
        model_type: 'rosettafold' | 'apm'
        device: torch device string
        d_t1d: t1d feature dimension (from preprocess)
        d_t2d: t2d feature dimension (from preprocess)
    """
    import os
    from omegaconf import OmegaConf
    
    # Get the directory where this module is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(os.path.dirname(current_dir), 'config')
    
    # Load base configuration
    base_config_path = os.path.join(config_dir, 'base.yaml')
    base_conf = OmegaConf.load(base_config_path)
    
    # Load model-specific configuration and merge with base
    if model_type == 'apm_backbone':
        model_config_path = os.path.join(config_dir, 'apm.yaml')
        model_conf = OmegaConf.load(model_config_path)
        conf = OmegaConf.merge(base_conf, model_conf)
        conf.model.type = 'apm_backbone'
        print("loading APMBackboneWrapper")
        return APMBackboneWrapper(conf, device=device, d_t1d=d_t1d, d_t2d=d_t2d)
    elif model_type == 'apm':
        model_config_path = os.path.join(config_dir, 'apm.yaml')
        model_conf = OmegaConf.load(model_config_path)
        # Merge model-specific config with base config
        conf = OmegaConf.merge(base_conf, model_conf)
        # Override model type to ensure consistency
        conf.model.type = 'apm'
        print("loading APMWrapper")
        return APMWrapper(conf, device=device, d_t1d=d_t1d, d_t2d=d_t2d)
    elif model_type == 'apm_sidechain':
        model_config_path = os.path.join(config_dir, 'apm_sc.yaml')
        model_conf = OmegaConf.load(model_config_path)
        conf = OmegaConf.merge(base_conf, model_conf)
        conf.model.type = 'apm_sidechain'
        print("loading APMSidechainWrapper")
        return APMSidechainWrapper(conf, device=device, d_t1d=d_t1d, d_t2d=d_t2d)
    elif model_type == 'rosettafold':
        model_config_path = os.path.join(config_dir, 'RosettaModel.yaml')
        model_conf = OmegaConf.load(model_config_path)
        # Merge model-specific config with base config
        conf = OmegaConf.merge(base_conf, model_conf)
        # Override model type to ensure consistency
        conf.model.type = 'rosettafold'
        return RoseTTAFoldWrapper(conf, device=device, d_t1d=d_t1d, d_t2d=d_t2d)
    else:
        raise ValueError(f"Invalid model type: {model_type}. Supported types: 'apm', 'apm_backbone', 'apm_sidechain', 'rosettafold'")


