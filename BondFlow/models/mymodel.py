
import torch
from omegaconf import DictConfig, OmegaConf
from rfdiff.kinematics import get_init_xyz, xyz_to_t2d
from rfdiff.chemical import seq2chars, aa2num, num2aa, aa2long
from rfdiff.util_module import ComputeAllAtomCoords
import BondFlow.data.utils as iu
import BondFlow.data.SM_utlis as smu
from BondFlow.models.adapter import build_design_model
import logging
from rfdiff import util
from hydra.core.hydra_config import HydraConfig

from rfdiff.model_input_logger import pickle_function_call

import os, time, pickle
import logging
from rfdiff.util import writepdb_multi, writepdb
from BondFlow.data.link_utils import get_valid_links
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
from BondFlow.data.link_utils import _get_bond_info
import math

from BondFlow.models.interpolant import Interpolant
from BondFlow.models.interpolant import _centered_gaussian as interpolant_centered_gaussian
from BondFlow.models.interpolant import _uniform_so3 as interpolant_uniform_so3
from multiflow_data import utils as du
from BondFlow.models.layers import TimeEmbedding
from BondFlow.models.allatom_wrapper import AllAtomWrapper
from BondFlow.models.guidance import GuidanceManager, build_guidances
from BondFlow.models.Loss import BondCoherenceLoss, OpenFoldClashLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
SCRIPT_DIR=os.getcwd()

# For optional visualization (e.g., in notebooks). We deliberately do **not**
# force a specific backend here so that the environment (Jupyter, IDE, etc.)
# can choose an appropriate interactive backend.
try:
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    plt = None
    np = None

TOR_INDICES  = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES   = util.reference_angles

class MySampler:

    def __init__(self, conf: DictConfig,device='cpu') -> None:
        """
        Initialize sampler.
        Args:
            conf: Configuration.
        """
        self.initialize(conf,device=device)
        
    def initialize(self, conf: DictConfig,device='cpu') -> None:
        """
        Initialize sampler.
        Args:
            conf: Configuration
        
        - Selects appropriate model from input
        - Assembles Config from model checkpoint and command line overrides

        """
        self._log = logging.getLogger(__name__)
        self.device = device
        needs_model_reload = False

        self._conf = conf
        self.d_t1d=self._conf.preprocess.d_t1d
        self.d_t2d=self._conf.preprocess.d_t2d
        self.d_time=self._conf.preprocess.d_time
        self.sigma_perturb = self._conf.preprocess.sigma_perturb
        self.time_embedding = TimeEmbedding(d_embed=self.d_time).to(self.device)
        ################################
        ### Select Appropriate Model ###
        ################################



        #######################
        ### Assemble Config ###
        #######################
        # if self.ckpt_path is not None:
        #     self.ckpt = self.load_checkpoint()
        # else:
        #     self.ckpt = None
        model_type = getattr(self._conf.model, 'type', 'rosettafold')
        self.model = self.load_model(model_type)
        if self._conf.model.sidechain_model_type is not None:
            model_type = self._conf.model.sidechain_model_type
            self.sidechain_model = self.load_model(model_type)



        # Initialize helper objects
        self.inf_conf = self._conf.inference
        self.design_config = self._conf.design_config
        # self.denoiser_conf = self._conf.denoiser
        # self.ppi_conf = self._conf.ppi
        # self.potential_conf = self._conf.potentials
        # self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess
        # Initialize Interpolant
        self.interpolant = Interpolant(self._conf.interpolant,device = self.device)

        # if conf.inference.schedule_directory_path is not None:
        #     schedule_directory = conf.inference.schedule_directory_path
        # else:
        #     #schedule_directory = f"{SCRIPT_DIR}/../../schedules"
        #     schedule_directory = "./cache"

        # # Check for cache schedule
        # if not os.path.exists(schedule_directory):
        #     os.mkdir(schedule_directory)



        # self.diffuser = Diffuser(**self._conf.diffuser, device=self.device,cache_dir=schedule_directory)



        backend = getattr(self._conf.preprocess, 'allatom_backend', 'rfdiff')
        self.allatom = AllAtomWrapper(backend=backend, device=self.device).to(self.device)


        # Guidance (optional)
        try:
            guidance_cfg = getattr(self._conf, 'guidance', None)
        except Exception:
            guidance_cfg = None
        self.guidance_manager = GuidanceManager(build_guidances(guidance_cfg, device=self.device))

        
        
        
            # set default pdb
        #     script_dir = os.getcwd()
        #     self.inf_conf.input_pdb=os.path.join(script_dir, '../../examples/input_pdbs/1qys.pdb')
        # self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=True, center=False)
            
        # self.chain_idx = None

        ##############################
        ### Handle Partial Noising ###
        ##############################

        # if self.diffuser_conf.partial_T:
        #     assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
        #     self.t_step_input = int(self.diffuser_conf.partial_T)
        # else:
        #     self.t_step_input = int(self.diffuser_conf.T)
   
    def load_model(self,model_type):
        """Create design model from config via adapter factory (RF or APM)."""

        # Lazy import to avoid circular imports and keep coupling low

        # Get model type from configuration
        
        model = build_design_model(model_type, device=self.device, d_t1d=self.d_t1d, d_t2d=self.d_t2d)
        if self._conf.logging.inputs:
            pickle_dir = pickle_function_call(model, 'forward', 'inference')
            print(f'pickle_dir: {pickle_dir}')
        model = model.eval()

        return model

    def generate_crop_target_pdb(self, pdb_file,chain_id,crop_mode,crop_length=256,fixed_res=None):
        pdb_parsed = iu.process_target(pdb_file,
                                    parse_hetatom=False, 
                                    center=False,
                                    parse_link=True)
        
        contig, res_mask = iu.generate_crop_contigs(pdb_parsed, chain_id, mode=crop_mode, crop_length=crop_length, fixed_res=fixed_res)
        print(contig)
        contig_new = self._conf
        if crop_mode == 'complex':
            contig_new.design_config.bond_condition = ['B|B']
        else:
            contig_new.design_config.bond_condition = None
        contig_new.design_config.contigs = contig
        contig_new.design_config.partial_t = 0.1 # no use
        target = iu.Target(contig_new.design_config,pdb_parsed)
        target.res_mask = res_mask
        return target, contig_new

    def _center_motif(self, xyz_target, str_mask, res_mask, pdb_idx):
        """
        根据提供的掩码将 motif 居中。str_mask是scalfold
        在结合剂设计任务中，靶点蛋白链是固定的，不参与中心化。
        此函数支持批处理。
        """
        B, L, _, _ = xyz_target.shape
        
        # target_chain_indices 的形状为 (B, L)
        target_chain_indices = torch.zeros((B, L), dtype=torch.bool, device=xyz_target.device)

        for batch in range(B):
            # 识别靶点蛋白链
            chain_ids = sorted(list(set([res[0] for res in pdb_idx[batch]])))
            for chain_id in chain_ids:
                chain_indices = [i for i, res in enumerate(pdb_idx[batch]) if res[0] == chain_id]

                # 检查批次中每个样本的链是否为靶点
                # is_target_chain 的形状为 (B,)
                is_target_chain = ~str_mask[batch, chain_indices].any()
                # 将靶点链的索引设置为 True
                target_chain_indices[batch, chain_indices] = is_target_chain
        
        # 用于中心化的掩码应排除靶点蛋白
        # Motif 是那些非靶点蛋白且在 res_mask 中的部分
        # motif_mask 的形状为 (B, L)
        motif_mask = res_mask.float() * (~target_chain_indices).float() * (~str_mask).float()

        if motif_mask.sum() == 0:
            # 如果没有 motif，计算整体中心坐标
            motif_center = (xyz_target[:,:,1,:].nan_to_num() * res_mask.unsqueeze(-1).float()).sum(dim=1, keepdim=True)
            motif_center  = motif_center / (res_mask.sum(dim=-1, keepdim=True).unsqueeze(-1) + 1e-8)
        else:
            # motif_coords 的形状为 (B, 1, 3)
            motif_center = (xyz_target[:,:,1,:].nan_to_num() * motif_mask.unsqueeze(-1).float()).sum(dim=1, keepdim=True)
            motif_center  = motif_center / (motif_mask.sum(dim=-1, keepdim=True).unsqueeze(-1) + 1e-8)
        
        xyz_centered = xyz_target - motif_center.unsqueeze(2)
        # print("center of xyz__centered:",xyz_centered[:,:,1,:].mean(dim=(1)))
        epsilon = torch.randn(B, device=xyz_target.device) * self.sigma_perturb
        epsilon = epsilon[:,None,None,None]  # (B, 1, 1, 1)
        
        # 将扰动应用到中心化后的坐标上
        xyz_stochastically_centered = xyz_centered + epsilon
        # print("center of xyz_stochastically_centered:",xyz_stochastically_centered[:,:,1,:].mean(dim=(1)))
        return xyz_stochastically_centered
    
    def _center_global(self, xyz_target, str_mask, res_mask, pdb_idx):
        """
        根据提供的掩码将 全局 居中。str_mask是scalfold
        此函数支持批处理。
        """
        B, L, _, _ = xyz_target.shape
        
        
        motif_center = (xyz_target[:,:,1,:].nan_to_num() * res_mask.unsqueeze(-1).float()).sum(dim=1, keepdim=True)
        motif_center  = motif_center / (res_mask.sum(dim=-1, keepdim=True).unsqueeze(-1) + 1e-8)

        xyz_centered = xyz_target - motif_center.unsqueeze(2)
        print("motif_center",motif_center)
        # 验证中心化是否正确
        yuandian = (xyz_centered[:,:,1,:]*res_mask.unsqueeze(-1).float()).sum(dim=1)/ (res_mask.sum(dim=-1, keepdim=True) + 1e-8)
        print("yuandian",yuandian)
        # print("center of xyz__centered:",yuandian)
        epsilon = torch.randn((B,3), device=xyz_target.device) * self.sigma_perturb
        epsilon = epsilon[:,None,None,:]  # (B, 1, 1, 3)
        
        # 将扰动应用到中心化后的坐标上
        xyz_stochastically_centered = xyz_centered + epsilon
        yuandian = (xyz_stochastically_centered[:,:,1,:]*res_mask.unsqueeze(-1).float()).sum(dim=1)/ (res_mask.sum(dim=-1, keepdim=True) + 1e-8)
        # print("center of xyz_stochastically_centered:",yuandian)
        return xyz_stochastically_centered

    def sample_with_interpolant(self, xyz_target, seq_target, ss_target, res_mask, str_mask, 
                                seq_mask, bond_diffuse_mask, pdb_idx, t, head_mask=None, tail_mask=None, N_C_anchor=None):
        """
        使用 Interpolant 对输入进行加噪或采样。

        Args:
            xyz_target (Tensor): (B, L, 3, 3) 目标坐标 (N, CA, C)。
            seq_target (Tensor): (B, L) 目标序列。
            ss_target (Tensor): (B, L, L) 目标二级结构。
            res_mask (Tensor): (B, L) 氨基酸残基掩码。
            str_mask (Tensor): (B, L) 结构掩码。
            seq_mask (Tensor): (B, L) 序列掩码。
            bond_diffuse_mask (Tensor): (B, L, L) 二级结构扩散掩码。
            pdb_idx (list): PDB索引，用于识别链。
            t (Tensor): (B,) 扩散时间步。

        Returns:
            dict: 包含加噪/采样后数据的字典。
        """
        # 1. 中心化 motif
        xyz_centered = self._center_global(xyz_target, str_mask, res_mask, pdb_idx)

        # 2. 从坐标计算旋转矩阵
        rotmats = iu.get_R_from_xyz(xyz_centered).to(self.device)

        # 3. 调用 interpolant.corrupt_batch
        noised_batch = self.interpolant.corrupt_batch(
            trans_1=xyz_centered[:, :, 1, :],
            rotmats_1=rotmats,
            aatypes_1=seq_target,
            ss_1=ss_target,
            res_mask=res_mask,
            trans_diffuse_mask=str_mask.float(),
            rots_diffuse_mask=str_mask.float(),
            aatypes_diffuse_mask=seq_mask.float(),
            ss_diffuse_mask=bond_diffuse_mask.float(),
            t=t
        )

        xyz_noised = iu.get_xyz_from_RT(noised_batch['rotmats_t'],noised_batch['trans_t'])
        
        # 将head_mask和tail_mask位置的张量分别复制为锚定Body残基的特征
        if (head_mask is not None or tail_mask is not None) and (N_C_anchor is not None):
            noised_batch['aatypes_t'] = iu.update_nc_node_features(noised_batch['aatypes_t'], N_C_anchor, head_mask, tail_mask)
            # 使用带offset的坐标更新函数
            xyz_noised = iu.update_nc_node_coordinates(xyz_noised, N_C_anchor, head_mask, tail_mask)
            xyz_centered = iu.update_nc_node_coordinates(xyz_centered, N_C_anchor, head_mask, tail_mask)
        
        return xyz_noised, noised_batch['aatypes_t'], noised_batch['ss_t'], xyz_centered, rotmats

    
    def _preprocess_batch(self, seq, xyz_t, bond_mat,rf_idx,pdb_idx,alpha,alpha_tor_mask,
                            t,str_mask=None,seq_mask=None, bond_mask=None,res_mask=None,
                            head_mask=None, tail_mask=None):
        
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
        ### Graph features ###
        #####################
        # Build diffusion distance features from bond matrix.
        # Use residue-level graph by default; optionally upgraded to atom-level
        # when head/tail masks and N/C anchors are available (training).
        # res_dist_matrix, meta = self.bond_mat_2_dist_mat(
        #     bond_mat,
        #     rf_idx,
        #     res_mask,
        #     head_mask=head_mask,
        #     tail_mask=tail_mask,
        #     N_C_anchor=None,  # training path passes this explicitly
        # )


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

    # def bond_mat_2_dist_mat(
    #     self,
    #     bond_mat,
    #     rf_idx,
    #     res_mask,
    #     head_mask=None,
    #     tail_mask=None,
    #     N_C_anchor=None,
    # ):
    #     """
    #     Build pairwise diffusion-distance features from a bond matrix.

    #     - If only (bond_mat, rf_idx, res_mask) are provided, fall back to the
    #       original residue-level graph (L nodes).
    #     - If head_mask / tail_mask / N_C_anchor are provided (training with
    #       explicit N/C functional nodes), upgrade to a 3L-node atomic graph:
    #         * For each residue position i, create (N_i, CA_i, C_i).
    #         * Add intra-residue edges N_i-CA_i-C_i.
    #         * Add peptide edges C_i-N_{i+1} where rf_idx indicates sequence
    #           adjacency and both residues are valid.
    #         * For each non-zero entry in bond_mat[i,j], connect the appropriate
    #           backbone atoms:
    #             - body node      -> CA_i
    #             - head functional -> N_owner (via N_C_anchor, dim=0)
    #             - tail functional -> C_owner (via N_C_anchor, dim=1)
    #       Then run diffusion_distance_tensor on the 3L graph and average the
    #       resulting atomic features back to residue level (L x L).
    #     """
    #     B, L_total = bond_mat.shape[:2]

    #     use_atomic_graph = (
    #         head_mask is not None
    #         and tail_mask is not None
    #         and N_C_anchor is not None
    #     )

    #     # -----------------------------
    #     # Fallback: original behaviour
    #     # -----------------------------
    #     if not use_atomic_graph:
    #         _bond_mat = bond_mat.clone()

    #         # Remove self-loops
    #         eye = torch.eye(L_total, device=bond_mat.device).unsqueeze(0)
    #         _bond_mat = _bond_mat * (1 - eye)

    #         # Connect sequential nodes only within the same chain
    #         is_sequential = (rf_idx[:, 1:] == rf_idx[:, :-1] + 1)

    #         # Respect residue mask if provided (avoid linking masked residues)
    #         if res_mask is not None:
    #             res_mask_bool = res_mask.bool().squeeze(-1) if res_mask.dim() == 3 else res_mask.bool()
    #             is_sequential = is_sequential & res_mask_bool[:, :-1] & res_mask_bool[:, 1:]

    #         b_ids, i_ids = torch.where(is_sequential)
    #         if b_ids.numel() > 0:
    #             _bond_mat[b_ids, i_ids, i_ids + 1] = 1
    #             _bond_mat[b_ids, i_ids + 1, i_ids] = 1

    #         # Run diffusion distance on residue graph
    #         res_dist_matrix = smu.diffusion_distance_tensor(
    #             A_adj_batch=_bond_mat,
    #             times=self._conf.preprocess.diffusion_map_times,
    #             k=self._conf.preprocess.diffusion_map_features,
    #             skip_top=True,
    #             node_mask=res_mask.int(),
    #             rbf_num=100,
    #             rbf_gamma=None,
    #             k_ratio=0.6,
    #         )

    #         return res_dist_matrix, None

    #     # ---------------------------------------
    #     # Upgraded: atomic graph with 3L nodes
    #     # ---------------------------------------
    #     device = bond_mat.device
    #     dtype = bond_mat.dtype

    #     head_mask = head_mask.bool()
    #     tail_mask = tail_mask.bool()
    #     res_mask_bool = res_mask.bool().squeeze(-1) if res_mask.dim() == 3 else res_mask.bool()

    #     # body positions: valid residues that are not pure N/C functional nodes
    #     body_mask_bool = res_mask_bool & (~head_mask) & (~tail_mask)
    #     # per-sample body counts and max for batching
    #     body_counts = body_mask_bool.sum(dim=1)  # (B,)
    #     if int(body_counts.max().item()) == 0:
    #         # degenerate case: fall back to residue graph
    #         return self.bond_mat_2_dist_mat(bond_mat, rf_idx, res_mask)

    #     L_body_max = int(body_counts.max().item())
    #     N_atom = 3 * L_body_max

    #     # Owner mapping: for each position, which *body* residue index provides its backbone
    #     # First build global owner indices in [0, L_total)
    #     owner_global = torch.arange(L_total, device=device).view(1, L_total).expand(B, L_total).clone()

    #     # Precompute a "local body index" map for all batches:
    #     # body_local_all[b, i] = local index in [0, Lb) if position i is a body residue, else -1.
    #     # This avoids re-allocating and filling per-batch mappings inside loops.
    #     body_cumsum = body_mask_bool.long().cumsum(dim=1)
    #     body_local_all = body_cumsum - 1
    #     body_local_all = body_local_all.masked_fill(~body_mask_bool, -1)

    #     # Reusable identity for bond_mat masking (shared L_total across batch)
    #     eye_total = torch.eye(L_total, device=device, dtype=torch.bool)

    #     # Helper to update owner_global from N_C_anchor for functional nodes
    #     def _assign_owner_from_anchor(layer_idx, func_mask):
    #         # layer_idx: 0 for N-side anchor, 1 for C-side anchor
    #         nonlocal owner_global
    #         pos = torch.nonzero(func_mask, as_tuple=False)  # (N_func, 2) -> (b, i)
    #         if pos.numel() == 0:
    #             return
    #         b_idx = pos[:, 0]
    #         i_idx = pos[:, 1]
    #         # N_C_anchor[b, i, :, layer_idx] is anchor row for this functional node
    #         anchor_rows = N_C_anchor[b_idx, i_idx, :, layer_idx]  # (N_func, L_total)
    #         # Only allow body residues (非 head/tail 且有效) 作为 owner
    #         body_cols = body_mask_bool[b_idx]
    #         anchor_rows = anchor_rows & body_cols
    #         # 对于每个函数点，若存在多个 True，取第一个；若不存在，则保留原 owner(i)=i
    #         any_anchor = anchor_rows.any(dim=1)
    #         if any_anchor.any():
    #             # argmax 在 all-zero 行会返回 0，因此用 any_anchor 过滤
    #             idx_true = anchor_rows.float().argmax(dim=1)
    #             new_owner = torch.where(any_anchor, idx_true, i_idx)
    #             owner_global[b_idx, i_idx] = new_owner

    #     # N 端功能节点 -> 通过 N_C_anchor[..., 0] 找到所属残基
    #     _assign_owner_from_anchor(layer_idx=0, func_mask=head_mask)
    #     # C 端功能节点
    #     _assign_owner_from_anchor(layer_idx=1, func_mask=tail_mask)

    #     # Atom type per position:
    #     #   0 -> N, 1 -> CA, 2 -> C
    #     center_type = torch.full((B, L_total), 1, device=device, dtype=torch.long)  # 默认 CA
    #     center_type[head_mask] = 0  # head 功能节点视为 N
    #     center_type[tail_mask] = 2  # tail 功能节点视为 C

    #     # Build atomic adjacency and node mask with compressed body indices
    #     A_atom = torch.zeros((B, N_atom, N_atom), device=device, dtype=dtype)
    #     node_mask_atom = torch.zeros((B, N_atom), device=device, dtype=torch.bool)

    #     for b in range(B):
    #         Lb = int(body_counts[b].item())
    #         if Lb == 0:
    #             continue

    #         # Per-batch view of local body indices and indices of body residues
    #         body_local = body_local_all[b]
    #         body_idx = torch.nonzero(body_mask_bool[b], as_tuple=False).view(-1)

    #         # valid atomic nodes for this sample
    #         node_mask_atom[b, : 3 * Lb] = True

    #         # 1) Intra-residue edges: N_j-CA_j, CA_j-C_j for body residues j
    #         n_idx = 3 * torch.arange(Lb, device=device, dtype=torch.long)
    #         ca_idx = n_idx + 1
    #         c_idx = n_idx + 2
    #         A_atom[b, n_idx, ca_idx] = 1.0
    #         A_atom[b, ca_idx, n_idx] = 1.0
    #         A_atom[b, ca_idx, c_idx] = 1.0
    #         A_atom[b, c_idx, ca_idx] = 1.0

    #         # 2) Peptide bonds: connect C_i to N_{i+1} when rf_idx indicates adjacency
    #         is_sequential_b = (
    #             (rf_idx[b, 1:] == rf_idx[b, :-1] + 1)
    #             & res_mask_bool[b, :-1]
    #             & res_mask_bool[b, 1:]
    #         )
    #         i_ids = torch.nonzero(is_sequential_b, as_tuple=False).view(-1)
    #         if i_ids.numel() > 0:
    #             owner_i = body_local[i_ids]
    #             owner_ip1 = body_local[i_ids + 1]
    #             valid = (owner_i >= 0) & (owner_ip1 >= 0)
    #             owner_i = owner_i[valid]
    #             owner_ip1 = owner_ip1[valid]
    #             if owner_i.numel() > 0:
    #                 c_i = 3 * owner_i + 2
    #                 n_ip1 = 3 * owner_ip1 + 0
    #                 A_atom[b, c_i, n_ip1] = 1.0
    #                 A_atom[b, n_ip1, c_i] = 1.0

    #         # 3) Special bonds from bond_mat: map each entry (i,j) to appropriate atoms
    #         #    using owner_global + center_type (N/CA/C) compressed through body_local.
    #         bm_mask_b = bond_mat[b] > 0.5  # treat entries >=0.5 as connected
    #         # Remove self connections (i == j)
    #         bm_mask_b = bm_mask_b & (~eye_total)

    #         edges = torch.nonzero(bm_mask_b, as_tuple=False)
    #         if edges.numel() > 0:
    #             i_e = edges[:, 0]
    #             j_e = edges[:, 1]

    #             owner_i_global = owner_global[b, i_e]
    #             owner_j_global = owner_global[b, j_e]

    #             owner_i_local = body_local[owner_i_global]
    #             owner_j_local = body_local[owner_j_global]

    #             valid_owner = (owner_i_local >= 0) & (owner_j_local >= 0)
    #             if valid_owner.any():
    #                 owner_i_local = owner_i_local[valid_owner]
    #                 owner_j_local = owner_j_local[valid_owner]
    #                 type_i = center_type[b, i_e][valid_owner]
    #                 type_j = center_type[b, j_e][valid_owner]

    #                 atom_i = 3 * owner_i_local + type_i
    #                 atom_j = 3 * owner_j_local + type_j

    #                 A_atom[b, atom_i, atom_j] = 1.0
    #                 A_atom[b, atom_j, atom_i] = 1.0

    #     # 4) Run diffusion distance on atomic graph
    #     res_dist_atom = smu.diffusion_distance_tensor(
    #         A_adj_batch=A_atom,
    #         times=self._conf.preprocess.diffusion_map_times,
    #         k=self._conf.preprocess.diffusion_map_features,
    #         skip_top=True,
    #         node_mask=node_mask_atom.int(),
    #         rbf_num=100,
    #         rbf_gamma=None,
    #         k_ratio=0.6,
    #     )

    #     # 5) Aggregate atomic features back to residue level by averaging N/CA/C
    #     #    for each *body* residue pair, then scatter into full L_total x L_total.
    #     B_loc, N_atom_loc, _, F = res_dist_atom.shape
    #     assert N_atom_loc == N_atom, "Unexpected atomic feature shape."

    #     # per-batch result initialised as zeros
    #     res_dist_matrix = torch.zeros(
    #         B_loc, L_total, L_total, F, device=device, dtype=dtype
    #     )

    #     for b in range(B_loc):
    #         Lb = int(body_counts[b].item())
    #         if Lb == 0:
    #             continue
    #         # slice valid atomic block and reshape
    #         block = res_dist_atom[b, : 3 * Lb, : 3 * Lb, :]  # (3Lb, 3Lb, F)
    #         block_view = block.view(Lb, 3, Lb, 3, F)
    #         res_body = block_view.mean(dim=(1, 3))  # (Lb, Lb, F)

    #         body_idx = torch.nonzero(body_mask_bool[b], as_tuple=False).view(-1)
    #         res_dist_matrix[b, body_idx[:, None], body_idx[None, :], :] = res_body

    #     # Expose atomic adjacency for debugging / tests
    #     meta = {"atom_adj": A_atom}

    #     return res_dist_matrix, meta



    # def sample_step(self, *, t, x_t, seq_init, final_step, train=True):
    def sample_step(self, t_1, t_2, x_t_1, seq_t_1, bond_mat_t_1, fixed_batch_data, masks,
                    trans_sc=None, aatypes_sc=None, torsions_sc=None,
                    trans_1=None, rotmats_1=None, aatypes_1=None, ss_1=None):
        """
        Performs one step of the sampling process using the interpolant framework.

        Args:
            t_1 (float): Current time step (e.g., from 0 to 1).
            t_2 (float): Next time step.
            x_t_1 (torch.tensor): (B,L,14,3) The residue positions at time t_1.
            seq_t_1 (torch.tensor): (B,L) The sequence at time t_1.
            bond_mat_t_1 (torch.tensor): (B,L,L) The ss matrix at time t_1.
            fixed_batch_data (dict): Dictionary with non-changing data for the model.
            masks (dict): Dictionary with diffuse masks.

        Returns:
            px0 (torch.tensor): (B,L,14,3) The model's prediction of x0.
            x_t_2 (torch.tensor): (B,L,14,3) The updated positions of the next step.
            seq_t_2 (torch.tensor): (B,L) The updated sequence of the next step.
            bond_mat_t_2 (torch.tensor): (B,L,L) The updated ss matrix of the next step.
        """
        B, L = seq_t_1.shape[:2]
        device = x_t_1.device

        # Skip torsion computation to improve stability at early sampling
        B_local, L_local = seq_t_1.shape[:2]
        alpha = torch.zeros(B_local, L_local, 10, 2, device=device, dtype=torch.float32)
        alpha_tor_mask = torch.zeros(B_local, L_local, 10, device=device, dtype=torch.bool)
        res_mask = masks['res_mask']
        head_mask = masks['head_mask']
        tail_mask = masks['tail_mask']
        bond_mask = masks['bond_diffuse_mask']
        N_C_anchor = masks['N_C_anchor']
        if aatypes_1 is not None:
            aatypes_1 = aatypes_1.long()

        # Preprocess batch for model
        msa_full, xyz_t, alpha_t, idx, t1d, t2d, str_mask, seq_mask, bond_mask = \
            self._preprocess_batch(
                seq=seq_t_1, xyz_t=x_t_1, bond_mat=bond_mat_t_1,
                rf_idx=fixed_batch_data['rf_idx'], pdb_idx=fixed_batch_data['pdb_idx'],
                alpha=alpha, alpha_tor_mask=alpha_tor_mask,
                t=torch.full((B,), t_1, device=device, dtype=torch.float32),
                str_mask=masks['str_mask'], seq_mask=masks['seq_mask'],
                bond_mask=bond_mask,
                res_mask=res_mask,
                head_mask=head_mask,
                tail_mask=tail_mask
            )
        

        with torch.no_grad():
            # Use unified BaseDesignModel signature (works for both RF and APM wrappers)
            model_out = self.model(
                seq_noised=seq_t_1,
                xyz_noised=x_t_1[..., :14, :],
                bond_noised=bond_mat_t_1,
                rf_idx=idx,
                pdb_idx=fixed_batch_data['pdb_idx'],
                N_C_anchor=N_C_anchor,
                alpha_target=alpha,
                alpha_tor_mask=alpha_tor_mask,
                partial_T=torch.full((B,), t_1, device=device, dtype=torch.float32),
                str_mask=str_mask,
                seq_mask=seq_mask,
                bond_mask=bond_mask,
                head_mask=head_mask,
                tail_mask=tail_mask,
                res_mask=res_mask,
                use_checkpoint=False,
                trans_sc=trans_sc,
                aatypes_sc=aatypes_sc,
                torsions_sc=torsions_sc,
                trans_1=trans_1,
                rotmats_1= rotmats_1,
                aatypes_1= aatypes_1,
                bond_mat_1=ss_1,
            )

            # Unpack depending on wrapper type
            if isinstance(model_out, (list, tuple)) and len(model_out) == 4:
                # APMWrapper returns: pred_logits, xyz_pred, alpha_s, bond_matrix
                logits, xyz_pred, alpha_pred, bond_mat_pred = model_out
                px0_bb = xyz_pred.squeeze(1)
            else:
                # RoseTTAFoldWrapper passthrough RF outputs (8-tuple)
                msa_prev, pair_prev, px0_bb, state_prev, alpha_pred, logits, bond_mat_pred, _ = model_out

 





        # Guidance hook: pre_model (may adjust logits/px0_bb/alpha_pred/bond_mat_pred)
        t_1_tensor = torch.full((B,), t_1, device=device, dtype=torch.float32)
        model_raw = {
            'logits': logits,
            'px0_bb': px0_bb,
            'alpha_pred': alpha_pred,
            'bond_mat_pred': bond_mat_pred,
        }
        model_raw = self.guidance_manager.pre_model(
            model_raw,
            t_1=t_1_tensor,
            t_2=t_2,
            seq_t_1=seq_t_1,
            x_t_1=x_t_1,
            masks=masks,
            fixed_batch_data=fixed_batch_data,
        )
        logits = model_raw.get('logits', logits)
        px0_bb = model_raw.get('px0_bb', px0_bb)
        alpha_pred = model_raw.get('alpha_pred', alpha_pred)
        bond_mat_pred = model_raw.get('bond_mat_pred', bond_mat_pred)
     




        # All-atom prediction for x0
        res_mask_2d = res_mask.unsqueeze(1) * res_mask.unsqueeze(2)
        res_bond_mask_2d = res_mask_2d  * bond_mask
        
        pseq0 = torch.argmax(logits[...,:20], dim=-1)
        final_res_mask = res_mask.float() * (1-head_mask.float()) * (1-tail_mask.float())
        bond_mat_pred_sampled = smu.sample_permutation(bond_mat_pred, res_mask_2d)
        
   
        # fix mask part
        if aatypes_1 is not None:
            pseq0 = pseq0 * seq_mask.float() + (1 - seq_mask.float()) * aatypes_1
            print("change pseq0 by aatypes_1")
        if  trans_1 is not None and rotmats_1 is not None:
            print("change px0_bb by trans_1 and rotmats_1")
            xyz_1 = iu.get_xyz_from_RT(rotmats_1,trans_1)
            px0_bb = px0_bb * str_mask.float() + ( 1 - str_mask.float()) * xyz_1

        _, px0  = self.allatom(pseq0, px0_bb, alpha_pred, 
                            bond_mat=bond_mat_pred_sampled, 
                            link_csv_path=self.preprocess_conf.link_config, use_H=False,
                            res_mask=final_res_mask)
        # Normalize px0 shape to [B, L, 14, 3]
        if px0.dim() == 5 and px0.shape[1] == 1:
            px0 = px0.squeeze(1)
        px0 = px0[:, :, :14, :]
        print("pseq0",pseq0)
        # Prepare model_out for interpolant
        model_out_interpolant = {
            'pred_trans': px0[:, :, 1, :].nan_to_num(), 'pred_rotmats': iu.get_R_from_xyz(px0.nan_to_num()),
            'pred_aatypes': pseq0, 'pred_logits': logits, 'pred_ss': bond_mat_pred
        }






        # Guidance hook: pre_interpolant (may adjust pred_trans/pred_rotmats/logits/ss)
        model_out_interpolant = self.guidance_manager.pre_interpolant(
            model_out_interpolant,
            t_1=t_1_tensor,
            t_2=t_2,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            aatypes_1=aatypes_1,
            ss_1=ss_1,
            seq_t_1=seq_t_1,
            x_t_1=x_t_1,
            masks=masks,
            fixed_batch_data=fixed_batch_data,
        )











        # Convert current state to RT representation
        trans_t_1 = x_t_1[:, :, 1, :]
        rotmats_t_1 = iu.get_R_from_xyz(x_t_1)
        # Take a step with the interpolant for backbone
        trans_t_2, rotmats_t_2, aatypes_t_2, ss_t_2, _ = self.interpolant.sample_step(
            model_out_interpolant, t_1, t_2,
            trans_t_1, rotmats_t_1, aatypes_t_1=seq_t_1, ss_t_1=bond_mat_t_1,
            trans_diffuse_mask=masks['trans_diffuse_mask'],
            rots_diffuse_mask=masks['rots_diffuse_mask'],
            aatypes_diffuse_mask=masks['aatypes_diffuse_mask'],
            ss_diffuse_mask=bond_mask,
            trans_1=trans_1, rotmats_1=rotmats_1, aatypes_1=aatypes_1, ss_1=ss_1
        )

        # Guidance hook: post_step (may adjust trans_t_2/rotmats_t_2/aatypes_t_2/ss_t_2)
        step_out = {
            'trans_t_2': trans_t_2,
            'rotmats_t_2': rotmats_t_2,
            'aatypes_t_2': aatypes_t_2,
            'ss_t_2': ss_t_2,
        }
        step_out = self.guidance_manager.post_step(
            step_out,
            t_1=t_1_tensor,
            t_2=t_2,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            aatypes_1=aatypes_1,
            ss_1=ss_1,
            seq_t_1=seq_t_1,
            x_t_1=x_t_1,
            masks=masks,
            fixed_batch_data=fixed_batch_data,
        )
        trans_t_2 = step_out.get('trans_t_2', trans_t_2)
        rotmats_t_2 = step_out.get('rotmats_t_2', rotmats_t_2)
        aatypes_t_2 = step_out.get('aatypes_t_2', aatypes_t_2)
        ss_t_2 = step_out.get('ss_t_2', ss_t_2)




        
        # Build new backbone and full atom structure
        x_t_2_bb = iu.get_xyz_from_RT(rotmats_t_2, trans_t_2)
        # Ensure integer indices for sequence
        aatypes_t_2 = aatypes_t_2.to(torch.long)
        bond_mat_t_2_sampled = smu.sample_permutation(ss_t_2, res_mask_2d)
        _, x_t_2 = self.allatom(aatypes_t_2, x_t_2_bb, alpha_pred, bond_mat=bond_mat_t_2_sampled, 
                                link_csv_path=self.preprocess_conf.link_config, use_H=False, res_mask=final_res_mask)

        # Normalize x_t_2 shape to [B, L, 14, 3]
        if x_t_2.dim() == 5 and x_t_2.shape[1] == 1:
            x_t_2 = x_t_2.squeeze(1)
        x_t_2 = x_t_2[:, :, :14, :]
        
        # Prepare self-conditioning tensors for the next step
        new_trans_sc = px0[:, :, 1, :].nan_to_num()
        new_aatypes_sc = logits
        
        # Extract chi angles from alpha_s prediction for torsion self-conditioning
        alpha_s_squeezed = alpha_pred # [B, L, 10, 2]
        chi_sincos = alpha_s_squeezed[:, :, 3:7, :] # [B, L, 4, 2]
        # atan2(sin, cos) for angles
        new_torsions_sc = torch.atan2(chi_sincos[..., 1], chi_sincos[..., 0]) # [B, L, 4]
        new_sc_dict = {
            'trans_sc': new_trans_sc,
            'aatypes_sc': new_aatypes_sc,
            'torsions_sc': new_torsions_sc,
            'torsions_cs_sc': chi_sincos,
        }

        if (head_mask is not None or tail_mask is not None) and (N_C_anchor is not None):
            px0 = iu.update_nc_node_coordinates(px0, N_C_anchor, head_mask, tail_mask)
            x_t_2 = iu.update_nc_node_coordinates(x_t_2, N_C_anchor, head_mask, tail_mask)
            aatypes_t_2 = iu.update_nc_node_features(aatypes_t_2, N_C_anchor, head_mask, tail_mask)



        # 打印bond_mat_t_2_sampled非对角线元素为1的位置上的aatypes_t_2的残基对
        # 修复：num2aa 是列表而不是字典，因此使用索引而不是 .get()
        with torch.no_grad():
            bm = bond_mat_t_2_sampled
            aa = aatypes_t_2
            # 获取batch size和长度
            B, L, _ = bm.shape
            for b in range(B):
                # 仅取非对角线
                idxs = torch.nonzero((bm[b] >= 0.9) & (~torch.eye(L, dtype=torch.bool, device=bm.device)))
                for (i, j) in idxs:
                    if i > j:
                        resi = int(aa[b, i])
                        resj = int(aa[b, j])
                        resi_name = num2aa[resi] if 0 <= resi < len(num2aa) else str(resi)
                        resj_name = num2aa[resj] if 0 <= resj < len(num2aa) else str(resj)
                        print(f"Batch t_2 {b} | Bond({i},{j}) : {resi_name} - {resj_name}")

        with torch.no_grad():
            bm = bond_mat_pred_sampled
            aa = pseq0
            # 获取batch size和长度
            B, L, _ = bm.shape
            for b in range(B):
                # 仅取非对角线
                idxs = torch.nonzero((bm[b] >= 0.9) & (~torch.eye(L, dtype=torch.bool, device=bm.device)))
                for (i, j) in idxs:
                    if i > j:
                        resi = int(aa[b, i])
                        resj = int(aa[b, j])
                        resi_name = num2aa[resi] if 0 <= resi < len(num2aa) else str(resi)
                        resj_name = num2aa[resj] if 0 <= resj < len(num2aa) else str(resj)
                        print(f"Batch pred {b} | Bond({i},{j}) : {resi_name} - {resj_name}")

        return px0, x_t_2, aatypes_t_2, ss_t_2, alpha_pred, new_sc_dict

    def refine_sidechain(self, seq, x, bond_mat, rf_idx, pdb_idx, alpha_alt_target, 
                        alpha_tor_mask, str_mask, seq_mask, head_mask, tail_mask, bond_mask, res_mask, N_C_anchor=None):
        self.sidechain_model.eval()

        if (head_mask is not None or tail_mask is not None) and (N_C_anchor is not None):
            seq = iu.update_nc_node_features(seq, N_C_anchor, head_mask, tail_mask)
            x = iu.update_nc_node_coordinates(x, N_C_anchor, head_mask, tail_mask)
        
        # APMSidechainWrapper 会根据 bond_mat / rf_idx / res_mask 内部构建 res_dist_embed，
        # 这里不再在 sampler 中调用 bond_mat_2_dist_mat。
        torsion_angles = self.sidechain_model(
            seq_noised=seq,
            xyz_noised=x,
            bond_noised=bond_mat,
            rf_idx=rf_idx,
            pdb_idx=pdb_idx,
            res_dist_matrix=None,
            alpha_target=alpha_alt_target,
            alpha_tor_mask=alpha_tor_mask,
            partial_T=None,
            str_mask=str_mask,
            seq_mask=seq_mask,
            head_mask=head_mask,
            tail_mask=tail_mask,
            bond_mask=bond_mask,
            res_mask=res_mask,
            use_checkpoint=False,
        )


        final_res_mask = res_mask.float() * (1 - head_mask.float()) * (1 - tail_mask.float())
        _ , all_atom_coords = self.allatom(
            seq, x[..., :3, :], torsion_angles,  # 主链coordinates, 只用前3个原子
            link_csv_path=self.preprocess_conf.link_config,
            use_H=False,
            bond_mat=bond_mat,
            res_mask= final_res_mask,
        )
        _ , all_atom_coords_withCN = self.allatom(
            seq, x[..., :3, :], torsion_angles,  # 主链coordinates, 只用前3个原子
            link_csv_path=self.preprocess_conf.link_config,
            use_H=False,
            bond_mat=bond_mat,
            res_mask= res_mask,
        )

        return torsion_angles,all_atom_coords,all_atom_coords_withCN

    def _write_refined_outputs(
        self,
        out_pdb_dir: str,
        file_prefix: str,
        num_batch: int,
        final_seq: torch.Tensor,
        final_bond_sampled: torch.Tensor,
        refine_x: torch.Tensor,
        refine_x_withCN: torch.Tensor,
        fixed_batch_data: dict,
        head_mask: torch.Tensor,
        tail_mask: torch.Tensor,
    ):
        """统一写出精修后的 PDB 与键信息。"""
        post_refine_dir = out_pdb_dir if out_pdb_dir is not None else "."
        pdb_idx_all = fixed_batch_data["pdb_idx"]

        for b in range(num_batch):
            filename = os.path.join(
                post_refine_dir,
                f"{file_prefix}final_refined_structure_{b}.pdb",
            )
            pdb_idx = pdb_idx_all
            res_pdb_idx = [int(res[1]) + 1 for res in pdb_idx[b]]
            chain_pdb_idx = [res[0] for res in pdb_idx[b]]
            writepdb(
                filename,
                refine_x[b],
                final_seq[b],
                idx_pdb=res_pdb_idx,
                chain_idx=chain_pdb_idx,
                bond_mat=final_bond_sampled[b],
                link_csv_path=self.preprocess_conf.link_config,
            )
            self._save_bond_info(
                post_refine_dir,
                b,
                final_bond_sampled[b],
                pdb_idx[b],
                final_seq[b],
                refine_x_withCN[b],
                include_invalid=True,
                filename="bonds_final_refined_structure",
                head_mask=head_mask[b] if head_mask is not None else None,
                tail_mask=tail_mask[b] if tail_mask is not None else None,
            )

    def refine_sidechain_by_gd(
        self,
        seq,
        x,
        bond_mat,
        rf_idx,
        pdb_idx,
        alpha_init,
        str_mask,
        seq_mask,
        head_mask,
        tail_mask,
        bond_mask,
        res_mask,
        N_C_anchor,
        num_steps: int = 100,
        lr: float = 1e-1,
        lr_min: float = 1e-2,
        w_bond: float = 1.0,
        w_clash: float = 0.2,
    ):
        """
        使用梯度下降直接优化侧链扭转角。

        - 初始化自主模型预测的 alpha_init (B, L, 10, 2)，只优化 chi1-4 (通道 3:7)。
        - 损失为 BondCoherenceLoss + OpenFoldClashLoss。
        """
        device = self.device

        # 保证输入在当前设备且不反传回 backbone / 采样过程
        seq = seq.detach().to(device)
        x = x.detach().to(device)
        bond_mat = bond_mat.detach().to(device)
        rf_idx = rf_idx.to(device)
        # pdb_idx 只是传给 allatom / 保存，不参与梯度
        alpha_init = alpha_init.detach().to(device)
        str_mask = str_mask.to(device)
        seq_mask = seq_mask.to(device)
        bond_mask = bond_mask.to(device) if bond_mask is not None else torch.ones_like(bond_mat, dtype=torch.bool, device=device)
        res_mask = res_mask.to(device)
        head_mask = head_mask.to(device) if head_mask is not None else torch.zeros_like(res_mask, dtype=torch.bool, device=device)
        tail_mask = tail_mask.to(device) if tail_mask is not None else torch.zeros_like(res_mask, dtype=torch.bool, device=device)

        B, L = seq.shape[:2]

        # 从 alpha_init 中取出 chi1-4 的 cos/sin，转换为角度作为优化变量
        chi_init = alpha_init[:, :, 3:7, :]  # [B, L, 4, 2]
        # 约定：alpha[..., 0]=cos, alpha[..., 1]=sin
        phi_init = torch.atan2(chi_init[..., 1], chi_init[..., 0])  # [B, L, 4]
        phi = phi_init.clone().detach().requires_grad_(True)

        # 惰性构建 loss 模块

        self._bond_coherence_loss = BondCoherenceLoss(
            link_csv_path=self.preprocess_conf.link_config,device=device,    
            energy_w_angle=1,energy_w_dihedral=1,
        )

        self._openfold_clash_loss = OpenFoldClashLoss(
            link_csv_path=self.preprocess_conf.link_config,
            device=str(device), log_raw=False,
        )

        optimizer = torch.optim.Adam([phi], lr=lr)

        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_steps), eta_min=lr_min)

        final_res_mask = res_mask.float() * (1 - head_mask.float()) * (1 - tail_mask.float())

        # 记录历史“最好”的 phi，用于返回最优构象而不是最后一步
        # 评价标准：优先最大 is_valid 链接数，其次最小 loss
        best_valid_links = None
        best_loss = None
        best_phi = phi.detach().clone()

        with torch.enable_grad():  
            for it in range(num_steps):
                optimizer.zero_grad()
                
                # 将角度映射回 cos/sin，并写回到 10 通道 alpha 中（只改 chi1-4）
                chi_sincos = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)  # [B, L, 4, 2]
                alpha_opt = alpha_init.clone()
                alpha_opt[:, :, 3:7, :] = chi_sincos

                backbone_bb = x[..., :3, :]
                # 只调用一次 allatom，使用带 CN 的全原子坐标
                _, all_atom_coords_withCN = self.allatom(
                    seq,
                    backbone_bb,
                    alpha_opt,
                    bond_mat=bond_mat,
                    link_csv_path=self.preprocess_conf.link_config,
                    use_H=False,
                    res_mask=res_mask,
                )
                # ClashLoss 直接复用相同坐标，通过 res_mask 控制参与的残基
                loss_bond = self._bond_coherence_loss(
                    N_C_anchor,
                    bond_matrix=bond_mat,
                    res_mask=res_mask,
                    all_atom_coords=all_atom_coords_withCN,
                    aatype=seq,
                    head_mask=head_mask,
                    tail_mask=tail_mask,
                    silent=True,
                )
                loss_clash = self._openfold_clash_loss(
                    N_C_anchor,
                    allatom_xyz=all_atom_coords_withCN,
                    seq_pred=seq,
                    res_mask=final_res_mask,
                    bond_mat=bond_mat,
                    head_mask=head_mask,
                    tail_mask=tail_mask,
                )
                

                loss = w_bond * loss_bond + w_clash * loss_clash
                loss.backward()
                print("total loss", loss.item())
                print("loss_bond", loss_bond.item())
                print("loss_clash", loss_clash.item())
                print("step", it, "lr", optimizer.param_groups[0]["lr"])

                # # 统计当前构象中 is_valid 的键数量（跨 batch 求和）
                # with torch.no_grad():
                #     total_valid_links = 0
                #     for b in range(B):
                #         links = get_valid_links(
                #             seq[b],
                #             all_atom_coords_withCN[b],
                #             bond_mat[b],
                #             self.preprocess_conf.link_config,
                #             head_mask=head_mask[b] if head_mask is not None else None,
                #             tail_mask=tail_mask[b] if tail_mask is not None else None,
                #             include_invalid=False,
                #         )
                #         total_valid_links += sum(1 for lk in links if lk.get("is_valid", False))
                #     print(f"step {it} total_valid_links",total_valid_links)
                #     # 更新历史最优：优先更多有效键，其次更小 loss
                #     if (
                #         best_valid_links is None
                #         or total_valid_links > best_valid_links
                #         or (
                #             total_valid_links == best_valid_links
                #             and (best_loss is None or loss.item() < best_loss)
                #         )
                #     ):
                #         best_valid_links = int(total_valid_links)
                #         best_loss = float(loss.item())
                #         best_phi = phi.detach().clone()

                optimizer.step()
                scheduler.step()

        # 使用历史最优的 phi 生成扭转角和全原子坐标（带 / 不带 N/C 端约束）
        with torch.no_grad():
            # 若优化过程中从未更新 best_phi，则退回当前 phi
            best_phi = None
            phi_final = best_phi if best_phi is not None else phi.detach()
            chi_sincos = torch.stack([torch.cos(phi_final), torch.sin(phi_final)], dim=-1)
            torsion_angles = alpha_init.clone()
            torsion_angles[:, :, 3:7, :] = chi_sincos

            backbone_bb = x[..., :3, :]
            _, all_atom_coords = self.allatom(
                seq,
                backbone_bb,
                torsion_angles,
                bond_mat=bond_mat,
                link_csv_path=self.preprocess_conf.link_config,
                use_H=False,
                res_mask=final_res_mask,
            )
            _, all_atom_coords_withCN = self.allatom(
                seq,
                backbone_bb,
                torsion_angles,
                bond_mat=bond_mat,
                link_csv_path=self.preprocess_conf.link_config,
                use_H=False,
                res_mask=res_mask,
            )

        return torsion_angles.detach(), all_atom_coords.detach(), all_atom_coords_withCN.detach()

    def _sample_loop(
        self,
        x_t_1,
        seq_t_1,
        bond_mat_t_1,
        ts,
        fixed_batch_data,
        masks,
        num_batch,
        num_res,
        record_trajectory: bool = True,
        eps_t: float = 5e-4,
        out_pdb_dir: str = None,
        file_prefix: str = "",
        tqdm_desc: str = "Sampling progress",
        trans_1: torch.Tensor = None,
        rotmats_1: torch.Tensor = None,
        aatypes_1: torch.Tensor = None,
        ss_1: torch.Tensor = None,
        N_C_add: bool = False,
    ):
        # Self-conditioning initialization
        num_tokens = 21  # Or get from model config
        trans_sc = torch.zeros(num_batch, num_res, 3, device=self.device)
        aatypes_sc = torch.zeros(num_batch, num_res, num_tokens, device=self.device)
        torsions_sc = torch.zeros(num_batch, num_res, 4, device=self.device)
        torsions_cs_sc = torch.zeros(num_batch, num_res, 4, 2, device=self.device)
        torsions_cs_sc[..., 0] = 1.0  # cos=1, sin=0
        head_mask = masks['head_mask']
        tail_mask = masks['tail_mask']
        res_mask = masks['res_mask']
        final_res_mask = res_mask.float() * (1-head_mask.float()) * (1-tail_mask.float())
        
        # Trajectory recording
        traj = {
            'px0': [], 'x': [], 'seq': [], 'bond': [],
        }
        
        final_px0 = None
        t_1 = ts[0]





        i = 0
        j = 11
        bond_mat_temp = torch.zeros_like(bond_mat_t_1)
        bond_mat_temp[:, i, j] = 1
        bond_mat_temp[:, j, i] = 1
        # 除了i和j,将bond_mat_temp对角线设为1
        diag_indices = torch.arange(bond_mat_temp.shape[1], device=bond_mat_temp.device)
        # Set diagonal to 1
        bond_mat_temp[:, diag_indices, diag_indices] = 1
        # Set (i, i) and (j, j) back to 0
        bond_mat_temp[:, i, i] = 0
        bond_mat_temp[:, j, j] = 0
        ss_1 = bond_mat_temp
        masks['bond_mask'] = torch.ones_like(bond_mat_t_1)
        masks['bond_mask'][:, i, :] = 0
        masks['bond_mask'][:, j, :] = 0
        masks['bond_mask'][:, :, i] = 0
        masks['bond_mask'][:, :, j] = 0
        masks['bond_diffuse_mask'] = torch.ones_like(bond_mat_t_1)
        masks['bond_diffuse_mask'][:, i, :] = 0
        masks['bond_diffuse_mask'][:, j, :] = 0
        masks['bond_diffuse_mask'][:, :, i] = 0
        masks['bond_diffuse_mask'][:, :, j] = 0

        masks['bond_mask'] = torch.zeros_like(bond_mat_t_1)
        masks['bond_diffuse_mask'] = torch.zeros_like(bond_mat_t_1)
        # ss_1 = torch.eye(bond_mat_t_1.shape[1], device=bond_mat_t_1.device)
        # ss_1 = ss_1.unsqueeze(0).repeat(num_batch, 1, 1)
        ss_1 = ss_1.to(self.device)



        # Run model
        self.model.eval()
        # Integrate over time
        for t_2 in tqdm(ts[1:], desc=tqdm_desc, unit="step"):
            t_2_val = torch.clamp(t_2, max=1.0 - eps_t)
            t_1_safe = torch.clamp(t_1, max=1.0 - eps_t)

            px0, x_t_2, seq_t_2, bond_mat_t_2, alpha_pred, new_sc_dict = self.sample_step(
                t_1_safe, t_2_val, x_t_1, seq_t_1, bond_mat_t_1,
                fixed_batch_data=fixed_batch_data,
                masks=masks,
                trans_sc=trans_sc,
                aatypes_sc=aatypes_sc,
                torsions_sc=torsions_sc,
                trans_1=trans_1,
                rotmats_1=rotmats_1,
                aatypes_1=aatypes_1,
                ss_1=ss_1,
            )
            # if N_C_add:

                
            if record_trajectory:
                traj['px0'].append(px0.detach().clone())
                traj['x'].append(x_t_2.detach().clone())
                traj['seq'].append(seq_t_2.detach().clone())
                traj['bond'].append(bond_mat_t_2.detach().clone() if bond_mat_t_2 is not None else None)

            x_t_1, seq_t_1, bond_mat_t_1 = x_t_2, seq_t_2, bond_mat_t_2
            trans_sc = new_sc_dict['trans_sc']
            aatypes_sc = new_sc_dict['aatypes_sc']
            torsions_sc = new_sc_dict['torsions_sc']
            torsions_cs_sc = new_sc_dict.get('torsions_cs_sc', torsions_cs_sc)
            final_px0 = px0
            t_1 = t_2_val

        # Final processing
        final_seq_alt = torch.argmax(aatypes_sc[..., :20], dim=-1)
        final_seq = torch.where(seq_t_1 == int(du.MASK_TOKEN_INDEX), final_seq_alt, seq_t_1.clone())
        final_bond = bond_mat_t_1

        res_mask_2d = masks['res_mask'].unsqueeze(1) * masks['res_mask'].unsqueeze(2)
        final_bond_sampled = smu.sample_permutation(final_bond, res_mask_2d)

 


        _, final_x = self.allatom(
            final_seq, x_t_1[..., :3, :], alpha_pred, bond_mat=final_bond_sampled,
            link_csv_path=self.preprocess_conf.link_config, use_H=False,
            res_mask=final_res_mask,
        )
        _, final_x_withCN = self.allatom(
            final_seq, x_t_1[..., :3, :], alpha_pred, bond_mat=final_bond_sampled,
            link_csv_path=self.preprocess_conf.link_config, use_H=False,
            res_mask=res_mask,
        )


        if record_trajectory:
            traj['px0'] = torch.stack(traj['px0']) if traj['px0'] else None
            traj['x'] = torch.stack(traj['x']) if traj['x'] else None
            traj['seq'] = torch.stack(traj['seq']) if traj['seq'] else None
            if all(b is not None for b in traj['bond']):
                traj['bond'] = torch.stack(traj['bond'])
            else:
                traj['bond'] = None

        if out_pdb_dir is not None:
            # Separate subfolders for pre-refine and post-refine outputs
            pre_refine_dir = os.path.join(out_pdb_dir, "pre_refine")
            post_refine_dir = os.path.join(out_pdb_dir, "post_refine")
            os.makedirs(pre_refine_dir, exist_ok=True)
            os.makedirs(post_refine_dir, exist_ok=True)
            for b in range(num_batch):
                filename = os.path.join(pre_refine_dir, f"{file_prefix}final_structure_{b}.pdb")
                pdb_idx = fixed_batch_data['pdb_idx']
                res_pdb_idx = [int(res[1]) + 1 for res in pdb_idx[b]]
                chain_pdb_idx = [res[0] for res in pdb_idx[b]]
                writepdb(
                    filename, final_x[b], final_seq[b],
                    idx_pdb=res_pdb_idx, chain_idx=chain_pdb_idx,
                    bond_mat=final_bond_sampled[b], link_csv_path=self.preprocess_conf.link_config
                )
                self._save_bond_info(
                    pre_refine_dir, b, final_bond_sampled[b], pdb_idx[b], final_seq[b], final_x_withCN[b],
                    include_invalid=True,
                    head_mask=head_mask[b] if head_mask is not None else None,
                    tail_mask=tail_mask[b] if tail_mask is not None else None,
                )
                if record_trajectory and traj['x'] is not None:
                    traj_filename = os.path.join(pre_refine_dir, f"{file_prefix}trajectory_{b}.pdb")
                    writepdb_multi(
                        traj_filename, traj['x'][:, b], torch.zeros(num_res, device=self.device),
                        traj['seq'][:, b], chain_ids=[res[0] for res in pdb_idx[b]], use_hydrogens=False,
                        bond_mat=final_bond_sampled[b], link_csv_path=self.preprocess_conf.link_config
                    )
                    traj['px0']
                    traj_filename = os.path.join(pre_refine_dir, f"{file_prefix}px0_trajectory_{b}.pdb")
                    writepdb_multi(
                        traj_filename, traj['px0'][:, b], torch.zeros(num_res, device=self.device),
                        traj['seq'][:, b], chain_ids=[res[0] for res in pdb_idx[b]], use_hydrogens=False,
                        bond_mat=final_bond_sampled[b], link_csv_path=self.preprocess_conf.link_config
                    )
        print("final_bond_sampled",final_bond_sampled)
        sidechain_model_type = getattr(self._conf.model, "sidechain_model_type", None)
        # 如果配置了 sidechain_model_type，则使用深度学习侧链模型精修；
        # 否则退回到基于 BondCoherence + Clash 的梯度下降扭转角优化。
        if sidechain_model_type is not None:
            print("start refine_sidechain (model)-------------------------------")
            B,L = final_seq.shape[:2]
            alpha_tor_mask = torch.zeros(B,L, 10, device=self.device, dtype=torch.bool)
            torsion_angles ,refine_x,refine_x_withCN = self.refine_sidechain(
                final_seq, final_x, final_bond_sampled, fixed_batch_data['rf_idx'], 
                fixed_batch_data['pdb_idx'], alpha_pred, alpha_tor_mask, 
                masks['str_mask'], masks['seq_mask'], masks['head_mask'], 
                masks['tail_mask'], masks['bond_mask'], masks['res_mask'],
                masks.get('N_C_anchor', None)
            )
        else:
            print("start refine_sidechain (gradient descent)---------------------")
            B, L = final_seq.shape[:2]
            start_time = time.time()
            torsion_angles, refine_x, refine_x_withCN = self.refine_sidechain_by_gd(
                final_seq,
                final_x_withCN,
                final_bond_sampled,
                fixed_batch_data['rf_idx'],
                fixed_batch_data['pdb_idx'],
                alpha_pred,
                masks['str_mask'],
                masks['seq_mask'],
                masks['head_mask'],
                masks['tail_mask'],
                masks['bond_mask'],
                masks['res_mask'],
                masks['N_C_anchor'],
            )
            end_time = time.time()
            print(f"Refine sidechain time: {end_time - start_time} seconds")
        # 公共的输出逻辑：不管是 sidechain 模型还是 GD，都统一在这里写精修结果
        self._write_refined_outputs(
            out_pdb_dir=post_refine_dir if out_pdb_dir is not None else ".",
            file_prefix=file_prefix,
            num_batch=num_batch,
            final_seq=final_seq,
            final_bond_sampled=final_bond_sampled,
            refine_x=refine_x,
            refine_x_withCN=refine_x_withCN,
            fixed_batch_data=fixed_batch_data,
            head_mask=head_mask,
            tail_mask=tail_mask,
        )


        if record_trajectory:
            return final_px0, final_x, final_seq, final_bond_sampled, traj
        else:
            return final_px0, final_x, final_seq, final_bond_sampled

    def sample_from_prior(
        self,
        num_batch: int,
        num_res: int,
        *,
        num_timesteps: int = None,
        rf_idx: torch.Tensor = None,
        pdb_idx = None,
        res_mask: torch.Tensor = None,
        str_mask: torch.Tensor = None,
        seq_mask: torch.Tensor = None,
        bond_mask: torch.Tensor = None,
        record_trajectory: bool = True,
        eps_t: float = 5e-4,
        out_pdb_dir: str = None,
        N_C_add = True,
    ):
        """Run a full interpolant-based sampling loop from a simple prior to the final result.

        Returns:
            final_px0: [B, L, 14, 3]
            final_x:   [B, L, 14, 3]
            final_seq: [B, L] (long)
            final_bond: [B, L, L]
            traj (optional): dict of lists for visualization
        """
        device = self.device

        # Defaults for indices and masks
        if rf_idx is None:
            rf_idx = torch.arange(num_res, device=device, dtype=torch.long)[None].repeat(num_batch, 1)
        if pdb_idx is None:
            # Build simple single-chain index list per batch: [("A", i), ...]
            pdb_idx = [[("A", int(i)) for i in range(num_res)] for _ in range(num_batch)]
        if res_mask is None:
            res_mask = torch.ones(num_batch, num_res, dtype=torch.bool, device=device)
        if str_mask is None:
            str_mask = torch.ones(num_batch, num_res, dtype=torch.bool, device=device)
        if seq_mask is None:
            seq_mask = torch.ones(num_batch, num_res, dtype=torch.bool, device=device)
        # if bond_mask is None:
        #     bond_mask = torch.ones(num_batch, num_res, num_res, dtype=torch.bool, device=device)

  

        # Time schedule
        if num_timesteps is None:
            num_timesteps = int(self.interpolant._sample_cfg.num_timesteps)
        t_min = float(self.interpolant._cfg.min_t)
        # Avoid using exactly 1.0 to prevent division by (1 - t)
        t_max = 1.0 - eps_t
        ts = torch.linspace(t_min, t_max, steps=num_timesteps, device=device)

        # Prior initialization
        trans_0 = interpolant_centered_gaussian(num_batch, num_res, device) * du.NM_TO_ANG_SCALE
        rotmats_0 = interpolant_uniform_so3(num_batch, num_res, device)
        x_bb = iu.get_xyz_from_RT(rotmats_0, trans_0)  # [B, L, 3, 3] (N, CA, C)
        # Bond prior: 直接调用 Interpolant 的 ss 先验函数，保持与训练腐蚀阶段一致
        bond_mat_t_1 = self.interpolant._sample_ss_prior(res_mask=res_mask.float())
        # Sequence prior consistent with interpolant aatype scheme
        if getattr(self.interpolant._aatypes_cfg, 'interpolant_type', 'masking') == 'masking':
            seq_t_1 = torch.full((num_batch, num_res), int(du.MASK_TOKEN_INDEX), device=device, dtype=torch.long)
        elif getattr(self.interpolant._aatypes_cfg, 'interpolant_type', 'masking') == 'uniform':
            seq_t_1 = torch.randint(low=0, high=int(getattr(du, 'NUM_TOKENS', 20)), size=(num_batch, num_res), device=device, dtype=torch.long)
        else:
            seq_t_1 = torch.randint(low=0, high=20, size=(num_batch, num_res), device=device, dtype=torch.long)

        # Expand to 14 atoms so downstream torsion computation expects xyz[...,3]
        # Use ideal torsions for all 10 channels: cos=1, sin=0
        alpha0 = torch.zeros(num_batch, num_res, 10, 2, device=device, dtype=torch.float32)
        alpha0[..., 0] = 1.0
        _, x_t_1 = self.allatom(seq_t_1, x_bb, alpha0, bond_mat=bond_mat_t_1, link_csv_path=self.preprocess_conf.link_config, use_H=False)  # [B, L, 14, 3]


        head_mask = torch.zeros(num_batch, num_res, dtype=torch.bool, device=device)
        tail_mask = torch.zeros(num_batch, num_res, dtype=torch.bool, device=device)
        if N_C_add:
            x_t_1 = torch.cat([x_bb[:, :1, :, :], x_bb, x_bb[:, -1:, :, :]], dim=1)
            seq_t_1 = torch.cat([seq_t_1[:, :1], seq_t_1, seq_t_1[:, -1:]], dim=1)
            str_mask = torch.cat([str_mask[:, :1], str_mask, str_mask[:, -1:]], dim=1)
            seq_mask = torch.cat([seq_mask[:, :1], seq_mask, seq_mask[:, -1:]], dim=1)
            res_mask = torch.cat([res_mask[:, :1], res_mask, res_mask[:, -1:]], dim=1)
            rf_idx = torch.cat([rf_idx[:, :1], rf_idx, rf_idx[:, -1:]], dim=1)
            bond_mat_t_1 = self.interpolant._sample_ss_prior(res_mask=res_mask.float())
            for i in range(num_batch):
                pdb_idx[i] = [pdb_idx[i][0]] + pdb_idx[i] + [pdb_idx[i][-1]]
            if bond_mask is None:
                bond_mask = torch.ones(num_batch, num_res+2, num_res+2, dtype=torch.bool, device=device)
              # Diffuse masks per modality
            num_batch,num_res = seq_t_1.shape[:2]
            head_mask = torch.zeros(num_batch, num_res, dtype=torch.bool, device=device)
            tail_mask = torch.zeros(num_batch, num_res, dtype=torch.bool, device=device)
            head_mask[:, 0] = True
            tail_mask[:, -1] = True
            N_C_anchor = torch.zeros(num_batch, num_res, num_res, 2, dtype=torch.bool, device=device)
            N_C_anchor[:, 0, 1, 0] = True
            N_C_anchor[:, -1, -2, 1] = True
            N_C_anchor[:, 1, 0, 0] = True
            N_C_anchor[:, -2, -1, 1] = True
        trans_diffuse_mask = str_mask.float()
        rots_diffuse_mask = str_mask.float()
        aatypes_diffuse_mask = seq_mask.float()
        bond_diffuse_mask = bond_mask.float()

                # Fixed batch data
        fixed_batch_data = {
            'rf_idx': rf_idx,
            'pdb_idx': pdb_idx,
        }
        masks = {
            'str_mask': str_mask,
            'seq_mask': seq_mask,
            'bond_mask': bond_mask,
            'res_mask': res_mask,
            'trans_diffuse_mask': trans_diffuse_mask,
            'rots_diffuse_mask': rots_diffuse_mask,
            'aatypes_diffuse_mask': aatypes_diffuse_mask,
            'bond_diffuse_mask': bond_diffuse_mask,
            'head_mask': head_mask,
            'tail_mask': tail_mask,
            'N_C_anchor': N_C_anchor,
        }
        
        return self._sample_loop(
            x_t_1, seq_t_1, bond_mat_t_1, ts,
            fixed_batch_data, masks,
            num_batch, num_res,
            record_trajectory=record_trajectory,
            eps_t=eps_t,
            out_pdb_dir=out_pdb_dir,
            file_prefix="",
            tqdm_desc="Sampling from prior"
            #ss_1=ss_1,
        )

    def sample_from_partial(
        self,
        xyz_target,
        seq_target,
        ss_target,
        num_batch: int,
        num_res: int,
        N_C_anchor,
        *,
        partial_t: float,  # 开始时间步，0.0表示从头开始，1.0表示从完全噪声开始
        num_timesteps: int = None,
        rf_idx: torch.Tensor = None,
        pdb_idx = None,
        res_mask: torch.Tensor = None,
        str_mask: torch.Tensor = None,
        seq_mask: torch.Tensor = None,
        bond_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        tail_mask: torch.Tensor = None,
        record_trajectory: bool = True,
        eps_t: float = 5e-4,
        out_pdb_dir: str = None,
    ):
        """Run partial diffusion sampling from a specified timestep.

        Args:
            xyz_target (Tensor): (B, L, 3, 3) 目标坐标 (N, CA, C)
            seq_target (Tensor): (B, L) 目标序列
            ss_target (Tensor): (B, L, L) 目标键合矩阵
            num_batch (int): 批次大小
            num_res (int): 残基数量
            partial_t (float): 开始时间步 (0.0-1.0)，0.0表示从头开始，1.0表示从完全噪声开始
            num_timesteps (int): 总时间步数
            rf_idx (Tensor): RF索引
            pdb_idx: PDB索引
            res_mask (Tensor): 残基掩码
            str_mask (Tensor): 结构掩码
            seq_mask (Tensor): 序列掩码
            bond_mask (Tensor): 键掩码
            record_trajectory (bool): 是否记录轨迹
            eps_t (float): 时间步容差
            out_pdb_dir (str): 输出目录

        Returns:
            final_px0: [B, L, 14, 3]
            final_x:   [B, L, 14, 3]
            final_seq: [B, L] (long)
            final_bond: [B, L, L]
            traj (optional): dict of lists for visualization
        """
        device = self.device

        # Defaults for indices and masks
        if rf_idx is None:
            rf_idx = torch.arange(num_res, device=device, dtype=torch.long)[None].repeat(num_batch, 1)
        if pdb_idx is None:
            # Build simple single-chain index list per batch: [("A", i), ...]
            pdb_idx = [[("A", int(i)) for i in range(num_res)] for _ in range(num_batch)]
        if res_mask is None:
            res_mask = torch.ones(num_batch, num_res, dtype=torch.bool, device=device)
        if str_mask is None:
            str_mask = torch.ones(num_batch, num_res, dtype=torch.bool, device=device)
        if seq_mask is None:
            seq_mask = torch.ones(num_batch, num_res, dtype=torch.bool, device=device)
        if bond_mask is None:
            bond_mask = torch.ones(num_batch, num_res, num_res, dtype=torch.bool, device=device)

        # Diffuse masks per modality
        trans_diffuse_mask = str_mask.float()
        rots_diffuse_mask = str_mask.float()
        aatypes_diffuse_mask = seq_mask.float()
        bond_diffuse_mask = bond_mask.float()

        # Time schedule
        if num_timesteps is None:
            num_timesteps = int(self.interpolant._sample_cfg.num_timesteps)
        t_min = float(self.interpolant._cfg.min_t)
        # Avoid using exactly 1.0 to prevent division by (1 - t)
        t_max = 1.0 - eps_t
        
        # 计算从partial_t开始的时间步
        total_timesteps = num_timesteps
        start_timestep = int(partial_t * total_timesteps)
        remaining_timesteps = total_timesteps - start_timestep
        
        # 创建从partial_t到1.0的时间序列
        ts = torch.linspace(partial_t, t_max, steps=remaining_timesteps, device=device)

        # 使用sample_with_interpolant对目标结构进行加噪到partial_t时间步
        print(f"Starting partial diffusion from t={partial_t} (timestep {start_timestep}/{total_timesteps})")
        
        # 确保输入张量的批次维度匹配
        if xyz_target.shape[0] != num_batch:
            xyz_target = xyz_target.repeat(num_batch, 1, 1, 1)
        if seq_target.shape[0] != num_batch:
            seq_target = seq_target.repeat(num_batch, 1)
        if ss_target.shape[0] != num_batch:
            ss_target = ss_target.repeat(num_batch, 1, 1)

        # 对目标结构进行加噪到partial_t时间步
        t_batch = torch.full((num_batch,), partial_t, device=device, dtype=torch.float32)
        xyz_noised, seq_noised, ss_noised, xyz_centered, rotmats = self.sample_with_interpolant(
            xyz_target, seq_target, ss_target, res_mask, str_mask, seq_mask, bond_diffuse_mask, pdb_idx, t_batch,
            head_mask=head_mask, tail_mask=tail_mask
        )

        # 初始化当前状态
        x_t_1_bb = xyz_noised  # [B, L, 3, 3]
        seq_t_1 = seq_noised  # [B, L]
        bond_mat_t_1 = ss_noised  # [B, L, L]

        # 扩展x_t_1到14原子以匹配allatom的期望
        # 使用理想扭转角初始化侧链
        alpha0 = torch.zeros(num_batch, num_res, 10, 2, device=device, dtype=torch.float32)
        alpha0[..., 0] = 1.0
        final_res_mask = res_mask.float() * (1 - head_mask.float()) * (1 - tail_mask.float())
        _, x_t_1 = self.allatom(seq_t_1, x_t_1_bb, alpha0, bond_mat=bond_mat_t_1, res_mask=final_res_mask,
                                   link_csv_path=self.preprocess_conf.link_config, use_H=False)
        
        # 确保形状正确
        if x_t_1.dim() == 5 and x_t_1.shape[1] == 1:
            x_t_1 = x_t_1.squeeze(1)
        x_t_1 = x_t_1[:, :, :14, :]  # [B, L, 14, 3]

        # Fixed batch data
        fixed_batch_data = {
            'rf_idx': rf_idx,
            'pdb_idx': pdb_idx,
        }

        masks = {
            'str_mask': str_mask,
            'seq_mask': seq_mask,
            'bond_mask': bond_mask,
            'res_mask': res_mask,
            'trans_diffuse_mask': trans_diffuse_mask,
            'rots_diffuse_mask': rots_diffuse_mask,
            'aatypes_diffuse_mask': aatypes_diffuse_mask,
            'bond_diffuse_mask': bond_diffuse_mask,
            'head_mask': head_mask,
            'tail_mask': tail_mask,
            'N_C_anchor': N_C_anchor,
        }

        return self._sample_loop(
            x_t_1, seq_t_1, bond_mat_t_1, ts,
            fixed_batch_data, masks,
            num_batch, num_res,
            record_trajectory=record_trajectory,
            eps_t=eps_t,
            out_pdb_dir=out_pdb_dir,
            file_prefix="partial_",
            tqdm_desc=f"Partial diffusion from t={partial_t:.2f}",
            trans_1=xyz_centered[:, :, 1, :],
            rotmats_1=rotmats,
            aatypes_1=seq_target,
            ss_1=ss_target,
        )

    def _save_bond_info(self, out_pdb_dir, batch_idx, bond_matrix, pdb_indices, sequence, coordinates, 
                        include_invalid=True,filename=None,head_mask=None,tail_mask=None):
        """Saves detailed bond information to a CSV file for a single item in a batch.

        Args:
            out_pdb_dir: Output directory.
            batch_idx: Batch index used in filename.
            bond_matrix: [L, L] final sampled bond adjacency (binary/float).
            pdb_indices: List[(chain_id, res_idx)] for each residue.
            sequence: [L] residue types (long, 0-20).
            coordinates: [L, 14, 3] all-atom coordinates for distance calculation.
            include_invalid: If True, also write bonds not present in spec using CA-CA distance.
        """
        if filename is None:
            bond_filename = os.path.join(out_pdb_dir, f"bonds_{batch_idx}.txt")
        else:
            bond_filename = os.path.join(out_pdb_dir, f"{filename}_{batch_idx}.txt")
        with open(bond_filename, 'w') as f:
            f.write(
                "res1_chain,res1_idx,res1_type,atom1_name,"
                "res2_chain,res2_idx,res2_type,atom2_name,"
                "distance,distance_ref,"
                "angle_i_deg,angle_i_ref_deg,"
                "angle_j_deg,angle_j_ref_deg,"
                "dihedral_1_deg,dihedral_1_ref_deg,"
                "dihedral_2_deg,dihedral_2_ref_deg,"
                "is_valid\n"
            )
            links = get_valid_links(
                sequence,
                coordinates,
                bond_matrix,
                self.preprocess_conf.link_config,
                head_mask=head_mask,
                tail_mask=tail_mask,
                include_invalid=include_invalid,
            )
            for link in links:
                res1_idx = int(link['i'])
                res2_idx = int(link['j'])
                pdb_info1 = pdb_indices[res1_idx - 1]
                pdb_info2 = pdb_indices[res2_idx - 1]
                res1_type_str = num2aa[link['res1_num']]
                res2_type_str = num2aa[link['res2_num']]
                atom1_name = link['atom1_name']
                atom2_name = link['atom2_name']
                distance = link.get('distance', None)
                distance_ref = link.get('distance_ref', None)

                # Geometry values are stored in radians; convert to degrees for human readability
                def _to_deg(val):
                    return math.degrees(float(val)) if val is not None else None

                angle_i = _to_deg(link.get('angle_i', None))
                angle_i_ref = _to_deg(link.get('angle_i_ref', None))
                angle_j = _to_deg(link.get('angle_j', None))
                angle_j_ref = _to_deg(link.get('angle_j_ref', None))
                dihedral_1 = _to_deg(link.get('dihedral_1', None))
                dihedral_1_ref = _to_deg(link.get('dihedral_1_ref', None))
                dihedral_2 = _to_deg(link.get('dihedral_2', None))
                dihedral_2_ref = _to_deg(link.get('dihedral_2_ref', None))
                is_valid = bool(link.get('is_valid', False))

                def _fmt(val, fmt_str):
                    return (fmt_str % val) if val is not None else ""

                f.write(
                    f"{pdb_info1[0]},{res1_idx},{res1_type_str},{atom1_name},"
                    f"{pdb_info2[0]},{res2_idx},{res2_type_str},{atom2_name},"
                    f"{_fmt(distance, '%.3f')},"
                    f"{_fmt(distance_ref, '%.3f')},"
                    f"{_fmt(angle_i, '%.2f')},"
                    f"{_fmt(angle_i_ref, '%.2f')},"
                    f"{_fmt(angle_j, '%.2f')},"
                    f"{_fmt(angle_j_ref, '%.2f')},"
                    f"{_fmt(dihedral_1, '%.2f')},"
                    f"{_fmt(dihedral_1_ref, '%.2f')},"
                    f"{_fmt(dihedral_2, '%.2f')},"
                    f"{_fmt(dihedral_2_ref, '%.2f')},"
                    f"{int(is_valid)}\n"
                )