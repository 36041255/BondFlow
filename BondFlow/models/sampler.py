import torch
from omegaconf import DictConfig, OmegaConf
import random
#from rfdiff.kinematics import get_init_xyz, xyz_to_t2d
from rfdiff.chemical import aa2num, num2aa, aa2long
#from rfdiff.util_module import ComputeAllAtomCoords
import BondFlow.data.utils as iu
import BondFlow.data.SM_utlis as smu
from BondFlow.models.adapter import build_design_model
import os, time
import numpy as np
from rfdiff.util import writepdb_multi, writepdb
from BondFlow.data.link_utils import get_valid_links
from tqdm import tqdm
import math

from BondFlow.models.interpolant import Interpolant
from BondFlow.models.interpolant import _centered_gaussian as interpolant_centered_gaussian
from BondFlow.models.interpolant import _uniform_so3 as interpolant_uniform_so3
from multiflow_data import utils as du
from multiflow_data.so3_utils import sample_uniform as so3_sample_uniform
from BondFlow.models.allatom_wrapper import (
    AllAtomWrapper,
    apply_bidirectional_anchor_update,
    apply_head_phi_rotation,
    apply_tail_psi_rotation,
)
from BondFlow.models.guidance import GuidanceManager, build_guidances
from BondFlow.models.Loss import BondCoherenceLoss, OpenFoldClashLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
SCRIPT_DIR=os.getcwd()

class Sampler:

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
        self.device = device
        self._conf = conf
        self.sigma_perturb = self._conf.preprocess.sigma_perturb
        self.rotation_perturb = self._conf.preprocess.rotation_perturb

        model_type = getattr(self._conf.model, 'type', 'apm_backbone')
        self.model = self.load_model(model_type)
        if self._conf.model.sidechain_model_type is not None:
            model_type = self._conf.model.sidechain_model_type
            self.sidechain_model = self.load_model(model_type)

        if self._conf.inference.seed is not None:
            set_reproducible(self._conf.inference.seed)

        # Initialize helper objects
        self.inf_conf = self._conf.inference
        self.design_config = self._conf.design_config
        self.preprocess_conf = self._conf.preprocess
        # Initialize Interpolant
        self.interpolant = Interpolant(self._conf.interpolant,device = self.device)

        # Loss Functions
        self._bond_coherence_loss = BondCoherenceLoss(
            link_csv_path=self.preprocess_conf.link_config, device=device,    
            energy_w_angle=1, energy_w_dihedral=1,#energy_jsd_gamma=1,
        )
        self._openfold_clash_loss = OpenFoldClashLoss(
            link_csv_path=self.preprocess_conf.link_config, device=str(device), log_raw=False,
            debug_print_pairs = True,
        )


        backend = getattr(self._conf.preprocess, 'allatom_backend', 'apm')
        self.allatom = AllAtomWrapper(backend=backend, device=self.device).to(self.device)


        # Guidance (optional)
        try:
            guidance_cfg = getattr(self._conf, 'guidance', None)
        except Exception:
            guidance_cfg = None
        self.guidance_manager = GuidanceManager(build_guidances(guidance_cfg, device=self.device))

   
    def load_model(self,model_type):  
        # 加载 model_config 并修改 ckpt_path（如果已解析为绝对路径）
        model_config_path = self._conf.model.model_config_path
        model_config = OmegaConf.load(model_config_path)
        
        # 如果主配置中已经有解析好的 ckpt_path（绝对路径），使用它覆盖 model.yaml 中的相对路径
        if hasattr(self._conf.model, 'ckpt_path') and self._conf.model.ckpt_path:
            model_config.model.ckpt_path = self._conf.model.ckpt_path
        
        model = build_design_model(model_type, device=self.device, model_config=model_config)
        model = model.eval()

        return model

    def _center_xyz_with_chain_and_hotspots(
        self,
        xyz_target: torch.Tensor,
        res_mask: torch.Tensor,
        str_mask: torch.Tensor,
        chain_ids: torch.Tensor | None = None,
        hotspots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        根据链信息与 hotspot / 固定结构掩码对坐标进行中心化。

        规则：
          - 单链：有固定结构 (str_mask=False) 时，以其 CA 几何中心为原点；否则用整体 CA 中心。
          - 多链：若提供 hotspot，则以 hotspot CA 平均中心为原点；
                  否则用固定结构部分的 CA 中心；
                  若仍不存在，则退化为整体 CA 中心。
        """
        B, L, _, _ = xyz_target.shape
        device = xyz_target.device
        dtype = xyz_target.dtype

        # ---- 1) 判定每个样本是单链还是多链 (batch 张量操作) ----
        if chain_ids is None:
            single_flag = torch.ones(B, 1, dtype=torch.bool, device=device)
        else:
            valid = chain_ids > 0  # 忽略 0 这一类占位 id
            has_valid = valid.any(dim=1, keepdim=True)  # [B,1]

            # 为 masked min/max 构造大/小哨兵值
            big_pos = torch.iinfo(chain_ids.dtype).max
            big_neg = torch.iinfo(chain_ids.dtype).min
            chain_min = torch.where(
                valid, chain_ids, torch.full_like(chain_ids, big_pos)
            ).amin(dim=1, keepdim=True)
            chain_max = torch.where(
                valid, chain_ids, torch.full_like(chain_ids, big_neg)
            ).amax(dim=1, keepdim=True)

            multi_flag = has_valid & (chain_max != chain_min)
            single_flag = ~multi_flag  # [B,1]

        # ---- 2) 单链 / 多链下的中心掩码 (全张量实现) ----
        fixed_mask = (~str_mask.bool()) & res_mask.bool()  # [B,L]
        fixed_any = fixed_mask.any(dim=1, keepdim=True)  # [B,1]

        # 单链情形的中心掩码：优先固定结构，否则整体
        center_single = torch.where(fixed_any, fixed_mask, res_mask.bool())  # [B,L]

        # 多链情形：优先 hotspot，其次固定结构，否则整体
        single_hotspot_center = torch.zeros(B, 1, dtype=torch.bool, device=device)
        if hotspots is not None:
            if hotspots.dtype == torch.bool:
                h_mask = hotspots & res_mask.bool()
            else:
                h_mask = (hotspots > 0) & res_mask.bool()
            h_any = h_mask.any(dim=1, keepdim=True)  # [B,1]
            h_count = h_mask.sum(dim=1, keepdim=True)
            single_hotspot_center = (~single_flag) & h_any & (h_count == 1)
            center_multi = torch.where(h_any, h_mask, center_single)  # [B,L]
        else:
            center_multi = center_single

        # 根据 single_flag 在单链 / 多链规则之间选择
        center_mask = torch.where(single_flag, center_single, center_multi)  # [B,L]

        # 极端情况：某个样本 center_mask 全 False，则退化为“全部残基”
        valid_any = center_mask.any(dim=1, keepdim=True)  # [B,1]
        center_mask = torch.where(
            valid_any, center_mask, torch.ones_like(center_mask)
        )  # [B,L]

        # ---- 3) 基于 center_mask 计算几何中心并平移 ----
        ca_coords = xyz_target[:, :, 1, :]  # [B, L, 3]
        mask_f = center_mask.unsqueeze(-1).to(dtype)  # [B, L, 1]
        denom = mask_f.sum(dim=1, keepdim=True) + 1e-8  # [B, 1, 1]
        center = (ca_coords * mask_f).sum(dim=1, keepdim=True) / denom  # [B, 1, 3]

        # 扩展到原始维度 [B, 1, 1, 3] 以便从所有原子坐标中减去
        xyz_centered = xyz_target - center.unsqueeze(2)

        # If a multi-chain sample is centered on exactly one hotspot residue, that
        # hotspot CA lands exactly at the origin. The PDB writer treats all-zero
        # coordinates as invalid, so add a tiny global offset to preserve the atom.
        if single_hotspot_center.any():
            tiny_shift = torch.tensor([1e-6, 0.0, 0.0], dtype=dtype, device=device)
            xyz_centered = xyz_centered + single_hotspot_center.to(dtype).view(B, 1, 1, 1) * tiny_shift.view(1, 1, 1, 3)

        return xyz_centered

    # def generate_crop_target_pdb(self, pdb_file,chain_id,crop_mode,crop_length=256,fixed_res=None):
    #     pdb_parsed = iu.process_target(pdb_file,
    #                                 parse_hetatom=False, 
    #                                 center=False,
    #                                 parse_link=True)
        
    #     contig, res_mask = iu.generate_crop_contigs(pdb_parsed, chain_id, mode=crop_mode, crop_length=crop_length, fixed_res=fixed_res)
    #     print(contig)
    #     contig_new = self._conf
    #     if crop_mode == 'complex':
    #         contig_new.design_config.bond_condition = ['B|B']
    #     else:
    #         contig_new.design_config.bond_condition = None
    #     contig_new.design_config.contigs = contig
    #     contig_new.design_config.partial_t = 0.1 # no use
    #     target = iu.Target(contig_new.design_config,pdb_parsed)
    #     target.res_mask = res_mask
    #     return target, contig_new

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
        # 验证中心化是否正确
        yuandian = (xyz_centered[:,:,1,:]*res_mask.unsqueeze(-1).float()).sum(dim=1)/ (res_mask.sum(dim=-1, keepdim=True) + 1e-8)
        # print("center of xyz__centered:",yuandian)
        epsilon = torch.randn((B,3), device=xyz_target.device) * self.sigma_perturb
        epsilon = epsilon[:,None,None,:]  # (B, 1, 1, 3)
        
        # 将扰动应用到中心化后的坐标上
        xyz_stochastically_centered = xyz_centered + epsilon
        yuandian = (xyz_stochastically_centered[:,:,1,:]*res_mask.unsqueeze(-1).float()).sum(dim=1)/ (res_mask.sum(dim=-1, keepdim=True) + 1e-8)
        # print("center of xyz_stochastically_centered:",yuandian)
        return xyz_stochastically_centered

    def sample_with_interpolant(
        self,
        xyz_target,
        seq_target,
        ss_target,
        res_mask,
        str_mask,
        seq_mask,
        bond_diffuse_mask,
        hotspots,
        t,
        chain_ids,
        head_mask=None,
        tail_mask=None,
        N_C_anchor=None,
        
        
    ):
        """
        使用 Interpolant 对输入进行加噪或采样。

        Args:
            xyz_target (Tensor): (B, L, 3, 3) 目标坐标 (N, CA, C)。
            seq_target (Tensor): (B, L) 目标序列。
            ss_target (Tensor): (B, L, L) 目标二级结构。
            res_mask (Tensor): (B, L) 氨基酸残基掩码。
            str_mask (Tensor): (B, L) 结构掩码 (True=可扰动, False=固定结构)。
            seq_mask (Tensor): (B, L) 序列掩码。
            bond_diffuse_mask (Tensor): (B, L, L) 二级结构扩散掩码。
            pdb_idx (list): (为了兼容保留, 不再用于识别链)。
            t (Tensor): (B,) 扩散时间步。
            chain_ids (Tensor, optional): (B, L) full_chain_ids, 用于区分单链/多链。
            hotspots (Tensor, optional): (B, L) hotspot 掩码 (>0 视为 True)。

        Returns:
            dict: 包含加噪/采样后数据的字典。
        """
        # 1. 中心化 + 随机旋转 & 平移扰动
        #
        #   - 单链: 若存在固定结构 (str_mask=False), 以其 CA 几何中心为原点; 否则用整体几何中心。
        #   - 多链: 若提供 hotspot, 以 hotspot CA 平均中心为原点; 否则用固定结构部分的 CA 中心;
        #           若均不存在则退化为整体几何中心。
        xyz_centered = self._center_xyz_with_chain_and_hotspots(
            xyz_target, res_mask, str_mask, chain_ids=chain_ids, hotspots=hotspots
        )
        B = xyz_centered.shape[0]
        device = xyz_centered.device
        dtype = xyz_centered.dtype

        # 随机旋转 (batch-wise)
        R = so3_sample_uniform(B).to(device=device, dtype=dtype)  # [B, 3, 3]
        if self.rotation_perturb:
            # R[b] @ xyz_centered[b, l, k, :]
            xyz_rot = torch.einsum("bij,blkj->blki", R, xyz_centered)
        else:
            xyz_rot = xyz_centered

        # 各向同性平移扰动
        epsilon = torch.randn(B, 1, 1, 3, device=device, dtype=dtype) * self.sigma_perturb
        xyz_centered = xyz_rot + epsilon
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
            rotmats = iu.get_R_from_xyz(xyz_centered).to(self.device)
        return xyz_noised, noised_batch['aatypes_t'], noised_batch['ss_t'], xyz_centered, rotmats

 
    def sample_step(self, t_1, t_2, x_t_1, seq_t_1, bond_mat_t_1, fixed_batch_data, masks,
                    trans_sc=None, aatypes_sc=None, torsions_sc=None,
                    trans_1=None, rotmats_1=None, aatypes_1=None, ss_1=None,
                    compute_full_graph=True): # <--- 新增参数
        """
        Performs one step of the sampling process using the interpolant framework.
        
        Args:
            compute_full_graph (bool): If False, skips expensive allatom reconstruction and 
                                       bond permutation sampling. Used for intermediate steps
                                       when trajectory recording is not needed.
        """
        B, L = seq_t_1.shape[:2]
        device = x_t_1.device

        # Skip torsion computation to improve stability at early sampling
        B_local, L_local = seq_t_1.shape[:2]
        alpha = torch.zeros(B_local, L_local, 10, 2, device=device, dtype=torch.float32)
        alpha_tor_mask = torch.zeros(B_local, L_local, 10, device=device, dtype=torch.bool)
        res_mask = masks['res_mask'].to(device)
        head_mask = masks['head_mask'].to(device)
        tail_mask = masks['tail_mask'].to(device)
        bond_mask = masks['bond_diffuse_mask'].to(device)
        N_C_anchor = masks['N_C_anchor'].to(device)
        str_mask=masks['str_mask'].to(device)
        seq_mask=masks['seq_mask'].to(device)
        rf_idx=fixed_batch_data['rf_idx'].to(device)
        hotspots=fixed_batch_data['hotspots'].to(device)
        if aatypes_1 is not None:
            aatypes_1 = aatypes_1.long()
        
        with torch.no_grad():
            # Use unified BaseDesignModel signature (works for both RF and APM wrappers)
            model_out = self.model(
                seq_noised=seq_t_1,
                xyz_noised=x_t_1[..., :14, :],
                bond_noised=bond_mat_t_1,
                rf_idx=rf_idx,
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
                chain_ids=fixed_batch_data['chain_num_ids'],
                use_checkpoint=False,
                trans_sc=trans_sc,
                aatypes_sc=aatypes_sc,
                torsions_sc=torsions_sc,
                trans_1=trans_1,
                rotmats_1= rotmats_1,
                aatypes_1= aatypes_1,
                bond_mat_1=ss_1,
                # Optional PLM / full-structure context for APMBackboneWrapper
                origin_pdb_idx=fixed_batch_data['origin_pdb_idx'],
                pdb_seq_full=fixed_batch_data['pdb_seq_full'],
                pdb_idx_full=fixed_batch_data['pdb_idx_full'],
                pdb_core_id=fixed_batch_data['pdb_core_id'],
                hotspots=hotspots,
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
            alpha_pred=alpha_pred,
            allatom=self.allatom,
        )
        logits = model_raw.get('logits', logits)
        px0_bb = model_raw.get('px0_bb', px0_bb)
        alpha_pred = model_raw.get('alpha_pred', alpha_pred)
        bond_mat_pred = model_raw.get('bond_mat_pred', bond_mat_pred)

        # All-atom prediction for x0
        res_mask_2d = res_mask.unsqueeze(1) * res_mask.unsqueeze(2)
        
        pseq0 = torch.argmax(logits[...,:20], dim=-1)
        final_res_mask = res_mask.float() * (1-head_mask.float()) * (1-tail_mask.float())

        # ==============================================================================
        # OPTIMIZATION: Skip expensive AllAtom and Permutation Sampling if not required
        # ==============================================================================
        if compute_full_graph:
            bond_mat_pred_sampled = smu.sample_permutation(bond_mat_pred, res_mask_2d)
            if aatypes_1 is not None:
                pseq0 = pseq0 * seq_mask.float() + (1 - seq_mask.float()) * aatypes_1
            
            # 处理 Inpainting/Masking (Full Atom)
            if trans_1 is not None and rotmats_1 is not None:
                xyz_1 = iu.get_xyz_from_RT(rotmats_1, trans_1)
                px0_bb_masked = px0_bb * str_mask[...,None,None].float() + ( 1 - str_mask[...,None,None].float()) * xyz_1
            else:
                px0_bb_masked = px0_bb

            _, px0  = self.allatom(pseq0, px0_bb_masked, alpha_pred, 
                                bond_mat=bond_mat_pred_sampled, 
                                link_csv_path=self.preprocess_conf.link_config, use_H=False,
                                res_mask=final_res_mask,
                                head_mask=head_mask,
                                tail_mask=tail_mask,
                                N_C_anchor=N_C_anchor,)

        else:
            # FAST PATH: Construct minimal px0 (N, CA, C) and fill rest with NaN
            # This is sufficient for self-conditioning (which uses CA) and next step t2d
            bond_mat_pred_sampled = bond_mat_pred # No sampling
            
            # Construct placeholder [B, L, 14, 3]
            px0 = torch.full((B, L, 14, 3), float('nan'), device=device)
            
            # Apply Inpainting/Masking manually to backbone
            if trans_1 is not None and rotmats_1 is not None:
                xyz_1 = iu.get_xyz_from_RT(rotmats_1, trans_1) # [B, L, 3, 3]
                # Apply mask to the N, CA, C coordinates
                px0_bb_masked = px0_bb * str_mask[...,None,None].float() + \
                                (1 - str_mask[...,None,None].float()) * xyz_1
            else:
                px0_bb_masked = px0_bb

            px0[:, :, :3, :] = px0_bb_masked # Fill backbone

        
        # Normalize px0 shape to [B, L, 14, 3]
        if px0.dim() == 5 and px0.shape[1] == 1:
            px0 = px0.squeeze(1)
        px0 = px0[:, :, :14, :]

        # Prepare model_out for interpolant
        model_out_interpolant = {
            'pred_trans': px0[:, :, 1, :].nan_to_num(), # Safe: CA exists
            'pred_rotmats': iu.get_R_from_xyz(px0.nan_to_num()), # Safe: N,CA,C exist
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
            alpha_pred=alpha_pred,
            allatom=self.allatom,
        )
        trans_t_2 = step_out.get('trans_t_2', trans_t_2)
        rotmats_t_2 = step_out.get('rotmats_t_2', rotmats_t_2)
        aatypes_t_2 = step_out.get('aatypes_t_2', aatypes_t_2)
        ss_t_2 = step_out.get('ss_t_2', ss_t_2)

        # Build new backbone and full atom structure
        x_t_2_bb = iu.get_xyz_from_RT(rotmats_t_2, trans_t_2)
        # Ensure integer indices for sequence
        aatypes_t_2 = aatypes_t_2.to(torch.long)
        
        # ==============================================================================
        # OPTIMIZATION: Skip expensive AllAtom and Permutation Sampling for Next Step
        # ==============================================================================
        if compute_full_graph:
            bond_mat_t_2_sampled = smu.sample_permutation(ss_t_2, res_mask_2d)
            _, x_t_2 = self.allatom(aatypes_t_2, x_t_2_bb, alpha_pred, bond_mat=bond_mat_t_2_sampled, 
                                    link_csv_path=self.preprocess_conf.link_config, use_H=False, 
                                    res_mask=final_res_mask,
                                    head_mask=head_mask,
                                    tail_mask=tail_mask,
                                    N_C_anchor=N_C_anchor,)
        else:
            # FAST PATH
            bond_mat_t_2_sampled = ss_t_2 # No sampling needed for next iteration's input
            x_t_2 = torch.full((B, L, 14, 3), float('nan'), device=device)
            x_t_2[:, :, :3, :] = x_t_2_bb # Fill backbone (N, CA, C)

        # Normalize x_t_2 shape to [B, L, 14, 3]
        if x_t_2.dim() == 5 and x_t_2.shape[1] == 1:
            x_t_2 = x_t_2.squeeze(1)
        x_t_2 = x_t_2[:, :, :14, :]
        
        # Prepare self-conditioning tensors for the next step
        # Note: px0 contains NaNs in fast mode, but index 1 (CA) is valid, which is what we need.
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

        # Handle Anchor Updates
        # Note: Even in fast mode with NaNs, N (0), CA (1), C (2) are present, 
        # so updating their coords via anchors is valid and necessary.
        if (head_mask is not None or tail_mask is not None) and (N_C_anchor is not None):
            px0 = iu.update_nc_node_coordinates(px0, N_C_anchor, head_mask, tail_mask)
            x_t_2 = iu.update_nc_node_coordinates(x_t_2, N_C_anchor, head_mask, tail_mask)
            aatypes_t_2 = iu.update_nc_node_features(aatypes_t_2, N_C_anchor, head_mask, tail_mask)


           
        # fix mask part
        if aatypes_1 is not None:
            aatypes_t_2 = aatypes_t_2 * seq_mask.float() + (1 - seq_mask.float()) * aatypes_1
            aatypes_t_2 = aatypes_t_2.to(torch.long)

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
    def _build_full_chain_structure(
        self,
        design_xyz: torch.Tensor,
        design_seq: torch.Tensor,
        origin_pdb_idx: list,
        pdb_parsed: dict = None,
        pdb_idx_full: list = None,
        pdb_seq_full = None,
        design_pdb_idx: list = None,  
        head_mask: torch.Tensor = None,  
        tail_mask: torch.Tensor = None, 
    ):
        """
        构建完整链结构 (修复版：基于 Chain 边界自动剔除虚拟节点)
        """
        if pdb_parsed is None or pdb_idx_full is None:
            # 如果没有原始PDB或完整链信息，返回设计区域本身
            return design_xyz, design_seq, origin_pdb_idx, head_mask, tail_mask
        
        L_full = len(pdb_idx_full)
        
        # 1. 获取原始PDB信息
        if isinstance(pdb_parsed["xyz_14"], torch.Tensor):
            orig_xyz = pdb_parsed["xyz_14"].clone()
        else:
            orig_xyz = torch.from_numpy(pdb_parsed["xyz_14"]) 
        
        orig_pdb_idx = pdb_parsed["pdb_idx"] 
        orig_idx_to_pos = {idx: pos for pos, idx in enumerate(orig_pdb_idx)}
        
        # 2. 初始化完整链张量
        full_xyz = torch.full((L_full, 14, 3), float('nan'), dtype=design_xyz.dtype, device=design_xyz.device)
        
        if isinstance(pdb_seq_full, torch.Tensor):
            full_seq = pdb_seq_full.clone()
        elif isinstance(pdb_seq_full, np.ndarray):
            full_seq = torch.from_numpy(pdb_seq_full)
        else:
            full_seq = torch.tensor(pdb_seq_full, dtype=torch.long)
            
        full_head_mask = torch.zeros(L_full, dtype=torch.bool, device=design_xyz.device)
        full_tail_mask = torch.zeros(L_full, dtype=torch.bool, device=design_xyz.device)

        # 3. 计算 offset (保持不变)
        design_ca_coords = design_xyz[:, 1, :] 
        offset_candidates = []
        for design_idx, origin_idx in enumerate(origin_pdb_idx):
            if origin_idx != ('?', '-1') and origin_idx in orig_idx_to_pos:
                orig_pos = orig_idx_to_pos[origin_idx]
                orig_ca = orig_xyz[orig_pos, 1, :].to(design_xyz.device)
                design_ca = design_ca_coords[design_idx]
                if not torch.isnan(design_ca).any() and not torch.isnan(orig_ca).any():
                    offset_candidates.append(design_ca - orig_ca)
        
        if offset_candidates:
            design_offset = torch.stack(offset_candidates).mean(dim=0)
        else:
            design_offset = torch.zeros(3, dtype=design_xyz.dtype, device=design_xyz.device)

        # ==================== 核心修复逻辑开始 ====================
        # 4. 识别真实节点 (Real Nodes)
        # 逻辑：在 design_pdb_idx 中，连续相同的 Chain ID 组成一个块。
        # 每个块的 第一个 (Head) 和 最后一个 (Tail) 都是虚拟节点。
        
        is_real_node = [False] * len(design_xyz)
        
        if design_pdb_idx is not None:
            current_chain = None
            chain_start_idx = 0
            
            for i, pdb_idx_item in enumerate(design_pdb_idx):
                chain_id = pdb_idx_item[0] # 获取 Chain ID (例如 'A', 'B', '?')
                
                if chain_id != current_chain:
                    # 链发生了变化 (或刚开始)
                    if current_chain is not None:
                        # 上一条链结束了，i-1 是上一条链的 Tail (虚拟)
                        is_real_node[i-1] = False
                    
                    # 当前是新链的开始，i 是新链的 Head (虚拟)
                    is_real_node[i] = False
                    
                    current_chain = chain_id
                else:
                    # 链中间的残基，暂定为 Real
                    is_real_node[i] = True
            
            # 处理最后一条链的 Tail
            if len(design_pdb_idx) > 0:
                is_real_node[-1] = False
        
        # 5. 构建映射，仅使用 Real Nodes
        # design_pdb_idx_to_design_idx = {}
        # if design_pdb_idx is not None:
        #     for design_idx, pdb_idx_item in enumerate(design_pdb_idx):
        #         if design_idx < len(design_xyz):
        #             if is_real_node[design_idx]:
        #                 # 只有真实节点才进入映射表，这样 Tail 就不会覆盖 Real
        #                 design_pdb_idx_to_design_idx[pdb_idx_item] = design_idx

        # # 6. 填充 PDB
        # # 创建一个只包含 Real Nodes 索引的列表，用于顺序填充 New_ Residues
        # real_design_indices = [i for i, is_real in enumerate(is_real_node) if is_real]
        # real_idx_ptr = 0 # 指向 real_design_indices 的指针

        # for full_idx, full_pdb_idx_item in enumerate(pdb_idx_full):
        #     mapped_idx = -1
            
        #     # 情况 A: 可以通过 PDB ID 直接匹配
        #     if full_pdb_idx_item in design_pdb_idx_to_design_idx:
        #         mapped_idx = design_pdb_idx_to_design_idx[full_pdb_idx_item]
                
        #     # 情况 B: 原始 PDB 结构 (固定且不在设计区)
        #     elif full_pdb_idx_item in orig_idx_to_pos:
        # 5. 构建映射，仅使用 Real Nodes
        # [修复]: 使用 origin_pdb_idx (包含真实的 ID 如 'B/79' 或 'NEW1') 替代 design_pdb_idx (重编号 ID)
        # 这样可以确保 pdb_idx_full 中的 ID 能正确匹配到设计区域的索引，避免 Case B 和 Case C 的指针冲突
        design_pdb_idx_to_design_idx = {}
        if origin_pdb_idx is not None:
            for design_idx, origin_item in enumerate(origin_pdb_idx):
                if design_idx < len(design_xyz):
                    if is_real_node[design_idx]:
                        # 只有真实节点才进入映射表
                        # 注意：如果 origin_item 重复(如N/C端add)，is_real_node 会过滤掉虚拟的那个
                        design_pdb_idx_to_design_idx[origin_item] = design_idx
        
        # 保留基于 design_pdb_idx 的映射作为备选（兼容旧逻辑，防止某些特殊情况 origin 为空）
        if design_pdb_idx is not None and not design_pdb_idx_to_design_idx:
             for design_idx, pdb_idx_item in enumerate(design_pdb_idx):
                if design_idx < len(design_xyz):
                    if is_real_node[design_idx]:
                        design_pdb_idx_to_design_idx[pdb_idx_item] = design_idx

        # 6. 填充 PDB
        # 创建一个只包含 Real Nodes 索引的列表，用于顺序填充 New_ Residues
        real_design_indices = [i for i, is_real in enumerate(is_real_node) if is_real]
        real_idx_ptr = 0 # 指向 real_design_indices 的指针

        for full_idx, full_pdb_idx_item in enumerate(pdb_idx_full):
            mapped_idx = -1
            
            # 情况 A: 可以通过 PDB ID 直接匹配 (现在 Fixed Residues 和 New Residues 都能在这里匹配)
            if full_pdb_idx_item in design_pdb_idx_to_design_idx:
                mapped_idx = design_pdb_idx_to_design_idx[full_pdb_idx_item]
                
            # 情况 B: 原始 PDB 结构 (固定且不在设计区)
            elif full_pdb_idx_item in orig_idx_to_pos:
                orig_pos = orig_idx_to_pos[full_pdb_idx_item]
                full_xyz[full_idx] = orig_xyz[orig_pos].to(design_xyz.device)
                continue # 已填充，跳过后续
            
            # 情况 C: 无法匹配 ID (通常是 New_ 生成的残基)，使用顺序填充
            else:
                # 尝试通过 origin_pdb_idx 验证是否为 New Residue
                # 这里简化逻辑：如果它不在 Origin PDB 里，且我们需要填坑，就从 real_design_indices 里取下一个
                if real_idx_ptr < len(real_design_indices):
                    # 检查当前指针指向的残基是否真的是 New Residue?
                    # 这一步比较难精确判断，但通常顺序是对齐的。
                    # 我们检查 design_pdb_idx 里的 ID 是否也是未知的/特殊的
                    d_idx = real_design_indices[real_idx_ptr]
                    d_pdb = design_pdb_idx[d_idx]
                    
                    # 如果 design 中的 ID 也是无法识别的 (例如 '?', '-1') 或者与目标 full_pdb_idx_item 对应
                    # 这里假设 New_ Residue 都是按顺序排列的
                    mapped_idx = d_idx
                    real_idx_ptr += 1 # 消耗一个真实残基

            # 执行数据填充
            if mapped_idx != -1 and mapped_idx < len(design_xyz):
                full_xyz[full_idx] = design_xyz[mapped_idx] - design_offset.unsqueeze(0).unsqueeze(0)
                full_seq[full_idx] = design_seq[mapped_idx]
                
                # 迁移 Mask (虽然现在可能不需要了，但为了兼容性保留)
                # 注意：这里需要 1D Mask
                head_mask_1d = head_mask[0] if head_mask is not None and head_mask.dim() > 1 else head_mask
                tail_mask_1d = tail_mask[0] if tail_mask is not None and tail_mask.dim() > 1 else tail_mask
                
                if head_mask_1d is not None and mapped_idx < len(head_mask_1d):
                    full_head_mask[full_idx] = head_mask_1d[mapped_idx]
                if tail_mask_1d is not None and mapped_idx < len(tail_mask_1d):
                    full_tail_mask[full_idx] = tail_mask_1d[mapped_idx]

        # ==================== 核心修复逻辑结束 ====================

        print(f"WriteFullChains: Real nodes identified: {sum(is_real_node)}/{len(design_xyz)}")
        return full_xyz, full_seq, pdb_idx_full, full_head_mask, full_tail_mask


    def _write_pdb_structure(
        self,
        output_dir: str,
        file_prefix: str,
        batch_idx: int,
        start_index: int,
        final_seq: torch.Tensor,
        final_bond_sampled: torch.Tensor,
        xyz_coords: torch.Tensor,  # 设计区域的坐标
        xyz_coords_withCN: torch.Tensor,  # 包含虚拟节点的坐标（用于bond_info）
        fixed_batch_data: dict,
        head_mask: torch.Tensor,
        tail_mask: torch.Tensor,
        nc_anchor: torch.Tensor = None,
        pdb_parsed: dict = None,
        bond_info_filename: str = "bonds",
    ):
        """通用的PDB结构写入方法，处理完整链和设计区域两种模式。
        
        Args:
            output_dir: 输出目录
            file_prefix: 文件名前缀
            batch_idx: 批次索引
            start_index: 起始索引
            final_seq: 最终序列 [B, L]
            final_bond_sampled: 最终键矩阵 [B, L, L]
            xyz_coords: 坐标（设计区域模式使用）[B, L, 14, 3]
            xyz_coords_withCN: 坐标（包含虚拟节点，用于bond_info）[B, L, 14, 3]
            fixed_batch_data: 固定批次数据
            head_mask: 头部掩码
            tail_mask: 尾部掩码
            nc_anchor: N/C锚点
            pdb_parsed: 解析的PDB数据
            bond_info_filename: bond信息文件名前缀
        """
        out_i = int(start_index) + int(batch_idx)
        filename = os.path.join(output_dir, f"{file_prefix}structure_{out_i}.pdb")
        pdb_idx = fixed_batch_data["pdb_idx"]
        
        # 检查是否需要写入完整链
        write_full_chains = getattr(self.design_config, 'write_full_chains', False)
        
        if write_full_chains and pdb_parsed is not None:
            # 写入完整链
            origin_pdb_idx = fixed_batch_data.get('origin_pdb_idx', [pdb_idx[batch_idx]])[batch_idx]
            pdb_idx_full_batch = fixed_batch_data.get('pdb_idx_full', [pdb_idx[batch_idx]])[batch_idx]
            pdb_seq_full_batch = fixed_batch_data.get('pdb_seq_full', [final_seq[batch_idx].cpu()])[batch_idx]
            design_pdb_idx = pdb_idx[batch_idx]  # 设计区域的pdb_idx
            # 获取设计区域的head_mask和tail_mask
            head_mask_batch = head_mask[batch_idx].cpu() if head_mask is not None else None
            tail_mask_batch = tail_mask[batch_idx].cpu() if tail_mask is not None else None
            full_xyz, full_seq, full_pdb_idx, full_head_mask, full_tail_mask = self._build_full_chain_structure(
                xyz_coords_withCN[batch_idx].cpu(),
                final_seq[batch_idx].cpu(),
                origin_pdb_idx,
                pdb_parsed,
                pdb_idx_full=pdb_idx_full_batch,
                pdb_seq_full=pdb_seq_full_batch,
                design_pdb_idx=design_pdb_idx,
                head_mask=head_mask_batch,
                tail_mask=tail_mask_batch,
            )
            # 过滤虚拟节点：创建掩码，True表示保留，False表示过滤
            keep_mask = torch.ones(len(full_pdb_idx), dtype=torch.bool)
            if full_head_mask is not None:
                keep_mask = keep_mask & (~full_head_mask.bool())
            if full_tail_mask is not None:
                keep_mask = keep_mask & (~full_tail_mask.bool())
            
            # 应用掩码，过滤掉虚拟节点
            filtered_xyz = full_xyz[keep_mask]
            filtered_seq = full_seq[keep_mask]
            filtered_pdb_idx = [full_pdb_idx[i] for i in range(len(full_pdb_idx)) if keep_mask[i]]
            
            # 为每个链重新编号残基（从1开始）
            # 收集所有唯一的链ID（按照在filtered_pdb_idx中出现的顺序）
            unique_chains = []
            chain_id_to_new_id = {}
            for res in filtered_pdb_idx:
                chain_id = res[0]
                if chain_id not in chain_id_to_new_id:
                    # 按照出现的顺序分配新的链ID
                    new_chain_id = chr(ord('A') + len(unique_chains))
                    chain_id_to_new_id[chain_id] = new_chain_id
                    unique_chains.append(chain_id)
            
            # 为每个链重新编号残基（从1开始）
            res_pdb_idx = []
            chain_pdb_idx = []
            chain_res_counters = {chain_id: 0 for chain_id in unique_chains}
            
            for res in filtered_pdb_idx:
                chain_id = res[0]
                new_chain_id = chain_id_to_new_id[chain_id]
                chain_res_counters[chain_id] += 1
                chain_pdb_idx.append(new_chain_id)
                res_pdb_idx.append(chain_res_counters[chain_id])
            
            # 构建完整链的bond_mat
            full_bond_mat = torch.zeros(len(full_pdb_idx), len(full_pdb_idx), 
                                       device=final_bond_sampled.device, 
                                       dtype=final_bond_sampled.dtype)
            origin_to_design_idx = {}
            for design_idx, origin_idx in enumerate(origin_pdb_idx):
                if origin_idx != ('?', '-1'):
                    origin_to_design_idx[origin_idx] = design_idx
            
            for orig_i, orig_idx_i in enumerate(full_pdb_idx):
                if orig_idx_i not in origin_to_design_idx:
                    continue
                design_i = origin_to_design_idx[orig_idx_i]
                for orig_j, orig_idx_j in enumerate(full_pdb_idx):
                    if orig_idx_j not in origin_to_design_idx:
                        continue
                    design_j = origin_to_design_idx[orig_idx_j]
                    full_bond_mat[orig_i, orig_j] = final_bond_sampled[batch_idx, design_i, design_j]
            
            # 过滤bond_mat，只保留非虚拟节点的键
            filtered_bond_mat = full_bond_mat[keep_mask][:, keep_mask]
            writepdb(
                filename,
                filtered_xyz,
                filtered_seq,
                idx_pdb=res_pdb_idx,
                chain_idx=chain_pdb_idx,
                bond_mat=filtered_bond_mat,
                link_csv_path=self.preprocess_conf.link_config,
                head_mask=None,  # 已经过滤了虚拟节点，不需要再传递掩码
                tail_mask=None,
                nc_anchor=None,
            )
        else:
            # 原有逻辑：只写入设计区域
            res_pdb_idx = [int(res[1]) for res in pdb_idx[batch_idx]]
            chain_pdb_idx = [res[0] for res in pdb_idx[batch_idx]]
            writepdb(
                filename,
                xyz_coords[batch_idx],
                final_seq[batch_idx],
                idx_pdb=res_pdb_idx,
                chain_idx=chain_pdb_idx,
                bond_mat=final_bond_sampled[batch_idx],
                link_csv_path=self.preprocess_conf.link_config,
                head_mask=head_mask[batch_idx] if head_mask is not None else None,
                tail_mask=tail_mask[batch_idx] if tail_mask is not None else None,
                nc_anchor=nc_anchor[batch_idx] if nc_anchor is not None else None,
            )
        
        # 保存bond信息
        self._save_bond_info(
            output_dir,
            out_i,
            final_bond_sampled[batch_idx],
            pdb_idx[batch_idx],
            final_seq[batch_idx],
            xyz_coords_withCN[batch_idx],
            include_invalid=True,
            filename=bond_info_filename,
            head_mask=head_mask[batch_idx] if head_mask is not None else None,
            tail_mask=tail_mask[batch_idx] if tail_mask is not None else None,
        )

    def _write_refined_outputs(
        self,
        out_pdb_dir: str,
        file_prefix: str,
        num_batch: int,
        start_index: int,
        final_seq: torch.Tensor,
        final_bond_sampled: torch.Tensor,
        refine_x: torch.Tensor,
        refine_x_withCN: torch.Tensor,
        fixed_batch_data: dict,
        head_mask: torch.Tensor,
        tail_mask: torch.Tensor,
        nc_anchor: torch.Tensor = None,
        pdb_parsed: dict = None,
    ):
        """统一写出精修后的 PDB 与键信息。"""
        post_refine_dir = out_pdb_dir if out_pdb_dir is not None else "."
        
        for b in range(num_batch):
            self._write_pdb_structure(
                output_dir=post_refine_dir,
                file_prefix=f"{file_prefix}final_refined_",
                batch_idx=b,
                start_index=start_index,
                final_seq=final_seq,
                final_bond_sampled=final_bond_sampled,
                xyz_coords=refine_x,
                xyz_coords_withCN=refine_x_withCN,
                fixed_batch_data=fixed_batch_data,
                head_mask=head_mask,
                tail_mask=tail_mask,
                nc_anchor=nc_anchor,
                pdb_parsed=pdb_parsed,
                bond_info_filename="bonds_final_refined_structure",
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
        chain_ids,
        num_steps: int = 100,
        lr: float = 1e-1,
        lr_min: float = 1e-2,
        w_bond: float = 1,
        w_clash: float = 0.5,
    ):
        """
        Gradient descent optimization with Bi-Directional backbone updates AND explicit O-atom rotation.
        
        New Logic:
        1. Identify Body Anchors for Head/Tail.
        2. Optimize 'anchor_delta' to rotate Body N (Head) or C (Tail).
        3. Overwrite allatom-generated O coordinates with these optimized positions.
        """
        device = self.device
        import BondFlow.data.utils as iu

        # Detach inputs
        seq = seq.detach().to(device)
        x = x.detach().to(device)
        bond_mat = bond_mat.detach().to(device)
        alpha_init = alpha_init.detach().to(device)
        
        # Ensure masks are boolean
        head_mask = head_mask.bool().to(device)
        tail_mask = tail_mask.bool().to(device)
        res_mask = res_mask.to(device)
        final_res_mask = res_mask.float() * (~head_mask).float() * (~tail_mask).float()

        # --- [Logic] Map Virtual Masks to Body Anchor Masks ---
        # body_head_mask: The residue that the Virtual N-term is attached to.
        # body_tail_mask: The residue that the Virtual C-term is attached to.
        body_head_mask = torch.matmul(head_mask.unsqueeze(1).float(), N_C_anchor[..., 0].float()).squeeze(1).bool()
        body_tail_mask = torch.matmul(tail_mask.unsqueeze(1).float(), N_C_anchor[..., 1].float()).squeeze(1).bool()
        # 1. Setup Optimization Variables
        # Sidechain Chi
        chi_init = alpha_init[:, :, 3:7, :]
        chi_angles = torch.atan2(chi_init[..., 1], chi_init[..., 0]).detach().requires_grad_(True)

        # Backbone N/C Rotation (Rigid body movement of N or C)
        anchor_delta = torch.zeros(seq.shape, device=device, dtype=torch.float32, requires_grad=True)

        # [New] Phi/psi chain-level rotations around Head/Tail body anchors
        # phi_prev_delta: controls C(i-1)-N(i)-CA(i)-C(i) by rotating upstream (<= i-1) around N(i)->CA(i)
        # psi_next_delta: controls N(j)-CA(j)-C(j)-N(j+1) by rotating downstream (>= j+1) around CA(j)->C(j)
        phi_prev_delta = torch.zeros(seq.shape, device=device, dtype=torch.float32, requires_grad=True)
        psi_next_delta = torch.zeros(seq.shape, device=device, dtype=torch.float32, requires_grad=True)

        # 2. Setup Optimizer
        optimizer = torch.optim.Adam(
            [chi_angles, anchor_delta, phi_prev_delta, psi_next_delta],
            lr=lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_steps), eta_min=lr_min)



        # 3. Optimization Loop
        with torch.enable_grad():  
            for it in range(num_steps):
                optimizer.zero_grad()
                # A. Prepare Torsions
                chi_sincos = torch.stack([torch.cos(chi_angles), torch.sin(chi_angles)], dim=-1)
                alpha_opt = alpha_init.clone()
                alpha_opt[:, :, 3:7, :] = chi_sincos

                # B. Backbone Update (N/C positions + chain-level phi/psi)
                backbone_bb = x[..., :3, :]
                # 1) Local N/C rotation on body anchors
                backbone_bb = apply_bidirectional_anchor_update(
                    backbone_bb, anchor_delta, body_head_mask, body_tail_mask
                )

                backbone_bb = apply_head_phi_rotation(
                    backbone_bb, phi_prev_delta, body_head_mask, chain_ids
                )

                # 3) Chain-level downstream psi(j) rotation
                # 直接调用修改后的原函数，传入 chain_ids
                backbone_bb = apply_tail_psi_rotation(
                    backbone_bb, psi_next_delta, body_tail_mask, chain_ids
                )

                # C. All-Atom Generation
                _, all_atom_coords_withCN = self.allatom(
                    seq,
                    backbone_bb, 
                    alpha_opt,
                    bond_mat=bond_mat,
                    link_csv_path=self.preprocess_conf.link_config,
                    use_H=False,
                    res_mask=res_mask,
                    head_mask=head_mask,
                    tail_mask=tail_mask,
                    N_C_anchor=N_C_anchor,
                )

                # D. Sync Virtual Nodes
                # Update Head/Tail virtual node coordinates to match the modified Body anchors
                all_atom_coords_calc = iu.update_nc_node_coordinates(
                    all_atom_coords_withCN, N_C_anchor, head_mask, tail_mask, apply_offset=False
                )

                # E. Loss Calculation
                loss_bond = self._bond_coherence_loss(
                    N_C_anchor,
                    bond_matrix=bond_mat,
                    res_mask=res_mask,
                    all_atom_coords=all_atom_coords_calc, 
                    aatype=seq,
                    head_mask=head_mask,
                    tail_mask=tail_mask,
                    silent=True,
                )
                
                loss_clash = self._openfold_clash_loss(
                    N_C_anchor,
                    allatom_xyz=all_atom_coords_calc,
                    seq_pred=seq,
                    res_mask=final_res_mask, 
                    bond_mat=bond_mat,
                    head_mask=head_mask,
                    tail_mask=tail_mask,
                )

                loss = w_bond * loss_bond + w_clash * loss_clash
                loss.backward()

                if it % 5 == 0:
                    print(f"Step {it}: Loss={loss.item():.4f} (Bond={loss_bond.item():.4f}, Clash={loss_clash.item():.4f})")
                
                optimizer.step()
                scheduler.step()

        # 4. Final Reconstruction (No Grad)
        with torch.no_grad():
            chi_final = chi_angles.detach()
            chi_sincos = torch.stack([torch.cos(chi_final), torch.sin(chi_final)], dim=-1)
            torsion_angles = alpha_init.clone()
            torsion_angles[:, :, 3:7, :] = chi_sincos

            backbone_bb_final = x[..., :3, :]
            backbone_bb_final = apply_bidirectional_anchor_update(
                backbone_bb_final, anchor_delta.detach(), body_head_mask, body_tail_mask
            )
            backbone_bb_final = apply_head_phi_rotation(
                backbone_bb_final, phi_prev_delta.detach(), body_head_mask, chain_ids
            )
            backbone_bb_final = apply_tail_psi_rotation(
                backbone_bb_final, psi_next_delta.detach(), body_tail_mask, chain_ids
            )
            
            # Rebuild and Re-rotate
            _, all_atom_coords_final = self.allatom(
                seq, backbone_bb_final, torsion_angles, bond_mat=bond_mat,
                link_csv_path=self.preprocess_conf.link_config, 
                use_H=False, res_mask=final_res_mask,
                head_mask=head_mask,
                tail_mask=tail_mask,
                N_C_anchor=N_C_anchor,
            )
            
            # Final Sync
            all_atom_coords_withNC_final = iu.update_nc_node_coordinates(
                all_atom_coords_final, N_C_anchor, head_mask, tail_mask, apply_offset=False
            )

        return torsion_angles.detach(), all_atom_coords_final.detach(), all_atom_coords_withNC_final.detach()

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
        start_index: int = 0,
        tqdm_desc: str = "Sampling progress",
        trans_1: torch.Tensor = None,
        rotmats_1: torch.Tensor = None,
        aatypes_1: torch.Tensor = None,
        ss_1: torch.Tensor = None,
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
        N_C_anchor = masks['N_C_anchor']
        final_res_mask = res_mask.float() * (1-head_mask.float()) * (1-tail_mask.float())
        
        # Trajectory recording
        traj = {
            'px0': [], 'x': [], 'seq': [], 'bond': [],
        }
        
        final_px0 = None
        t_1 = ts[0]

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
                compute_full_graph=record_trajectory
            )


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
            head_mask=head_mask,
            tail_mask=tail_mask,
            N_C_anchor=N_C_anchor,
        )
        _, final_x_withCN = self.allatom(
            final_seq, x_t_1[..., :3, :], alpha_pred, bond_mat=final_bond_sampled,
            link_csv_path=self.preprocess_conf.link_config, use_H=False,
            res_mask=res_mask,
            head_mask=head_mask,
            tail_mask=tail_mask,
            N_C_anchor=N_C_anchor,
        )
        final_x_withCN = iu.update_nc_node_coordinates(final_x_withCN, N_C_anchor, head_mask, tail_mask,apply_offset=False)


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
            
            pdb_parsed = fixed_batch_data.get('pdb_parsed', None)
            
            for b in range(num_batch):
                # 使用通用方法写入pre-refine结构
                self._write_pdb_structure(
                    output_dir=pre_refine_dir,
                    file_prefix=f"{file_prefix}final_",
                    batch_idx=b,
                    start_index=start_index,
                    final_seq=final_seq,
                    final_bond_sampled=final_bond_sampled,
                    xyz_coords=final_x_withCN,
                    xyz_coords_withCN=final_x_withCN,
                    fixed_batch_data=fixed_batch_data,
                    head_mask=head_mask,
                    tail_mask=tail_mask,
                    nc_anchor=masks.get('N_C_anchor', None),
                    pdb_parsed=pdb_parsed,
                    bond_info_filename="bonds",
                )
                
                # 写入轨迹（仅pre-refine阶段）
                if record_trajectory and traj['x'] is not None:
                    out_i = int(start_index) + int(b)
                    pdb_idx = fixed_batch_data['pdb_idx']
                    traj_filename = os.path.join(pre_refine_dir, f"{file_prefix}trajectory_{out_i}.pdb")
                    writepdb_multi(
                        traj_filename, traj['x'][:, b], torch.zeros(num_res, device=self.device),
                        traj['seq'][:, b], chain_ids=[res[0] for res in pdb_idx[b]], use_hydrogens=False,
                        bond_mat=final_bond_sampled[b], link_csv_path=self.preprocess_conf.link_config,
                        head_mask=head_mask[b] if head_mask is not None else None,
                        tail_mask=tail_mask[b] if tail_mask is not None else None,
                        nc_anchor=masks['N_C_anchor'][b] if 'N_C_anchor' in masks else None,
                    )
                    traj_filename = os.path.join(pre_refine_dir, f"{file_prefix}px0_trajectory_{out_i}.pdb")
                    writepdb_multi(
                        traj_filename, traj['px0'][:, b], torch.zeros(num_res, device=self.device),
                        traj['seq'][:, b], chain_ids=[res[0] for res in pdb_idx[b]], use_hydrogens=False,
                        bond_mat=final_bond_sampled[b], link_csv_path=self.preprocess_conf.link_config,
                        head_mask=head_mask[b] if head_mask is not None else None,
                        tail_mask=tail_mask[b] if tail_mask is not None else None,
                        nc_anchor=masks['N_C_anchor'][b] if 'N_C_anchor' in masks else None,
                    )

        base_matrix, valid_mask,_,_ = self._bond_coherence_loss.compute_consistency_base(
                                                            bond_matrix=final_bond_sampled,
                                                            all_atom_coords=final_x_withCN,
                                                            res_mask=res_mask,
                                                            aatype=final_seq,
                                                            head_mask=head_mask,
                                                            tail_mask=tail_mask,
                                                            nc_anchor=N_C_anchor
                                                        )
        
        refine_bond_matrix = final_bond_sampled * (base_matrix < 0.65).float()
        refine_bond_matrix = smu.make_sub_doubly2doubly_stochastic(refine_bond_matrix)
        sidechain_model_type = getattr(self._conf.model, "sidechain_model_type", None)
        # 如果配置了 sidechain_model_type，则使用深度学习侧链模型精修；
        # 否则退回到基于 BondCoherence + Clash 的梯度下降扭转角优化。
        if sidechain_model_type is not None:
            print("start refine_sidechain (model)-------------------------------")
            B,L = final_seq.shape[:2]
            alpha_tor_mask = torch.zeros(B,L, 10, device=self.device, dtype=torch.bool)
            torsion_angles ,refine_x,refine_x_withCN = self.refine_sidechain(
                final_seq, final_x, refine_bond_matrix, fixed_batch_data['rf_idx'], 
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
                refine_bond_matrix,
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
                fixed_batch_data['chain_num_ids']
            )
            end_time = time.time()
            print(f"Refine sidechain time: {end_time - start_time} seconds")
        # 公共的输出逻辑：不管是 sidechain 模型还是 GD，都统一在这里写精修结果
        # 获取原始PDB数据（如果可用）
        pdb_parsed = fixed_batch_data.get('pdb_parsed', None)
        self._write_refined_outputs(
            out_pdb_dir=post_refine_dir if out_pdb_dir is not None else ".",
            file_prefix=file_prefix,
            num_batch=num_batch,
            start_index=int(start_index),
            final_seq=final_seq,
            final_bond_sampled=refine_bond_matrix,
            refine_x=refine_x_withCN,
            refine_x_withCN=refine_x_withCN,
            fixed_batch_data=fixed_batch_data,
            head_mask=head_mask,
            tail_mask=tail_mask,
            nc_anchor=masks.get('N_C_anchor', None),
            pdb_parsed=pdb_parsed,
        )

        if record_trajectory:
            return final_px0, final_x, final_seq, refine_bond_matrix, traj
        else:
            return final_px0, final_x, final_seq, refine_bond_matrix

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
        seq_init: torch.Tensor = None,   # [B, L] integer aa tokens
        ss_init: torch.Tensor = None,    # [B, L, L] initial / fixed bond matrix
        xyz_init: torch.Tensor = None,   # [B, L, 3, 3] initial / fixed xyz coordinates
        head_mask: torch.Tensor = None,  # [B, L] optional override
        tail_mask: torch.Tensor = None,  # [B, L] optional override
        N_C_anchor: torch.Tensor = None, # [B, L, L, 2] optional override
        record_trajectory: bool = True,
        eps_t: float = 5e-4,
        out_pdb_dir: str = None,
        # --- APMBackboneWrapper / PLM-related optional inputs ---
        origin_pdb_idx = None,
        pdb_seq_full = None,
        pdb_idx_full = None,
        pdb_core_id = None,
        chain_ids: torch.Tensor = None,
        hotspots: torch.Tensor = None,
        start_index: int = 0,
        pdb_parsed: dict = None,  # 原始PDB数据，用于写入完整链

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
            pdb_idx = [[("A", int(i) + 1) for i in range(num_res)] for _ in range(num_batch)]
        if res_mask is None:
            res_mask = torch.ones(num_batch, num_res, dtype=torch.bool, device=device)
        if str_mask is None:
            str_mask = torch.ones(num_batch, num_res, dtype=torch.bool, device=device)
        if seq_mask is None:
            seq_mask = torch.ones(num_batch, num_res, dtype=torch.bool, device=device)
        if hotspots is None:
            hotspots = torch.zeros(num_batch, num_res, dtype=torch.bool, device=device)
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
        trans_0 = interpolant_centered_gaussian(num_batch, num_res, device) #* du.NM_TO_ANG_SCALE
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
        # Default N/C-related masks (can be overridden by caller)
        if head_mask is None:
            head_mask = torch.zeros(num_batch, num_res, dtype=torch.bool, device=device)
        else:
            head_mask = head_mask.to(device)
        if tail_mask is None:
            tail_mask = torch.zeros(num_batch, num_res, dtype=torch.bool, device=device)
        else:
            tail_mask = tail_mask.to(device)
        if N_C_anchor is None:
            N_C_anchor = torch.zeros(num_batch, num_res, num_res, 2, dtype=torch.bool, device=device)
            # Minimal default anchor: first<->second for N, last<->second-last for C
            if num_res >= 2:
                N_C_anchor[:, 0, 1, 0] = True
                N_C_anchor[:, 1, 0, 0] = True
                N_C_anchor[:, -1, -2, 1] = True
                N_C_anchor[:, -2, -1, 1] = True
        else:
            N_C_anchor = N_C_anchor.to(device)

        # Inject fixed initial sequence / bond values for non-diffused positions.
        # This is critical for using YAML contigs (seq_FIX) and bond_condition (:...:FIX)
        # even when sampling from prior.
        if seq_init is not None and seq_mask is not None:
            seq_init = seq_init.to(device)
            # where seq_mask is False (fixed), copy from seq_init
            seq_t_1 = torch.where(seq_mask.bool(), seq_t_1, seq_init)
        # --- IMPORTANT: keep New_ close to hotspot/fixed-frame ---
        #
        # If the fixed/target chain is in an absolute PDB frame (e.g. centered around (100,100,100))
        # but the prior Gaussian for the designable part is centered around the origin,
        # then a naive "center around hotspots" step would shift the *entire* system so that
        # hotspots become 0,0,0 — and would push the New_ residues to ~(-100,-100,-100).
        #
        # What we want instead:
        #   1) Compute the hotspot/fixed center from xyz_init
        #   2) Translate the Gaussian prior backbone by that center (so New_ starts near the target)
        #   3) Then apply the usual centering, so both fixed and New_ end up near the origin.
        if chain_ids is not None:
            chain_num_ids_for_center = chain_ids.to(device)
        else:
            chain_num_ids_for_center = torch.full((num_batch, num_res), 1, dtype=torch.long, device=device)

        xyz_init_centered = None
        if xyz_init is not None:
            xyz_init = xyz_init.to(device)
            # Use the same centering rule as training to derive the reference shift vector.
            xyz_init_centered = self._center_xyz_with_chain_and_hotspots(
                xyz_init, res_mask, str_mask, chain_ids=chain_num_ids_for_center, hotspots=hotspots
            )
            # Recover the translation that was subtracted (use CA of residue 0 as a stable representative)
            center_vec = (xyz_init[:, 0, 1, :] - xyz_init_centered[:, 0, 1, :])  # [B,3]
            x_bb = x_bb + center_vec[:, None, None, :]

        # Now inject fixed xyz into the (shifted) prior backbone
        if xyz_init is not None and str_mask is not None:
            x_bb = torch.where(str_mask[:, :, None, None].bool(), x_bb, xyz_init)

        if ss_init is not None and bond_mask is not None:
            ss_init = ss_init.to(device=device, dtype=bond_mat_t_1.dtype)
            fixed_edges = ~bond_mask.bool()
            if fixed_edges.any():
                bond_mat_t_1 = bond_mat_t_1.clone()
                bond_mat_t_1[fixed_edges] = ss_init[fixed_edges]
                bond_mat_t_1 = smu.make_sub_doubly2doubly_stochastic(bond_mat_t_1)

        # Finally, center the whole system (training-consistent)
        x_bb = self._center_xyz_with_chain_and_hotspots(
            x_bb, res_mask, str_mask, chain_ids=chain_num_ids_for_center, hotspots=hotspots
        )
        _, x_t_1 = self.allatom(seq_t_1, x_bb, alpha0, bond_mat=bond_mat_t_1, 
                                link_csv_path=self.preprocess_conf.link_config, use_H=False,
                                head_mask=head_mask,
                                tail_mask=tail_mask,
                                N_C_anchor=N_C_anchor,)  # [B, L, 14, 3]


        trans_diffuse_mask = str_mask.float()
        rots_diffuse_mask = str_mask.float()
        aatypes_diffuse_mask = seq_mask.float()
        bond_diffuse_mask = bond_mask.float()

        # ------------------------------------------------------------------
        # Auto-generate APMBackboneWrapper-related metadata if not provided.
        # For pure prior sampling, we treat the design window as the full
        # structure, so full-sequence / origin mapping are 1:1 with design.
        # ------------------------------------------------------------------
        B_cur, L_cur = seq_t_1.shape[:2]

        # origin_pdb_idx: [B][L] list of (chain, res_idx) for each design pos
        if origin_pdb_idx is None:
            origin_pdb_idx = []
            for b in range(B_cur):
                if pdb_idx is None or len(pdb_idx[b]) < L_cur:
                    origin_list = [("A", int(i)) for i in range(L_cur)]
                else:
                    origin_list = [
                        (str(res[0]), int(res[1])) for res in pdb_idx[b][:L_cur]
                    ]
                origin_pdb_idx.append(origin_list)

        # pdb_idx_full: [B][L_full] full-structure PDB indices
        if pdb_idx_full is None:
            if pdb_idx is None:
                pdb_idx_full = [[("A", int(i)) for i in range(L_cur)] for _ in range(B_cur)]
            else:
                pdb_idx_full = [list(pdb_idx[b]) for b in range(B_cur)]

        # pdb_seq_full: [B] full-structure sequences (tensor or np.array)
        if pdb_seq_full is None:
            pdb_seq_full = [seq_t_1[b].detach().clone() for b in range(B_cur)]

        # pdb_core_id: [B] arbitrary identifiers (used only as a non-None key)
        if pdb_core_id is None:
            pdb_core_id = [f"prior_{b}" for b in range(B_cur)]

        # Chain ids for APMBackboneWrapper (fallback: single chain if not provided)
        if chain_ids is not None:
            chain_num_ids = chain_ids.to(device)
        else:
            chain_num_ids = torch.full((num_batch, num_res), 1, dtype=torch.long, device=device)

        # Fixed batch data (passed through to adapter / APMBackboneWrapper)
        fixed_batch_data = {
            'rf_idx': rf_idx,
            'pdb_idx': pdb_idx,
            'chain_num_ids': chain_num_ids,
            'origin_pdb_idx': origin_pdb_idx,
            'pdb_seq_full': pdb_seq_full,
            'pdb_idx_full': pdb_idx_full,
            'pdb_core_id': pdb_core_id,
            'hotspots': hotspots,
            'pdb_parsed': pdb_parsed,  # 原始PDB数据，用于写入完整链
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
            aatypes_1 = seq_init,
            ss_1 = ss_init,
            trans_1 = (xyz_init_centered if xyz_init_centered is not None else xyz_init)[:,:,1,:],
            rotmats_1 = iu.get_R_from_xyz(xyz_init_centered if xyz_init_centered is not None else xyz_init).to(self.device),
            record_trajectory=record_trajectory,
            eps_t=eps_t,
            out_pdb_dir=out_pdb_dir,
            file_prefix="",
            start_index=int(start_index),
            tqdm_desc="Sampling from prior",
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
        # --- APMBackboneWrapper / PLM-related optional inputs ---
        origin_pdb_idx = None,
        pdb_seq_full = None,
        pdb_idx_full = None,
        pdb_core_id = None,
        chain_ids: torch.Tensor = None,
        hotspots: torch.Tensor = None,
        start_index: int = 0,
        pdb_parsed: dict = None,  # 原始PDB数据，用于写入完整链
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
            xyz_target,
            seq_target,
            ss_target,
            res_mask,
            str_mask,
            seq_mask,
            bond_diffuse_mask,
            hotspots,
            t_batch,
            chain_ids,
            head_mask=head_mask,
            tail_mask=tail_mask,
            N_C_anchor=N_C_anchor,
            
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
                                   link_csv_path=self.preprocess_conf.link_config, use_H=False,
                                    head_mask=head_mask,
                                    tail_mask=tail_mask,
                                    N_C_anchor=N_C_anchor,)
        x_t_1 = iu.update_nc_node_coordinates(x_t_1, N_C_anchor, head_mask, tail_mask)

        # 确保形状正确
        if x_t_1.dim() == 5 and x_t_1.shape[1] == 1:
            x_t_1 = x_t_1.squeeze(1)
        x_t_1 = x_t_1[:, :, :14, :]  # [B, L, 14, 3]

        # Chain ids for APMBackboneWrapper (fallback: single chain if not provided)
        if chain_ids is not None:
            chain_num_ids = chain_ids.to(device)
        else:
            chain_num_ids = torch.full((num_batch, num_res), 1, dtype=torch.long, device=device)

        # Fixed batch data (passed through to adapter / APMBackboneWrapper)
        fixed_batch_data = {
            'rf_idx': rf_idx,
            'pdb_idx': pdb_idx,
            'chain_num_ids': chain_num_ids,
            # 以下字段应由调用方在有真实结构时提供（例如 cyclize_from_pdb）；
            # 在缺失时，可以保持为 None，由下游模型自行决定是否使用 PLM。
            'origin_pdb_idx': origin_pdb_idx,
            'pdb_seq_full': pdb_seq_full,
            'pdb_idx_full': pdb_idx_full,
            'pdb_core_id': pdb_core_id,
            'hotspots': hotspots,
            'pdb_parsed': pdb_parsed,  # 原始PDB数据，用于写入完整链
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
            start_index=int(start_index),
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
                pdb_info1 = pdb_indices[res1_idx]
                pdb_info2 = pdb_indices[res2_idx]
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
                    f"{pdb_info1[0]},{pdb_info1[1]},{res1_type_str},{atom1_name},"
                    f"{pdb_info2[0]},{pdb_info2[1]},{res2_type_str},{atom2_name},"
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

def set_reproducible(seed: int = 1234):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CuDNN determinism
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
