import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from rfdiff.chemical import aa2long, aabonds, num2aa, aa2num
from multiflow_data import so3_utils as su
from apm.openfold.utils.loss import sidechain_loss as _sidechain_loss
from BondFlow.data.link_utils import LinkInfo
from torch import nn

# ----------------------
# Shared utility helpers
# ----------------------
def infer_termini_masks(
    res_mask: torch.Tensor,
    head_mask: torch.Tensor = None,
    tail_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Public helper to infer simple head/tail masks when not provided.
    - head: first valid residue per batch
    - tail: last valid residue per batch
    Shapes: res_mask [B,L] -> returns two [B,L] bool tensors.
    """
    B, L = res_mask.shape
    device = res_mask.device
    if head_mask is not None and tail_mask is not None:
        return head_mask.to(device=device, dtype=torch.bool), tail_mask.to(device=device, dtype=torch.bool)

    inferred_head = torch.zeros((B, L), dtype=torch.bool, device=device)
    inferred_tail = torch.zeros((B, L), dtype=torch.bool, device=device)
    for i in range(B):
        valid_indices = torch.where(res_mask[i])[0]
        if len(valid_indices) > 0:
            inferred_head[i, int(valid_indices[0].item())] = True
            inferred_tail[i, int(valid_indices[-1].item())] = True
    return inferred_head, inferred_tail

def compute_terminal_body_maps(
    res_mask: torch.Tensor,
    head_mask: torch.Tensor,
    tail_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given per-batch residue validity and head/tail masks, compute mapping from
    each head index to the first valid body residue after it (head_to_body_idx),
    and from each tail index to the last valid body residue before it (tail_to_body_idx).

    Inputs: res_mask, head_mask, tail_mask: [B, L] (bool or float)
    Returns:
      - head_to_body_idx: [B, L] long; -1 where not a head; otherwise target index
      - tail_to_body_idx: [B, L] long; -1 where not a tail; otherwise target index
    """
    B, L = res_mask.shape
    device = res_mask.device
    valid = (res_mask > 0.5).to(torch.bool)
    hmask = head_mask.to(torch.bool)
    tmask = tail_mask.to(torch.bool)

    head_to_body = torch.full((B, L), -1, dtype=torch.long, device=device)
    tail_to_body = torch.full((B, L), -1, dtype=torch.long, device=device)

    for b in range(B):
        valid_idx = torch.where(valid[b])[0]
        # Heads: map to first valid j > h, fallback to min(h+1, L-1)
        heads_b = torch.where(hmask[b])[0]
        for h in heads_b.tolist():
            j_gt = valid_idx[valid_idx > h]
            hb = int(j_gt[0].item()) if j_gt.numel() > 0 else min(h + 1, L - 1)
            head_to_body[b, h] = hb
        # Tails: map to last valid j < t, fallback to max(t-1, 0)
        tails_b = torch.where(tmask[b])[0]
        for t in tails_b.tolist():
            j_lt = valid_idx[valid_idx < t]
            tb = int(j_lt[-1].item()) if j_lt.numel() > 0 else max(t - 1, 0)
            tail_to_body[b, t] = tb

    return head_to_body, tail_to_body




class SidechainFAPELoss(nn.Module):
    """
    A minimal PyTorch wrapper around APM/OpenFold sidechain FAPE.

    It provides two interfaces:
      - forward(...): takes pre-built rigid-group frames and atom14 positions
      - forward_from_backbone(...): builds per-residue backbone frames from N,CA,C
    """
    def __init__(self,
                 clamp_distance: float = 10.0,
                 length_scale: float = 10,
                 eps: float = 1e-8,
                 bond_threshold: float = 0.5,
                 bond_weight: float = 10):
        super().__init__()
        self.clamp_distance = float(clamp_distance)
        self.length_scale = float(length_scale)
        self.eps = float(eps)
        # Emphasis config for bonded sidechains (residue-level weighting)
        self.bond_threshold = float(bond_threshold)
        self.bond_weight = float(bond_weight)

    def forward(self,
                sidechain_frames_4x4: torch.Tensor,
                pred_atom14_pos: torch.Tensor,
                rigidgroups_gt_frames: torch.Tensor,
                rigidgroups_alt_gt_frames: torch.Tensor,
                rigidgroups_gt_exists: torch.Tensor,
                renamed_atom14_gt_positions: torch.Tensor,
                renamed_atom14_gt_exists: torch.Tensor,
                alt_naming_is_better: torch.Tensor,
                res_mask: torch.Tensor = None,
                bond_mat: torch.Tensor = None) -> torch.Tensor:
        # Ensure sequence/steps axis exists at dim 0 for APM sidechain_loss
        if sidechain_frames_4x4.dim() == 5:
            sidechain_frames_seq = sidechain_frames_4x4.unsqueeze(0)  # [1, L, G, 4, 4]
        elif sidechain_frames_4x4.dim() == 6:
            sidechain_frames_seq = sidechain_frames_4x4
        else:
            raise ValueError("sidechain_frames_4x4 must be [L,G,4,4] or [T,L,G,4,4]")

        if pred_atom14_pos.dim() == 4:
            sidechain_atom_pos_seq = pred_atom14_pos.unsqueeze(0)    # [1, L, 14, 3]
        elif pred_atom14_pos.dim() == 5:
            sidechain_atom_pos_seq = pred_atom14_pos                 # [T, L, 14, 3]
        else:
            raise ValueError("pred_atom14_pos must be [L,14,3] or [T,L,14,3]")

        # --- Sanitize numeric issues before calling OpenFold sidechain_loss ---
        # 1) Replace NaN/Inf in positions with zeros (safe no-op under masks)
        sidechain_atom_pos_seq = sidechain_atom_pos_seq.nan_to_num(0.0)
        renamed_atom14_gt_positions = renamed_atom14_gt_positions.nan_to_num(0.0)

        # 2) Replace NaN/Inf in 4x4 frames and enforce homogeneous identity on invalid frames
        def _sanitize_frames(frames: torch.Tensor) -> torch.Tensor:
            frames = frames.nan_to_num(0.0)
            # Detect invalid transforms (any non-finite element within the 4x4)
            finite_mask = torch.isfinite(frames).all(dim=-1).all(dim=-1)
            if not bool(finite_mask.all()):
                I = torch.eye(4, device=frames.device, dtype=frames.dtype)
                frames = torch.where(finite_mask.unsqueeze(-1).unsqueeze(-1), frames, I)
            # Enforce bottom row [0,0,0,1] for all frames to avoid degenerate homogeneous transforms
            frames = frames.clone()
            frames[..., 3, :3] = 0.0
            frames[..., 3, 3] = 1.0
            return frames

        sidechain_frames_seq = _sanitize_frames(sidechain_frames_seq)
        rigidgroups_gt_frames = _sanitize_frames(rigidgroups_gt_frames)
        rigidgroups_alt_gt_frames = _sanitize_frames(rigidgroups_alt_gt_frames)

        # Apply residue mask to frames/atoms existence masks if provided
        if res_mask is not None:
            res_mask = res_mask.to(rigidgroups_gt_exists.device, dtype=rigidgroups_gt_exists.dtype)
            rigidgroups_gt_exists = rigidgroups_gt_exists * res_mask[..., None]
            renamed_atom14_gt_exists = renamed_atom14_gt_exists * res_mask[..., None]

        # Build per-residue weights from bond_mat (emphasize bonded residues),
        # modeled after TorsionLossLegacy, and integrate them into masks
        if bond_mat is not None:
            bm = bond_mat.detach()
            B, L = bm.shape[0], bm.shape[-1]
            if res_mask is not None:
                res_valid = res_mask.to(dtype=torch.bool, device=bm.device)
            else:
                res_valid = torch.ones((B, L), dtype=torch.bool, device=bm.device)

            pair_valid = (res_valid.unsqueeze(1) & res_valid.unsqueeze(2))
            eye = torch.eye(L, device=bm.device, dtype=torch.bool).unsqueeze(0)
            pair_valid = pair_valid & (~eye)

            bm_thr = bm * (bm > self.bond_threshold).float()
            bm_thr = bm_thr * pair_valid.float()

            row_sum = bm_thr.sum(dim=-1)
            col_sum = bm_thr.sum(dim=-2)
            res_weights = (row_sum + col_sum) / 2.0  # [B, L]
            weights_res = 1.0 + res_weights * self.bond_weight  # [B, L]

            # Scale existence masks by residue weights to emphasize bonded residues
            rigidgroups_gt_exists = rigidgroups_gt_exists * weights_res.unsqueeze(-1)
            renamed_atom14_gt_exists = renamed_atom14_gt_exists * weights_res.unsqueeze(-1)

        fape = _sidechain_loss(
            sidechain_frames=sidechain_frames_seq,
            sidechain_atom_pos=sidechain_atom_pos_seq,
            rigidgroups_gt_frames=rigidgroups_gt_frames,
            rigidgroups_alt_gt_frames=rigidgroups_alt_gt_frames,
            rigidgroups_gt_exists=rigidgroups_gt_exists,
            renamed_atom14_gt_positions=renamed_atom14_gt_positions,
            renamed_atom14_gt_exists=renamed_atom14_gt_exists,
            alt_naming_is_better=alt_naming_is_better,
            clamp_distance=self.clamp_distance,
            length_scale=self.length_scale,
            eps=self.eps,
        )

        return torch.mean(fape)

    @staticmethod
    def _build_backbone_frames_4x4(xyz_bb: torch.Tensor) -> torch.Tensor:
        """
        Build per-residue backbone homogeneous transforms from N, CA, C coordinates.
        xyz_bb: [B, L, 3, 3] (N, CA, C)
        Returns: [B, L, 1, 4, 4]
        """
        B, L = xyz_bb.shape[:2]
        N = xyz_bb[:, :, 0]
        CA = xyz_bb[:, :, 1]
        C = xyz_bb[:, :, 2]
        x_axis = (C - CA)
        x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)
        v2 = (N - CA)
        v2_proj = (x_axis * torch.sum(v2 * x_axis, dim=-1, keepdim=True))
        y_axis = v2 - v2_proj
        y_axis = y_axis / (torch.norm(y_axis, dim=-1, keepdim=True) + 1e-8)
        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [B,L,3,3]
        T = CA  # [B,L,3]
        H = torch.zeros((B, L, 4, 4), dtype=xyz_bb.dtype, device=xyz_bb.device)
        H[:, :, :3, :3] = R
        H[:, :, :3, 3] = T
        H[:, :, 3, 3] = 1.0
        return H.unsqueeze(2)  # [B,L,1,4,4]

    def forward_from_backbone(self,
                              xyz_bb_pred: torch.Tensor,
                              pred_atom14_pos: torch.Tensor,
                              xyz_bb_gt: torch.Tensor,
                              atom14_gt_pos: torch.Tensor,
                              res_mask: torch.Tensor = None,
                              bond_mat: torch.Tensor = None) -> torch.Tensor:
        """
        Convenience wrapper: build frames from backbone N,CA,C and compute sidechain FAPE.
        Inputs:
          - xyz_bb_pred: [B, L, 3, 3] predicted backbone (N,CA,C)
          - pred_atom14_pos: [B, L, 14, 3] predicted atom14 positions
          - xyz_bb_gt: [B, L, 3, 3] ground-truth backbone (N,CA,C)
          - atom14_gt_pos: [B, L, 14, 3] ground-truth atom14 positions
          - res_mask: [B, L] optional residue mask
        """
        xyz_bb_pred = xyz_bb_pred.nan_to_num(0.0)
        xyz_bb_gt = xyz_bb_gt.nan_to_num(0.0)
        pred_atom14_pos = pred_atom14_pos.nan_to_num(0.0)
        atom14_gt_pos = atom14_gt_pos.nan_to_num(0.0)

        sidechain_frames_4x4 = self._build_backbone_frames_4x4(xyz_bb_pred)
        rigidgroups_gt_frames = self._build_backbone_frames_4x4(xyz_bb_gt)
        rigidgroups_alt_gt_frames = rigidgroups_gt_frames
        rigidgroups_gt_exists = torch.ones(
            (*xyz_bb_gt.shape[:2], 1), dtype=pred_atom14_pos.dtype, device=pred_atom14_pos.device
        )
        renamed_atom14_gt_positions = atom14_gt_pos
        renamed_atom14_gt_exists = torch.ones_like(pred_atom14_pos[..., 0])
        alt_naming_is_better = torch.zeros_like(rigidgroups_gt_exists[..., 0])

        return self.forward(
            sidechain_frames_4x4=sidechain_frames_4x4,
            pred_atom14_pos=pred_atom14_pos,
            rigidgroups_gt_frames=rigidgroups_gt_frames,
            rigidgroups_alt_gt_frames=rigidgroups_alt_gt_frames,
            rigidgroups_gt_exists=rigidgroups_gt_exists,
            renamed_atom14_gt_positions=renamed_atom14_gt_positions,
            renamed_atom14_gt_exists=renamed_atom14_gt_exists,
            alt_naming_is_better=alt_naming_is_better,
            res_mask=res_mask,
            bond_mat=bond_mat,
        )
        
class LFrameLoss(nn.Module):
    def __init__(self, w_trans=1.0, w_rot=1.0, d_clamp=10.0, gamma=1.02, eps=1e-4):
        """
        LFrame Loss Module with Masking
        Args:
            w_trans: 平移损失的权重
            w_rot: 旋转损失的权重
            d_clamp: 平移距离的截断值
            gamma: 时间步加权的指数因子（>1.0）
            eps: 数值稳定常数
        """
        super().__init__()
        self.w_trans = w_trans
        self.w_rot = w_rot
        self.d_clamp = d_clamp
        self.gamma = gamma
        self.eps = eps
        self.register_buffer('time_weights', None)
    
    def compute_rotation_matrix(self, coords,mask):
        """
        从原子坐标计算旋转矩阵
        Args:
            coords: [..., 3, 3] 张量，最后两维表示原子(N, CA, C)的坐标
        Returns:
            rotation_matrix: [..., 3, 3] 旋转矩阵
        """
        # 提取原子坐标 (N, CA, C)
        N = coords[..., 0, :]  # [..., 3]
        CA = coords[..., 1, :]  # [..., 3]
        C = coords[..., 2, :]  # [..., 3]
        
        # 计算局部坐标系基向量
        v1 = C - CA  # x轴: C->CA
        v2 = N - CA  # 临时向量
        
        # Gram-Schmidt正交化
        r1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + self.eps)  # x轴单位向量
        
        # 计算y轴分量
        v2_proj = torch.sum(v2 * r1, dim=-1, keepdim=True) * r1
        r2 = v2 - v2_proj
        r2 = r2 / (torch.norm(r2, dim=-1, keepdim=True) + self.eps)  # y轴单位向量
        r2_norm = torch.norm(r2, dim=-1)
        if torch.any((r2_norm < 1e-6) & mask ) : #mask will be near collinear
            print("Warning: Near-collinear atoms detected!")
        # 计算z轴（叉积）
        r3 = torch.cross(r1, r2, dim=-1)
        
        # 组合旋转矩阵 [r1, r2, r3]^T
        return torch.stack([r1, r2, r3], dim=-1)

    def forward(self, pred, target, noise,mask=None, t = None):
        """
        使用"全局平均"原则计算带Mask的Frame损失
        Args:
            pred: [B, T, L, 3, 3] 预测的原子坐标 (N, CA, C)
            target: [B, T, L, 3, 3] 真实的原子坐标
            noise: [B, T, L, 3, 3] 加噪的原子坐标
            mask: [B, L] bool或int型掩码，True/1表示有效位置
        Returns:
            loss: 标量损失值
        """
        B, T, L, _, _ = pred.shape

        # 1. 处理掩码 (与之前相同)
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=pred.device)
        
        # 扩展掩码到时间步维度 [B, T, L]
        # 注意：这里我们保留了最细粒度的mask，这是方法二的关键
        mask_3d = mask.unsqueeze(1).expand(-1, T, -1)

        # 2. 计算残基级的平移和旋转损失 (与之前基本相同)
        # 2.1 平移损失
        pred_CA = pred[..., 1, :]
        target_CA = target[..., 1, :]
        trans_dist = torch.norm(pred_CA - target_CA, dim=-1)
        clamp_mask = torch.rand_like(trans_dist) < 1
        clamped_dist = torch.min(trans_dist, torch.full_like(trans_dist, self.d_clamp))
        trans_loss = torch.where(clamp_mask, clamped_dist, trans_dist) ** 2
        trans_loss = self.w_trans * trans_loss  # 形状: [B, T, L]

        # --- 旋转损失 (修改为论文形式) ---
        # 1. 从坐标计算旋转矩阵
        pred_rot = self.compute_rotation_matrix(pred, mask_3d)
        noisy_rot = self.compute_rotation_matrix(noise, mask_3d)
        target_rot = self.compute_rotation_matrix(target, mask_3d)

        # 3. 使用对数映射到切空间
        pred_log_rot = su.calc_rot_vf(noisy_rot,pred_rot)
        target_log_rot = su.calc_rot_vf(noisy_rot,target_rot)

        # 4. 计算切空间中的L2距离平方
        # 这完全匹配了论文中的核心思想: ||log_{R_t}(R_pred) - log_{R_t}(R_target)||^2
        rot_loss = self.w_rot * torch.sum((pred_log_rot - target_log_rot)**2, dim=-1) 

        # 3. 合并残基级损失
        residue_loss = trans_loss + rot_loss # 形状: [B, T, L]

        # 4. 生成时间步权重
        if self.time_weights is None or self.time_weights.shape[0] != T:
            self.time_weights = torch.tensor(
                [self.gamma ** i for i in range(T)],
                device=residue_loss.device,
                dtype=residue_loss.dtype
            ) # 形状: [T]

        # 5. 【核心修改】应用时间和残基掩码，并计算全局加权和
        # 将时间权重扩展到 [1, T, 1] 以便与 [B, T, L] 的张量广播
        template_weights_expanded = self.time_weights.view(1, T, 1)
        
        # 将掩码转为float类型用于乘法
        mask_3d_float = mask_3d.float()

        # noise time step t
        if t is not None:
            time_weights_expanded = (1/(1 - t)).view(B, 1, 1)
        else:
            time_weights_expanded = torch.ones((B, 1, 1), device=pred.device)
        
        # 计算损失的分子：将每个有效位置的损失乘以其对应的时间权重，然后全部相加
        # (residue_loss * mask_3d_float) 确保只计算有效位置的损失
        numerator = (residue_loss * mask_3d_float * template_weights_expanded * time_weights_expanded).sum() 

        # 计算损失的分母：每个有效位置的"贡献"是其时间权重，将所有有效位置的贡献相加
        # 这相当于计算了加权后的有效元素总数
        denominator = (mask_3d_float * template_weights_expanded).sum()
        # 6. 计算最终的全局平均损失
        # 添加一个小的epsilon防止除以零
        eps = 1e-8
        final_loss = numerator / (denominator + eps)

        # 7. 【可选】为了日志记录，用同样的方式计算各个分量的损失
        # 注意：这些计算不参与反向传播，只是为了监控
        with torch.no_grad():
            trans_numerator = (trans_loss * mask_3d_float * template_weights_expanded).sum()
            rot_numerator = (rot_loss * mask_3d_float * template_weights_expanded).sum()
            
            avg_trans_loss = trans_numerator / (denominator + eps)
            avg_rot_loss = rot_numerator / (denominator + eps)

            print(f"R loss: {avg_rot_loss.item():.4f}")
            print(f"T loss: {avg_trans_loss.item():.4f}")

        # 11. 返回最终的标量损失
        return final_loss


class MultiBinCrossEntropy(nn.Module):
    """
    多分桶交叉熵损失模块（支持批量处理）
    计算给定角度/距离类型的分桶交叉熵损失
    """
    def __init__(self, bin_count, min_val=None, max_val=None, periodic=False):
        """
        Args:
            bin_count (int): 分桶数量
            min_val (float): 值范围最小值
            max_val (float): 值范围最大值
            periodic (bool): 是否为周期性角度（如二面角）
        """
        super().__init__()
        self.bin_count = bin_count
        self.min_val = min_val
        self.max_val = max_val
        self.periodic = periodic
        self.bin_width = (max_val - min_val) / bin_count if min_val is not None else None
        
    def compute_bin_index(self, values):
        """计算值对应的分桶索引（支持批量）"""
        # 处理NaN/Inf
        nan_mask = torch.isnan(values)
        inf_mask = torch.isinf(values)
        invalid_mask = nan_mask | inf_mask
        
        if invalid_mask.any():
            # 安全替换非法值
            values = torch.where(invalid_mask, 
                                torch.tensor(0.0, device=values.device), 
                                values)

        if self.periodic:
            # 处理周期性角度（如二面角）
            values = torch.remainder(values - self.min_val, 2 * math.pi) + self.min_val
        else:
            # 截断到有效范围
            values = torch.clamp(values, self.min_val, self.max_val)
        
        # 计算分桶索引
        bin_index = torch.floor((values - self.min_val) / self.bin_width)
        return bin_index.long().clamp(0, self.bin_count - 1)
    
    def forward(self, logits, values, mask=None):
        """
        Args:
            logits (Tensor): 模型预测的logits [B, bin_count, L, L]
            values (Tensor): 实际值 [B, L, L]
            mask (Tensor): 3D掩码 [B, L, L]，True表示需要计算损失的位置
            
        Returns:
            Tensor: 标量损失值
        """
        B, C, L, _ = logits.shape

        if mask is None:
            mask = torch.ones(B, L, L, dtype=torch.bool, device=logits.device)

        # 计算目标分桶索引 [B, L, L]
        target = self.compute_bin_index(values).long()

        loss_orig = F.cross_entropy(logits, target, reduction='none')
        loss_orig = loss_orig.nan_to_num() * mask.float()  
        
        # 计算有效元素的平均损失
        loss = loss_orig.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)) # [B]
        return loss.mean()

class L2DLoss(nn.Module):
    """
    L2D损失模块（支持批量处理）
    计算四个几何特征的分桶交叉熵损失
    """
    def __init__(self):
        super().__init__()
        
        # 初始化四个分桶损失模块
        self.dist_loss = MultiBinCrossEntropy(
            bin_count=37, min_val=0.0, max_val=19.0, periodic=False
        )
        self.omega_loss = MultiBinCrossEntropy(
            bin_count=37, min_val=-math.pi, max_val=math.pi, periodic=True
        )
        self.theta_loss = MultiBinCrossEntropy(
            bin_count=37, min_val=-math.pi, max_val=math.pi, periodic=True
        )
        self.phi_loss = MultiBinCrossEntropy(
            bin_count=19, min_val=0.0, max_val=math.pi, periodic=False
        )
    
    def forward(self, logits_dist, logits_omega, logits_theta, logits_phi,
                            dist, omega, theta, phi, mask_2d=None):
        """
        Args:
            logits_dist: 距离分布logits [B, 37, L, L]
            logits_omega: Omega角度logits [B, 37, L, L]
            logits_theta: Theta角度logits [B, 37, L, L]
            logits_phi: Phi角度logits [B, 19, L, L]
            z0: 坐标张量 [B, L, 3, 3] (N, Cα, Cβ)
            mask_2d: 3D掩码 [B, L, L]，True表示需要计算损失的位置
            
        Returns:
            Tensor: 总损失值
        """
        # # 分离坐标以避免梯度传播
       
        
        # # 提取原子坐标 [B, L, 3]
        # N, Ca, Cb = z0[..., 0, :], z0[..., 1, :], z0[..., 2, :]
        
        # # 创建默认掩码（排除对角元素）[B, L, L]
        
        # # ====================== 距离矩阵计算 ======================
        # # [B, L, 1, 3] - [B, 1, L, 3] = [B, L, L, 3]
        # dist = torch.norm(
        #     Cb.unsqueeze(2) - Cb.unsqueeze(1), 
        #     dim=-1
        # )
        # # 按论文要求截断在18.5Å
        # dist = torch.clamp(dist, max=18.5)
        
        # # ====================== 角度计算辅助函数 ======================
        # def dihedral_angle(a, b, c, d):
        #     """
        #     计算四个点a-b-c-d的二面角（批量处理）
        #     Args:
        #         a, b, c, d: 坐标张量 [B, L, 3]
        #     Returns:
        #         二面角矩阵 [B, L, L]
        #     """
        #     # 扩展维度以便计算所有残基对 [B, L, 1, 3] 和 [B, 1, L, 3]
        #     a = a.unsqueeze(2)  # [B, L, 1, 3]
        #     b = b.unsqueeze(2)  # [B, L, 1, 3]
        #     c = c.unsqueeze(1)  # [B, 1, L, 3]
        #     d = d.unsqueeze(1)  # [B, 1, L, 3]
            
        #     # 计算向量
        #     v1 = b - a  # [B, L, 1, 3]
        #     v2 = c - b  # [B, L, L, 3] (广播)
        #     v3 = d - c  # [B, L, L, 3] (广播)
            
        #     # 计算法向量
        #     n1 = torch.cross(v1, v2, dim=-1)  # [B, L, L, 3]
        #     n2 = torch.cross(v2, v3, dim=-1)  # [B, L, L, 3]
            
        #     # 归一化法向量
        #     n1 = n1 / (torch.norm(n1, dim=-1, keepdim=True) + 1e-8)
        #     n2 = n2 / (torch.norm(n2, dim=-1, keepdim=True) + 1e-8)
            
        #     # 计算二面角
        #     v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)
        #     sin_theta = torch.sum(torch.cross(n1, n2, dim=-1) * v2_norm, dim=-1)
        #     cos_theta = torch.sum(n1 * n2, dim=-1)
        #     return torch.atan2(sin_theta, cos_theta)  # [B, L, L]
        
        # def planar_angle(a, b, c):
        #     """
        #     计算点b处a-b-c的平面角（批量处理）
        #     Args:
        #         a, b, c: 坐标张量 [B, L, 3]
        #     Returns:
        #         平面角矩阵 [B, L, L]
        #     """
        #     # 扩展维度 [B, L, 1, 3] 和 [B, 1, L, 3]
        #     a = a.unsqueeze(2)  # [B, L, 1, 3]
        #     b = b.unsqueeze(2)  # [B, L, 1, 3]
        #     c = c.unsqueeze(1)  # [B, 1, L, 3]
            
        #     # 计算向量
        #     v1 = a - b  # [B, L, 1, 3]
        #     v2 = c - b  # [B, L, L, 3] (广播)
            
        #     # 计算点积和范数
        #     dot_product = torch.sum(v1 * v2, dim=-1)  # [B, L, L]
        #     norm_product = torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)
            
        #     # 计算余弦值并确保在有效范围内
        #     cos_phi = dot_product / (norm_product + 1e-8)
        #     cos_phi = torch.clamp(cos_phi, -1.0, 1.0)
        #     return torch.acos(cos_phi)  # [B, L, L]
        
        # # ====================== 计算四种几何特征 ======================
        # # Omega: Dihedral(Cα_i, Cβ_i, Cα_j, Cβ_j) [B, L, L]
        # omega = dihedral_angle(Ca, Cb, Ca, Cb)
        
        # # Theta: Dihedral(N_i, Cα_i, Cβ_i, Cβ_j) [B, L, L]
        # theta = dihedral_angle(N, Ca, Cb, Cb)
        
        # # Phi: Planar angle at Cβ_i (Cα_i, Cβ_i, Cβ_j) [B, L, L]
        # phi = planar_angle(Ca, Cb, Cb)
        B, L, _ = mask_2d.shape
        if mask_2d is None:
            mask_2d = ~torch.eye(L, dtype=torch.bool, device=logits_dist.device)
            mask_2d = mask_2d.unsqueeze(0).expand(B, -1, -1)  # 扩展到批量
        else:
            mask_2d = mask_2d & ~torch.eye(L, dtype=torch.bool, device=logits_dist.device).unsqueeze(0).expand(B, -1, -1)

        # ====================== 计算各项损失 ======================
        #查看dist中的nan并输出mask
        dist = torch.clamp(dist, max=18.5) 
        mask_dist = ~torch.isnan(dist) & mask_2d
        loss_dist = self.dist_loss(logits_dist, dist, mask_dist)

        mask_omega = ~torch.isnan(omega) & mask_2d
        loss_omega = self.omega_loss(logits_omega, omega, mask_omega)

        mask_theta = ~torch.isnan(theta) & mask_2d
        loss_theta = self.theta_loss(logits_theta, theta, mask_theta)
        
        mask_phi = ~torch.isnan(phi) & mask_2d
        loss_phi = self.phi_loss(logits_phi, phi, mask_phi)

        print("loss_dist:", loss_dist.item())
        print("loss_omega:", loss_omega.item())
        print("loss_theta:", loss_theta.item())
        print("loss_phi:", loss_phi.item())
        return loss_dist + loss_omega + loss_theta + loss_phi

    
class LseqLoss(nn.Module):

    def __init__(self):
        super().__init__()

    
    def forward(self, seq_pred, seq_orig,mask=None):

        """
        Args:
            seq_pred: [B, L, K] 预测的序列噪声logits
            seq_orig: [B, L] 原始序列（整数编码）
            mask: [B, L] 布尔型掩码，True表示有效位置
        Returns:
            loss: 标量损失值
        """
        B,L,K = seq_pred.shape
        if mask is None:
            mask = torch.ones(B, L ,dtype=torch.bool, device=seq_pred.device)

        # 确保输入是 torch.Tensor
        seq_pred = seq_pred.float()  # 确保是浮点数
        seq_orig = seq_orig.long()   # 确保是整数（类别标签）

        # 展平预测和原始序列以计算交叉熵
        # [B, L, K] -> [B*L, K]
        seq_pred_flat = seq_pred.view(-1, seq_pred.size(-1))
        # [B, L] -> [B*L]
        seq_orig_flat = seq_orig.view(-1)
        # [B, L] -> [B*L]
        mask_flat = mask.view(-1).bool()

        if mask_flat.sum() == 0:
            # 如果没有有效位置，返回零损失
            return torch.tensor(0.0, device=seq_pred.device)
         
        # 只计算有效位置的损失
        valid_pred = seq_pred_flat[mask_flat]
        valid_orig = seq_orig_flat[mask_flat]

        # 计算交叉熵损失
        loss = F.cross_entropy(valid_pred, valid_orig, reduction='mean')
        return loss

class FAPELoss(nn.Module):
    """
    FAPE (Frame-Align-Point-Error) Loss 的 PyTorch 实现。
    
    该实现包含了两个掩码：
    1. res_mask: 标记有效的残基 (1) 与 padding (0)。
    2. str_mask: 标记需要计算 loss 的残基 (1) 与固定的 motif (0)。
    """
    def __init__(self, clamp_distance: float = 10.0, eps: float = 1e-8):
        """
        Args:
            clamp_distance (float): 距离误差的截断值（单位：埃），
                                    用于防止梯度爆炸。
            eps (float): 用于数值稳定性的一个小常数。
        """
        super().__init__()
        self.clamp_distance = clamp_distance
        self.eps = eps

    def compute_rotation_matrix_svd(self, coords):
        """
        从原子坐标使用SVD稳健地计算旋转矩阵。
        """
        # 1. 提取并中心化定义坐标系的向量
        N = coords[..., 0, :]
        CA = coords[..., 1, :]
        C = coords[..., 2, :]
        v1 = C - CA
        v2 = N - CA
        
        # 2. 构建协方差矩阵的输入矩阵 A
        A = torch.stack([v1, v2], dim=-1) # A 的形状为 [..., 3, 2]

        # 3. 执行奇异值分解 (SVD)
        try:
            U, S, Vh = torch.linalg.svd(A)
        except torch.linalg.LinAlgError as e:
            print(f"SVD failed: {e}. Returning identity matrix.")
            shape = A.shape[:-2] + (3, 3)
            return torch.eye(3, device=A.device, dtype=A.dtype).expand(shape)

        # 4. 保证结果是"真"旋转矩阵（处理反射情况）
        det_U = torch.det(U)
        R = U.clone()
        
        fix_mask = det_U < 0

        R[..., :, -1][fix_mask] *= -1
        # --------------------------

        return R
    
    def _create_frames(self, backbone_coords: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        从骨架原子坐标构建局部参考系。
        
        Args:
            backbone_coords (torch.Tensor): 骨架原子坐标，形状 (B, L, 3, 3)。
                                            原子顺序为 N, Ca, C。
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - rotations (torch.Tensor): 旋转矩阵，形状 (B, L, 3, 3)。
                - translations (torch.Tensor): 平移向量 (Ca 坐标)，形状 (B, L, 3)。
        """
        # (B, L, 3)
        N_coords = backbone_coords[:, :, 0, :]
        Ca_coords = backbone_coords[:, :, 1, :]
        C_coords = backbone_coords[:, :, 2, :]
        rotmats = self.compute_rotation_matrix_svd(backbone_coords).float()
        
        # 平移向量就是 Ca 的坐标
        translations = Ca_coords
        
        return rotmats, translations

    def _frame_transform(self,
                         rotations: torch.Tensor,
                         translations: torch.Tensor,
                         points: torch.Tensor) -> torch.Tensor:
        """
        将点云变换到给定的局部坐标系下。
        
        Args:
            rotations (torch.Tensor): 旋转矩阵 (B, L_frames, 3, 3)。
            translations (torch.Tensor): 平移向量 (B, L_frames, 3)。
            points (torch.Tensor): 要变换的点 (B, L_points, 3)。
        
        Returns:
            torch.Tensor: 变换后的点，形状 (B, L_frames, L_points, 3)。
        """
        # 扩展维度以进行广播
        # rotations:    (B, L_frames, 1, 3, 3)
        # translations: (B, L_frames, 1, 3)
        # points:       (B, 1, L_points, 3)
        rotations = rotations.unsqueeze(-3)
        translations = translations.unsqueeze(-2)
        points = points.unsqueeze(-3)

        # 应用变换: p' = R^T * (p - t)
        inv_rotations = rotations.transpose(-1, -2)
        transformed_points = torch.matmul(inv_rotations, (points - translations).unsqueeze(-1)).squeeze(-1)

        return transformed_points

    def forward(self,
                pred_coords: torch.Tensor,
                true_coords: torch.Tensor,
                res_mask: torch.Tensor,
                str_mask: torch.Tensor) -> torch.Tensor:
        """
        计算带有掩码的 FAPE Loss。
        
        Args:
            pred_coords (torch.Tensor): 预测的骨架坐标 (B, L, 3, 3) for (N, Ca, C)。
            true_coords (torch.Tensor): 真实的骨架坐标 (B, L, 3, 3) for (N, Ca, C)。
            res_mask (torch.Tensor):    残基掩码 (B, L)，1 表示有效，0 表示 padding。
            str_mask (torch.Tensor):    结构掩码 (B, L)，1 表示计算 loss，0 表示固定 motif。
            
        Returns:
            torch.Tensor: 计算出的 FAPE Loss (标量)。
        """
        # 1. 从真实坐标和预测坐标中提取 Ca 原子坐标作为要变换的点
        # (B, L, 3)
        true_points = true_coords[:, :, 1, :]
        pred_points = pred_coords[:, :, 1, :]
        
        # 2. 为真实坐标和预测坐标构建局部坐标系
        # (B, L, 3, 3), (B, L, 3)
        true_rot, true_trans = self._create_frames(true_coords)
        pred_rot, pred_trans = self._create_frames(pred_coords)

        # 3. 将所有 Ca 点变换到每个残基的局部坐标系下
        # (B, L, L, 3)
        true_transformed = self._frame_transform(true_rot, true_trans, true_points)
        pred_transformed = self._frame_transform(pred_rot, pred_trans, pred_points)

        # 4. 计算变换后点对之间的 L2 距离
        # (B, L, L)
        distance_error = torch.linalg.norm(true_transformed - pred_transformed, dim=-1)

        # 5. 应用截断 (Clamping)
        clamped_error = torch.clamp(distance_error, max=self.clamp_distance)

        # 6. 构建并应用掩码
        res_mask_2d = res_mask.unsqueeze(-1) * res_mask.unsqueeze(-2)
        
        # str_mask: (B, L) -> (B, L, L)
        str_mask_2d = str_mask.unsqueeze(-2)
        
        # 合并两个掩码
        # 最终掩码的形状为 (B, L, L)
        final_mask = res_mask_2d * str_mask_2d

        # 将掩码应用于 loss
        masked_error = clamped_error * final_mask

        # 7. 计算最终的 loss (在所有有效对上求平均)
        loss = torch.sum(masked_error) / (torch.sum(final_mask) + self.eps)
        
        return loss

class BondCoherenceLoss(nn.Module):
    """
    键-结构-序列自洽损失（简化版）。

    约束：
    - 仅对 bond 概率高于给定阈值的残基对施加几何与序列惩罚。
    - 不将几何与序列软目标投影到双随机空间；直接在掩码域内归一化或使用对数项。
    - 几何项由 CA 远距 hinge 改为依据 link.csv 指定原子对的距离 MSE 惩罚：
      对于每个有效残基对 (i,j)，若 link.csv 指定了 (atom_i, atom_j, ref_dist)，
      则惩罚 (‖x_i(atom_i) - x_j(atom_j)‖ - ref_dist)^2。
    - 序列惩罚项使用 - bond_matrix * log(S_ij)，其中 S_ij 来自 `link.csv` 的相容度软目标。
    - 对角线表示"无连接"，不参与 pair 惩罚。
    - 主链骨架约束：当预测两残基之间成键时，约束其 Cα-Cα 距离不要过远（铰链损失）。
    """
    def __init__(self,
                 link_csv_path: str = None,
                 link_info: LinkInfo = None,
                 device: torch.device = 'cpu',
                 lambda_geom: float = 0,
                 lambda_seq: float = 2,
                 lambda_adjacent: float = 0.5,
                 adjacent_sep_weights: list = [1],
                 bond_dist_tol_factor: float = 1.2,
                 use_seq_logits: bool = True,
                 eps: float = 1e-6,
                 bond_threshold: float = 0.75,
                 lambda_ca_backbone: float = 0,
                 ca_max_distance: float = 12,
                 t_geom_threshold: float = 0.8,
                 lambda_entropy: float = 0,
                 geom_mse_cap_value: float = 15,
                 ca_max_distance_cap_value: float = 16,
                 lambda_angle: float = 0,
                 lambda_dihedral: float = 0,
                 angle_cap_value: float = 1.5, #58°
                 dihedral_cap_value: float = 1.5, #58°
                 angle_eps: float = 1e-5,
                 # --- Energy-based consistency (geometry->prob) ---
                 lambda_energy_bce: float = 1,
                 energy_temperature: float = 1,
                 energy_w_dist: float = 2,
                 energy_w_angle: float = 0.5,
                 energy_w_dihedral: float = 0.5,
                 # Hardness-aware JSD focusing
                 energy_jsd_gamma: float = 2.0,
                 energy_jsd_topk_ratio: float = 0.0,
                 # Distance metric for energy-based consistency: 'jsd' | 'w1' | 'w2'
                 energy_distance_metric: str = 'jsd',
                 ):
        """
        Args:
            link_csv_path: 兼容度 CSV 文件路径 (如果提供了 link_info，则可选)。
            link_info: 预初始化的 LinkInfo 对象 (优先使用)。
            lambda_geom_hinge: 几何远距离 hinge 惩罚系数。
            lambda_seq: 序列不相容惩罚系数，实施 -Σ bond_ij log S_ij。
            use_seq_logits: True 则优先使用 `seq_logits`（softmax 后）；否则使用 `seq_labels`。
            compat_default: link.csv 未覆盖到的氨基酸对的默认小正值，避免 log(0)。
            eps: 数值稳定常数。
            bond_threshold: 只有当 bond_matrix[i,j] > bond_threshold 时，才对其施加几何和序列惩罚。
                          设置为 0.0 则恢复惩罚所有非零 bond 的行为。
        """
        super().__init__()
        self.lambda_geom = lambda_geom
        self.lambda_seq = lambda_seq
        self.lambda_adjacent = lambda_adjacent
        self.bond_dist_tol_factor = bond_dist_tol_factor
        self.use_seq_logits = use_seq_logits
        self.eps = eps
        self.bond_threshold = bond_threshold
        self.link_csv_path = link_csv_path
        self.lambda_ca_backbone = lambda_ca_backbone
        self.ca_max_distance = ca_max_distance
        self.ca_max_distance_cap_value = ca_max_distance_cap_value
        self.t_geom_threshold = t_geom_threshold
        self.lambda_entropy = lambda_entropy
        self.geom_mse_cap_value = float(geom_mse_cap_value)
        self.lambda_angle = lambda_angle
        self.lambda_dihedral = lambda_dihedral
        self.angle_cap_value = float(angle_cap_value)
        self.dihedral_cap_value = float(dihedral_cap_value)
        self.angle_eps = float(angle_eps)
        # Energy-based BCE hyperparameters
        self.lambda_energy_bce = float(lambda_energy_bce)
        self.energy_temperature = float(energy_temperature)
        self.energy_w_dist = float(energy_w_dist)
        self.energy_w_angle = float(energy_w_angle)
        self.energy_w_dihedral = float(energy_w_dihedral)
        self.energy_jsd_gamma = float(energy_jsd_gamma)
        self.energy_jsd_topk_ratio = float(energy_jsd_topk_ratio)
        # energy_distance_metric: 'jsd' (Jensen–Shannon), 'w1' (|p-q|), 'w2' ((p-q)^2)
        self.energy_distance_metric = str(energy_distance_metric).lower()

        # 邻近序列间隔惩罚的权重列表：
        # 若提供 [w1, w2, ..., wk]，则对 |i-j|=1..k 分别施加惩罚并按权重加权。
        # 若为 None 或空，则退化为仅 |i-j|=1 的原始行为。
        if adjacent_sep_weights is None:
            adj_weights_tensor = torch.tensor([], dtype=torch.float32)
        else:
            adj_weights_tensor = torch.tensor(adjacent_sep_weights, dtype=torch.float32)
        self.register_buffer('adjacent_sep_weights', adj_weights_tensor)

        if link_info is not None:
            self.link_info = link_info
        elif link_csv_path is not None:
            # For backward compatibility
            self.link_info = LinkInfo(link_csv_path, device=device, compat_default=self.eps)
        else:
            self.link_info = None

        if self.link_info is not None:
            compat = self.link_info.compat_matrix
        else:
            # 如果两者都未提供，则创建一个默认的相容性矩阵
            print("Warning: Neither link_csv_path nor link_info provided to BondCoherenceLoss. Using default compatibility.")
            aa_list = list(aa2num.keys())
            K = len(aa_list) - 1 # 排除 MASK_IDX
            compat = torch.full((K, K), self.eps, device=device)

        self.register_buffer('compat_matrix', compat.to(device))

    def _get_termini_masks(
        self,
        res_mask: torch.Tensor,
        head_mask: torch.Tensor = None,
        tail_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = res_mask.shape
        device = res_mask.device

        if head_mask is not None and tail_mask is not None:
            return head_mask.bool(), tail_mask.bool()

        # Fallback heuristic: find first and last valid residue in each batch item
        # This is an approximation and works best for single, contiguous chains.
        inferred_head = torch.zeros_like(res_mask, dtype=torch.bool)
        inferred_tail = torch.zeros_like(res_mask, dtype=torch.bool)

        for i in range(B):
            valid_indices = torch.where(res_mask[i])[0]
            if len(valid_indices) > 0:
                inferred_head[i, valid_indices[0]] = True
                inferred_tail[i, valid_indices[-1]] = True
        
        return inferred_head, inferred_tail

    def _angle(self, p1, p2, p3):
        """Computes angle p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        # Normalize with eps for stability
        v1_u = F.normalize(v1, p=2, dim=-1, eps=self.angle_eps)
        v2_u = F.normalize(v2, p=2, dim=-1, eps=self.angle_eps)
        # cos(theta) and sin(theta) via cross product magnitude
        cos_theta = torch.sum(v1_u * v2_u, dim=-1)
        sin_theta = torch.norm(torch.cross(v1_u, v2_u, dim=-1), dim=-1)
        # Use atan2(sin, cos) to avoid infinite gradients near |cos|=1
        return torch.atan2(sin_theta, cos_theta + self.eps)

    def _dihedral(self, p0, p1, p2, p3):
        """Computes dihedral p0-p1-p2-p3"""
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = F.normalize(b1, p=2, dim=-1, eps=self.angle_eps)
        v = b0 - torch.sum(b0 * b1_norm, dim=-1, keepdim=True) * b1_norm
        w = b2 - torch.sum(b2 * b1_norm, dim=-1, keepdim=True) * b1_norm
        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.cross(b1_norm, v, dim=-1) * w, dim=-1)
        return torch.atan2(y, x + self.eps)

    def _compute_all_geom_losses(
        self,
        aatype: torch.Tensor,
        all_atom_coords: torch.Tensor,
        res_mask: torch.Tensor,
        head_mask: torch.Tensor = None,
        tail_mask: torch.Tensor = None,
    ) -> dict:
        B, L, A, _ = all_atom_coords.shape
        device = all_atom_coords.device

        if self.link_info is None or not self.link_info.bond_spec:
            zeros = all_atom_coords.new_zeros((B, L, L))
            valid = torch.zeros((B, L, L), dtype=torch.bool, device=device)
            return {
                'dist_mse': zeros, 'dist_valid': valid,
                'angle_i_err_sq': zeros, 'angle_i_valid': valid,
                'angle_j_err_sq': zeros, 'angle_j_valid': valid,
                'dihedral_1_err_sq': zeros, 'dihedral_1_valid': valid,
                'dihedral_2_err_sq': zeros, 'dihedral_2_valid': valid,
            }

        pair_mask = (res_mask.unsqueeze(1).bool() & res_mask.unsqueeze(2).bool())
        eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & (~eye)
        
        # Single-rule per pair: pre-allocate per-pair tensors
        all_rules_p_i = torch.full((B, L, L), -1, dtype=torch.long, device=device)
        all_rules_p_j = torch.full((B, L, L), -1, dtype=torch.long, device=device)
        all_rules_a_i = torch.full((B, L, L), -1, dtype=torch.long, device=device)
        all_rules_a_j = torch.full((B, L, L), -1, dtype=torch.long, device=device)
        all_rules_d_1_i = torch.full((B, L, L), -1, dtype=torch.long, device=device)
        all_rules_d_1_j = torch.full((B, L, L), -1, dtype=torch.long, device=device)
        all_rules_d_2_i = torch.full((B, L, L), -1, dtype=torch.long, device=device)
        all_rules_d_2_j = torch.full((B, L, L), -1, dtype=torch.long, device=device)

        all_rules_ref_dist = torch.zeros((B, L, L), dtype=all_atom_coords.dtype, device=device)
        all_rules_ref_angle_i = torch.zeros((B, L, L), dtype=all_atom_coords.dtype, device=device)
        all_rules_ref_angle_j = torch.zeros((B, L, L), dtype=all_atom_coords.dtype, device=device)
        all_rules_ref_dihedral_1 = torch.zeros((B, L, L), dtype=all_atom_coords.dtype, device=device)
        all_rules_ref_dihedral_2 = torch.zeros((B, L, L), dtype=all_atom_coords.dtype, device=device)
        
        all_rules_d_1_planar = torch.zeros((B, L, L), dtype=torch.bool, device=device)
        all_rules_d_2_planar = torch.zeros((B, L, L), dtype=torch.bool, device=device)

        all_rules_mask = torch.zeros((B, L, L), dtype=torch.bool, device=device)

        _aa2long = [tuple(a.strip() if a is not None else None for a in row) for row in aa2long]
        for (r1, r2), rules in self.link_info.bond_spec.items():
            for k, rule in enumerate(rules):
                def get_atom_idx(res_num, atom_name):
                    if not atom_name: return -1
                    try: return _aa2long[res_num].index(atom_name)
                    except ValueError: return -1

                a1_idx = get_atom_idx(r1, rule.get('atom1'))
                a2_idx = get_atom_idx(r2, rule.get('atom2'))
                
                if a1_idx != -1 and a2_idx != -1:
                    # --- Termini-aware gating ---
                    # Backbone rules (N/C) are only permitted on true termini; sidechain rules are disallowed on terminal clones.
                    atom1_name = (rule.get('atom1') or '').strip().upper()
                    atom2_name = (rule.get('atom2') or '').strip().upper()

                    head_bool = head_mask.bool()
                    tail_bool = tail_mask.bool()
                    non_terminal = ~(head_bool | tail_bool)

                    # Gate for residue i (atom1)
                    if atom1_name == 'N':
                        gate_i = head_bool
                    elif atom1_name == 'C':
                        gate_i = tail_bool
                    else:
                        # sidechain atoms must not be on terminal clones
                        gate_i = non_terminal

                    # Gate for residue j (atom2)
                    if atom2_name == 'N':
                        gate_j = head_bool
                    elif atom2_name == 'C':
                        gate_j = tail_bool
                    else:
                        gate_j = non_terminal

                    # Base type match, then apply termini gating per side, and valid pair mask (off-diagonal, valid residues)
                    pair_indices = (aatype == r1).unsqueeze(2) & (aatype == r2).unsqueeze(1)
                    pair_indices = pair_indices & gate_i.unsqueeze(2) & gate_j.unsqueeze(1) & pair_mask
                    
                    # Only set where not already set by a previous rule
                    new_pairs = pair_indices & (~all_rules_mask)
                    all_rules_p_i[new_pairs] = a1_idx
                    all_rules_p_j[new_pairs] = a2_idx
                    all_rules_ref_dist[new_pairs] = rule.get('dist', 0.0)
                    all_rules_mask[new_pairs] = True

                    # Angle i
                    if rule.get('angle_i_ref') is not None and rule.get('angle_i_anchor') is not None:
                        anchor_idx = get_atom_idx(r1, rule['angle_i_anchor'])
                        if anchor_idx != -1:
                            all_rules_a_i[new_pairs] = anchor_idx
                            all_rules_ref_angle_i[new_pairs] = rule['angle_i_ref']
                    
                    # Angle j
                    if rule.get('angle_j_ref') is not None and rule.get('angle_j_anchor') is not None:
                        anchor_idx = get_atom_idx(r2, rule['angle_j_anchor'])
                        if anchor_idx != -1:
                            all_rules_a_j[new_pairs] = anchor_idx
                            all_rules_ref_angle_j[new_pairs] = rule['angle_j_ref']

                    # Dihedral 1
                    if rule.get('dihedral_1_ref') is not None and rule.get('dihedral_1_anchor_i') and rule.get('dihedral_1_anchor_j'):
                        anchor_i_idx = get_atom_idx(r1, rule['dihedral_1_anchor_i'])
                        anchor_j_idx = get_atom_idx(r2, rule['dihedral_1_anchor_j'])
                        if anchor_i_idx != -1 and anchor_j_idx != -1:
                            all_rules_d_1_i[new_pairs] = anchor_i_idx
                            all_rules_d_1_j[new_pairs] = anchor_j_idx
                            all_rules_ref_dihedral_1[new_pairs] = rule['dihedral_1_ref']
                            all_rules_d_1_planar[new_pairs] = rule['dihedral_1_planar']

                    # Dihedral 2
                    if rule.get('dihedral_2_ref') is not None and rule.get('dihedral_2_anchor_i') and rule.get('dihedral_2_anchor_j'):
                        anchor_i_idx = get_atom_idx(r1, rule['dihedral_2_anchor_i'])
                        anchor_j_idx = get_atom_idx(r2, rule['dihedral_2_anchor_j'])
                        if anchor_i_idx != -1 and anchor_j_idx != -1:
                            all_rules_d_2_i[new_pairs] = anchor_i_idx
                            all_rules_d_2_j[new_pairs] = anchor_j_idx
                            all_rules_ref_dihedral_2[new_pairs] = rule['dihedral_2_ref']
                            all_rules_d_2_planar[new_pairs] = rule['dihedral_2_planar']

        
        # Helper to gather atom coords by single index per pair
        def gather_from_source(atom_indices, is_from_i):
            idx = atom_indices.clamp(min=0)  # (B,L,L)
            src = all_atom_coords.unsqueeze(2).expand(-1, -1, L, -1, -1) if is_from_i else \
                  all_atom_coords.unsqueeze(1).expand(-1, L, -1, -1, -1)  # (B,L,L,A,3)
            coords = torch.gather(src, 3, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 3)).squeeze(3)
            return coords.nan_to_num(0.0)  # (B,L,L,3)

        p_i = gather_from_source(all_rules_p_i, True)
        p_j = gather_from_source(all_rules_p_j, False)
        a_i = gather_from_source(all_rules_a_i, True)
        a_j = gather_from_source(all_rules_a_j, False)
        d_1_i = gather_from_source(all_rules_d_1_i, True)
        d_1_j = gather_from_source(all_rules_d_1_j, False)
        d_2_i = gather_from_source(all_rules_d_2_i, True)
        d_2_j = gather_from_source(all_rules_d_2_j, False)

        # -- Calculate Distances --
        dist = torch.norm(p_i - p_j, dim=-1)  # (B,L,L)
        dist_mse = (dist - all_rules_ref_dist).pow(2)
        dist_valid = all_rules_mask & (all_rules_p_i >= 0) & (all_rules_p_j >= 0) & pair_mask

        # -- Calculate Angles --
        angle_i_cand = self._angle(a_i, p_i, p_j)
        angle_i_err_sq = (angle_i_cand - all_rules_ref_angle_i).pow(2)
        angle_i_valid = dist_valid & (all_rules_a_i >= 0)

        angle_j_cand = self._angle(p_i, p_j, a_j)
        angle_j_err_sq = (angle_j_cand - all_rules_ref_angle_j).pow(2)
        angle_j_valid = dist_valid & (all_rules_a_j >= 0)
        
        # -- Calculate Dihedrals --
        dihedral_1_cand = self._dihedral(d_1_i, p_i, p_j, d_1_j)
        diff_1 = dihedral_1_cand - all_rules_ref_dihedral_1
        # Wrap to principal value in [-pi, pi] (2π-periodic)
        wrapped_1 = torch.remainder(diff_1 + torch.pi, 2 * torch.pi) - torch.pi
        # If planar: treat φ and φ+π as equivalent, wrap to [-π/2, π/2]
        planar_wrapped_1 = torch.remainder(diff_1 + 0.5 * torch.pi, torch.pi) - 0.5 * torch.pi
        dihedral_1_err_sq = torch.where(all_rules_d_1_planar, planar_wrapped_1.pow(2), wrapped_1.pow(2))
        dihedral_1_valid = dist_valid & (all_rules_d_1_i >= 0) & (all_rules_d_1_j >= 0)

        dihedral_2_cand = self._dihedral(d_2_i, p_i, p_j, d_2_j)
        diff_2 = dihedral_2_cand - all_rules_ref_dihedral_2
        wrapped_2 = torch.remainder(diff_2 + torch.pi, 2 * torch.pi) - torch.pi
        planar_wrapped_2 = torch.remainder(diff_2 + 0.5 * torch.pi, torch.pi) - 0.5 * torch.pi
        dihedral_2_err_sq = torch.where(all_rules_d_2_planar, planar_wrapped_2.pow(2), wrapped_2.pow(2))
        dihedral_2_valid = dist_valid & (all_rules_d_2_i >= 0) & (all_rules_d_2_j >= 0)

        return {
            'dist_mse': dist_mse, 'dist_valid': dist_valid,
            'angle_i_err_sq': angle_i_err_sq, 'angle_i_valid': angle_i_valid,
            'angle_j_err_sq': angle_j_err_sq, 'angle_j_valid': angle_j_valid,
            'dihedral_1_err_sq': dihedral_1_err_sq, 'dihedral_1_valid': dihedral_1_valid,
            'dihedral_2_err_sq': dihedral_2_err_sq, 'dihedral_2_valid': dihedral_2_valid,
            "dihedral_1_cand": dihedral_1_cand, "dihedral_2_cand": dihedral_2_cand, 
            "angle_i_cand": angle_i_cand, "angle_j_cand": angle_j_cand,
            # Extras for detailed debugging/printing
            "dist_val": dist,
            "rules_mask": all_rules_mask,
            "rules_p_i": all_rules_p_i, "rules_p_j": all_rules_p_j,
            "rules_a_i": all_rules_a_i, "rules_a_j": all_rules_a_j,
            "rules_d_1_i": all_rules_d_1_i, "rules_d_1_j": all_rules_d_1_j,
            "rules_d_2_i": all_rules_d_2_i, "rules_d_2_j": all_rules_d_2_j,
            "rules_d_1_planar": all_rules_d_1_planar, "rules_d_2_planar": all_rules_d_2_planar,
            "ref_dist": all_rules_ref_dist,
            "ref_angle_i": all_rules_ref_angle_i, "ref_angle_j": all_rules_ref_angle_j,
            "ref_dihedral_1": all_rules_ref_dihedral_1, "ref_dihedral_2": all_rules_ref_dihedral_2,
        }


    def _mask_off_diagonal(self, M: torch.Tensor) -> torch.Tensor:
        B, L, _ = M.shape
        eye = torch.eye(L, device=M.device, dtype=torch.bool).unsqueeze(0)
        return M.masked_fill(eye, 0.0)

    def _build_res_mask_2d(self, res_mask: torch.Tensor) -> torch.Tensor:
        return res_mask.unsqueeze(-1).bool() & res_mask.unsqueeze(-2).bool()


    def forward(self,
                bond_matrix: torch.Tensor,
                res_mask: torch.Tensor,
                seq_logits: torch.Tensor = None,
                seq_labels: torch.Tensor = None,
                true_seq: torch.Tensor = None,
                mask_2d: torch.Tensor = None,
                all_atom_coords: torch.Tensor = None,
                aatype: torch.Tensor = None,
                t: torch.Tensor = None,
                head_mask: torch.Tensor = None,
                tail_mask: torch.Tensor = None,
                true_bond_matrix: torch.Tensor = None,
                detach_bond = False,
                return_terms: bool = False,
                silent: bool = False
                ) -> torch.Tensor:
        B, L, _ = bond_matrix.shape
        device = bond_matrix.device

        if mask_2d is None:
            mask_2d = self._build_res_mask_2d(res_mask)
        eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
        pair_mask = mask_2d & (~eye)

        # --- Validate/prepare head/tail masks ---
        # If not provided, infer a coarse head/tail per batch item
        if head_mask is None or tail_mask is None:
            head_mask, tail_mask = self._get_termini_masks(res_mask, head_mask, tail_mask)
        head_mask = head_mask.to(device=device, dtype=torch.bool)
        tail_mask = tail_mask.to(device=device, dtype=torch.bool)
        assert head_mask.shape == (B, L) and tail_mask.shape == (B, L), "head_mask and tail_mask must be [B,L]"

        # Build termini-related masks: include head/tail and their adjacent body residues
        # head_to_body_idx, tail_to_body_idx = compute_terminal_body_maps(res_mask, head_mask, tail_mask)
        # head_body_mask = torch.zeros_like(head_mask, dtype=torch.bool)
        # tail_body_mask = torch.zeros_like(tail_mask, dtype=torch.bool)
        # for b in range(B):
        #     hb_idx = torch.where(head_to_body_idx[b] >= 0)[0]
        #     tb_idx = torch.where(tail_to_body_idx[b] >= 0)[0]
        #     if hb_idx.numel() > 0:
        #         head_body_mask[b, head_to_body_idx[b, hb_idx]] = True
        #     if tb_idx.numel() > 0:
        #         tail_body_mask[b, tail_to_body_idx[b, tb_idx]] = True
        termini_related_mask = head_mask | tail_mask # | head_body_mask | tail_body_mask

        # 创建并应用 bond 阈值掩码
        bond_thresh_mask = bond_matrix > self.bond_threshold

        final_penalty_mask = pair_mask & bond_thresh_mask
        if true_bond_matrix is not None:
            final_penalty_mask = final_penalty_mask & true_bond_matrix.bool()
        final_penalty_mask_float = final_penalty_mask.float()
        if detach_bond:
            bond_matrix_reweight = bond_matrix.detach() 
        else:
            bond_matrix_reweight = bond_matrix
        # 基于 link.csv 指定原子对的几何 MSE 惩罚（加权平均归一化）
        geom_mse = bond_matrix_reweight.new_tensor(0.0)
        angle_term = bond_matrix_reweight.new_tensor(0.0)
        dihedral_term = bond_matrix_reweight.new_tensor(0.0)

        if all_atom_coords is not None and aatype is not None:
            geom_losses = self._compute_all_geom_losses(
                aatype, all_atom_coords, res_mask, head_mask,tail_mask
            )
            
            mse_per_pair = geom_losses['dist_mse']
            valid_dist = geom_losses['dist_valid']
            angle_i_err_sq = geom_losses['angle_i_err_sq']
            valid_angle_i = geom_losses['angle_i_valid']
            angle_j_err_sq = geom_losses['angle_j_err_sq']
            valid_angle_j = geom_losses['angle_j_valid']
            dihedral_1_err_sq = geom_losses['dihedral_1_err_sq']
            valid_dihedral_1 = geom_losses['dihedral_1_valid']
            dihedral_2_err_sq = geom_losses['dihedral_2_err_sq']
            valid_dihedral_2 = geom_losses['dihedral_2_valid']
            if not silent:
                print("valid_pairs:", valid_dist.sum().item())
                print("final_penalty_mask_float:",final_penalty_mask_float.sum().item())
                print("valid")
                print("bond_matrix_reweight:",bond_matrix_reweight.sum().item())
            
            # 输出超过阈值的bond_mat的平均
            if not silent:
                above_thresh_bonds = bond_matrix[final_penalty_mask_float.bool()]
                if above_thresh_bonds.numel() > 0:
                    print("bond_matrix above threshold mean:", above_thresh_bonds.float().mean().item())
                else:
                    print("bond_matrix above threshold mean: no bonds above threshold")
            
            # Helper for capping, weighting, and reducing loss terms
            def compute_term(error_sq, valid_mask, cap_value, t, term_name="term"):
                # Cap the error
                error_sq = error_sq.nan_to_num(0.0)
                coef = (cap_value / error_sq.clamp_min(self.eps)).detach()
                error_sq_capped = torch.where(error_sq <= cap_value, error_sq, coef * error_sq)
                
                # Weight and gate by time
                weight = bond_matrix_reweight * final_penalty_mask_float * valid_mask.float()
                if t is not None:
                    if t.dim() == 0: t = t.view(1)
                    t_mask = (t > self.t_geom_threshold).float().view(-1, 1, 1)
                    weight = weight * t_mask / (1 - self.t_geom_threshold)
                    if not silent:
                        print(f"[{term_name}] t={t.item():.3f}, t_mask={t_mask.item():.3f}, valid_count={valid_mask.sum().item()}, weight_sum={weight.sum().item():.3f}")
                elif not silent:
                    print(f"[{term_name}] t=None, valid_count={valid_mask.sum().item()}, weight_sum={weight.sum().item():.3f}")
                num = (error_sq_capped * weight).sum()
                den = (final_penalty_mask_float * valid_mask.float()).sum()
                return num / (den + self.eps)

            # --- Distance Term ---
            geom_mse = compute_term(mse_per_pair, valid_dist, self.geom_mse_cap_value, t, "geom_mse")
            if not silent:
                print("mse_per_pair (valid & penalized):", mse_per_pair[(final_penalty_mask_float * valid_dist.float()).bool()])

            # --- Angle Term ---
            angle_err_sq = (angle_i_err_sq * valid_angle_i.float() + angle_j_err_sq * valid_angle_j.float())
            valid_angle = valid_angle_i | valid_angle_j
            angle_term = compute_term(angle_err_sq, valid_angle, self.angle_cap_value, t, "angle_term")
            
            # --- Dihedral Term ---
            dihedral_err_sq = (dihedral_1_err_sq * valid_dihedral_1.float() + dihedral_2_err_sq * valid_dihedral_2.float())
            valid_dihedral = valid_dihedral_1 | valid_dihedral_2
            dihedral_term = compute_term(dihedral_err_sq, valid_dihedral, self.dihedral_cap_value, t, "dihedral_term")

            # --- Energy-based geometry -> probability consistency (p_geom vs bond_matrix) ---
            energy_jsd_term = bond_matrix_reweight.new_tensor(0.0)
            if self.lambda_energy_bce > 0:
                # Build combined per-pair errors with capping and dimensionless scaling
                def _cap_err(err_sq, cap_val):
                    err_sq = err_sq.nan_to_num(0.0)
                    coef = (cap_val / err_sq.clamp_min(self.eps)).detach()
                    return torch.where(err_sq <= cap_val, err_sq, coef * err_sq)

                # Distance
                err_d = _cap_err(mse_per_pair, self.geom_mse_cap_value)
                u_d = err_d #/ max(self.geom_mse_cap_value, self.eps)

                # Angle: average of available i/j, else 0
                num_a = (valid_angle_i.float() + valid_angle_j.float()).clamp_min(1.0)
                avg_angle_err = (angle_i_err_sq * valid_angle_i.float() + angle_j_err_sq * valid_angle_j.float()) / num_a
                err_a = _cap_err(avg_angle_err, self.angle_cap_value)
                u_a = err_a #/ max(self.angle_cap_value, self.eps)

                # Dihedral: average of available 1/2, else 0
                num_h = (valid_dihedral_1.float() + valid_dihedral_2.float()).clamp_min(1.0)
                avg_dih_err = (dihedral_1_err_sq * valid_dihedral_1.float() + dihedral_2_err_sq * valid_dihedral_2.float()) / num_h
                err_h = _cap_err(avg_dih_err, self.dihedral_cap_value)
                u_h = err_h #/ max(self.dihedral_cap_value, self.eps)

                # Total energy U (dimensionless)
                U = self.energy_w_dist * u_d + self.energy_w_angle * u_a + self.energy_w_dihedral * u_h

                # Convert to probability via Boltzmann-like mapping
                T = max(self.energy_temperature, self.eps)
                p_geom = torch.exp(-U / T).clamp(self.eps, 1.0 - self.eps)

                # Compute divergence/distance between bond_matrix (p) and p_geom (q)
                # Masking: valid pairs with any geometric signal, off-diagonal
                energy_valid = valid_dist | valid_angle | valid_dihedral
                energy_mask = pair_mask & energy_valid

                if energy_mask.any():
                    p = bond_matrix.clamp(self.eps, 1.0 - self.eps)
                    q = p_geom  # no detach: allow mutual learning

                    # Select metric: 'jsd' (default), 'w1' (|p-q|), 'w2' ((p-q)^2)
                    metric = getattr(self, 'energy_distance_metric', 'jsd')

                    if metric == 'jsd':
                        m = 0.5 * (p + q)
                        kl_pm = p * torch.log(p / m.clamp(self.eps, 1.0)) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - m).clamp(self.eps, 1.0))
                        kl_qm = q * torch.log(q / m.clamp(self.eps, 1.0)) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - m).clamp(self.eps, 1.0))
                        base = 0.5 * (kl_pm + kl_qm)
                    elif metric == 'w1':
                        base = (p - q).abs()
                    elif metric == 'w2':
                        base = (p - q) ** 2
                    else:
                        # Fallback to JSD on invalid option
                        m = 0.5 * (p + q)
                        kl_pm = p * torch.log(p / m.clamp(self.eps, 1.0)) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - m).clamp(self.eps, 1.0))
                        kl_qm = q * torch.log(q / m.clamp(self.eps, 1.0)) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - m).clamp(self.eps, 1.0))
                        base = 0.5 * (kl_pm + kl_qm)

                    # Hardness-aware weighting: focal-style w = base^gamma
                    weights = (base.clamp_min(self.eps)).detach() ** self.energy_jsd_gamma
                    weights = weights * energy_mask.float().detach()

                    # Optional time gating, aligned with geometric terms
                    if t is not None:
                        if t.dim() == 0: t = t.view(1)
                        t_mask = (t > self.t_geom_threshold).float().view(-1, 1, 1)
                        weights_before_gate = weights.sum()
                        weights = weights * t_mask / (1 - self.t_geom_threshold)
                        if not silent:
                            print(f"[energy_bce/{metric}] t={t.item():.3f}, t_mask={t_mask.item():.3f}, weights_before_gate={weights_before_gate.item():.3f}, weights_after_gate={weights.sum().item():.3f}")

                    # Detailed printing for pairs with bond > threshold
                    if not silent and False:
                        try:
                            # Gather helper tensors from geom_losses if available
                            dist_val = geom_losses.get('dist_val', None)
                            angle_i_cand = geom_losses.get('angle_i_cand', None)
                            angle_j_cand = geom_losses.get('angle_j_cand', None)
                            dihedral_1_cand = geom_losses.get('dihedral_1_cand', None)
                            dihedral_2_cand = geom_losses.get('dihedral_2_cand', None)
                            ref_dist = geom_losses.get('ref_dist', None)
                            ref_angle_i = geom_losses.get('ref_angle_i', None)
                            ref_angle_j = geom_losses.get('ref_angle_j', None)
                            ref_dihedral_1 = geom_losses.get('ref_dihedral_1', None)
                            ref_dihedral_2 = geom_losses.get('ref_dihedral_2', None)
                            rules_p_i = geom_losses.get('rules_p_i', None)
                            rules_p_j = geom_losses.get('rules_p_j', None)
                            rules_a_i = geom_losses.get('rules_a_i', None)
                            rules_a_j = geom_losses.get('rules_a_j', None)
                            rules_d_1_i = geom_losses.get('rules_d_1_i', None)
                            rules_d_1_j = geom_losses.get('rules_d_1_j', None)
                            rules_d_2_i = geom_losses.get('rules_d_2_i', None)
                            rules_d_2_j = geom_losses.get('rules_d_2_j', None)
                            rules_d_1_planar = geom_losses.get('rules_d_1_planar', None)
                            rules_d_2_planar = geom_losses.get('rules_d_2_planar', None)

                            def _aa_name(idx: int) -> str:
                                try:
                                    if isinstance(num2aa, (list, tuple)):
                                        if 0 <= idx < len(num2aa):
                                            return str(num2aa[idx])
                                        return str(idx)
                                    if isinstance(num2aa, dict):
                                        return str(num2aa.get(idx, idx))
                                except Exception:
                                    pass
                                return str(idx)

                            # Candidate mask: energy_mask and bond > threshold
                            cand_mask = energy_mask & (bond_matrix > self.bond_threshold)
                            print(f"[energy_bce/{metric}] Detailed pairs (bond > {self.bond_threshold:.2f}):")
                            for b in range(B):
                                idx_pairs = cand_mask[b].nonzero(as_tuple=False)
                                for ij in idx_pairs:
                                    i = int(ij[0].item()); j = int(ij[1].item())
                                    # AA types and names
                                    ri = int(aatype[b, i].item()) if aatype is not None else -1
                                    rj = int(aatype[b, j].item()) if aatype is not None else -1
                                    name_i = _aa_name(ri)
                                    name_j = _aa_name(rj)
                                    # Atom indices and names (safe)
                                    def _atom_name(res_num: int, atom_idx: int) -> str:
                                        try:
                                            if atom_idx < 0: return "-"
                                            nm = aa2long[res_num][atom_idx]
                                            return str(nm).strip() if nm is not None else f"atom{atom_idx}"
                                        except Exception:
                                            return f"atom{atom_idx}"
                                    pi_idx = int(rules_p_i[b, i, j].item()) if rules_p_i is not None else -1
                                    pj_idx = int(rules_p_j[b, i, j].item()) if rules_p_j is not None else -1
                                    ai_idx = int(rules_a_i[b, i, j].item()) if rules_a_i is not None else -1
                                    aj_idx = int(rules_a_j[b, i, j].item()) if rules_a_j is not None else -1
                                    d1i_idx = int(rules_d_1_i[b, i, j].item()) if rules_d_1_i is not None else -1
                                    d1j_idx = int(rules_d_1_j[b, i, j].item()) if rules_d_1_j is not None else -1
                                    d2i_idx = int(rules_d_2_i[b, i, j].item()) if rules_d_2_i is not None else -1
                                    d2j_idx = int(rules_d_2_j[b, i, j].item()) if rules_d_2_j is not None else -1
                                    pi_name = _atom_name(ri, pi_idx)
                                    pj_name = _atom_name(rj, pj_idx)
                                    ai_name = _atom_name(ri, ai_idx)
                                    aj_name = _atom_name(rj, aj_idx)
                                    d1i_name = _atom_name(ri, d1i_idx)
                                    d1j_name = _atom_name(rj, d1j_idx)
                                    d2i_name = _atom_name(ri, d2i_idx)
                                    d2j_name = _atom_name(rj, d2j_idx)

                                    # Values (dist/angle/dihedral)
                                    def _to_deg(x):
                                        try:
                                            return float(x.item()) * 180.0 / math.pi
                                        except Exception:
                                            try:
                                                return float(x) * 180.0 / math.pi
                                            except Exception:
                                                return None
                                    def _fmt(val, suffix=""):
                                        return f"{val:.2f}{suffix}" if (val is not None and not math.isnan(val)) else "NA"

                                    d_val = float(dist_val[b, i, j].item()) if dist_val is not None else float('nan')
                                    d_ref = float(ref_dist[b, i, j].item()) if ref_dist is not None else float('nan')
                                    ai_val = _to_deg(angle_i_cand[b, i, j]) if angle_i_cand is not None else None
                                    aj_val = _to_deg(angle_j_cand[b, i, j]) if angle_j_cand is not None else None
                                    ai_ref = _to_deg(ref_angle_i[b, i, j]) if ref_angle_i is not None else None
                                    aj_ref = _to_deg(ref_angle_j[b, i, j]) if ref_angle_j is not None else None
                                    d1_val = _to_deg(dihedral_1_cand[b, i, j]) if dihedral_1_cand is not None else None
                                    d2_val = _to_deg(dihedral_2_cand[b, i, j]) if dihedral_2_cand is not None else None
                                    d1_ref = _to_deg(ref_dihedral_1[b, i, j]) if ref_dihedral_1 is not None else None
                                    d2_ref = _to_deg(ref_dihedral_2[b, i, j]) if ref_dihedral_2 is not None else None
                                    d1_planar = bool(rules_d_1_planar[b, i, j].item()) if rules_d_1_planar is not None else False
                                    d2_planar = bool(rules_d_2_planar[b, i, j].item()) if rules_d_2_planar is not None else False

                                    pij = float(p[b, i, j].item())
                                    qij = float(q[b, i, j].item())
                                    baseij = float(base[b, i, j].item())

                                    print(f"  b={b} i={i}({name_i}) j={j}({name_j})  atoms: {pi_name}-{pj_name}  "
                                          f"d={_fmt(d_val,'Å')} ref={_fmt(d_ref,'Å')}  "
                                          f"ai[{ai_name}]={_fmt(ai_val,'°')} ref={_fmt(ai_ref,'°')}  "
                                          f"aj[{aj_name}]={_fmt(aj_val,'°')} ref={_fmt(aj_ref,'°')}  "
                                          f"d1[{d1i_name},{d1j_name}]={_fmt(d1_val,'°')} ref={_fmt(d1_ref,'°')} planar={d1_planar}  "
                                          f"d2[{d2i_name},{d2j_name}]={_fmt(d2_val,'°')} ref={_fmt(d2_ref,'°')} planar={d2_planar}  "
                                          f"p={pij:.3f} q={qij:.3f} base={baseij:.4f}")
                        except Exception as e:
                            try:
                                print(f"[energy_bce/{metric}] detailed pair print failed: {e}")
                            except Exception:
                                pass

                    # Optional top-k focusing by ratio over masked elements
                    if self.energy_jsd_topk_ratio > 0:
                        k = int((weights > 0).sum().item() * float(self.energy_jsd_topk_ratio))
                        if k > 0:
                            flat_w = weights.view(-1)
                            flat_base = base.view(-1)
                            masked_idx = (flat_w > 0).nonzero(as_tuple=False).squeeze(-1)
                            if masked_idx.numel() > 0:
                                sel_vals = flat_base[masked_idx]
                                topk_vals, topk_idx = torch.topk(sel_vals, k=min(k, sel_vals.numel()))
                                keep = torch.zeros_like(flat_w)
                                keep[masked_idx[topk_idx]] = 1.0
                                weights = (flat_w * keep).view_as(weights)

                    weighted_loss = base * weights
                    denom_e = weights.sum() + self.eps
                    energy_jsd_term = weighted_loss.sum() / denom_e
                    if not silent:
                        print(f"[energy_bce/{metric}] base_sum={base.sum().item():.3f}, weighted_sum={weighted_loss.sum().item():.3f}, denom={denom_e.item():.3f}")
                        try:
                            print(f"[energy_bce/{metric}] energy_jsd_term={float(energy_jsd_term.item()):.6f}")
                        except Exception:
                            pass
            else:
                energy_jsd_term = bond_matrix_reweight.new_tensor(0.0)

            
        # If only geometric terms are requested, return them early
        if return_terms:
            try:
                g = float(geom_mse.detach().item()) if isinstance(geom_mse, torch.Tensor) else float(geom_mse)
            except Exception:
                g = float(geom_mse)
            try:
                a = float(angle_term.detach().item()) if isinstance(angle_term, torch.Tensor) else float(angle_term)
            except Exception:
                a = float(angle_term)
            try:
                d = float(dihedral_term.detach().item()) if isinstance(dihedral_term, torch.Tensor) else float(dihedral_term)
            except Exception:
                d = float(dihedral_term)
            return {"geom_mse": g, "angle_term": a, "dihedral_term": d}

        # 主链 Cα 距离铰链损失（只在预测成键的残基对上约束）
        ca_backbone_term = bond_matrix.new_tensor(0.0)
        if all_atom_coords is not None and self.lambda_ca_backbone > 0:
            # 全部残基的 CA 坐标固定在原子维度索引 1
            # all_atom_coords: (B, L, A, 3)
            ca_coords = all_atom_coords[:, :, 1, :]  # (B, L, 3)
            ca_coord_ok = ~torch.isnan(ca_coords).any(dim=-1)  # (B, L)
            ca_coords = ca_coords.nan_to_num(0.0)
            # 有效成对掩码（CA 坐标有效的 pair）。final_penalty_mask 已包含非对角与残基掩码
            ca_pair_valid = (ca_coord_ok.unsqueeze(1) & ca_coord_ok.unsqueeze(2))

            # Cα-Cα 距离
            diff_ca = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)
            dist_ca = torch.linalg.norm(diff_ca, dim=-1)

            # 铰链损失：max(0, d_CA - d_max)^2
            hinge_ca = F.relu(dist_ca - self.ca_max_distance) ** 2

            ca_cap_value = self.ca_max_distance_cap_value
            coef = (ca_cap_value / hinge_ca.clamp_min(self.eps)).detach()
            hinge_ca = torch.where(hinge_ca <= ca_cap_value, hinge_ca, coef * hinge_ca)
            ca_weight = bond_matrix_reweight * final_penalty_mask_float * ca_pair_valid.float()
            ca_num = (hinge_ca * ca_weight).sum()
            ca_den = (final_penalty_mask_float * ca_pair_valid).sum()
            ca_backbone_term = ca_num / (ca_den + self.eps)

        # 相邻残基间的bond惩罚扩展：鼓励近邻 (|i-j|=1..k) 不形成 bond，支持按距离加权
        adjacency_term = bond_matrix.new_tensor(0.0)
        if self.lambda_adjacent > 0:
            if self.adjacent_sep_weights.numel() == 0:
                # 兼容旧行为：仅 |i-j| = 1
                idx = torch.arange(L - 1, device=device)
                adj_mask = torch.zeros((L, L), dtype=torch.bool, device=device)
                adj_mask[idx, idx + 1] = True
                adj_mask[idx + 1, idx] = True
                adj_mask = adj_mask.unsqueeze(0).expand(B, -1, -1)
                adj_mask = adj_mask & pair_mask
                adj_mask_f = adj_mask.float()
                num_adj = adj_mask_f.sum().clamp_min(1.0)
                adjacency_term = (
                    -torch.log((1 - bond_matrix).clamp(self.eps, 1.0 - self.eps))
                    * adj_mask_f
                ).sum() / num_adj
            else:
                # 扩展到 |i-j| = d, d=1..k
                max_d = min(L - 1, int(self.adjacent_sep_weights.numel()))
                if max_d > 0:
                    neg_log_prob = -torch.log((1.0 - bond_matrix).clamp(self.eps, 1.0 - self.eps))
                    for d in range(1, max_d + 1):
                        w_d = self.adjacent_sep_weights[d - 1]
                        if w_d <= 0:
                            continue
                        sep_mask = torch.zeros((L, L), dtype=torch.bool, device=device)
                        # 上下偏移 d 的两条对角线
                        i = torch.arange(L - d, device=device)
                        sep_mask[i, i + d] = True
                        sep_mask[i + d, i] = True
                        sep_mask = sep_mask.unsqueeze(0).expand(B, -1, -1)
                        sep_mask = sep_mask & pair_mask
                        sep_mask_f = sep_mask.float()
                        num_pairs = sep_mask_f.sum().clamp_min(1.0)
                        term_d = (neg_log_prob * sep_mask_f).sum() / num_pairs
                        adjacency_term = adjacency_term + w_d * term_d

        # 序列不相容惩罚：- bond * log(S_ij)
        seq_term = bond_matrix_reweight.new_tensor(0.0)
        if self.lambda_seq > 0 and (seq_logits is not None or seq_labels is not None):
            Kc = self.compat_matrix.shape[0]
            if seq_logits is not None and self.use_seq_logits:
                probs = F.softmax(seq_logits, dim=-1)
            else:
                assert seq_labels is not None, "seq_labels must be provided if not using seq_logits"
                # Clamp labels to valid range to avoid CUDA device asserts in one_hot
                seq_labels_safe = seq_labels.long().clamp(0, Kc - 1)
                probs = F.one_hot(seq_labels_safe, num_classes=Kc).float()
            
            if probs.shape[-1] != Kc:
                # 确保维度匹配
                p_Kc = probs.shape[-1]
                if p_Kc > Kc:
                    probs = probs[..., :Kc]
                else: # p_Kc < Kc
                    pad = torch.zeros(*probs.shape[:-1], Kc - p_Kc, device=device, dtype=probs.dtype)
                    probs = torch.cat([probs, pad], dim=-1)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            Cp = torch.matmul(probs, self.compat_matrix)
            S = torch.matmul(Cp, probs.transpose(-1, -2))
            S = S.clamp_min(self.eps) # 保证 > 0, 避免 log(0)
            
            # 序列不相容惩罚（加权平均归一化）
            seq_weight = bond_matrix_reweight * final_penalty_mask_float
            # Exclude termini and their adjacent body residues from sequence compatibility
            exclude_2d = ~(termini_related_mask.unsqueeze(1) | termini_related_mask.unsqueeze(2))
            seq_weight = seq_weight * exclude_2d.float()
            seq_num = (-(torch.log(S)) * seq_weight).sum()
            seq_den = (final_penalty_mask_float * exclude_2d.float()).sum()
            seq_term = seq_num / (seq_den + self.eps)
            # --- Debug printing of residue pairs by bond category during seq_term ---
            try:
                if self.link_info is not None and (seq_logits is not None or seq_labels is not None):
                    with torch.no_grad():
                        # Use predicted labels if logits provided, else provided labels
                        if seq_logits is not None and self.use_seq_logits:
                            seq_types = probs.argmax(dim=-1)  # [B, L]
                        else:
                            # keep consistent with clamped labels used for one_hot
                            seq_types = seq_labels_safe

                        # Consider pairs that are penalized (bond > threshold and valid, off-diagonal)
                        penalized_mask = final_penalty_mask.bool()

                        def classify_rules(rule_list):
                            """Classify a rule list into a bond category by atom names only."""
                            # Normalize atom names
                            def norm(atom_name):
                                return (atom_name or '').strip().upper()
                            for r in rule_list:
                                a1, a2 = norm(r.get('atom1')), norm(r.get('atom2'))
                                if a1 == 'SG' and a2 == 'SG':
                                    return 'Disulfide'
                            for r in rule_list:
                                a1, a2 = norm(r.get('atom1')), norm(r.get('atom2'))
                                if ((a1 in ('OG', 'OG1', 'OH') and a2 in ('CG', 'CD')) or
                                    (a2 in ('OG', 'OG1', 'OH') and a1 in ('CG', 'CD'))):
                                    return 'Lactone'
                            for r in rule_list:
                                a1, a2 = norm(r.get('atom1')), norm(r.get('atom2'))
                                if (a1 == 'NZ' and a2 in ('CG', 'CD')) or (a2 == 'NZ' and a1 in ('CG', 'CD')):
                                    return 'Amide'
                            for r in rule_list:
                                a1, a2 = norm(r.get('atom1')), norm(r.get('atom2'))
                                if a1 in ('N', 'C') or a2 in ('N', 'C'):
                                    return 'Head/Tail'
                            return 'Other'

                        # Track unique type pairs and instance counts
                        categories_types = {
                            'Disulfide': set(),
                            'Lactone': set(),
                            'Amide': set(),
                            'Head/Tail': set(),
                            'Other': set(),
                        }
                        categories_counts = {
                            'Disulfide': 0,
                            'Lactone': 0,
                            'Amide': 0,
                            'Head/Tail': 0,
                            'Other': 0,
                        }

                        for b in range(B):
                            idx_pairs = penalized_mask[b].nonzero(as_tuple=False)
                            for ij in idx_pairs:
                                i, j = int(ij[0].item()), int(ij[1].item())
                                r1 = int(seq_types[b, i].item())
                                r2 = int(seq_types[b, j].item())

                                rules_12 = list(self.link_info.bond_spec.get((r1, r2), [])) if (self.link_info is not None and self.link_info.bond_spec) else []
                                rules_21 = list(self.link_info.bond_spec.get((r2, r1), [])) if (self.link_info is not None and self.link_info.bond_spec) else []
                                rule_list = rules_12 + rules_21

                                category = classify_rules(rule_list)
                                # Robust AA index->name mapping for list or dict num2aa
                                def _aa_name(idx: int) -> str:
                                    try:
                                        if isinstance(num2aa, (list, tuple)):
                                            if 0 <= idx < len(num2aa):
                                                return str(num2aa[idx])
                                            return str(idx)
                                        if isinstance(num2aa, dict):
                                            return str(num2aa.get(idx, idx))
                                    except Exception:
                                        pass
                                    return str(idx)
                                name1 = _aa_name(r1)
                                name2 = _aa_name(r2)
                                pair_name = f"{name1}-{name2}"
                                category_key = category if category in categories_types else 'Other'
                                categories_types[category_key].add(pair_name)
                                categories_counts[category_key] = categories_counts.get(category_key, 0) + 1

                        # Print summary (limit list length to avoid spam)
                        for cat, pairs in categories_types.items():
                            count = int(categories_counts.get(cat, 0))
                            if count == 0:
                                continue
                            sample = sorted(list(pairs))[:32]
                            print(f"[seq_term] {cat}: {count} instances, {len(pairs)} unique types -> {sample}")
            except Exception as e:
                print(f"[seq_term] pair categorization print failed: {e}")

            # --- Debug printing of true residue pairs by bond category (if true_seq provided) ---
            try:
                if (true_seq is not None) and (self.link_info is not None):
                    with torch.no_grad():
                        Kc = int(self.compat_matrix.shape[0])
                        seq_types_true = true_seq.long().clamp(0, Kc - 1)
                        penalized_mask = final_penalty_mask.bool()

                        def classify_rules(rule_list):
                            def norm(atom_name):
                                return (atom_name or '').strip().upper()
                            for r in rule_list:
                                a1, a2 = norm(r.get('atom1')), norm(r.get('atom2'))
                                if a1 == 'SG' and a2 == 'SG':
                                    return 'Disulfide'
                            for r in rule_list:
                                a1, a2 = norm(r.get('atom1')), norm(r.get('atom2'))
                                if ((a1 in ('OG', 'OG1', 'OH') and a2 in ('CG', 'CD')) or
                                    (a2 in ('OG', 'OG1', 'OH') and a1 in ('CG', 'CD'))):
                                    return 'Lactone'
                            for r in rule_list:
                                a1, a2 = norm(r.get('atom1')), norm(r.get('atom2'))
                                if (a1 == 'NZ' and a2 in ('CG', 'CD')) or (a2 == 'NZ' and a1 in ('CG', 'CD')):
                                    return 'Amide'
                            for r in rule_list:
                                a1, a2 = norm(r.get('atom1')), norm(r.get('atom2'))
                                if a1 in ('N', 'C') or a2 in ('N', 'C'):
                                    return 'Head/Tail'
                            return 'Other'

                        categories_types = {
                            'Disulfide': set(),
                            'Lactone': set(),
                            'Amide': set(),
                            'Head/Tail': set(),
                            'Other': set(),
                        }
                        categories_counts = {
                            'Disulfide': 0,
                            'Lactone': 0,
                            'Amide': 0,
                            'Head/Tail': 0,
                            'Other': 0,
                        }

                        for b in range(B):
                            idx_pairs = penalized_mask[b].nonzero(as_tuple=False)
                            for ij in idx_pairs:
                                i, j = int(ij[0].item()), int(ij[1].item())
                                r1 = int(seq_types_true[b, i].item())
                                r2 = int(seq_types_true[b, j].item())

                                rules_12 = list(self.link_info.bond_spec.get((r1, r2), [])) if (self.link_info is not None and self.link_info.bond_spec) else []
                                rules_21 = list(self.link_info.bond_spec.get((r2, r1), [])) if (self.link_info is not None and self.link_info.bond_spec) else []
                                rule_list = rules_12 + rules_21

                                category = classify_rules(rule_list)

                                def _aa_name(idx: int) -> str:
                                    try:
                                        if isinstance(num2aa, (list, tuple)):
                                            if 0 <= idx < len(num2aa):
                                                return str(num2aa[idx])
                                            return str(idx)
                                        if isinstance(num2aa, dict):
                                            return str(num2aa.get(idx, idx))
                                    except Exception:
                                        pass
                                    return str(idx)

                                name1 = _aa_name(r1)
                                name2 = _aa_name(r2)
                                pair_name = f"{name1}-{name2}"
                                category_key = category if category in categories_types else 'Other'
                                categories_types[category_key].add(pair_name)
                                categories_counts[category_key] = categories_counts.get(category_key, 0) + 1

                        for cat, pairs in categories_types.items():
                            count = int(categories_counts.get(cat, 0))
                            if count == 0:
                                continue
                            sample = sorted(list(pairs))[:32]
                            print(f"[seq_true] {cat}: {count} instances, {len(pairs)} unique types -> {sample}")
            except Exception as e:
                print(f"[seq_true] pair categorization print failed: {e}")
    
        # 熵正则项：对称双随机矩阵的行/列分布熵（排除对角、掩码无效项）
        entropy_term = bond_matrix.new_tensor(0.0)
        if self.lambda_entropy > 0:
            valid_pairs_f = mask_2d.float()  # off-diagonal & valid residues
            p_valid = (bond_matrix * valid_pairs_f)
            # row-wise normalized distribution (exclude diagonal)
            row_sums = p_valid.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            q_row = p_valid / row_sums
            row_entropy = -(q_row.clamp_min(self.eps) * q_row.clamp_min(self.eps).log()).sum(dim=-1)  # [B, L]
            # column-wise normalized distribution (exclude diagonal)
            col_sums = p_valid.sum(dim=-2, keepdim=True).clamp_min(self.eps)
            q_col = p_valid / col_sums
            col_entropy = -(q_col.clamp_min(self.eps) * q_col.clamp_min(self.eps).log()).sum(dim=-2)  # [B, L]
            res_mask_f = res_mask.float()
            denom = res_mask_f.sum().clamp_min(1.0)
            row_term = (row_entropy * res_mask_f).sum() / denom
            col_term = (col_entropy * res_mask_f).sum() / denom
            entropy_term = 0.5 * (row_term + col_term)

        if not silent:
            print("seq_term",seq_term.item())
            print("geom_mse",geom_mse.item() if isinstance(geom_mse, torch.Tensor) else float(geom_mse))
            print("angle_term", angle_term.item() if isinstance(angle_term, torch.Tensor) else float(angle_term))
            print("dihedral_term", dihedral_term.item() if isinstance(dihedral_term, torch.Tensor) else float(dihedral_term))
            print("energy_jsd_term", float(energy_jsd_term.item()))
            print("adjacency_term", adjacency_term.item())
            print("ca_backbone_term", ca_backbone_term.item())
            print("entropy_term", entropy_term.item())

        loss = (
            self.lambda_geom * geom_mse
            + self.lambda_angle * angle_term
            + self.lambda_dihedral * dihedral_term
            + self.lambda_seq * seq_term
            + self.lambda_energy_bce * energy_jsd_term
            + self.lambda_adjacent * adjacency_term
            + self.lambda_ca_backbone * ca_backbone_term
            + self.lambda_entropy * entropy_term
        )
        return loss


class DSMCrossEntropyLoss(nn.Module):
    """
    一个功能全面的损失函数，结合了：
    1. Focal Loss: 专注于困难样本。
    2. 归一化: 消除 gamma 对损失尺度的影响。
    3. 对角线/非对角线加权: 解决目标矩阵主要为单位矩阵时的平凡解问题。
    """
    def __init__(self, eps=1e-6, gamma=2, off_diagonal_weight=1):
        """
        Args:
            eps (float): 一个很小的数，用于防止 log(0) 和除以零。
            gamma (float): Focal Loss 的聚焦参数。值越大，对困难样本的关注度越高。
            off_diagonal_weight (float): 非对角线正样本 (target=1) 的权重。
                                           用于提高非恒等匹配的重要性。
        """
        super().__init__()
        self.eps = eps
        self.gamma = gamma
        self.off_diagonal_weight = off_diagonal_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            input (torch.Tensor): 预测的概率矩阵，形状为 (B, N, N)。
            target (torch.Tensor): 真实的目标矩阵，形状为 (B, N, N)。
            mask_2d (torch.Tensor): 一个布尔类型的2D掩码，形状为 (B, N, N)。
                                    `True` 表示该位置参与损失计算。

        Returns:
            torch.Tensor: 计算出的标量损失值。
        """
        B, N, _ = input.shape
        if mask_2d is None:
            mask_2d = torch.ones_like(input, dtype=torch.bool, device=input.device)

        # --- 1) 数值稳定的概率与互补概率 ---
        p = input.clamp(self.eps, 1.0 - self.eps)
        q = 1.0 - p

        # --- 2) Focal 因子（正负类分开计算） ---
        pos_focal = torch.pow(1.0 - p, self.gamma)
        neg_focal = torch.pow(p, self.gamma)

        # --- 3) 非对角正样本权重矩阵（仅用于正类） ---
        identity_matrix = torch.eye(N, device=input.device, dtype=target.dtype)
        identity_matrix = identity_matrix.unsqueeze(0).expand_as(target)
        pos_class_weight = torch.where(identity_matrix == 1, self.off_diagonal_weight, self.off_diagonal_weight)
        neg_class_weight = torch.ones_like(target)  # 负类统一权重=1

        # --- 4) 正负类对称二元交叉熵（带Focal与类权重），并应用掩码 ---
        mask_f = mask_2d.float()
        pos_loss = -target * pos_focal * torch.log(p) * pos_class_weight * mask_f
        neg_loss = -(1.0 - target) * neg_focal * torch.log(q) * neg_class_weight * mask_f
        loss_elements = pos_loss + neg_loss

        # --- 5) 归一化（仅用正样本权重作为分母） ---
        #normalizer = (target * pos_class_weight * pos_focal * mask_f).sum().detach()
        normalizer = (target * pos_class_weight * pos_focal * mask_f).sum().detach()
        loss = loss_elements.sum() / (normalizer + self.eps)

        # 调试信息：观测真实正样本的非对角预测均值
        try:
            offdiag_mask = (target * (1 - identity_matrix.float()) == 1)
            denom = torch.clamp(offdiag_mask.sum(), min=0.95)
            print("real offdiagonal: ", p[offdiag_mask].sum() / denom, "denom: ", denom)
        except Exception:
            pass
                # 额外调试信息：打印基于 mask 的 ROC-AUC（带平局处理）
        try:
            with torch.no_grad():
                valid_mask = mask_2d.bool()
                y_true = (target[valid_mask] > 0.5).float()
                y_score = p[valid_mask]
                n = y_true.numel()
                if n > 0:
                    n_pos = int((y_true == 1).sum().item())
                    n_neg = int(n - n_pos)
                else:
                    n_pos, n_neg = 0, 0

                if n_pos > 0 and n_neg > 0:
                    # 使用秩统计（Mann-Whitney U），对相同分数采用平均秩
                    s, order = torch.sort(y_score)
                    ranks = torch.arange(1, s.numel() + 1, device=s.device, dtype=s.dtype)

                    if s.numel() == 1:
                        mean_ranks_orig = ranks
                    else:
                        # 识别相同分数组并计算组平均秩
                        diff = torch.ne(s[1:], s[:-1])
                        group_ids = torch.zeros_like(s, dtype=torch.long)
                        group_ids[1:] = diff
                        group_ids = torch.cumsum(group_ids, dim=0)
                        num_groups = int(group_ids[-1].item()) + 1
                        sums = torch.zeros(num_groups, device=s.device, dtype=s.dtype)
                        counts = torch.zeros(num_groups, device=s.device, dtype=s.dtype)
                        sums = sums.scatter_add(0, group_ids, ranks)
                        counts = counts.scatter_add(0, group_ids, torch.ones_like(ranks))
                        mean_ranks_per_group = sums / counts.clamp_min(1)
                        mean_ranks_sorted = mean_ranks_per_group[group_ids]
                        mean_ranks_orig = torch.empty_like(mean_ranks_sorted)
                        mean_ranks_orig[order] = mean_ranks_sorted

                    sum_ranks_pos = mean_ranks_orig[y_true == 1].sum()
                    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg + self.eps)
                    print(f"ROC-AUC: {float(auc.item()):.4f} (pos={n_pos}, neg={n_neg})")
                else:
                    print("ROC-AUC: NA (insufficient pos/neg)")
        except Exception:
            pass


        return loss

class ClashLoss(nn.Module):
    """
    计算一个极简的冲突损失。

    此版本会惩罚任何两个不同的原子，只要它们的距离小于设定的阈值，
    无论它们之间是否存在化学键。
    """
    def __init__(self, min_dis=1.15, clash_cap_value: float = 20, eps: float = 1e-8):
        """
        初始化ClashLoss模块。

        Args:
            min_dis (float): 允许的最小距离（单位：埃）。
                             距离小于此值的原子对将受到惩罚。
        """
        super().__init__()
        self.min_dis = min_dis
        self.clash_cap_value = clash_cap_value
        self.eps = eps
        # 假设 _precompute_data() 方法会被调用且其依赖的全局变量存在
        self._precompute_data()

    def _precompute_data(self):
        """
        Pre-computes and stores data required for the loss calculation,
        such as atom masks and bond masks for each amino acid type.
        (This method is unchanged from the original).
        """
        # This part requires your global variables: num2aa, aa2long, aabonds
        # The logic remains the same as in your original code.
        num_aa_types = len(num2aa)
        num_atoms = 14 # Use 14 for heavy atoms only

        atom_masks = []
        bond_masks = []

        for i in range(num_aa_types):
            # Atom mask: True for existing atoms among the first 14 slots
            atom_mask = [name is not None for name in aa2long[i][:num_atoms]]
            atom_masks.append(torch.tensor(atom_mask, dtype=torch.bool))

            # Bond mask: True for bonded atom pairs
            bond_mask = torch.zeros((num_atoms, num_atoms), dtype=torch.bool)
            if i < len(aabonds):
                # Create a map only for the first 14 heavy atoms
                atom_map = {name.strip(): j for j, name in enumerate(aa2long[i][:num_atoms]) if name is not None}
                for bond_pair in aabonds[i]:
                    a1, a2 = bond_pair
                    # Only consider bonds where both atoms are heavy atoms
                    if a1.strip() in atom_map and a2.strip() in atom_map:
                        idx1, idx2 = atom_map[a1.strip()], atom_map[a2.strip()]
                        bond_mask[idx1, idx2] = True
                        bond_mask[idx2, idx1] = True
            bond_masks.append(bond_mask)

        self.register_buffer('aa_atom_masks', torch.stack(atom_masks))
        self.register_buffer('aa_bond_masks', torch.stack(bond_masks))

    def forward(self, allatom_xyz, seq_pred, res_mask):
        """
        计算极简的冲突损失。

        Args:
            allatom_xyz (torch.Tensor): 原子坐标张量，形状 (B, L, 14, 3)。
            seq_pred (torch.Tensor): 预测的氨基酸序列索引，形状 (B, L)。
            res_mask (torch.Tensor): 残基掩码张量，形状 (B, L)。

        Returns:
            torch.Tensor: 计算出的冲突损失。
        """
        B, L, N_atom, _ = allatom_xyz.shape
        device = allatom_xyz.device
        
        if N_atom != 14:
            raise ValueError(f"ClashLoss expects N_atom=14 (heavy atoms), but got {N_atom}")
        
        # 1. 计算所有原子之间的成对距离（先清理 NaN，避免传播）
        #    注意：我们用掩码屏蔽无效原子（包含 NaN 坐标或类型不存在），
        #    因此将 NaN 用 0 填充不会产生伪零距离的贡献。
        coord_valid = ~torch.isnan(allatom_xyz).any(dim=-1)  # [B, L, N_atom]
        coords_clean = allatom_xyz.nan_to_num(0.0)
        xyz_flat = coords_clean.reshape(B, L * N_atom, 3)
        dists = torch.cdist(xyz_flat, xyz_flat, p=2)

        # 将预计算的掩码移动到正确的设备
        aa_atom_masks = self.aa_atom_masks.to(device)
        # self.aa_bond_masks 不再需要，可以注释掉
        
        # a. 处理氨基酸类型中不存在的原子 + 坐标缺失（NaN）
        atom_mask_per_res = aa_atom_masks[seq_pred] & coord_valid
        atom_mask_flat = atom_mask_per_res.reshape(B, L * N_atom)
        valid_atom_mask = atom_mask_flat.unsqueeze(2) & atom_mask_flat.unsqueeze(1)

        # b. 处理被padding的残基
        res_mask_flat = res_mask.repeat_interleave(N_atom, dim=1).bool()
        res_mask_2d = res_mask_flat.unsqueeze(2) & res_mask_flat.unsqueeze(1)

        # c. 合并掩码
        # 【核心修改点】我们不再需要 bond_mask，直接合并其他掩码
        # 旧逻辑: final_mask = valid_atom_mask & res_mask_2d & (~bond_mask_flat)
        final_mask = valid_atom_mask & res_mask_2d
        
        # d. 必须排除原子与自身的冲突（对角线）
        final_mask.diagonal(dim1=-2, dim2=-1).fill_(False)

        # 3. 计算冲突惩罚值
        clash_values = torch.nn.functional.relu(self.min_dis - dists) ** 2
        clash_loss = (clash_values * final_mask).sum(dim=(-1,-2))
        if self.clash_cap_value is not None:
            coef = (self.clash_cap_value / clash_loss.clamp_min(self.eps)).detach()
            clash_loss = torch.where(clash_loss <= self.clash_cap_value, clash_loss, coef * clash_loss)
        clash_loss = clash_loss.sum() / B # 按批次大小归一化

        return clash_loss
    
class OpenFoldClashLoss(nn.Module):
    """
    OpenFold-style clash loss with link.csv-aware tolerance relaxation for bonded residue pairs.

    - Between-residue clashes: follows OpenFold formulation, but for residue pairs
      that are predicted bonded (bond_mat > threshold) AND whose specific atom pairs
      are allowed by link.csv, we relax the lower-bound to the CSV avg_distance and
      apply smaller soft/hard tolerances.
    - Within-residue clashes: penalize van der Waals overlaps among atoms in the
      same residue (simple VdW lower bound, excluding identical-atom pairs).
    """
    def __init__(self,
                 link_csv_path: str = "/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/link.csv",
                 bond_threshold: float = 0.5,
                 overlap_tolerance_soft: float = 1.5,
                 overlap_tolerance_hard: float = 1.5,
                 link_tol_soft: float = 0.20,
                 link_tol_hard: float = 0.10,
                 eps: float = 1e-10,
                 device: str = 'cpu',
                 include_within: bool = False,
                 log_raw: bool = True,
                 treat_adjacent_as_bonded: bool = True,
                 peptide_cn_refdist: float = 1.33,
                 debug_print_pairs: bool = False,
                 debug_pairs_topk: int = 20,
                 debug_print_batch_idx: int = 0):
        super().__init__()
        self.link_csv_path = link_csv_path
        self.bond_threshold = float(bond_threshold)
        self.overlap_tolerance_soft = float(overlap_tolerance_soft)
        self.overlap_tolerance_hard = float(overlap_tolerance_hard)
        self.link_tol_soft = float(link_tol_soft)
        self.link_tol_hard = float(link_tol_hard)
        self.eps = float(eps)
        self.device = device
        self.include_within = bool(include_within)
        self.log_raw = bool(log_raw)
        self.treat_adjacent_as_bonded = bool(treat_adjacent_as_bonded)
        self.peptide_cn_refdist = float(peptide_cn_refdist)
        self.debug_print_pairs = bool(debug_print_pairs)
        self.debug_pairs_topk = int(debug_pairs_topk)
        self.debug_print_batch_idx = int(debug_print_batch_idx)

        self._precompute_static()
        self._compile_link_rules()

    def _precompute_static(self) -> None:
        """
        Precompute per-AA atom14 existence mask and per-atom14 VdW radii.
        Uses rfdiff.chemical.aa2long for atom name lists (first 14 heavy atoms).
        """
        # van der Waals radii [Å]
        vdw = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}

        K = len(num2aa)
        num_atoms = 14

        aa_atom_exists = []   # [K, 14] bool
        aa_atom_radius = []   # [K, 14] float
        aa_isN_rows = []      # [K, 14] bool
        aa_isC_rows = []      # [K, 14] bool

        for i in range(K):
            names = aa2long[i][:num_atoms]
            exists_row = []
            radius_row = []
            row_isN = []
            row_isC = []
            for name in names:
                if name is None:
                    exists_row.append(False)
                    radius_row.append(0.0)
                    row_isN.append(False)
                    row_isC.append(False)
                else:
                    clean = name.strip().upper()
                    exists_row.append(True)
                    elem = clean[0] if len(clean) > 0 else 'C'
                    radius_row.append(float(vdw.get(elem, 1.70)))
                    row_isN.append(clean == 'N')
                    row_isC.append(clean == 'C')
            aa_atom_exists.append(torch.tensor(exists_row, dtype=torch.bool))
            aa_atom_radius.append(torch.tensor(radius_row, dtype=torch.float32))
            aa_isN_rows.append(torch.tensor(row_isN, dtype=torch.bool))
            aa_isC_rows.append(torch.tensor(row_isC, dtype=torch.bool))

        self.register_buffer('aa_atom_exists', torch.stack(aa_atom_exists).to(self.device))  # [K,14]
        self.register_buffer('aa_atom_radius', torch.stack(aa_atom_radius).to(self.device))  # [K,14]
        self.register_buffer('aa_isN_mask', torch.stack(aa_isN_rows).to(self.device))        # [K,14]
        self.register_buffer('aa_isC_mask', torch.stack(aa_isC_rows).to(self.device))        # [K,14]

    def _compile_link_rules(self) -> None:
        """
        Compile link.csv rules to per-(res1,res2,atom14_i,atom14_j) masks and reference distances.
        """
        K = self.aa_atom_exists.shape[0]
        num_atoms = self.aa_atom_exists.shape[1]

        allowed = torch.zeros((K, K, num_atoms, num_atoms), dtype=torch.bool)
        refdist = torch.zeros((K, K, num_atoms, num_atoms), dtype=torch.float32)

        if self.link_csv_path is None:
            self.register_buffer('link_allowed_mask', allowed)
            self.register_buffer('link_ref_dist', refdist)
            return

        link = LinkInfo(self.link_csv_path)
        # Build fast lookup for atom14 index per AA using aa2long first-14 names
        atom_index_per_aa = []  # list[dict[str,int]] of length K
        for i in range(K):
            name_to_idx = {}
            for j, nm in enumerate(aa2long[i][:num_atoms]):
                if nm is None:
                    continue
                name_to_idx[nm.strip().upper()] = j
            atom_index_per_aa.append(name_to_idx)

        for (r1, r2), rules in (link.bond_spec or {}).items():
            if r1 >= K or r2 >= K:
                continue
            idx_map_1 = atom_index_per_aa[r1]
            idx_map_2 = atom_index_per_aa[r2]
            for rule in rules:
                a1 = (rule.get('atom1') or '').strip().upper()
                a2 = (rule.get('atom2') or '').strip().upper()
                try:
                    dist_val = float(rule.get('dist'))
                except Exception:
                    continue
                if a1 in idx_map_1 and a2 in idx_map_2:
                    i14 = idx_map_1[a1]
                    j14 = idx_map_2[a2]
                    allowed[r1, r2, i14, j14] = True
                    refdist[r1, r2, i14, j14] = dist_val
        self.register_buffer('link_allowed_mask', allowed)
        self.register_buffer('link_ref_dist', refdist)

    def _build_atom_exists(self, seq_pred: torch.Tensor, coord_valid: torch.Tensor) -> torch.Tensor:
        """Build (B,L,14) atom-existence mask combining AA template and coord validity."""
        # aa mask by seq
        B, L = seq_pred.shape
        K = self.aa_atom_exists.shape[0]
        seq_safe = seq_pred.clamp(min=0, max=K - 1)
        aa_mask = self.aa_atom_exists[seq_safe]         # [B,L,14]
        return aa_mask & coord_valid

    def _build_atom_radius(self, seq_pred: torch.Tensor) -> torch.Tensor:
        """Build (B,L,14) per-atom VdW radii by AA and atom index."""
        B, L = seq_pred.shape
        K = self.aa_atom_radius.shape[0]
        seq_safe = seq_pred.clamp(min=0, max=K - 1)
        return self.aa_atom_radius[seq_safe]

    def _gather_link_masks(self, seq_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build (B,L,L,14,14) link_allowed and ref distances from (K,K,14,14) by indexing (seq_i, seq_j).
        """
        B, L = seq_pred.shape
        K = self.link_allowed_mask.shape[0]
        device = seq_pred.device
        seq_safe = seq_pred.clamp(min=0, max=K - 1)

        # Move buffers to same device as indices for advanced indexing
        allowed_buf = self.link_allowed_mask.to(device)
        ref_buf = self.link_ref_dist.to(device)

        allowed_list = []
        ref_list = []
        for b in range(B):
            sb = seq_safe[b]
            # [L,L,14,14]
            allowed_ij = allowed_buf[sb.unsqueeze(1), sb.unsqueeze(0)]
            ref_ij = ref_buf[sb.unsqueeze(1), sb.unsqueeze(0)]
            allowed_list.append(allowed_ij)
            ref_list.append(ref_ij)
        allowed = torch.stack(allowed_list, dim=0)
        ref = torch.stack(ref_list, dim=0)
        return allowed, ref

    def forward(self,
                allatom_xyz: torch.Tensor,   # (B,L,14,3)
                seq_pred: torch.Tensor,      # (B,L)
                res_mask: torch.Tensor,      # (B,L)
                bond_mat: torch.Tensor = None,
                head_mask: torch.Tensor = None,
                tail_mask: torch.Tensor = None) -> torch.Tensor:
        B, L, A, _ = allatom_xyz.shape
        assert A == 14, "OpenFoldClashLoss expects 14 heavy atoms per residue"
        device = allatom_xyz.device
        seq_pred = seq_pred.long()

        # Sanitize coords and existence
        coords = allatom_xyz.nan_to_num(0.0)
        nonzero_mask = (coords.abs().sum(dim=-1) > 1e-6)
        coord_valid = (~torch.isnan(allatom_xyz).any(dim=-1)) & nonzero_mask   # (B,L,14)
        atom_exists = self._build_atom_exists(seq_pred, coord_valid).to(dtype=coords.dtype)
        atom_radius = self._build_atom_radius(seq_pred).to(dtype=coords.dtype)

        # Pairwise distances: (B,L,L,14,14)
        dxyz = (
            coords[:, :, None, :, None, :] - coords[:, None, :, None, :, :]
        )
        dists = torch.sqrt(self.eps + (dxyz ** 2).sum(dim=-1))

        # Valid pair mask: atom exists on both sides, valid residues, exclude diagonal residue pairs, use upper triangle
        atom_pair_mask = (
            atom_exists[:, :, None, :, None] * atom_exists[:, None, :, None, :]
        )
        res_pair_mask = (res_mask[:, :, None].bool() & res_mask[:, None, :].bool())
        eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
        upper = torch.triu(torch.ones((L, L), dtype=torch.bool, device=device), diagonal=1).unsqueeze(0)
        # Exclude termini residues from regular clashes
        if head_mask is None or tail_mask is None:
            head_mask, tail_mask = infer_termini_masks(res_mask)
        else:
            head_mask = head_mask.to(device=device, dtype=torch.bool)
            tail_mask = tail_mask.to(device=device, dtype=torch.bool)
        termini_mask = head_mask | tail_mask
        exclude_termini_pairs = ~(termini_mask.unsqueeze(1) | termini_mask.unsqueeze(2))
        pair_mask_2d = res_pair_mask & (~eye) & upper & exclude_termini_pairs  # (B,L,L)
        valid_mask = atom_pair_mask * pair_mask_2d.unsqueeze(-1).unsqueeze(-1)

        # Base lower bound: VdW radii sum
        r_i = atom_radius[:, :, None, :, None]
        r_j = atom_radius[:, None, :, None, :]
        lower_bound = r_i + r_j  # (B,L,L,14,14)

        # Base tolerances
        tol_soft = dists.new_full((B, L, L, A, A), float(self.overlap_tolerance_soft))
        tol_hard = dists.new_full((B, L, L, A, A), float(self.overlap_tolerance_hard))

        # Link-based relaxation on bonded residue pairs + adjacent CN relaxation
        if bond_mat is not None and (self.link_allowed_mask.any() or self.treat_adjacent_as_bonded):
            connected = (bond_mat > self.bond_threshold) & (~eye)  # (B,L,L)
            link_allowed, link_ref = self._gather_link_masks(seq_pred)  # (B,L,L,14,14)
            # termini-gated N/C atoms
            Kmask = self.aa_atom_exists.shape[0]
            seq_safe_mask = seq_pred.clamp(min=0, max=Kmask - 1)
            isN = torch.zeros((B, L, 14), dtype=torch.bool, device=device)
            isC = torch.zeros((B, L, 14), dtype=torch.bool, device=device)
            # cheap detection by atom names in aa2long
            for b in range(B):
                for i in range(L):
                    aa = int(seq_safe_mask[b, i].item())
                    for a in range(14):
                        nm = aa2long[aa][a]
                        if nm is None:
                            continue
                        nm_s = str(nm).strip().upper()
                        if nm_s == 'N':
                            isN[b, i, a] = True
                        elif nm_s == 'C':
                            isC[b, i, a] = True
            gate_i = (~isN | head_mask.unsqueeze(-1)) & (~isC | tail_mask.unsqueeze(-1))
            gate_j = (~isN | head_mask.unsqueeze(-1)) & (~isC | tail_mask.unsqueeze(-1))
            gate_pair = gate_i.unsqueeze(2).unsqueeze(-1) & gate_j.unsqueeze(1).unsqueeze(-2)
            link_mask = link_allowed & gate_pair & connected.unsqueeze(-1).unsqueeze(-1)

            # Override lower bound and tolerances where link applies
            lower_bound = torch.where(link_mask, link_ref.to(lower_bound.dtype), lower_bound)
            tol_soft = torch.where(link_mask, tol_soft.new_tensor(self.link_tol_soft), tol_soft)
            tol_hard = torch.where(link_mask, tol_hard.new_tensor(self.link_tol_hard), tol_hard)

            # --- Head/Tail bridging semantics ---
            # If bond_mat indicates links to head/tail markers, map them to actual backbone atoms:
            #   head link -> use N on the first body residue after head
            #   tail link -> use C on the last body residue before tail
            # Build per-batch bridging pairs (2D) and apply CSV- and CN-based relaxations accordingly
            bridge_pairs_csv = torch.zeros((B, L, L), dtype=torch.bool, device=device)
            bridge_pairs_cn = torch.zeros((B, L, L), dtype=torch.bool, device=device)

            # Collect all head/tail indices per batch (support multi-chain)
            heads_per_b = [torch.where(head_mask[b])[0].tolist() for b in range(B)]
            tails_per_b = [torch.where(tail_mask[b])[0].tolist() for b in range(B)]

            # Compute head/tail -> body index maps using shared helper
            head_to_body_idx, tail_to_body_idx = compute_terminal_body_maps(res_mask, head_mask, tail_mask)

            # Helper to set upper-triangular index
            def _set_pair(mask2d, b, i, j, val=True):
                ii = int(i)
                jj = int(j)
                if ii == jj:
                    return
                a = min(ii, jj)
                c = max(ii, jj)
                mask2d[b, a, c] = val

            # Build CSV relax pairs where bonds attach to terminal markers
            for b in range(B):
                # handle all head-tail pairs
                for h in heads_per_b[b]:
                    hb = int(head_to_body_idx[b, h].item()) if head_to_body_idx[b, h] >= 0 else min(h + 1, L - 1)
                    # any bonds to/from head marker propagate to hb
                    conn_to_head = connected[b, :, h] | connected[b, h, :]
                    ks_h = torch.where(conn_to_head)[0]
                    for k in ks_h.tolist():
                        if k == h:
                            continue
                        _set_pair(bridge_pairs_csv, b, hb, k, True)
                    for t in tails_per_b[b]:
                        tb = int(tail_to_body_idx[b, t].item()) if tail_to_body_idx[b, t] >= 0 else max(t - 1, 0)
                        # cyclic head-tail bond triggers CN relax between tb and hb
                        if bool(connected[b, h, t] or connected[b, t, h] or connected[b, hb, tb] or connected[b, tb, hb]):
                            _set_pair(bridge_pairs_cn, b, tb, hb, True)
                # handle bonds to/from tail markers
                for t in tails_per_b[b]:
                    tb = int(tail_to_body_idx[b, t].item()) if tail_to_body_idx[b, t] >= 0 else max(t - 1, 0)
                    conn_to_tail = connected[b, :, t] | connected[b, t, :]
                    ks_t = torch.where(conn_to_tail)[0]
                    for k in ks_t.tolist():
                        if k == t:
                            continue
                        _set_pair(bridge_pairs_csv, b, tb, k, True)

            # Apply CN lower bound specifically to C(i)-N(j) atom pairs for cyclic bridge
            if bridge_pairs_cn.any():
                # atom-level mask: C@i-N@j OR N@i-C@j (both orientations)
                C_i = isC.unsqueeze(2).unsqueeze(-1)  # [B,L,1,14,1]
                N_j = isN.unsqueeze(1).unsqueeze(-2)  # [B,1,L,1,14]
                N_i = isN.unsqueeze(2).unsqueeze(-1)
                C_j = isC.unsqueeze(1).unsqueeze(-2)
                cn_atom_mask = (C_i & N_j) | (N_i & C_j)  # [B,L,L,14,14]
                bridge_cn_mask = bridge_pairs_cn.unsqueeze(-1).unsqueeze(-1) & cn_atom_mask
                cn_lb = lower_bound.new_full(lower_bound.shape, float(self.peptide_cn_refdist))
                lower_bound = torch.where(bridge_cn_mask, cn_lb, lower_bound)
                tol_soft = torch.where(bridge_cn_mask, tol_soft.new_tensor(self.link_tol_soft), tol_soft)
                tol_hard = torch.where(bridge_cn_mask, tol_hard.new_tensor(self.link_tol_hard), tol_hard)

                # Debug: print head/tail mapping and whether CN bridge applied for selected batch
                if getattr(self, 'debug_print_pairs', False):
                    try:
                        bdbg = int(max(0, min(getattr(self, 'debug_print_batch_idx', 0), B - 1)))
                        heads_dbg = heads_per_b[bdbg]
                        tails_dbg = tails_per_b[bdbg]
                        print(f"[ClashDebug] heads={heads_dbg} tails={tails_dbg}")
                        for hd in heads_dbg:
                            hb = int(head_to_body_idx[bdbg, hd].item()) if head_to_body_idx[bdbg, hd] >= 0 else (hd+1 if hd+1<L else L-1)
                            for td in tails_dbg:
                                tb = int(tail_to_body_idx[bdbg, td].item()) if tail_to_body_idx[bdbg, td] >= 0 else (td-1 if td-1>=0 else 0)
                                conn_ht = bool(connected[bdbg, hd, td].item()) if L>max(hd,td) else False
                                conn_hbtb = bool(connected[bdbg, hb, tb].item()) if L>max(hb,tb) else False
                                a = min(tb, hb); c = max(tb, hb)
                                print(f"[ClashDebug] h={hd}->hb={hb} t={td}->tb={tb} conn[h,t]={conn_ht} conn[hb,tb]={conn_hbtb} bridge_cn={bool(bridge_pairs_cn[bdbg,a,c].item())}")
                    except Exception:
                        pass

            # Apply CSV lower bounds for head/tail bridging to arbitrary residues
            if bridge_pairs_csv.any():
                bridge_csv_mask = bridge_pairs_csv.unsqueeze(-1).unsqueeze(-1)
                bridge_link_mask = link_allowed & bridge_csv_mask
                lower_bound = torch.where(bridge_link_mask, link_ref.to(lower_bound.dtype), lower_bound)
                tol_soft = torch.where(bridge_link_mask, tol_soft.new_tensor(self.link_tol_soft), tol_soft)
                tol_hard = torch.where(bridge_link_mask, tol_hard.new_tensor(self.link_tol_hard), tol_hard)

            # Adjacent CN relaxation (unconditional when enabled)
            adj = torch.zeros((L, L), dtype=torch.bool, device=device)
            idx = torch.arange(L - 1, device=device)
            adj[idx, idx + 1] = True
            adj = adj.unsqueeze(0).expand(B, -1, -1)
            if self.treat_adjacent_as_bonded:
                # approximate: set CN lower bound and small tolerance
                # use per-AA CN refdist via aa2long mapping (C index 2? we already use ref from link path above normally)
                cn_lb = lower_bound.new_full(lower_bound.shape, float(self.peptide_cn_refdist))
                cn_mask = adj.unsqueeze(-1).unsqueeze(-1)
                lower_bound = torch.where(cn_mask, cn_lb, lower_bound)
                tol_soft = torch.where(cn_mask, tol_soft.new_tensor(self.link_tol_soft), tol_soft)
                tol_hard = torch.where(cn_mask, tol_hard.new_tensor(self.link_tol_hard), tol_hard)

        # Compute soft overlap error (OpenFold style)
        # relu(lower_bound - tol_soft - dists)
        dists_to_low_error = valid_mask * torch.nn.functional.relu(lower_bound - tol_soft - dists)

        # Mean over violating pairs only
        between_sum = dists_to_low_error.sum()
        denom = (dists_to_low_error > 0).sum().clamp_min(1)
        between_mean = between_sum / denom

        # Within-residue simple VdW clash (optional)
        if self.include_within:
            # (B,L,14,14)
            d_intra = torch.sqrt(self.eps + (coords[:, :, :, None, :] - coords[:, :, None, :, :]).pow(2).sum(dim=-1))
            exist_intra = atom_exists.bool()
            mask_intra = (exist_intra[:, :, :, None] & exist_intra[:, :, None, :])
            # Exclude identical atoms
            mask_intra.diagonal(dim1=-2, dim2=-1).fill_(False)
            r_i_intra = atom_radius[:, :, :, None]
            r_j_intra = atom_radius[:, :, None, :]
            lb_intra = r_i_intra + r_j_intra
            tol_intra = d_intra.new_full(d_intra.shape, float(self.overlap_tolerance_soft))
            err_intra = mask_intra * torch.nn.functional.relu(lb_intra - tol_intra - d_intra)
            within_sum = err_intra.sum()
            denom_intra = (err_intra > 0).sum().clamp_min(1)
            within_mean = within_sum / denom_intra
        else:
            within_sum = between_sum.new_tensor(0.0)
            denom_intra = between_sum.new_tensor(1.0)
            within_mean = between_sum.new_tensor(0.0)

        if self.log_raw:
            try:
                print(f"[ClashLoss] between_sum(no_div)={between_sum.item():.6f} denom={denom.item():.0f}")
                print(f"[ClashLoss] within_sum(no_div)={within_sum.item():.6f} denom={denom_intra.item():.0f}")
                print(f"[ClashLoss] total_sum(no_div)={(between_sum + within_sum).item():.6f}")
            except Exception:
                pass

        # Optional: print top-K clashing atom pairs and their residues (debug)
        if getattr(self, 'debug_print_pairs', False):
            try:
                with torch.no_grad():
                    b = int(max(0, min(getattr(self, 'debug_print_batch_idx', 0), B - 1)))
                    viol = (lower_bound - tol_soft - dists)[b]
                    mask_b = (valid_mask[b] > 0) & (viol > 0)
                    num = int(mask_b.sum().item())
                    if num == 0:
                        print(f"[ClashPairs] batch={b}: no clashes")
                    else:
                        idxs = mask_b.nonzero(as_tuple=False)
                        vals = viol[mask_b]
                        k = int(max(1, min(getattr(self, 'debug_pairs_topk', 20), vals.numel())))
                        topv, topi = torch.topk(vals.view(-1), k=k)
                        # Helper to map AA index to name
                        def _aa_name(idx: int) -> str:
                            try:
                                if isinstance(num2aa, (list, tuple)):
                                    if 0 <= idx < len(num2aa):
                                        return str(num2aa[idx])
                                    return str(idx)
                                if isinstance(num2aa, dict):
                                    return str(num2aa.get(idx, idx))
                            except Exception:
                                pass
                            return str(idx)
                        sb = seq_pred[b].detach().to('cpu')
                        for rank in range(k):
                            i, j, ai, aj = [int(x) for x in idxs[topi[rank]].tolist()]
                            ri = int(sb[i].item())
                            rj = int(sb[j].item())
                            name_i = _aa_name(ri)
                            name_j = _aa_name(rj)
                            try:
                                atom_i = aa2long[ri][ai]
                            except Exception:
                                atom_i = None
                            try:
                                atom_j = aa2long[rj][aj]
                            except Exception:
                                atom_j = None
                            atom_i = str(atom_i).strip() if atom_i is not None else f"atom{ai}"
                            atom_j = str(atom_j).strip() if atom_j is not None else f"atom{aj}"
                            d_val = float(dists[b, i, j, ai, aj].item())
                            lb_val = float(lower_bound[b, i, j, ai, aj].item())
                            tol_val = float(tol_soft[b, i, j, ai, aj].item())
                            v_val = float(topv[rank].item())
                            print(f"[ClashPair] b={b} i={i}({name_i}) {atom_i}  --  j={j}({name_j}) {atom_j} | d={d_val:.2f}Å, lower={lb_val:.2f}Å, tol={tol_val:.2f}Å, viol={v_val:.2f}Å")
            except Exception as e:
                try:
                    print(f"[ClashPairs] debug print failed: {e}")
                except Exception:
                    pass

        return between_mean + within_mean

class TorsionLoss(nn.Module):
    """
    一个用于计算蛋白质扭转角损失的PyTorch模块。

    该损失函数处理了角度的周期性、化学对称性，并应用掩码忽略不存在的角度。
    """
    def __init__(self,
                 planar_loss_weight: float = 0.0,
                 sidechain_only: bool = True,
                 use_cosine_loss: bool = False,
                 huber_delta: float = 0.0,
                 focal_gamma: float = 0.0,
                 channel_weights: list = None,
                 use_bond_weighting: bool = False,
                 bond_threshold: float = 0.5,
                 bond_weight: float = 10):
        """
        初始化函数。

        Args:
            planar_loss_weight (float): 平面性损失的权重。如果设为0，则不计算平面性损失。
        """
        super(TorsionLoss, self).__init__()
        self.planar_loss_weight = planar_loss_weight
        self.sidechain_only = sidechain_only
        # Robust/angular loss options
        self.use_cosine_loss = bool(use_cosine_loss)
        self.huber_delta = float(huber_delta)
        self.focal_gamma = float(focal_gamma)
        # Optional per-channel static weights
        if channel_weights is None:
            self.register_buffer('channel_weights', None, persistent=False)
        else:
            self.register_buffer('channel_weights', torch.tensor(channel_weights, dtype=torch.float32), persistent=False)
        # Optional bond-based emphasis (reuse legacy behavior)
        self.use_bond_weighting = bool(use_bond_weighting)
        self.bond_threshold = float(bond_threshold)
        self.bond_weight = float(bond_weight)

    def forward(self, pred_torsions, true_torsions, true_torsions_alt, tors_mask, tors_planar=None, bond_mat=None):
        """
        前向传播函数。

        Args:
            pred_torsions (torch.Tensor): 预测的扭转角，维度为 (B, L, 10, 2)。
            true_torsions (torch.Tensor): 真实的扭转角，维度为 (B, L, 10, 2)。
            true_torsions_alt (torch.Tensor): 真实的可翻转扭转角，维度为 (B, L, 10, 2)。
            tors_mask (torch.Tensor): 布尔掩码，标记哪些扭转角是存在的，维度为 (B, L, 10)。
            tors_planar (torch.Tensor): 布尔掩码，标记哪些扭转角应为平面，维度为 (B, L, 10)。

        Returns:
            torch.Tensor: 计算出的总损失，是一个标量。
        """
        # 选取范围：若仅计算侧链，则只选 χ1–χ4 通道 (3:7)
        if self.sidechain_only:
            channel_slice = slice(3, 7)
            pred_torsions = pred_torsions[:, :, channel_slice, :]
            true_torsions = true_torsions[:, :, channel_slice, :]
            true_torsions_alt = true_torsions_alt[:, :, channel_slice, :]
            tors_mask = tors_mask[:, :, channel_slice]
            if tors_planar is not None:
                tors_planar = tors_planar[:, :, channel_slice]

        # 确保掩码为浮点类型
        tors_mask_float = tors_mask.float()

        # --- 1) 切空间上的主扭转角误差（S1 上的对数映射）---
        # 规范化到单位圆，避免网络未严格输出单位向量导致的偏差
        eps = 1e-6
        def _normalize(u: torch.Tensor) -> torch.Tensor:
            # 数值稳定的归一化，并对无效/被mask的条目使用安全的单位向量以避免 atan2(0,0)
            norm = torch.norm(u, dim=-1, keepdim=True)
            unit = u / torch.clamp(norm, min=eps)
            default = torch.zeros_like(unit)
            default[..., 0] = 1.0
            small = norm < eps
            mask_invalid = small | (~tors_mask.bool().unsqueeze(-1))
            unit = torch.where(mask_invalid, default, unit)
            return unit

        pred_unit = _normalize(pred_torsions)
        true_unit = _normalize(true_torsions)
        true_alt_unit = _normalize(true_torsions_alt)

        # 对于 S1，log_{p}(q) 的长度即为沿着 p 切向方向从 p 到 q 的最小有向角
        # 通过 atan2(det, dot) 稳定地计算有向角误差，范围在 (-pi, pi]
        def _angle_error(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
            # p, q: (..., 2) 均为单位向量
            dot = (p * q).sum(dim=-1).clamp(-1.0, 1.0)
            det = p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]
            return torch.atan2(det, dot)  # (...)

        err_true = _angle_error(pred_unit, true_unit)
        err_alt = _angle_error(pred_unit, true_alt_unit)

        # 选择与化学对称性一致的最小角误差
        angle_err_abs = torch.minimum(err_true.abs(), err_alt.abs())  # (B, L, 10)
        angle_err_sq = angle_err_abs ** 2

        # --- 2) 构造难例感知/稳健的角度损失 ---
        # 三种形式（按优先级）：Huber（若 delta>0）; cosine（1-cos）；默认平方误差
        if self.huber_delta > 0:
            d = self.huber_delta
            # SmoothL1 on absolute angle error
            angle_loss = torch.where(angle_err_abs <= d,
                                     0.5 * (angle_err_abs ** 2),
                                     d * (angle_err_abs - 0.5 * d))
        elif self.use_cosine_loss:
            # 1 - cos(Δθ)，对大角度差更稳健
            angle_loss = 1.0 - torch.cos(angle_err_abs)
        else:
            angle_loss = angle_err_sq

        # --- 3) 权重：基础mask × 可选bond强调 × 可选通道权重 × 可选focal ---
        weights = tors_mask_float

        # 3.1 bond-based residue emphasis（与 Legacy 一致）
        if self.use_bond_weighting and (bond_mat is not None):
            bm = bond_mat.detach()
            B, L = bm.shape[:2]
            # 只统计有效残基两两组合，忽略对角
            res_valid = tors_mask.any(dim=-1)  # [B, L]
            pair_valid = (res_valid.unsqueeze(1) & res_valid.unsqueeze(2))
            eye = torch.eye(L, device=bm.device, dtype=torch.bool).unsqueeze(0)
            pair_valid = pair_valid & (~eye)
            # 阈值筛选
            bm_thr = bm * (bm > self.bond_threshold).float()
            bm_thr = bm_thr * pair_valid.float()
            row_sum = bm_thr.sum(dim=-1)  # [B, L]
            col_sum = bm_thr.sum(dim=-2)  # [B, L]
            res_weights = (row_sum + col_sum) / 2.0  # [B, L]
            base_plus_emphasis = 1.0 + res_weights * self.bond_weight  # [B, L]
            C = tors_mask.shape[-1]
            weights = weights * base_plus_emphasis.unsqueeze(-1).expand(-1, -1, C)

        # 3.2 static per-channel weights（如 [chi1, chi2, chi3, chi4]）
        if self.channel_weights is not None:
            cw = self.channel_weights
            C = tors_mask.shape[-1]
            if cw.numel() != C:
                # 自动适配：若提供了4通道而当前不是4通道，则广播为均值
                if cw.numel() == 4 and C != 4:
                    cw_eff = torch.ones(C, dtype=cw.dtype, device=weights.device) * (cw.mean())
                else:
                    cw_eff = torch.ones(C, dtype=cw.dtype, device=weights.device)
            else:
                cw_eff = cw.to(device=weights.device, dtype=weights.dtype)
            weights = weights * cw_eff.view(1, 1, -1)

        # 3.3 Focal-style 难例权重（按角误差大小）
        if self.focal_gamma > 0:
            focal_w = (angle_err_abs / math.pi).clamp_min(0.0).clamp_max(1.0) ** self.focal_gamma
            weights = weights * focal_w

        # 掩码并做平均（带权）
        denom = weights.sum() + eps
        average_torsion_loss = (angle_loss * weights).sum() / denom

        total_loss = average_torsion_loss

        # --- 2. (可选) 计算平面性损失 ---
        if self.planar_loss_weight > 0 and tors_planar is not None:
            # 平面角（0或180度）的 sin 分量应接近 0。
            # 使用规范化后的预测向量的 y 分量（sin）来度量平面性
            sin_sq_loss = _normalize(pred_torsions)[..., 1] ** 2
            
            # 应用平面掩码
            tors_planar_float = tors_planar.float()
            # 与主损失一致的权重（但仅作用于被标注为平面的通道）
            planar_weights = weights * tors_planar_float
            denom_planar = planar_weights.sum() + 1e-8
            average_planar_loss = (sin_sq_loss * planar_weights).sum() / denom_planar
            
            # 将平面损失加权后计入总损失
            total_loss = total_loss + self.planar_loss_weight * average_planar_loss

        return total_loss


class TorsionLossLegacy(nn.Module):
    """
    备份的原始版本：在欧氏空间 (cos, sin) 上计算扭转角的 L2 损失，
    同时考虑化学对称性的替代角，以及可选的平面性约束。
    """
    def __init__(self, planar_loss_weight=0.0, sidechain_only: bool = True, bond_threshold: float = 0.75, 
                bond_weight: float = 0, focal_gamma: float = 1):
        super(TorsionLossLegacy, self).__init__()
        self.planar_loss_weight = planar_loss_weight
        self.sidechain_only = sidechain_only
        self.bond_threshold = bond_threshold
        self.bond_weight = bond_weight
        self.focal_gamma = float(focal_gamma)

    def forward(self, pred_torsions, true_torsions, true_torsions_alt, tors_mask, tors_planar=None, 
                bond_mat: torch.Tensor = None, return_split: bool = False):
        """
        Args:
            pred_torsions (torch.Tensor): (B, L, 10, 2)
            true_torsions (torch.Tensor): (B, L, 10, 2)
            true_torsions_alt (torch.Tensor): (B, L, 10, 2)
            tors_mask (torch.Tensor): (B, L, 10) 布尔掩码
            tors_planar (torch.Tensor, optional): (B, L, 10) 布尔掩码
            bond_mat (torch.Tensor, optional): (B, L, L) 双随机矩阵，表示成键概率

        Returns:
            torch.Tensor: 标量损失
        """
        # 选取范围：若仅计算侧链，则只选 χ1–χ4 通道 (3:7)
        if self.sidechain_only:
            channel_slice = slice(3, 7)
            pred_torsions = pred_torsions[:, :, channel_slice, :]
            true_torsions = true_torsions[:, :, channel_slice, :]
            true_torsions_alt = true_torsions_alt[:, :, channel_slice, :]
            tors_mask = tors_mask[:, :, channel_slice]
            if tors_planar is not None:
                tors_planar = tors_planar[:, :, channel_slice]

        tors_mask_float = tors_mask.float()

        # --- 计算基于 bond_mat 的每残基权重（强调 bond>threshold 的位置）---
        # 说明：将成对概率转为每残基重要性权重，行/列求和并平均；仅统计超过阈值的概率；不回传梯度
        weights_expanded = None
        if bond_mat is not None:
            bm = bond_mat.detach()
            # 只统计有效残基两两组合，忽略对角
            res_valid = tors_mask.any(dim=-1)  # [B, L]
            pair_valid = (res_valid.unsqueeze(1) & res_valid.unsqueeze(2))
            eye = torch.eye(bm.shape[-1], device=bm.device, dtype=torch.bool).unsqueeze(0)
            pair_valid = pair_valid & (~eye)

            # 阈值筛选并对无效位置清零
            bm_thr = bm * (bm > self.bond_threshold).float()
            bm_thr = bm_thr * pair_valid.float()

            # 每残基权重：行/列求和并平均，范围大致在 [0, 1]
            row_sum = bm_thr.sum(dim=-1)  # [B, L]
            col_sum = bm_thr.sum(dim=-2)  # [B, L]
            res_weights = (row_sum + col_sum)/2  # [B, L]

            # 直接在基线 1 上叠加残基权重（未归一化）
            base_plus_emphasis = 1.0 + res_weights * self.bond_weight

            # 展开到通道维度，与 (B, L, C) 的逐元素损失相乘
            C = tors_mask.shape[-1]
            weights_expanded = base_plus_emphasis.unsqueeze(-1).expand(-1, -1, C)
        else:
            weights_expanded = torch.ones_like(tors_mask_float)

        # 主扭转角损失：对 (cos, sin) 的欧氏距离做 L2
        loss_true = torch.sum((pred_torsions - true_torsions) ** 2, dim=-1)   # (B, L, 10)
        loss_alt = torch.sum((pred_torsions - true_torsions_alt) ** 2, dim=-1)  # (B, L, 10)
        min_loss = torch.min(loss_true, loss_alt)

        # 针对 min_loss 的 Focal 加权（归一化到 [0,1]，单位圆上最大欧氏距离平方为4）
        if getattr(self, 'focal_gamma', 0.0) > 0:
            focal_weights = weights_expanded * (min_loss.detach() ** self.focal_gamma)
        else:
            focal_weights = weights_expanded

        # 使用加权的全局平均（仅对有效与被强调的位置计数），Focal 加权仅作用于主损失
        weighted = min_loss * tors_mask_float * focal_weights
        denom = (tors_mask_float * focal_weights).sum() + 1e-8
        total_loss = weighted.sum() / denom

        # 基于 bond_mat 阈值的两部分损失（大于阈值 / 小于等于阈值）
        loss_high = total_loss
        loss_low = total_loss
        if bond_mat is not None:
            bm = bond_mat.detach()
            B, L = bm.shape[:2]
            res_valid = tors_mask.any(dim=-1)  # [B, L]
            pair_valid = (res_valid.unsqueeze(1) & res_valid.unsqueeze(2))
            eye = torch.eye(L, device=bm.device, dtype=torch.bool).unsqueeze(0)
            pair_valid = pair_valid & (~eye)
            # 分离高/低阈值部分
            bm_high = (bm > self.bond_threshold).float() * bm * pair_valid.float()
            bm_low = (bm <= self.bond_threshold).float() * bm * pair_valid.float()
            # 残基级聚合
            row_h = bm_high.sum(dim=-1)
            col_h = bm_high.sum(dim=-2)
            res_w_high = (row_h + col_h) / 2.0  # [B, L]
            row_l = bm_low.sum(dim=-1)
            col_l = bm_low.sum(dim=-2)
            res_w_low = (row_l + col_l) / 2.0   # [B, L]
            # 叠加到基线1
            C = tors_mask.shape[-1]
            w_high = (1.0 + res_w_high * self.bond_weight).unsqueeze(-1).expand(-1, -1, C)
            w_low = (1.0 + res_w_low * self.bond_weight).unsqueeze(-1).expand(-1, -1, C)
            # Focal（与主损失一致）
            if getattr(self, 'focal_gamma', 0.0) > 0:
                focal_factor = (min_loss.detach() ** self.focal_gamma)
                w_high = w_high * focal_factor
                w_low = w_low * focal_factor
            # 计算两部分损失
            num_h = (min_loss * tors_mask_float * w_high).sum()
            den_h = (tors_mask_float * w_high).sum() + 1e-8
            num_l = (min_loss * tors_mask_float * w_low).sum()
            den_l = (tors_mask_float * w_low).sum() + 1e-8
            loss_high = num_h / den_h
            loss_low = num_l / den_l

        # 可选的平面性损失：惩罚 sin 分量
        if self.planar_loss_weight > 0 and tors_planar is not None:
            tors_planar_float = tors_planar.float()
            sin_sq_loss = pred_torsions[..., 1] ** 2
            masked_planar = sin_sq_loss * tors_planar_float * weights_expanded
            denom_planar = (tors_planar_float * weights_expanded).sum() + 1e-8
            total_loss = total_loss + self.planar_loss_weight * (masked_planar.sum() / denom_planar)

        if return_split:
            return total_loss, min_loss, 
        return total_loss

