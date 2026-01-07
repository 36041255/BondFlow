import torch 
import torch.distributed  as dist 
import torch.nn  as nn 
import torch.optim  as optim 
from torch.nn.parallel  import DistributedDataParallel as DDP 

from BondFlow.data.dataloader import get_dataloader
# from BondFlow.models.adapter import build_plm_encoder
import torch.distributed as dist
from BondFlow.models.Loss import *
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import os

import os, time, pickle
import torch
import copy
from omegaconf import OmegaConf
import logging
import numpy as np
import random
import csv
from BondFlow.models.sampler import *
import BondFlow.data.SM_utlis as smu
from hydra import initialize, compose
from multiflow_data import utils as du
import time
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
data_dir = "/home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/ALL_MMCIF/train_data5"
cluster_file = os.path.join(data_dir,"LINKAF_tmp/cluster.tsv")
pdb_dir =  os.path.join(data_dir,"LINKAF_CIF")

data_term_dir = os.path.join(data_dir,"/LINK_TERM_MONO_CIF")
data_term_file = "/home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/ALL_MMCIF/train_data5/LINK_TERM_MONO_tmp/cluster.tsv"

pdb_com_list_path = "/home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/ALL_MMCIF/train_data5/COMPLEX_tmp/cluster.tsv"
pdb_com_dir = "/home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/ALL_MMCIF/train_data5/COMPLEX_CIF"

data_files = [cluster_file ,data_term_file,pdb_com_list_path]
data_dirs = [pdb_dir,data_term_dir,pdb_com_dir]
sampling_ratios=[{'monomer': 1},{'monomer': 1},{'complex_space': 1}]
dataset_probs=[0.1, 0.0,0.9]

cache_dir = "/home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/ALL_MMCIF/data_cache/"
crop_size = 256
val_ratio=0.025
seed=43  # 随机种子分割验证集和训练集
mask_bond_prob = 0.5
mask_seq_str_prob = 0.4
nc_pos_prob=0.3
hotspot_prob=1

parser = argparse.ArgumentParser(description="My Protein Design Training Script")
parser.add_argument('--checkpoint_dir', type=str, required=True,
                    help='Directory to save checkpoints and logs.')
args = parser.parse_args()
checkpoint_dir = args.checkpoint_dir  # 使用从命令行传入的路径


print(f"Python script is running.")
print(f"All outputs will be saved in: {checkpoint_dir}")

log_file_path = os.path.join(checkpoint_dir, "train_log.txt")
csv_log_file_path = os.path.join(checkpoint_dir, "train_loss.csv")
partial_T_threshold = 0.25
# Ratio of samples to draw from (partial_T_threshold, 1 - eps_t)
partial_T_high_ratio = 0.75
#w_frame, w_seq, w_bond, w_clash, w_fape, w_torsion,w_bond_coh = 1, 1, 1, 0.05 /(1-partial_T_threshold), 0.25, 1/(1-partial_T_threshold), 2 /(1-partial_T_threshold)
w_frame, w_seq, w_bond, w_clash, w_fape, w_torsion,w_bond_coh = 0.75, 1, 1, 0.75 , 0.25, 0.75, 1
# Additional loss weight for BondCoherenceLoss
batch_size = 8
total_batch_size = 64
samples_num=20000
lr = 5e-5
weight_decay = 1e-5
grad_clip_norm = 100
ema_decay = 0.99
use_ema = False
# Learning rates for parameter groups (APMWrapper only)
# lr_backbone applies to the native APM backbone; lr_added applies to newly added heads
# lr_backbone = 1e-5
# lr_added = 1e-4
eps_t = 1e-3
epochs = 50
validation_frequency = 50 # 每50次梯度更新后进行验证，您可以根据需要调整

config_file = "/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/base.yaml"
resume_from_checkpoint = None

# 选择的T是否一个batch内一样
same_T_in_batch = True
# 延迟启用部分损失项的轮数
clash_delay_epochs = 0
bond_coh_delay_epochs = 0

def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(conf,device='cpu') -> None:  # 移除类型标注中的 HydraConfig
    log = logging.getLogger(__name__)
    if conf.inference.seed is not None:
        make_deterministic(seed)

    # Check for available GPU and print result of check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    # Initialize sampler and target/contig.
    sampler = Sampler(conf,device=device)  # 使用传入的 device 参数
    return sampler

def sample_partial_t(
    batch_size: int,
    device: torch.device,
    same_T_in_batch: bool,
    partial_T_threshold: float,
    eps_t: float,
    partial_T_high_ratio: float,
):
    """Sample diffusion time t (partial_T) using a two-interval mixed uniform distribution.

    - With probability partial_T_high_ratio, sample uniformly from [partial_T_threshold, 1 - eps_t]
    - Otherwise sample uniformly from [0, partial_T_threshold]
    - If same_T_in_batch is True, all samples in the batch use the same t
    """
    high_range = (1 - eps_t) - partial_T_threshold
    low_range = partial_T_threshold
    if same_T_in_batch:
        if high_range <= 0:
            t_single = torch.rand((1,), device=device) * max(low_range, 0.0)
        elif low_range <= 0:
            t_single = partial_T_threshold + torch.rand((1,), device=device) * max(high_range, 0.0)
        else:
            u = torch.rand((1,), device=device)
            high_sample = partial_T_threshold + torch.rand((1,), device=device) * high_range
            low_sample = torch.rand((1,), device=device) * low_range
            t_single = torch.where(u < partial_T_high_ratio, high_sample, low_sample)
        partial_T = t_single.repeat(batch_size)
    else:
        if high_range <= 0:
            partial_T = torch.rand((batch_size,), device=device) * max(low_range, 0.0)
        elif low_range <= 0:
            partial_T = partial_T_threshold + torch.rand((batch_size,), device=device) * max(high_range, 0.0)
        else:
            u = torch.rand((batch_size,), device=device)
            high_sample = partial_T_threshold + torch.rand((batch_size,), device=device) * high_range
            low_sample = torch.rand((batch_size,), device=device) * low_range
            partial_T = torch.where(u < partial_T_high_ratio, high_sample, low_sample)
    print(f"partial_T: {partial_T}")
    return partial_T

def compute_importance_weight(
    partial_T: torch.Tensor,
    partial_T_threshold: float,
    eps_t: float,
    partial_T_high_ratio: float,
):
    """Compute importance-sampling weight that debiases the mixed t sampling to target Uniform[0, 1-eps_t]."""
    high_range = (1 - eps_t) - partial_T_threshold
    low_range = partial_T_threshold
    p_true = 1.0 / max(1.0 - eps_t, 1e-8)
    low_mask = (partial_T < partial_T_threshold).float()
    high_mask = 1.0 - low_mask
    low_den = max(low_range, 1e-8)
    high_den = max(high_range, 1e-8)
    p_sample = low_mask * ((1.0 - partial_T_high_ratio) / low_den) + high_mask * (partial_T_high_ratio / high_den)
    return (p_true / p_sample).detach()

def compute_w_nld(
    partial_T: torch.Tensor,
    partial_T_threshold: float,
    eps_t: float,
):
    """Compute normalized linear time weight w_nld with E[w_nld]=1 under Uniform[0, 1-eps_t]."""
    t0 = partial_T_threshold
    t1 = 1.0 - eps_t
    den = max(t1 - t0, 1e-8)
    t1_den = max(t1, 1e-8)
    w_raw = torch.clamp((partial_T - t0) / den, min=0.0, max=1.0)
    ew = (t1 - t0) / (2.0 * t1_den)
    return w_raw / max(ew, 1e-8)

def model_forward(sampler, model_fn, batch_data, criterion_frame, criterion_seq,
                   criterion_bond, criterion_FAPE, criterion_clash, criterion_torsion,
                   criterion_bond_coh):
    
    # 1. Extract data from batch and move to device
    xyz_orig = batch_data['full_xyz'].to(sampler.device) # (B, L, 14, 3)
    seq_target = batch_data['full_seq'].to(sampler.device)
    bond_matrix_target = batch_data['full_bond_matrix'].to(sampler.device)
    res_mask = batch_data['res_mask'].to(sampler.device)
    str_mask = batch_data['full_mask_str'].to(sampler.device)
    seq_mask = batch_data['full_mask_seq'].to(sampler.device)
    bond_mask = batch_data['full_bond_mask'].to(sampler.device)
    head_mask = batch_data['full_head_mask'].to(sampler.device)
    tail_mask = batch_data['full_tail_mask'].to(sampler.device)
    N_C_anchor = batch_data['full_N_C_anchor'].to(sampler.device)
    pdb_idx = batch_data['full_pdb_idx']
    rf_idx = batch_data['full_rf_idx'].to(sampler.device)
    alpha_target = batch_data['full_alpha'].to(sampler.device)
    alpha_alt_target = batch_data['full_alpha_alt'].to(sampler.device)
    alpha_tor_mask = batch_data['full_alpha_tor_mask'].to(sampler.device)
    pdb_id = batch_data['pdb_id']
    chain_ids = batch_data['full_chain_ids'].to(sampler.device)
    # Mapping back to original PDB so that the model can compute full-chain PLM
    # embeddings lazily inside the forward pass.
    origin_pdb_idx = batch_data['full_origin_pdb_idx']  # list of lists of (chain, res)
    pdb_seq_full = batch_data['pdb_seq_full']           # list of np.ndarray[int] (one per sample)
    pdb_idx_full = batch_data['pdb_idx_full']           # list of lists of (chain, res)
    pdb_core_id = batch_data['pdb_core_id']             # list of str (pdb basename)
    final_res_mask = res_mask.float() * (1 - head_mask.float()) * (1 - tail_mask.float())
    hotspots = batch_data['full_hotspot'].to(sampler.device)
    B, L = seq_target.shape


    # 2. Get random timestep (mixture sampling) and compute weights
    partial_T = sample_partial_t(
        batch_size=B,
        device=sampler.device,
        same_T_in_batch=same_T_in_batch,
        partial_T_threshold=partial_T_threshold,
        eps_t=eps_t,
        partial_T_high_ratio=partial_T_high_ratio,
    )

    # Importance Sampling weight for biased t sampling
    is_weight = compute_importance_weight(
        partial_T=partial_T,
        partial_T_threshold=partial_T_threshold,
        eps_t=eps_t,
        partial_T_high_ratio=partial_T_high_ratio,
    )

    # Normalized Linear Increasing time weight (w_nld)
    w_nld = compute_w_nld(
        partial_T=partial_T,
        partial_T_threshold=partial_T_threshold,
        eps_t=eps_t,
    )

    # 3. Noise the data using interpolant's sample method
    xyz_noised, seq_noised, bond_noised,xyz_centered,rotmats =  sampler.sample_with_interpolant( xyz_orig[:,:,:3,:], 
                                                                            seq_target, 
                                                                            bond_matrix_target, 
                                                                            res_mask, 
                                                                            str_mask, 
                                                                            seq_mask,
                                                                            bond_mask, 
                                                                            hotspots, 
                                                                            partial_T,
                                                                            chain_ids,
                                                                            N_C_anchor=N_C_anchor,
                                                                            head_mask=head_mask,
                                                                            tail_mask=tail_mask)

    # 5. Model forward pass（加入 CUDA 同步，统计更真实的 forward 耗时）
    torch.cuda.synchronize()
    start_time = time.time()
    try:
        logits_aa, xyz_pred, alpha_s, bond_matrix = model_fn(
                seq_noised=seq_noised,
                xyz_noised=xyz_noised,
                bond_noised=bond_noised,
                rf_idx=rf_idx,
                pdb_idx=pdb_idx,
                alpha_target=alpha_target,
                alpha_tor_mask=alpha_tor_mask,
                partial_T=partial_T,
                str_mask=str_mask,
                seq_mask=seq_mask,
                bond_mask=bond_mask,
                res_mask=res_mask,
                head_mask=head_mask,
                tail_mask=tail_mask,
                N_C_anchor=N_C_anchor,
                trans_1=xyz_centered[:,:,1,:].to(xyz_noised.dtype),
                rotmats_1= rotmats.to(xyz_noised.dtype),
                aatypes_1=seq_target.long(),
                bond_mat_1=bond_matrix_target,
                chain_ids=chain_ids,
                origin_pdb_idx=origin_pdb_idx,
                pdb_seq_full=pdb_seq_full,
                pdb_idx_full=pdb_idx_full,
                pdb_core_id=pdb_core_id,
                hotspots=hotspots,
                use_checkpoint=False,
            )
    except RuntimeError as e:
        with open(log_file_path, 'a') as f:
            pdb_ids = " ".join(pdb_id)
            log_message = f"model forward error {pdb_ids}\n"
            f.write(log_message)
            f.write(str(e) + '\n')
            print(log_message)
        raise e

    torch.cuda.synchronize()
    model_forward_time = time.time() - start_time
    print(f"model forward time: {model_forward_time}")

    # 6. Loss 计算计时（同样加入同步，避免异步导致低估）
    torch.cuda.synchronize()
    loss_start_time = time.time()


    seq_pred = torch.argmax(logits_aa, dim=-1)
    # fix mask part:
    seq_noised_mask = (seq_noised == du.MASK_TOKEN_INDEX)
    seq_res_mask = seq_mask.float() * res_mask.float() * seq_noised_mask.float()
    seq_pred = seq_pred * seq_res_mask.float() + (1 - seq_res_mask.float()) * seq_target.long()

    str_res_mask = str_mask.float() * res_mask.float() 
    times = xyz_pred.size(1)
    # Reshape str_res_mask to broadcast with xyz_pred: (B, L) -> (B, 1, L, 1, 1)
    str_res_mask_expanded = str_res_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    xyz_pred = xyz_pred * str_res_mask_expanded + ( 1 - str_res_mask_expanded) * xyz_centered.unsqueeze(1).expand(-1,times,-1,-1,-1)[...,:3,:]


    # 计算frame损失
    target_times =xyz_centered.unsqueeze(1).expand(-1,times,-1,-1,-1)[...,:3,:]
    noise_times = xyz_noised.unsqueeze(1).expand(-1,times,-1,-1,-1)[...,:3,:]
    
    loss_frame = criterion_frame(xyz_pred, target_times, noise_times,mask = str_res_mask.bool(),t=partial_T)

    loss_FAPE = criterion_FAPE(xyz_pred[:,-1,...],xyz_centered,final_res_mask, str_mask)

    # 计算序列损失        
    loss_seq = criterion_seq (logits_aa, seq_target, mask=seq_res_mask.bool())

    #计算键合矩阵的loss
    mask_res_bond_2d = (res_mask.unsqueeze(2).float() * res_mask.unsqueeze(1).float() * bond_mask.float()).bool()
    loss_bond = criterion_bond(bond_matrix, bond_matrix_target, mask_res_bond_2d)

    # partial_T: (B,) 或 (batch_size,)
    # 生成 mask，True 表示该样本参与 loss 计算
    partial_T_mask = (partial_T > partial_T_threshold)
    mask_res_2d = (res_mask.unsqueeze(2).float() * res_mask.unsqueeze(1).float()).bool()
    bond_matrix_sampled = smu.sample_permutation(bond_matrix, mask_res_2d)
    # RTframes,allatom_xyz = sampler.allatom(seq_pred,xyz_pred[:,-1,...],
    #                             alpha_s,use_H=False,bond_mat =bond_matrix_sampled,
    #                             link_csv_path="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/link.csv",
    #                             res_mask=final_res_mask)
    RTframes,allatom_xyz_withCN = sampler.allatom(seq_pred,xyz_pred[:,-1,...],
                            alpha_s,use_H=False,bond_mat =bond_matrix_sampled,
                            link_csv_path="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/link.csv",
                            res_mask=res_mask,
                            head_mask=head_mask,
                            tail_mask=tail_mask,
                            N_C_anchor=N_C_anchor,)
    # 计算原始 loss，保持掩码用于判断；same_T_in_batch 时用标量权重，否则逐样本加权

    loss_clash = criterion_clash(
        N_C_anchor,
        allatom_xyz_withCN,
        seq_pred,
        res_mask=final_res_mask*partial_T_mask[:,None].float(),
        bond_mat=bond_matrix_sampled,
        head_mask=head_mask,
        tail_mask=tail_mask,
    ) * w_nld.mean()



    alpha_tor_res_mask = alpha_tor_mask.float() * final_res_mask[:, :, None].float()*partial_T_mask[:,None,None].float()
    loss_torsion = criterion_torsion(
        alpha_s,
        alpha_target,
        alpha_alt_target,
        alpha_tor_res_mask,
        bond_mat = bond_matrix
    ) * w_nld.mean()


    # 计算Bond Coherence损失（结构-序列-键自洽）
    loss_bond_coh = criterion_bond_coh(
        bond_matrix=bond_matrix,
        res_mask=res_mask*partial_T_mask[:,None].float(),
        seq_logits=logits_aa,
        true_seq = seq_target,
        mask_2d=None,
        all_atom_coords = allatom_xyz_withCN,
        aatype = seq_pred,
        t = None,
        head_mask=head_mask,
        tail_mask=tail_mask,
        #true_bond_matrix=bond_matrix_target,
        nc_anchor=N_C_anchor,
        detach_bond=False,
    ) * w_nld.mean()

    # Apply importance weight to all t-dependent losses
    # Same-T-in-batch -> is_weight is effectively scalar per batch; otherwise, take batch-wise mean
    if is_weight.ndim > 0:
        # Reduce to scalar for stability when mixed within scalars returned by criteria
        is_w_scalar = is_weight.mean()
    else:
        is_w_scalar = is_weight

    loss_frame = loss_frame * is_w_scalar
    loss_seq = loss_seq * is_w_scalar
    loss_bond = loss_bond * is_w_scalar
    loss_FAPE = loss_FAPE * is_w_scalar
    loss_clash = loss_clash * is_w_scalar
    loss_torsion = loss_torsion * is_w_scalar
    loss_bond_coh = loss_bond_coh * is_w_scalar
    torch.cuda.synchronize()
    loss_end_time = time.time()
    loss_time = loss_end_time - loss_start_time
    print(f"loss time: {loss_time}")

    return loss_frame, loss_seq, loss_bond, loss_clash, loss_FAPE, loss_torsion, loss_bond_coh, partial_T, model_forward_time, loss_time

def validate_model(sampler, ddp_model, dataloader_val, criterion_frame, criterion_seq, criterion_bond, 
                    criterion_FAPE, criterion_clash, criterion_torsion, criterion_bond_coh, rank, world_size, epoch):
    """验证函数"""
    ddp_model.eval()
    val_loss_total = 0.0
    val_loss_frame = 0.0
    val_loss_seq = 0.0
    val_loss_bond = 0.0
    val_loss_clash = 0.0
    val_loss_fape = 0.0
    val_loss_torsion = 0.0
    val_loss_bond_coh = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader_val:
            if batch_data is None: continue
            loss_frame, loss_seq, loss_bond, loss_clash, loss_fape, loss_torsion, loss_bond_coh, _, _, _ = model_forward(
                sampler, ddp_model, batch_data, criterion_frame, criterion_seq, criterion_bond, criterion_FAPE, criterion_clash, criterion_torsion, criterion_bond_coh
            )
            
            # 根据epoch延迟启用部分loss
            w_clash_eff = 0.0 if epoch < clash_delay_epochs else w_clash
            w_bond_coh_eff = 0.0 if epoch < bond_coh_delay_epochs else w_bond_coh
            total_loss = w_frame*loss_frame + w_seq*loss_seq + w_bond*loss_bond + w_clash_eff*loss_clash + w_fape*loss_fape + w_torsion*loss_torsion + w_bond_coh_eff*loss_bond_coh
            val_loss_total += total_loss.item()
            val_loss_frame += loss_frame.item()
            val_loss_seq += loss_seq.item()
            val_loss_bond += loss_bond.item()
            val_loss_clash += loss_clash.item()
            val_loss_fape += loss_fape.item()
            val_loss_torsion += loss_torsion.item()
            val_loss_bond_coh += loss_bond_coh.item()
            num_batches += 1
    
    # 计算平均损失
    if num_batches > 0:
        val_loss_total /= num_batches
        val_loss_frame /= num_batches
        val_loss_seq /= num_batches
        val_loss_bond /= num_batches
        val_loss_clash /= num_batches
        val_loss_fape /= num_batches
        val_loss_torsion /= num_batches
        val_loss_bond_coh /= num_batches
    
    # 同步所有GPU的验证损失
    val_tensor = torch.tensor([val_loss_total, val_loss_frame, val_loss_seq, val_loss_bond, val_loss_clash, val_loss_fape, val_loss_torsion, val_loss_bond_coh], device=f'cuda:{rank}')
    dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
    global_val_losses = val_tensor / world_size
    
    return (
        global_val_losses[0].item(), global_val_losses[1].item(), global_val_losses[2].item(),
        global_val_losses[3].item(), global_val_losses[4].item(), global_val_losses[5].item(),
        global_val_losses[6].item(), global_val_losses[7].item()
    )


# 1. 初始化进程组 
def setup(rank, world_size):
    dist.init_process_group( 
        backend="nccl",  # NVIDIA GPU推荐使用nccl 
        init_method="tcp://127.0.0.1:12346",  # 本机通信 
        rank=rank,
        world_size=world_size 
    )
# 2. 清理进程组 
def cleanup():
    dist.destroy_process_group() 

def log_nan_loss(loss_tensor, loss_name, pdb_id, log_file_path):
    """
    Logs a message if the loss is NaN.

    Args:
        loss_tensor (torch.Tensor): The loss value.
        loss_name (str): The name of the loss (e.g., 'frame', 'seq').
        batch_data (dict): The batch data dictionary, containing 'pdb_id'.
        log_file_path (str): The path to the log file.
    
    Returns:
        bool: True if the loss was NaN, False otherwise.
    """
    if torch.isnan(loss_tensor):
        with open(log_file_path, 'a') as f:
            pdb_ids = " ".join(pdb_id)
            log_message = f"{loss_name} nan {pdb_ids} {loss_tensor.item()}\\n"
            f.write(log_message)
            print(log_message)
        return True
    return False
    
def configure_optimizers(model, learning_rate, weight_decay=0.001):
    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == 1 or
            name.endswith(".bias") or
            "layernorm" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
    )
    return optimizer

def setup_training(ddp_model, dataloader_train, accumulation_steps,device):
    """Initializes optimizer, scheduler, and loss criteria."""

    optimizer = configure_optimizers(ddp_model, lr, weight_decay = weight_decay)
    effective_steps_per_epoch = len(dataloader_train) // accumulation_steps
    print(f"Effective steps per epoch: {effective_steps_per_epoch}")
    scheduler = CosineAnnealingLR(optimizer, T_max=effective_steps_per_epoch * epochs, eta_min=1e-5)
    
    criterion_frame = LFrameLoss(w_trans=0.5, w_rot=1, gamma=1.1, d_clamp=15)
    criterion_seq = LseqLoss()
    criterion_bond = DSMCrossEntropyLoss()
    criterion_FAPE = FAPELoss(clamp_distance=15)
    criterion_clash = OpenFoldClashLoss(device=device)
    criterion_torsion = TorsionLossLegacy()
    # Bond coherence loss (uses link.csv in config)
    link_csv_path = "/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/link.csv"
    criterion_bond_coh = BondCoherenceLoss(link_csv_path=link_csv_path, device=device,
                    t_geom_threshold=partial_T_threshold)
    
    criteria = {
        'frame': criterion_frame, 'seq': criterion_seq, 'bond': criterion_bond,
        'FAPE': criterion_FAPE, 'clash': criterion_clash, 'torsion': criterion_torsion,
        'bond_coh': criterion_bond_coh,
    }
    
    return optimizer, scheduler, criteria


def create_ema_model(model, decay: float, device: str):
    """
    Create an EMA (Exponential Moving Average) copy of the model.
    EMA model is only used for evaluation / saving, not for backprop.
    """
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()
    return ema_model


@torch.no_grad()
def update_ema_model(model, ema_model, decay: float):
    """
    Update EMA model parameters: ema = decay * ema + (1 - decay) * model.
    """
    model_params = dict(model.named_parameters())
    ema_params = dict(ema_model.named_parameters())
    for name, param in model_params.items():
        if not param.requires_grad:
            continue
        if name not in ema_params:
            continue
        ema_param = ema_params[name]
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

def truncate_logs(checkpoint_epoch, checkpoint_step, csv_log_path, txt_log_path):
    """Truncates log files to the last entry before or at the checkpoint."""
    if not os.path.exists(csv_log_path):
        return

    # Truncate CSV log
    with open(csv_log_path, 'r+') as f:
        lines = f.readlines()
        header = lines[0]
        data_lines = lines[1:]
        
        last_valid_line_idx = -1
        for i, line in enumerate(data_lines):
            parts = line.strip().split(',')
            try:
                epoch, step = int(parts[0]), int(parts[1])
                if epoch < checkpoint_epoch or (epoch == checkpoint_epoch and step <= checkpoint_step):
                    last_valid_line_idx = i
                else:
                    break # Assuming logs are ordered
            except (ValueError, IndexError):
                continue

        if last_valid_line_idx != -1:
            f.seek(0)
            f.write(header)
            f.writelines(data_lines[:last_valid_line_idx + 1])
            f.truncate()
            print(f"CSV log truncated to epoch {checkpoint_epoch}, step {checkpoint_step}.")
        else: # Keep only header if no valid lines found
            f.seek(0)
            f.write(header)
            f.truncate()
            print("CSV log cleared, keeping header.")

    # Truncate text log (less precise, but better than nothing)
    # This is a simple implementation. A more robust one would parse the text log.
    if os.path.exists(txt_log_path):
        with open(txt_log_path, 'r+') as f:
            lines = f.readlines()
            # Find where to truncate. We'll keep all lines up to the one that announces saving the checkpoint.
            # A simple heuristic: find the last mention of the checkpoint epoch.
            # This is not perfect.
            # A better way is to find the line corresponding to the last valid CSV entry.
            # For now, we will just truncate based on CSV, and assume text log is for human reading.
            pass # For now, we only truncate the CSV which is the source of truth for plots.

def load_checkpoint(path, ddp_model, optimizer, scheduler, rank):
    """Loads a checkpoint from the given path."""
    start_epoch = 0
    best_val_loss = float('inf')
    total_epochs = epochs # Default to global epochs
    checkpoint_epoch, checkpoint_step = -1, -1
    if path and os.path.exists(path):
        checkpoint = torch.load(path, map_location=f'cuda:{rank}',weights_only=False)
        ddp_model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # 所有进程在加载状态后同步一次，保证状态一致
        if dist.is_initialized():
            dist.barrier()

        checkpoint_epoch = checkpoint['epoch']
        # best_model.pth is saved mid-epoch, epoch_X_best_loss.pth is saved at end of epoch
        if os.path.basename(path) == 'best_model.pth': # Resuming from a mid-epoch save
            start_epoch = checkpoint_epoch
            checkpoint_step = checkpoint['effective_step']
        else: # Resuming from an end-of-epoch save (e.g., epoch_X_best_loss.pth)
            start_epoch = checkpoint_epoch + 1
            checkpoint_step = -1 # Indicates start of a new epoch

        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        total_epochs = checkpoint.get('epochs', epochs) # Load epochs from checkpoint, fallback to global
        if rank == 0:
            print(f"Resuming training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}. Total epochs set to {total_epochs}.")
    return start_epoch, best_val_loss, total_epochs, checkpoint_epoch, checkpoint_step

# 5. 训练函数（每个GPU上独立执行）
def train_model(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)   # 绑定当前GPU 
    
    accumulation_steps = total_batch_size//world_size//batch_size  # 梯度累积步数


    config_path = os.path.dirname(config_file)
    config_path = os.path.relpath(config_path)
    config_name = os.path.basename(config_file).split(".yaml")[0]
    
    # 创建模型 + DDP包装 
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)
    # 调用主函数
    device = f"cuda:{rank}"
    sampler = main(cfg,device=device)
    model = sampler.model 
    ddp_model = DDP(model, device_ids=[rank],find_unused_parameters=False)
    # EMA model (kept on each rank, updated after successful optimizer step)
    ema_model = None
    if use_ema:
        ema_model = create_ema_model(ddp_model.module, decay=ema_decay, device=device)
    #plm_encoder = build_plm_encoder(sampler.model._folding_model,sampler.model._plm_type, device=f'cuda:{rank}')
    
    # 数据加载器 + 分布式采样器 
    dataloader_train, dataloader_val = get_dataloader(
        conf=cfg,
        batch_size=batch_size,
        pdb_list_path=data_files,
        pdb_dir=data_dirs,
        sampling_ratios=sampling_ratios,
        distributed=True,
        num_workers=8,
        crop_length=crop_size,
        device=f'cuda:{rank}',
        rank=rank,
        num_replicas=world_size,
        val_split=val_ratio,
        seed=seed,
        cache_dir = cache_dir,
        dataset_probs=dataset_probs,
        samples_num=samples_num,
        mask_bond_prob = mask_bond_prob,
        mask_seq_str_prob = mask_seq_str_prob,
        nc_pos_prob=nc_pos_prob,
        hotspot_prob=hotspot_prob,
        # plm_encoder = plm_encoder,
        # plm_max_chain_length=800,
    )

    # 优化器和损失函数 
    optimizer, scheduler, criteria = setup_training(ddp_model, dataloader_train, accumulation_steps,device=device)
    criterion_frame, criterion_seq, criterion_bond = criteria['frame'], criteria['seq'], criteria['bond']
    criterion_FAPE, criterion_clash, criterion_torsion = criteria['FAPE'], criteria['clash'], criteria['torsion']
    criterion_bond_coh = criteria['bond_coh']

    print("len(dataloader_train)",len(dataloader_train))
    print("len(dataloader_val)",len(dataloader_val))
    print("accumulation_steps",accumulation_steps)
    print("batch_size",batch_size)
    print("total_batch_size",total_batch_size)

    # 加载checkpoint
    start_epoch, best_val_loss, total_epochs, checkpoint_epoch, checkpoint_step = load_checkpoint(resume_from_checkpoint, ddp_model, optimizer, scheduler, rank)

    if rank == 0 and resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        truncate_logs(checkpoint_epoch, checkpoint_step, csv_log_file_path, log_file_path)

    timer = time.perf_counter()   # 高精度计时器（推荐）
    # 训练循环 
    optimizer.zero_grad()
    for epoch in range(start_epoch, total_epochs):
        dataloader_train.sampler.set_epoch(epoch)
        dataloader_val.sampler.set_epoch(epoch)

        is_resuming_epoch = (epoch == start_epoch and checkpoint_step > -1)
        if is_resuming_epoch:
            # checkpoint_step 是已完成的优化器步骤。我们需要从下一个步骤开始。
            # 第 N 个优化器步骤从第 (N-1) * accumulation_steps 个批次开始。
            # 所以我们要跳到第 checkpoint_step * accumulation_steps 个批次。
            start_batch_index = checkpoint_step * accumulation_steps
            if rank == 0:
                print(f"Resuming epoch {epoch} from effective_step {checkpoint_step}. Skipping to batch index {start_batch_index}.")
        else:
            start_batch_index = 0

        best_epoch_loss = float('inf')
        best_epoch_model_state_dict = None

        accumulated_loss = 0.0
        accumulated_seq_loss = 0.0
        accumulated_bond_loss = 0.0
        accumulated_clash_loss = 0.0
        accumulated_fape_loss = 0.0
        accumulated_frame_loss = 0.0
        accumulated_torsion_loss = 0.0
        accumulated_bond_coh_loss = 0.0
        # 累加一个“有效 step”内各个小 batch 的 forward / loss 时间
        accumulated_forward_time = 0.0
        accumulated_loss_time = 0.0

        ddp_model.train()
        step_wall_start_time = None  # 记录每个“有效 step”的起始时间
        for i, batch_data in enumerate(dataloader_train):      
            if i < start_batch_index:
                continue

            if batch_data is None:
                # 保持所有 rank 在该 step 上仍然进行一次反向和 all_reduce，避免 DDP 不同步
                print(f"Empty batch on rank {rank}, using zero loss.")
                device = f"cuda:{rank}"
                loss_frame = torch.tensor(0.0, device=device)
                loss_seq = torch.tensor(0.0, device=device)
                loss_bond = torch.tensor(0.0, device=device)
                loss_clash = torch.tensor(0.0, device=device)
                loss_fape = torch.tensor(0.0, device=device)
                loss_torsion = torch.tensor(0.0, device=device)
                loss_bond_coh = torch.tensor(0.0, device=device)
                partial_T = torch.tensor(0.0, device=device)
                pdb_id = ["EMPTY_BATCH"]
                batch_forward_time = 0.0
                batch_loss_time = 0.0
            else:
                print(f"++++++++++start batch{i} in {rank}+++++++++++++")
                loss_frame, loss_seq, loss_bond, loss_clash, loss_fape, loss_torsion, loss_bond_coh, partial_T, batch_forward_time, batch_loss_time = model_forward(
                    sampler, ddp_model, batch_data, criterion_frame, criterion_seq, criterion_bond,
                    criterion_FAPE, criterion_clash, criterion_torsion, criterion_bond_coh
                )
                pdb_id = batch_data['pdb_id']

            # 如果是这个“有效 step”的第一个小 batch，则记录 step 的起始时间
            if (i % accumulation_steps) == 0:
                step_wall_start_time = time.time()

            print("part T:", partial_T)
            # 计算总损失
            print(f"losses: frame={loss_frame.item():.4f}, seq={loss_seq.item():.4f}, bond={loss_bond.item():.4f}, clash={loss_clash.item():.4f}, fape={loss_fape.item():.4f}, torsion={loss_torsion.item():.4f}, bond_coh={loss_bond_coh.item():.4f}")

            losses = {
                "loss_frame": loss_frame,
                "loss_seq": loss_seq,
                "loss_bond": loss_bond,
                "loss_clash": loss_clash,
                "loss_fape": loss_fape,
                "loss_torsion": loss_torsion,
                "loss_bond_coh": loss_bond_coh,
            }

            # Iterate over the dictionary to get both the name (string) and value (tensor)
            for loss_name, loss_value in losses.items():
                log_nan_loss(loss_value, loss_name, pdb_id, log_file_path)

            accumulated_frame_loss += loss_frame.item()
            accumulated_seq_loss += loss_seq.item()
            accumulated_bond_loss += loss_bond.item()
            accumulated_clash_loss += loss_clash.item()
            accumulated_fape_loss += loss_fape.item()
            accumulated_torsion_loss += loss_torsion.item()
            accumulated_bond_coh_loss += loss_bond_coh.item()
            accumulated_forward_time += batch_forward_time
            accumulated_loss_time += batch_loss_time

            # 根据epoch延迟启用部分loss
            w_clash_eff = 0.0 if epoch < clash_delay_epochs else w_clash
            w_bond_coh_eff = 0.0 if epoch < bond_coh_delay_epochs else w_bond_coh
            total_loss = w_frame*loss_frame + w_seq*loss_seq + w_bond*loss_bond + w_clash_eff*loss_clash + w_fape*loss_fape + w_torsion*loss_torsion + w_bond_coh_eff*loss_bond_coh
            
            
            accumulated_loss += total_loss.item()

            scaled_loss = total_loss / accumulation_steps      

            is_last_accumulation_step = (i + 1) % accumulation_steps == 0
            if not is_last_accumulation_step:
                # 在 no_sync 上下文中执行反向传播，梯度只在本地累积
                with ddp_model.no_sync():
                    scaled_loss.backward()

            else:
                torch.cuda.synchronize()
                backward_start_time = time.time()
                # 最后一步，正常执行反向传播，DDP会同步所有累积的梯度
                scaled_loss.backward()
                torch.cuda.synchronize()
                backward_end_time = time.time()
                backward_time = backward_end_time - backward_start_time
                print(f"backward time: {backward_time}")
                # backward 之后开始计“通信 + 优化等其它 CPU/GPU 开销”的时间
                comm_optim_start_time = time.time()
                # 计算当前 GPU 上这个伪批量的平均 loss
                avg_pseudo_batch_loss_local = accumulated_loss / accumulation_steps
                avg_pseudo_batch_loss_seq = accumulated_seq_loss / accumulation_steps
                avg_pseudo_batch_loss_bond = accumulated_bond_loss / accumulation_steps
                avg_pseudo_batch_loss_clash = accumulated_clash_loss / accumulation_steps
                avg_pseudo_batch_loss_fape = accumulated_fape_loss / accumulation_steps
                avg_pseudo_batch_loss_frame = accumulated_frame_loss / accumulation_steps
                avg_pseudo_batch_loss_torsion = accumulated_torsion_loss / accumulation_steps
                avg_pseudo_batch_loss_bond_coh = accumulated_bond_coh_loss / accumulation_steps

                # 同步所有 GPU 的 loss
                loss_tensor = torch.tensor([avg_pseudo_batch_loss_local, 
                                            avg_pseudo_batch_loss_frame, 
                                            avg_pseudo_batch_loss_seq,
                                            avg_pseudo_batch_loss_bond,
                                            avg_pseudo_batch_loss_clash,
                                            avg_pseudo_batch_loss_fape,
                                            avg_pseudo_batch_loss_torsion,
                                            avg_pseudo_batch_loss_bond_coh], device=f'cuda:{rank}')

                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                global_avg_losses = loss_tensor / world_size
                global_avg_loss, global_avg_loss_frame, global_avg_loss_seq, global_avg_loss_bond, global_avg_loss_clash, global_avg_loss_fape, global_avg_loss_torsion, global_avg_loss_bond_coh = global_avg_losses.tolist()


                if rank == 0:              
                    effective_step = (i + 1) // accumulation_steps
                    elapsed_time = time.perf_counter()-timer
                    log_message = (f"Epoch: {epoch}, Step: {effective_step}, Time: {elapsed_time:.2f}, "
                                   f"Avg Loss: {global_avg_loss:.4f}, Frame: {global_avg_loss_frame:.4f}, "
                                   f"Seq: {global_avg_loss_seq:.4f}, Bond: {global_avg_loss_bond:.4f}, "
                                   f"Clash: {global_avg_loss_clash:.4f}, FAPE: {global_avg_loss_fape:.4f}, Torsion: {global_avg_loss_torsion:.4f}, "
                                   f"BondCoh: {global_avg_loss_bond_coh:.4f}")
                    print(log_message)
                    with open(log_file_path, 'a') as f:
                        f.write(log_message + '\n')
                    with open(csv_log_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, effective_step, 'train', f'{global_avg_loss:.6f}', f'{global_avg_loss_frame:.6f}', 
                                         f'{global_avg_loss_seq:.6f}', f'{global_avg_loss_bond:.6f}', f'{global_avg_loss_clash:.6f}', 
                                         f'{global_avg_loss_fape:.6f}', f'{global_avg_loss_torsion:.6f}', f'{global_avg_loss_bond_coh:.6f}', f'{elapsed_time:.2f}'])
                
                is_grad_nan_or_inf = torch.tensor([False], device=rank)
                for param in ddp_model.parameters():
                    if param.grad is not None and (torch.isinf(param.grad).any() or torch.isnan(param.grad).any()):
                        is_grad_nan_or_inf[0] = True
                        break # 发现一个就足够了，直接跳出循环

                # 在所有进程间同步检查结果 (只要有一个进程发现NaN，所有进程都跳过)
                dist.all_reduce(is_grad_nan_or_inf, op=dist.ReduceOp.MAX)

                if is_grad_nan_or_inf[0]:
                    # Log which rank had the NaN grad and its corresponding PDB ID
                    pdb_ids = " ".join(pdb_id)
                    effective_step = (i + 1) // accumulation_steps
                    nan_log_message = (f"Rank {rank} found NaN/Inf gradient. epoch:{epoch}, effective_step:{effective_step}, pdb_ids: {pdb_ids}. Skipping optimizer step for all ranks.")
                    print(nan_log_message)
                    with open(log_file_path, 'a') as f:
                        f.write(nan_log_message + '\n')
                    optimizer.zero_grad() # 仍然需要清空这些坏掉的梯度
                else:
                    nn.utils.clip_grad_norm_(ddp_model.parameters(), grad_clip_norm)
                    optimizer.step()
                    # Update EMA model after a successful optimizer.step()
                    if use_ema and ema_model is not None:
                        update_ema_model(ddp_model.module, ema_model, decay=ema_decay)
                    optimizer.zero_grad()

                scheduler.step()
                torch.cuda.empty_cache()

                # 计算“通信 + 优化 + 其它 CPU/GPU 操作”的时间
                torch.cuda.synchronize()
                comm_optim_end_time = time.time()
                comm_optim_time = comm_optim_end_time - comm_optim_start_time

                # 计算整个“有效 step”的总耗时
                if step_wall_start_time is not None:
                    step_wall_end_time = time.time()
                    step_wall_time = step_wall_end_time - step_wall_start_time
                else:
                    step_wall_time = float('nan')

                # 只在 rank 0 上打印详细时间分解，帮助分析 forward + loss + backward + 通信/优化 + 其它
                if rank == 0 and not (step_wall_time != step_wall_time):  # 过滤 NaN
                    total_forward_time = accumulated_forward_time
                    total_loss_time = accumulated_loss_time
                    known_sum = total_forward_time + total_loss_time + backward_time + comm_optim_time
                    other_time = max(step_wall_time - known_sum, 0.0)
                    print(
                        f"[Step timing] Epoch {epoch}, Step {effective_step}: "
                        f"total={step_wall_time:.3f}s, "
                        f"forward={total_forward_time:.3f}s, "
                        f"loss={total_loss_time:.3f}s, "
                        f"backward={backward_time:.3f}s, "
                        f"comm+optim={comm_optim_time:.3f}s, "
                        f"other={other_time:.3f}s"
                    )

                timer = time.perf_counter()

                # 每 validation_frequency*accumulation_steps 次梯度更新后进行一次验证
                if (i + 1) % (validation_frequency*accumulation_steps) == 0:
                    # Choose model for validation: EMA if enabled, otherwise current DDP model
                    model_for_val = ema_model if (use_ema and ema_model is not None) else ddp_model
                    val_loss_total, val_loss_frame, val_loss_seq, val_loss_bond, val_loss_clash, val_loss_fape, val_loss_torsion, val_loss_bond_coh = validate_model(
                        sampler, model_for_val, dataloader_val, criterion_frame, criterion_seq, criterion_bond, criterion_FAPE, criterion_clash, criterion_torsion, criterion_bond_coh, rank, world_size, epoch
                    )
                    
                    if rank == 0:
                        effective_step = (i + 1) // accumulation_steps
                        val_log_message = (f"Validation: Epoch: {epoch}, Step: {effective_step}, "
                                           f"Avg Loss: {val_loss_total:.4f}, Frame: {val_loss_frame:.4f}, "
                                           f"Seq: {val_loss_seq:.4f}, Bond: {val_loss_bond:.4f}, "
                                           f"Clash: {val_loss_clash:.4f}, FAPE: {val_loss_fape:.4f}, Torsion: {val_loss_torsion:.4f}, "
                                           f"BondCoh: {val_loss_bond_coh:.4f}")
                        print(val_log_message)
                        with open(log_file_path, 'a') as f:
                            f.write(val_log_message + '\n')
                        with open(csv_log_file_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([epoch, effective_step, 'validation', f'{val_loss_total:.6f}', f'{val_loss_frame:.6f}', 
                                             f'{val_loss_seq:.6f}', f'{val_loss_bond:.6f}', f'{val_loss_clash:.6f}', 
                                             f'{val_loss_fape:.6f}', f'{val_loss_torsion:.6f}', f'{val_loss_bond_coh:.6f}', 'N/A'])
                        
                        # 当前用于验证和保存的模型（EMA 或 原模型）
                        if use_ema and ema_model is not None:
                            model_for_save = ema_model
                        else:
                            model_for_save = ddp_model.module

                        # 检查是否是当前epoch中最好的训练损失
                        if val_loss_total < best_epoch_loss:
                            best_epoch_loss = val_loss_total
                            best_epoch_model_state_dict = model_for_save.state_dict()
                            best_epoch_effective_step = effective_step

                        # 检查是否是最佳验证损失并保存模型
                        if val_loss_total < best_val_loss:
                            best_val_loss = val_loss_total
                            best_model_state = {
                                'epoch': epoch,
                                'effective_step': effective_step,
                                # Save chosen model (EMA or raw) as the best model
                                'model_state_dict': model_for_save.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'crop_size': crop_size,
                                'batch_size': batch_size,
                                'best_val_loss': best_val_loss,
                                'val_loss_frame': val_loss_frame,
                                'val_loss_seq': val_loss_seq,
                                'val_loss_bond': val_loss_bond,
                                'val_loss_clash': val_loss_clash,
                                'val_loss_fape': val_loss_fape,
                                'val_loss_torsion': val_loss_torsion,
                                'val_loss_bond_coh': val_loss_bond_coh,
                                'config': cfg,
                                'epochs': total_epochs,
                            }
                            
                            # 保存最佳模型
                            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                            torch.save(best_model_state, best_model_path)
                            
                            best_log_message = f"New best model saved! Val Loss: {val_loss_total:.6f} -> {best_val_loss:.6f} at epoch {epoch}, step {effective_step}"
                            print(best_log_message)
                            with open(log_file_path, 'a') as f:
                                f.write(best_log_message + '\n')
                            
                    # 验证后切回训练模式
                    ddp_model.train()
                    torch.cuda.empty_cache()
                    timer = time.perf_counter() 

                accumulated_loss = 0.0
                accumulated_seq_loss = 0.0
                accumulated_bond_loss = 0.0
                accumulated_clash_loss = 0.0
                accumulated_fape_loss = 0.0
                accumulated_frame_loss = 0.0 
                accumulated_torsion_loss = 0.0
                accumulated_bond_coh_loss = 0.0
                accumulated_forward_time = 0.0
                accumulated_loss_time = 0.0

        # 在每个epoch结束时，保存该epoch中训练损失最小的模型
        if rank == 0 and best_epoch_model_state_dict is not None:
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}_best_loss.pth')
            epoch_checkpoint = {
                'epoch': epoch,
                'effective_step':best_epoch_effective_step,
                'model_state_dict': best_epoch_model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'crop_size': crop_size,
                'batch_size': batch_size,
                #'best_epoch_train_loss': best_epoch_loss,
                'best_val_loss': best_epoch_loss,
                'config': cfg,
                'epochs': total_epochs,
            }
            torch.save(epoch_checkpoint, epoch_checkpoint_path)
            epoch_save_log = f"Saved best model for epoch {epoch} with training loss {best_epoch_loss:.6f} to {epoch_checkpoint_path}"
            print(epoch_save_log)
            with open(log_file_path, 'a') as f:
                f.write(epoch_save_log + '\n')              
        
    cleanup()

# 6. 启动多进程训练 
if __name__ == "__main__":

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 如果不是从checkpoint恢复，则清空日志；否则追加日志
    log_mode = 'a' if resume_from_checkpoint else 'w'

    # 只有在不是从断点恢复时才复制配置文件
    if resume_from_checkpoint is None:
        #复制yaml文件
        os.system(f"cp {config_file} {checkpoint_dir}")

    with open(log_file_path, log_mode) as f:
        if log_mode == 'w':
            #记录一行超参数
            f.write(f"batch_size: {total_batch_size}, lr: {lr}, epochs: {epochs}, crop_size: {crop_size}, seed: {seed}, val_ratio: {val_ratio}\n")
            f.write(f"loss_weights: frame={w_frame}, seq={w_seq}, bond={w_bond}, clash={w_clash}, fape={w_fape}, torsion={w_torsion}, bond_coh={w_bond_coh}\n")
            # 记录sampling_ratios
            # sampling_ratios_str = ', '.join([f"{k}: {v}" for k, v in sampling_ratios.items()])
            # f.write(f"sampling_ratios: {sampling_ratios_str}\n")
            
    # 初始化CSV日志文件
    with open(csv_log_file_path, log_mode, newline='') as f:
        writer = csv.writer(f)
        if log_mode == 'w':
            writer.writerow(["epoch", "effective_step", "type", "total_loss", "frame_loss", "seq_loss", "bond_loss", "clash_loss", "fape_loss", "torsion_loss", "bond_coh_loss", "time"])    
    
    world_size = torch.cuda.device_count()   # 获取GPU数量 
    assert total_batch_size % (world_size * batch_size) == 0
    torch.multiprocessing.spawn( 
        train_model, args=(world_size,),
        nprocs=world_size, join=True 
    )


