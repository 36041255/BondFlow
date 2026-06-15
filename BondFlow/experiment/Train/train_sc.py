import torch 
import torch.distributed  as dist 
import torch.nn  as nn 
import torch.optim  as optim 
from torch.nn.parallel  import DistributedDataParallel as DDP 

from BondFlow.data.dataloader import get_dataloader
import torch.distributed as dist
from BondFlow.models.Loss import *
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import numpy as np

import os, time, pickle
import torch
from omegaconf import OmegaConf
import logging
import numpy as np
import random
import csv
from BondFlow.models.mymodel import *
from hydra import initialize, compose
from multiflow_data import utils as du
import math
import argparse



data_dir = "/home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/ALL_MMCIF/train_data4"
cluster_file = os.path.join(data_dir,"LINK_tmp/cluster.tsv")
pdb_dir =  os.path.join(data_dir,"LINK_CIF")
cache_dir = "/home/fit/lulei/WORK/xjt/Protein_design/CyclicPeptide/Dataset/ALL_MMCIF/data_cache/"
crop_size = 364
val_ratio=0.025
seed=42  # 随机种子分割验证集和训练集
dataloader_mask_bond_prob = 0.0
dataloader_mask_seq_str_prob = 0.0

parser = argparse.ArgumentParser(description="My Protein Design Training Script")
parser.add_argument('--checkpoint_dir', type=str, required=True,
                    help='Directory to save checkpoints and logs.')
args = parser.parse_args()
checkpoint_dir = args.checkpoint_dir # <- 使用从命令行传入的路径

print(f"Python script is running.")
print(f"All outputs will be saved in: {checkpoint_dir}")



log_file_path = os.path.join(checkpoint_dir, "train_log.txt")
csv_log_file_path = os.path.join(checkpoint_dir, "train_loss.csv")

w_frame, w_seq, w_bond, w_clash, w_fape, w_torsion,w_bond_coh, w_sc_fape = 0, 0, 0, 0.5 , 0.0, 1, 1, 0.25
# Additional loss weight for BondCoherenceLoss
batch_size = 8
total_batch_size = 64
lr = 1e-4
grad_clip_norm = 600
# Warmup配置
warmup_steps = 500  # warmup步数，可以根据需要调整（比如总步数的5-10%）
warmup_start_lr = 1e-5  # warmup起始学习率

# Learning rates for parameter groups (APMWrapper only)
# lr_backbone applies to the native APM backbone; lr_added applies to newly added heads
# lr_backbone = 1e-5
# lr_added = 1e-4
eps_t = 1e-3
epochs = 100
validation_frequency = 40 # 每50次梯度更新后进行验证，您可以根据需要调整
sampling_ratios={'monomer': 1, }
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
    if conf.inference.deterministic:
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
    sampler = MySampler(conf,device=device)  # 使用传入的 device 参数
    return sampler

 


def model_forward(sampler, model_fn, batch_data, criterion_frame, criterion_seq,
                   criterion_bond, criterion_FAPE, criterion_clash, criterion_torsion,
                   criterion_bond_coh, criterion_sc_fape):
    
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
    pdb_idx = batch_data['full_pdb_idx']
    rf_idx = batch_data['full_rf_idx'].to(sampler.device)
    alpha_target = batch_data['full_alpha'].to(sampler.device)
    alpha_alt_target = batch_data['full_alpha_alt'].to(sampler.device)
    alpha_tor_mask = batch_data['full_alpha_tor_mask'].to(sampler.device)
    pdb_id = batch_data['pdb_id']
    final_res_mask = res_mask.float() * (1 - head_mask.float()) * (1 - tail_mask.float())
    B, L = seq_target.shape

    # 3. Noise the data using interpolant's sample method
    xyz_centered = sampler._center_global(xyz_orig[:,:,:3,:], str_mask, res_mask, pdb_idx)

    # 4. Preprocess batch for the model
    res_dist_matrix, meta = sampler.bond_mat_2_dist_mat(bond_matrix_target, rf_idx, res_mask)
    
    # 5. Model forward pass
    try:
        torsion_angles = model_fn(
            seq_noised=seq_target,
            xyz_noised=xyz_centered,
            bond_noised=bond_matrix_target,
            rf_idx=rf_idx,
            pdb_idx=pdb_idx,
            res_dist_matrix=res_dist_matrix,
            alpha_target=alpha_target,
            alpha_tor_mask=alpha_tor_mask,
            partial_T=None,
            str_mask=str_mask,
            seq_mask=seq_mask,
            head_mask=head_mask,
            tail_mask=tail_mask,
            bond_mask=bond_mask,
            res_mask=res_mask,
            use_checkpoint=False,
            # trans_1=xyz_centered[:,:,1,:].to(xyz_noised.dtype),
            # rotmats_1= rotmats.to(xyz_noised.dtype),
            # aatypes_1=seq_target.long(),
        )
    except RuntimeError as e:
        with open(log_file_path, 'a') as f:
            pdb_ids = " ".join(pdb_id)
            log_message = f"model forward error {pdb_ids}\n"
            f.write(log_message)
            f.write(str(e) + '\n')
            print(log_message)
        raise e
    
    RTframes,allatom_xyz = sampler.allatom(seq_target,xyz_centered,
                                    torsion_angles,use_H=False,bond_mat=bond_matrix_target,
                                    link_csv_path="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/link.csv",
                                    res_mask=final_res_mask)

    loss_clash = criterion_clash(allatom_xyz, seq_target, final_res_mask,bond_matrix_target,head_mask=head_mask,tail_mask=tail_mask)

    alpha_tor_res_mask = alpha_tor_mask.float() * final_res_mask[:,:,None].float()
    loss_torsion = criterion_torsion(torsion_angles, alpha_target, alpha_alt_target, alpha_tor_res_mask,bond_mat=bond_matrix_target)
    # 计算Bond Coherence损失（结构-序列-键自洽）
    RTframes,allatom_xyz_withCN = sampler.allatom(seq_target,xyz_centered,
                            torsion_angles,use_H=False,bond_mat=bond_matrix_target,
                            link_csv_path="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/link.csv",
                            res_mask=res_mask)
    loss_bond_coh = criterion_bond_coh(
        bond_matrix=bond_matrix_target.float(),
        res_mask=res_mask,
        mask_2d=None,
        all_atom_coords = allatom_xyz_withCN,
        aatype = seq_target,
        t = None,
        head_mask=head_mask,
        tail_mask=tail_mask,
    )

    loss_sc_fape = criterion_sc_fape.forward_from_backbone(
        xyz_bb_pred=allatom_xyz[:,:,:3,:],
        pred_atom14_pos=allatom_xyz,
        xyz_bb_gt=xyz_orig[:,:,:3,:],
        atom14_gt_pos=xyz_orig,
        res_mask=final_res_mask,
        bond_mat=bond_matrix_target
    )

    loss_clash = loss_clash 
    loss_torsion = loss_torsion
    loss_bond_coh = loss_bond_coh


    return torch.tensor(0.0,device=sampler.device), torch.tensor(0.0,device=sampler.device), torch.tensor(0.0,device=sampler.device),\
    loss_clash,torch.tensor(0.0,device=sampler.device), loss_torsion, loss_bond_coh, loss_sc_fape, torch.tensor(1.0,device=sampler.device)


def validate_model(sampler, ddp_model, dataloader_val, criterion_frame, criterion_seq, criterion_bond, 
                    criterion_FAPE, criterion_clash, criterion_torsion, criterion_bond_coh, criterion_sc_fape, rank, world_size, epoch):
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
    val_loss_sc_fape = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader_val:
            if batch_data is None: continue
            loss_frame, loss_seq, loss_bond, loss_clash, loss_fape, loss_torsion, loss_bond_coh, loss_sc_fape, _ = model_forward(
                sampler, ddp_model, batch_data, criterion_frame, criterion_seq, criterion_bond, criterion_FAPE, criterion_clash, criterion_torsion, criterion_bond_coh, criterion_sc_fape
            )
            
            # 根据epoch延迟启用部分loss
            w_clash_eff = 0.0 if epoch < clash_delay_epochs else w_clash
            w_bond_coh_eff = 0.0 if epoch < bond_coh_delay_epochs else w_bond_coh
            total_loss = w_frame*loss_frame + w_seq*loss_seq + w_bond*loss_bond + w_clash_eff*loss_clash + w_fape*loss_fape + w_torsion*loss_torsion + w_bond_coh_eff*loss_bond_coh + w_sc_fape*loss_sc_fape
            val_loss_total += total_loss.item()
            val_loss_frame += loss_frame.item()
            val_loss_seq += loss_seq.item()
            val_loss_bond += loss_bond.item()
            val_loss_clash += loss_clash.item()
            val_loss_fape += loss_fape.item()
            val_loss_torsion += loss_torsion.item()
            val_loss_bond_coh += loss_bond_coh.item()
            val_loss_sc_fape += loss_sc_fape.item()
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
        val_loss_sc_fape /= num_batches
    
    # 同步所有GPU的验证损失
    val_tensor = torch.tensor([val_loss_total, val_loss_frame, val_loss_seq, val_loss_bond, val_loss_clash, val_loss_fape, val_loss_torsion, val_loss_bond_coh, val_loss_sc_fape], device=f'cuda:{rank}')
    dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
    global_val_losses = val_tensor / world_size
    
    return (
        global_val_losses[0].item(), global_val_losses[1].item(), global_val_losses[2].item(),
        global_val_losses[3].item(), global_val_losses[4].item(), global_val_losses[5].item(),
        global_val_losses[6].item(), global_val_losses[7].item(), global_val_losses[8].item()
    )


# 1. 初始化进程组 
def setup(rank, world_size):
    dist.init_process_group( 
        backend="nccl",  # NVIDIA GPU推荐使用nccl 
        init_method="tcp://127.0.0.1:12348",  # 本机通信 
        rank=rank,
        world_size=world_size 
    )
# 2. 清理进程组 
def cleanup():
    dist.destroy_process_group() 

def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, warmup_start_lr, max_lr, min_lr):
    """
    创建带warmup的余弦退火学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_steps: warmup步数
        total_steps: 总训练步数
        warmup_start_lr: warmup起始学习率
        max_lr: 最大学习率（warmup结束时的学习率）
        min_lr: 最小学习率（训练结束时的学习率）
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup阶段：线性增长
            return warmup_start_lr / max_lr + (1 - warmup_start_lr / max_lr) * current_step / warmup_steps
        else:
            # Cosine annealing阶段
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr / max_lr + (1 - min_lr / max_lr) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)

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
    
def setup_training(ddp_model, dataloader_train, accumulation_steps,device):
    """Initializes optimizer, scheduler, and loss criteria."""
    # Build optimizer with parameter groups when using APMWrapper
    # if hasattr(ddp_model, 'module') and \
    #    hasattr(ddp_model.module, 'apm') and \
    #    hasattr(ddp_model.module, 'torsion_head') and \
    #    hasattr(ddp_model.module, 'bond_pred'):
    #     backbone_params = ddp_model.module.apm.parameters()
    #     added_params = list(ddp_model.module.torsion_head.parameters()) + \
    #                    list(ddp_model.module.bond_pred.parameters())
    #     param_groups = [
    #         { 'params': backbone_params, 'lr': lr_backbone },
    #         { 'params': added_params, 'lr': lr_added },
    #     ]
    #     optimizer = optim.AdamW(param_groups, weight_decay=0)
    # else:
    optimizer = optim.AdamW(ddp_model.parameters(), lr=lr, weight_decay=1e-5)
    effective_steps_per_epoch = len(dataloader_train) // accumulation_steps
    total_steps = effective_steps_per_epoch * epochs
    print(f"Effective steps per epoch: {effective_steps_per_epoch}")

#scheduler = CosineAnnealingLR(optimizer, T_max=effective_steps_per_epoch * epochs, eta_min=1e-5)
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps} ({100*warmup_steps/total_steps:.1f}% of total)")
    
    # 使用带warmup的余弦退火调度器
    scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        warmup_start_lr=warmup_start_lr,
        max_lr=lr,
        min_lr=warmup_start_lr
    )
    
    criterion_frame = LFrameLoss(w_trans=0.5, w_rot=1, gamma=1.1, d_clamp=15)
    criterion_seq = LseqLoss()
    criterion_bond = DSMCrossEntropyLoss()
    criterion_FAPE = FAPELoss(clamp_distance=15)
    criterion_clash = OpenFoldClashLoss(device=device)
    criterion_torsion = TorsionLossLegacy()
    criterion_sc_fape = SidechainFAPELoss()
    # Bond coherence loss (uses link.csv in config)
    link_csv_path = "/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/link.csv"
    criterion_bond_coh = BondCoherenceLoss(link_csv_path=link_csv_path, device=device)
    
    criteria = {
        'frame': criterion_frame, 'seq': criterion_seq, 'bond': criterion_bond,
        'FAPE': criterion_FAPE, 'clash': criterion_clash, 'torsion': criterion_torsion,
        'bond_coh': criterion_bond_coh, 'sc_fape': criterion_sc_fape,
    }
    
    return optimizer, scheduler, criteria

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
    sampler = main(cfg,device=f"cuda:{rank}")
    model = sampler.sidechain_model
    ddp_model = DDP(model, device_ids=[rank],find_unused_parameters=False)

    # 数据加载器 + 分布式采样器 
    dataloader_train, dataloader_val = get_dataloader(
        conf=cfg,
        batch_size=batch_size,
        pdb_list_path=cluster_file,
        pdb_dir=pdb_dir,
        sampling_ratios=sampling_ratios,
        distributed=True,
        num_workers=4,
        crop_length=crop_size,
        device=f'cuda:{rank}',
        rank=rank,
        num_replicas=world_size,
        val_split=val_ratio,
        seed=seed,
        cache_dir = cache_dir,
        mask_bond_prob = dataloader_mask_bond_prob,
        mask_seq_str_prob=dataloader_mask_seq_str_prob
    )

    # 优化器和损失函数 
    optimizer, scheduler, criteria = setup_training(ddp_model, dataloader_train, accumulation_steps,device=f"cuda:{rank}")
    criterion_frame, criterion_seq, criterion_bond = criteria['frame'], criteria['seq'], criteria['bond']
    criterion_FAPE, criterion_clash, criterion_torsion = criteria['FAPE'], criteria['clash'], criteria['torsion']
    criterion_bond_coh = criteria['bond_coh']
    criterion_sc_fape = criteria['sc_fape']

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
        accumulated_sc_fape_loss = 0.0

        ddp_model.train()
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
                loss_sc_fape = torch.tensor(0.0, device=device)
                partial_T = torch.tensor(0.0, device=device)
                pdb_id = ["EMPTY_BATCH"]
            else:
                print(f"++++++++++start batch{i} in {rank}+++++++++++++")
                loss_frame, loss_seq, loss_bond, loss_clash, loss_fape, loss_torsion, loss_bond_coh, loss_sc_fape, partial_T = model_forward(
                    sampler, ddp_model, batch_data, criterion_frame, criterion_seq, criterion_bond,
                    criterion_FAPE, criterion_clash, criterion_torsion, criterion_bond_coh, criterion_sc_fape
                )
                pdb_id = batch_data['pdb_id']

            print("part T:", partial_T)
            # 计算总损失
            print(f"losses: frame={loss_frame.item():.4f}, seq={loss_seq.item():.4f}, bond={loss_bond.item():.4f}, clash={loss_clash.item():.4f}, fape={loss_fape.item():.4f}, torsion={loss_torsion.item():.4f}, bond_coh={loss_bond_coh.item():.4f}, sc_fape={loss_sc_fape.item():.4f}")

            losses = {
                "loss_frame": loss_frame,
                "loss_seq": loss_seq,
                "loss_bond": loss_bond,
                "loss_clash": loss_clash,
                "loss_fape": loss_fape,
                "loss_torsion": loss_torsion,
                "loss_bond_coh": loss_bond_coh,
                "loss_sc_fape": loss_sc_fape,
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
            accumulated_sc_fape_loss += loss_sc_fape.item()

            # 根据epoch延迟启用部分loss
            w_clash_eff = 0.0 if epoch < clash_delay_epochs else w_clash
            w_bond_coh_eff = 0.0 if epoch < bond_coh_delay_epochs else w_bond_coh
            total_loss = w_frame*loss_frame + w_seq*loss_seq + w_bond*loss_bond + w_clash_eff*loss_clash + w_fape*loss_fape + w_torsion*loss_torsion + w_bond_coh_eff*loss_bond_coh + w_sc_fape*loss_sc_fape
            
            
            accumulated_loss += total_loss.item()

            scaled_loss = total_loss / accumulation_steps      

            is_last_accumulation_step = (i + 1) % accumulation_steps == 0
            if not is_last_accumulation_step:
                # 在 no_sync 上下文中执行反向传播，梯度只在本地累积
                with ddp_model.no_sync():
                    scaled_loss.backward()

            else:
                # 最后一步，正常执行反向传播，DDP会同步所有累积的梯度
                scaled_loss.backward()

                # 计算当前 GPU 上这个伪批量的平均 loss
                avg_pseudo_batch_loss_local = accumulated_loss / accumulation_steps
                avg_pseudo_batch_loss_seq = accumulated_seq_loss / accumulation_steps
                avg_pseudo_batch_loss_bond = accumulated_bond_loss / accumulation_steps
                avg_pseudo_batch_loss_clash = accumulated_clash_loss / accumulation_steps
                avg_pseudo_batch_loss_fape = accumulated_fape_loss / accumulation_steps
                avg_pseudo_batch_loss_frame = accumulated_frame_loss / accumulation_steps
                avg_pseudo_batch_loss_torsion = accumulated_torsion_loss / accumulation_steps
                avg_pseudo_batch_loss_bond_coh = accumulated_bond_coh_loss / accumulation_steps
                avg_pseudo_batch_loss_sc_fape = accumulated_sc_fape_loss / accumulation_steps

                # 同步所有 GPU 的 loss
                loss_tensor = torch.tensor([avg_pseudo_batch_loss_local, 
                                            avg_pseudo_batch_loss_frame, 
                                            avg_pseudo_batch_loss_seq,
                                            avg_pseudo_batch_loss_bond,
                                            avg_pseudo_batch_loss_clash,
                                            avg_pseudo_batch_loss_fape,
                                            avg_pseudo_batch_loss_torsion,
                                            avg_pseudo_batch_loss_bond_coh,
                                            avg_pseudo_batch_loss_sc_fape], device=f'cuda:{rank}')

                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                global_avg_losses = loss_tensor / world_size
                global_avg_loss, global_avg_loss_frame, global_avg_loss_seq, global_avg_loss_bond, global_avg_loss_clash, global_avg_loss_fape, global_avg_loss_torsion, global_avg_loss_bond_coh, global_avg_loss_sc_fape = global_avg_losses.tolist()


                if rank == 0:              
                    effective_step = (i + 1) // accumulation_steps
                    elapsed_time = time.perf_counter()-timer
                    current_lr = optimizer.param_groups[0]['lr']
                    log_message = (f"Epoch: {epoch}, Step: {effective_step}, Time: {elapsed_time:.2f}, LR: {current_lr:.6e}, "
                                   f"Avg Loss: {global_avg_loss:.4f}, Frame: {global_avg_loss_frame:.4f}, "
                                   f"Seq: {global_avg_loss_seq:.4f}, Bond: {global_avg_loss_bond:.4f}, "
                                   f"Clash: {global_avg_loss_clash:.4f}, FAPE: {global_avg_loss_fape:.4f}, Torsion: {global_avg_loss_torsion:.4f}, "
                                   f"BondCoh: {global_avg_loss_bond_coh:.4f}, SC_FAPE: {global_avg_loss_sc_fape:.4f}")
                    print(log_message)
                    with open(log_file_path, 'a') as f:
                        f.write(log_message + '\n')
                    with open(csv_log_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, effective_step, 'train', f'{global_avg_loss:.6f}', f'{global_avg_loss_frame:.6f}', 
                                         f'{global_avg_loss_seq:.6f}', f'{global_avg_loss_bond:.6f}', f'{global_avg_loss_clash:.6f}', 
                                         f'{global_avg_loss_fape:.6f}', f'{global_avg_loss_torsion:.6f}', f'{global_avg_loss_bond_coh:.6f}', f'{global_avg_loss_sc_fape:.6f}', f'{elapsed_time:.2f}', f'{current_lr:.6e}'])
                
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
                    optimizer.zero_grad()

                scheduler.step()
                torch.cuda.empty_cache()
                timer = time.perf_counter()

                # 每 validation_frequency*accumulation_steps 次梯度更新后进行一次验证
                if (i + 1) % (validation_frequency*accumulation_steps) == 0:
                    val_loss_total, val_loss_frame, val_loss_seq, val_loss_bond, val_loss_clash, val_loss_fape, val_loss_torsion, val_loss_bond_coh, val_loss_sc_fape = validate_model(
                        sampler, ddp_model, dataloader_val, criterion_frame, criterion_seq, criterion_bond, criterion_FAPE, criterion_clash, criterion_torsion, criterion_bond_coh, criterion_sc_fape, rank, world_size, epoch
                    )
                    
                    if rank == 0:
                        effective_step = (i + 1) // accumulation_steps
                        current_lr = optimizer.param_groups[0]['lr']
                        val_log_message = (f"Validation: Epoch: {epoch}, Step: {effective_step}, LR: {current_lr:.6e}, "
                                           f"Avg Loss: {val_loss_total:.4f}, Frame: {val_loss_frame:.4f}, "
                                           f"Seq: {val_loss_seq:.4f}, Bond: {val_loss_bond:.4f}, "
                                           f"Clash: {val_loss_clash:.4f}, FAPE: {val_loss_fape:.4f}, Torsion: {val_loss_torsion:.4f}, "
                                           f"BondCoh: {val_loss_bond_coh:.4f}, SC_FAPE: {val_loss_sc_fape:.4f}")
                        print(val_log_message)
                        with open(log_file_path, 'a') as f:
                            f.write(val_log_message + '\n')
                        with open(csv_log_file_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([epoch, effective_step, 'validation', f'{val_loss_total:.6f}', f'{val_loss_frame:.6f}', 
                                             f'{val_loss_seq:.6f}', f'{val_loss_bond:.6f}', f'{val_loss_clash:.6f}', 
                                             f'{val_loss_fape:.6f}', f'{val_loss_torsion:.6f}', f'{val_loss_bond_coh:.6f}', f'{val_loss_sc_fape:.6f}', 'N/A', f'{current_lr:.6e}'])
                        
                        # 检查是否是当前epoch中最好的训练损失
                        if val_loss_total < best_epoch_loss:
                            best_epoch_loss = val_loss_total
                            best_epoch_model_state_dict = ddp_model.module.state_dict()
                            best_epoch_effective_step = effective_step

                        # 检查是否是最佳验证损失并保存模型
                        if val_loss_total < best_val_loss:
                            best_val_loss = val_loss_total
                            best_model_state = {
                                'epoch': epoch,
                                'effective_step': effective_step,
                                'model_state_dict': ddp_model.module.state_dict(),  # 注意使用module来获取原始模型
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
                                'val_loss_sc_fape': val_loss_sc_fape,
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
                accumulated_sc_fape_loss = 0.0

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
            f.write(f"warmup_steps: {warmup_steps}, warmup_start_lr: {warmup_start_lr}\n")
            f.write(f"loss_weights: frame={w_frame}, seq={w_seq}, bond={w_bond}, clash={w_clash}, fape={w_fape}, torsion={w_torsion}, bond_coh={w_bond_coh}, sc_fape={w_sc_fape}\n")
            # 记录sampling_ratios
            sampling_ratios_str = ', '.join([f"{k}: {v}" for k, v in sampling_ratios.items()])
            f.write(f"sampling_ratios: {sampling_ratios_str}\n")
            
    # 初始化CSV日志文件
    with open(csv_log_file_path, log_mode, newline='') as f:
        writer = csv.writer(f)
        if log_mode == 'w':
            writer.writerow(["epoch", "effective_step", "type", "total_loss", "frame_loss", "seq_loss", "bond_loss", "clash_loss", "fape_loss", "torsion_loss", "bond_coh_loss", "sc_fape_loss", "time", "learning_rate"])    
    
    world_size = torch.cuda.device_count()   # 获取GPU数量 
    assert total_batch_size % (world_size * batch_size) == 0
    torch.multiprocessing.spawn( 
        train_model, args=(world_size,),
        nprocs=world_size, join=True 
    )


