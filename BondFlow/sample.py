import os
import argparse
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from BondFlow.models.sampler import Sampler
import BondFlow.data.utils as iu

def resolve_relative_path(path, config_dir, check_exists=True):
    """
    解析相对路径，按以下优先级尝试：
    1. 如果已经是绝对路径，直接返回
    2. 相对于项目根目录（BondFlow/ 的父目录）
    3. 相对于配置文件所在目录
    4. 相对于当前工作目录
    
    Args:
        path: 路径字符串
        config_dir: 配置文件所在目录
        check_exists: 是否检查路径是否存在（对于输出路径，应该设为False）
        
    Returns:
        解析后的绝对路径（保留末尾的 / 如果原路径有）
    """
    if not path:
        return path
    
    if os.path.isabs(path):
        return path
    
    # 保存末尾的 / 或 os.sep
    ends_with_sep = path.endswith('/') or path.endswith(os.sep)
    
    # 1. 相对于项目根目录（假设配置文件在 BondFlow/config/ 下）
    project_root = os.path.dirname(os.path.dirname(config_dir))  # BondFlow/ 的父目录
    project_relative_path = os.path.join(project_root, path)
    if not check_exists or os.path.exists(project_relative_path):
        resolved = os.path.abspath(project_relative_path)
        # 如果原路径以 / 结尾，确保解析后的路径也以 / 结尾
        if ends_with_sep and not resolved.endswith(os.sep):
            resolved = resolved + os.sep
        return resolved
    
    # 2. 相对于配置文件所在目录
    config_relative_path = os.path.join(config_dir, path)
    if not check_exists or os.path.exists(config_relative_path):
        resolved = os.path.abspath(config_relative_path)
        if ends_with_sep and not resolved.endswith(os.sep):
            resolved = resolved + os.sep
        return resolved
    
    # 3. 相对于当前工作目录
    cwd_path = os.path.join(os.getcwd(), path)
    if not check_exists or os.path.exists(cwd_path):
        resolved = os.path.abspath(cwd_path)
        if ends_with_sep and not resolved.endswith(os.sep):
            resolved = resolved + os.sep
        return resolved
    
    # 如果都不存在，返回相对于项目根目录的路径（让后续代码处理错误）
    resolved = os.path.abspath(project_relative_path)
    if ends_with_sep and not resolved.endswith(os.sep):
        resolved = resolved + os.sep
    return resolved

def resolve_config_paths(cfg, config_dir):
    """
    解析配置文件中所有相对路径为绝对路径
    
    Args:
        cfg: OmegaConf 配置对象
        config_dir: 配置文件所在目录
    """
    # 解析 input_pdb
    if hasattr(cfg.design_config, 'input_pdb') and cfg.design_config.input_pdb:
        cfg.design_config.input_pdb = resolve_relative_path(cfg.design_config.input_pdb, config_dir)
    
    # 解析 link_config
    if hasattr(cfg.preprocess, 'link_config') and cfg.preprocess.link_config:
        cfg.preprocess.link_config = resolve_relative_path(cfg.preprocess.link_config, config_dir)
    
    # 解析 model.ckpt_path
    if hasattr(cfg.model, 'ckpt_path') and cfg.model.ckpt_path:
        cfg.model.ckpt_path = resolve_relative_path(cfg.model.ckpt_path, config_dir)
    
    # 解析 model.model_config_path
    if hasattr(cfg.model, 'model_config_path') and cfg.model.model_config_path:
        cfg.model.model_config_path = resolve_relative_path(cfg.model.model_config_path, config_dir)
    
    # 解析 inference.output_prefix（输出路径，不需要检查是否存在）
    if hasattr(cfg.inference, 'output_prefix') and cfg.inference.output_prefix:
        cfg.inference.output_prefix = resolve_relative_path(cfg.inference.output_prefix, config_dir, check_exists=False)
        
        # 计算 out_dir（从解析后的 output_prefix）
        out_prefix = cfg.inference.output_prefix
        if out_prefix.endswith('/') or out_prefix.endswith(os.sep):
            # 如果以 / 结尾，说明是目录路径，直接使用（去掉末尾的 /）
            out_dir = out_prefix.rstrip('/').rstrip(os.sep)
        else:
            # 如果没有以 / 结尾，可能是文件路径，取 dirname；或者就是目录路径
            # 检查是否是目录（如果存在的话）
            if os.path.exists(out_prefix) and os.path.isdir(out_prefix):
                out_dir = out_prefix
            else:
                # 假设是文件路径，取 dirname
                out_dir = os.path.dirname(out_prefix)
        
        # 将 out_dir 添加到配置中，方便后续使用
        cfg.inference.out_dir = out_dir

def run_sampling_worker(device_str, cfg_path, num_designs, num_cycle, num_timesteps, 
                        write_trajectory, out_dir, use_partial_diffusion, batch_range):
    """
    单个设备上的采样工作函数（支持CPU和GPU）
    
    Args:
        device_str: 设备字符串，如 'cuda:0', 'cpu', 'cuda'
        cfg_path: 配置文件路径
        num_designs: 每个批次的样本数
        num_cycle: 总批次数
        num_timesteps: 时间步数
        write_trajectory: 是否写入轨迹
        out_dir: 输出目录
        use_partial_diffusion: 是否使用部分扩散
        batch_range: (start_batch, end_batch) 该设备处理的批次范围
    """
    # 设置设备
    device = torch.device(device_str)
    if device.type == 'cuda':
        if ':' in device_str:
            gpu_id = int(device_str.split(':')[1])
            torch.cuda.set_device(gpu_id)
        else:
            gpu_id = 0
            torch.cuda.set_device(0)
        device_label = f"GPU {gpu_id}"
    else:
        device_label = "CPU"
    
    print(f"[{device_label}] Starting sampling on device {device}, processing batches {batch_range[0]} to {batch_range[1]-1}")
    
    cfg = OmegaConf.load(cfg_path)
    
    # 解析相对路径：将相对于配置文件所在目录的路径转换为绝对路径
    config_dir = os.path.dirname(os.path.abspath(cfg_path))
    resolve_config_paths(cfg, config_dir)
    
    # 从配置中获取 out_dir（已在 resolve_config_paths 中计算）
    out_dir = getattr(cfg.inference, 'out_dir', None)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    sampler = Sampler(cfg, device=device)
    
    start_batch, end_batch = batch_range
    num_batches_this_gpu = end_batch - start_batch
    
    if use_partial_diffusion:
        partial_t = cfg.design_config.partial_t
        
        # Build Target strictly from YAML
        pdb_parsed = iu.process_target(
            cfg.design_config.input_pdb,
            parse_hetatom=False,
            center=False,
            parse_link=True,
            link_csv_path=getattr(cfg.preprocess, 'link_config', None),
        )
        target = iu.Target(cfg.design_config, pdb_parsed, nc_pos_prob=0.0, inference=True)

        # Derive inputs from Target
        L = target.full_seq.shape[0]
        rf_idx = target.full_rf_idx[None, :].to(device)
        pdb_idx = [target.full_pdb_idx]
        res_mask = torch.ones(L, dtype=torch.bool, device=device)[None, :]
        str_mask = target.full_mask_str[None, :].to(device)
        seq_mask = target.full_mask_seq[None, :].to(device)
        bond_mask = target.full_bond_mask[None, :, :].to(device)
        head_mask = target.full_head_mask[None, :].to(device)
        tail_mask = target.full_tail_mask[None, :].to(device)
        N_C_anchor = target.full_N_C_anchor[None, :, :, :].to(device)
        chain_ids = target.full_chain_ids[None, :].to(device)
        hotspots = target.full_hotspot[None, :].to(device)
        
        # PLM / full-structure metadata
        if 'pdb_id' in pdb_parsed:
            pdb_core_id = [pdb_parsed['pdb_id']] * num_designs
        else:
            pdb_basename = os.path.splitext(os.path.basename(cfg.design_config.input_pdb))[0]
            pdb_core_id = [pdb_basename] * num_designs

        assert getattr(target, "pdb_seq_full", None) is not None, "Target.pdb_seq_full must be set in YAML"
        assert getattr(target, "pdb_idx_full", None) is not None, "Target.pdb_idx_full must be set in YAML"
        assert getattr(target, "full_origin_pdb_idx", None) is not None, "Target.full_origin_pdb_idx must be set in YAML"
        pdb_seq_full = [getattr(target, "pdb_seq_full", None)] * num_designs
        pdb_idx_full = [getattr(target, "pdb_idx_full", None)] * num_designs
        origin_pdb_idx = [target.full_origin_pdb_idx] * num_designs
        # INSERT_YOUR_CODE
        print("Target成员变量名：", list(vars(target).keys()))
        print("length:",L)
        print("str_mask:",str_mask)
        print("res_mask:",res_mask)
        print("seq_mask:",seq_mask)
        print("head_mask:",head_mask)
        print("tail_mask:",tail_mask)
        print("seq_full:",target.full_seq)
        torch.set_printoptions(profile="full")
        print("bond_mask:",bond_mask[0])
        torch.set_printoptions(profile="default") # rese
            
        print("bond_mask:",bond_mask[0,0,16],bond_mask[0,16,0])
        print("rf_idx:",rf_idx)
        print("pdb_idx:",pdb_idx)
        print("full_bond_matrix",target.full_bond_matrix)
        print("N_C_anchor",torch.where(N_C_anchor))
        print("origin_pdb_idx",origin_pdb_idx[0])
        print("pdb_core_id",pdb_core_id[0])
        print("pdb_idx_full",pdb_idx_full[0])
        print("pdb_seq_full",pdb_seq_full[0])
        # 准备目标数据用于 partial diffusion
        xyz_target = target.full_xyz[None, :, :3, :].to(device)
        seq_target = target.full_seq[None, :].to(device)
        ss_target = target.full_bond_matrix[None, :, :].to(device)
        
        print(f"[{device_label}] Processing {num_batches_this_gpu} batches (total: {num_batches_this_gpu * num_designs} designs)")
        for local_batch_i in range(num_batches_this_gpu):
            global_batch_i = start_batch + local_batch_i
            start_index = global_batch_i * num_designs
            print(f"[{device_label}] Batch {global_batch_i}/{num_cycle-1}, start_index={start_index}")
            
            sampler.sample_from_partial(
                xyz_target=xyz_target,
                seq_target=seq_target,
                ss_target=ss_target,
                num_batch=num_designs,
                num_res=L,
                N_C_anchor=N_C_anchor.repeat(num_designs, 1, 1, 1),
                partial_t=partial_t,
                num_timesteps=num_timesteps,
                rf_idx=rf_idx.repeat(num_designs, 1),
                pdb_idx=pdb_idx * num_designs,
                res_mask=res_mask.repeat(num_designs, 1),
                str_mask=str_mask.repeat(num_designs, 1),
                seq_mask=seq_mask.repeat(num_designs, 1),
                bond_mask=bond_mask.repeat(num_designs, 1, 1),
                head_mask=head_mask.repeat(num_designs, 1),
                tail_mask=tail_mask.repeat(num_designs, 1),
                record_trajectory=write_trajectory,
                out_pdb_dir=out_dir,
                chain_ids=chain_ids.repeat(num_designs, 1),
                origin_pdb_idx=origin_pdb_idx,
                pdb_seq_full=pdb_seq_full,
                pdb_idx_full=pdb_idx_full,
                pdb_core_id=pdb_core_id,
                hotspots=hotspots.repeat(num_designs, 1),
                start_index=int(start_index),
            )
    else:
        # Full diffusion from prior
        if getattr(cfg.design_config, "contigs", None) is not None:
            pdb_parsed = None
            if getattr(cfg.design_config, "input_pdb", None):
                pdb_parsed = iu.process_target(
                    cfg.design_config.input_pdb,
                    parse_hetatom=False,
                    center=False,
                    parse_link=True,
                    link_csv_path=getattr(cfg.preprocess, "link_config", None),
                )
            else:
                pdb_parsed = {"pdb_id": "prior", "chains": [], "pdb_idx": []}

            target = iu.Target(cfg.design_config, pdb_parsed, nc_pos_prob=0.0, inference=True)

            L = target.full_seq.shape[0]
            rf_idx = target.full_rf_idx[None, :].to(device)
            pdb_idx = [target.full_pdb_idx]
            res_mask = torch.ones(L, dtype=torch.bool, device=device)[None, :]
            str_mask = target.full_mask_str[None, :].to(device)
            seq_mask = target.full_mask_seq[None, :].to(device)
            bond_mask = target.full_bond_mask[None, :, :].to(device)
            head_mask = target.full_head_mask[None, :].to(device)
            tail_mask = target.full_tail_mask[None, :].to(device)
            N_C_anchor = target.full_N_C_anchor[None, :, :, :].to(device)
            chain_ids = target.full_chain_ids[None, :].to(device)
            hotspots = target.full_hotspot[None, :].to(device)

            seq_init = target.full_seq[None, :].to(device)
            ss_init = target.full_bond_matrix[None, :, :].to(device)
            xyz_init = target.full_xyz[None, :, :3, :].to(device)

            if pdb_parsed and "pdb_id" in pdb_parsed:
                pdb_core_id = [pdb_parsed["pdb_id"]] * num_designs
            else:
                pdb_core_id = ["prior"] * num_designs

            if not getattr(cfg.design_config, "input_pdb", None):
                origin_pdb_idx = [target.full_pdb_idx] * num_designs
                pdb_idx_full = [target.full_pdb_idx] * num_designs
                pdb_seq_full = [target.full_seq.detach().cpu()] * num_designs
            else:
                origin_pdb_idx = [target.full_origin_pdb_idx] * num_designs
                seq_full = getattr(target, "pdb_seq_full", None)
                idx_full = getattr(target, "pdb_idx_full", None)
                pdb_seq_full = [seq_full] * num_designs
                pdb_idx_full = [idx_full] * num_designs
            print("rf_idx",rf_idx)
            print("pdb_idx",pdb_idx)
            print("res_mask",res_mask)
            print("str_mask",str_mask)
            print("seq_mask",seq_mask)
            print("head_mask",head_mask)
            print("tail_mask",tail_mask)
            print("bond_mask",bond_mask)
            print("seq_init",seq_init)
            print("ss_init",ss_init)
            print("N_C_anchor",N_C_anchor)
            print("chain_ids",chain_ids)
            print("origin_pdb_idx",origin_pdb_idx[0])
            print("pdb_seq_full",pdb_seq_full[0])
            print("pdb_idx_full",pdb_idx_full[0])
            print("pdb_core_id",pdb_core_id)
            print("hotspots",hotspots)
            total = num_cycle * num_designs
            print(f"Running prior sampling in {num_cycle} batches x {num_designs} designs = {total} total")
            print(f"[{device_label}] Processing {num_batches_this_gpu} batches (total: {num_batches_this_gpu * num_designs} designs)")
            for local_batch_i in range(num_batches_this_gpu):
                global_batch_i = start_batch + local_batch_i
                start_index = global_batch_i * num_designs
                print(f"[{device_label}] Batch {global_batch_i}/{num_cycle-1}, start_index={start_index}")
                
                sampler.sample_from_prior(
                    num_batch=num_designs,
                    num_res=L,
                    num_timesteps=num_timesteps,
                    rf_idx=rf_idx.repeat(num_designs, 1),
                    pdb_idx=pdb_idx * num_designs,
                    res_mask=res_mask.repeat(num_designs, 1),
                    str_mask=str_mask.repeat(num_designs, 1),
                    seq_mask=seq_mask.repeat(num_designs, 1),
                    bond_mask=bond_mask.repeat(num_designs, 1, 1),
                    seq_init=seq_init.repeat(num_designs, 1),
                    ss_init=ss_init.repeat(num_designs, 1, 1),
                    xyz_init=xyz_init.repeat(num_designs, 1, 1, 1),
                    head_mask=head_mask.repeat(num_designs, 1),
                    tail_mask=tail_mask.repeat(num_designs, 1),
                    N_C_anchor=N_C_anchor.repeat(num_designs, 1, 1, 1),
                    record_trajectory=False,
                    out_pdb_dir=out_dir,
                    chain_ids=chain_ids.repeat(num_designs, 1),
                    origin_pdb_idx=origin_pdb_idx,
                    pdb_seq_full=pdb_seq_full,
                    pdb_idx_full=pdb_idx_full,
                    pdb_core_id=pdb_core_id,
                    hotspots=hotspots.repeat(num_designs, 1),
                    start_index=int(start_index),
                )
        else:
            # Legacy pure-prior mode
            assert cfg.design_config.length is not None, "design_config.length must be set in YAML"
            print(f"[{device_label}] Processing {num_batches_this_gpu} batches (total: {num_batches_this_gpu * num_designs} designs)")
            for local_batch_i in range(num_batches_this_gpu):
                global_batch_i = start_batch + local_batch_i
                start_index = global_batch_i * num_designs
                print(f"[{device_label}] Batch {global_batch_i}/{num_cycle-1}, start_index={start_index}")
                
                sampler.sample_from_prior(
                    num_batch=num_designs,
                    num_res=cfg.design_config.length,
                    num_timesteps=num_timesteps,
                    record_trajectory=False,
                    out_pdb_dir=out_dir,
                    start_index=int(start_index),
                )
    
    print(f"[{device_label}] Finished all batches")

def main():
    parser = argparse.ArgumentParser(description="Cyclize a segment using only YAML configuration")
    parser.add_argument("--cfg", default="./config/cyclize.yaml", help="Config yaml path")
    parser.add_argument("--device", default="auto", help="Device selection: 'auto', 'cpu', 'cuda', 'cuda:0', or multiple GPUs like 'cuda:0,cuda:1,cuda:2'")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    
    # 解析相对路径：将相对于配置文件所在目录的路径转换为绝对路径
    config_dir = os.path.dirname(os.path.abspath(args.cfg))
    resolve_config_paths(cfg, config_dir)
    
    num_designs = int(getattr(cfg.inference, 'num_designs', 1)) if hasattr(cfg, 'inference') else 1
    num_cycle = cfg.inference.num_cycle
    num_timesteps = int(cfg.interpolant.sampling.num_timesteps)
    write_trajectory = cfg.inference.write_trajectory

    assert cfg.inference.output_prefix is not None, "inference.output_prefix must be set in YAML"
    
    # 从配置中获取 out_dir（已在 resolve_config_paths 中计算）
    out_dir = getattr(cfg.inference, 'out_dir', None)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    use_partial_diffusion = getattr(cfg.design_config, 'use_partial_diffusion', False)
    assert cfg.design_config.use_partial_diffusion is not None, "design_config.use_partial_diffusion must be set in YAML"
    
    # Parse device argument to determine GPU list
    device_str = args.device.strip()
    
    # Check if multiple GPUs are specified (comma-separated)
    if ',' in device_str:
        # Multi-GPU mode: parse comma-separated GPU list
        gpu_strs = [g.strip() for g in device_str.split(',')]
        gpu_ids = []
        for gpu_str in gpu_strs:
            if gpu_str.startswith('cuda:'):
                gpu_id = int(gpu_str.split(':')[1])
                gpu_ids.append(gpu_id)
            elif gpu_str == 'cuda':
                # If just 'cuda', use GPU 0
                gpu_ids.append(0)
            else:
                raise ValueError(f"Invalid GPU specification: {gpu_str}. Use 'cuda:0', 'cuda:1', etc.")
        
        num_gpus = len(gpu_ids)
        available_gpus = torch.cuda.device_count()
        for gpu_id in gpu_ids:
            if gpu_id >= available_gpus:
                raise ValueError(f"GPU {gpu_id} not available. Only {available_gpus} GPUs detected.")
        
        print(f"Multi-GPU mode: Using {num_gpus} GPUs: {gpu_ids}")
        use_multiprocessing = True
        device = None  # Will be set per process
    elif device_str == "auto":
        # Single GPU mode: auto-select
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Single GPU mode: Selected device: {device}")
        num_gpus = 1
        use_multiprocessing = False
        gpu_ids = None
    elif device_str.startswith("cuda:"):
        # Single GPU mode: specific GPU
        gpu_id = int(device_str.split(':')[1])
        available_gpus = torch.cuda.device_count()
        if gpu_id >= available_gpus:
            raise ValueError(f"GPU {gpu_id} not available. Only {available_gpus} GPUs detected.")
        device = torch.device(device_str)
        print(f"Single GPU mode: Selected device: {device}")
        num_gpus = 1
        use_multiprocessing = False
        gpu_ids = None
    else:
        # Single GPU mode: cpu or cuda (default)
        device = torch.device(device_str)
        print(f"Single GPU mode: Selected device: {device}")
        num_gpus = 1
        use_multiprocessing = False
        gpu_ids = None
    
    if use_multiprocessing:
        # Multi-GPU parallel sampling
        # Distribute batches across GPUs
        batches_per_gpu = num_cycle // num_gpus  # 每个GPU至少处理的cycle数
        remainder = num_cycle % num_gpus  # 余数，需要分配给前几个GPU
        
        batch_ranges = []
        start = 0
        for i in range(num_gpus):
            # 前 remainder 个GPU多处理1个cycle
            end = start + batches_per_gpu + (1 if i < remainder else 0)
            batch_ranges.append((start, end))
            start = end
        
        print(f"\n{'='*60}")
        print(f"Multi-GPU Sampling Configuration:")
        print(f"  Total cycles: {num_cycle}")
        print(f"  Number of GPUs: {num_gpus}")
        print(f"  Base cycles per GPU: {batches_per_gpu}")
        if remainder > 0:
            print(f"  Extra cycles: {remainder} GPU(s) will process {batches_per_gpu + 1} cycles")
        print(f"{'='*60}")
        print(f"Batch distribution across {num_gpus} GPUs:")
        for i, gpu_id in enumerate(gpu_ids):
            start_batch, end_batch = batch_ranges[i]
            num_cycles_this_gpu = end_batch - start_batch
            print(f"  GPU {gpu_id}: cycles {start_batch} to {end_batch-1} ({num_cycles_this_gpu} cycles, {num_cycles_this_gpu * num_designs} designs)")
        print(f"{'='*60}\n")
        
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
        
        # Launch processes
        processes = []
        for i, gpu_id in enumerate(gpu_ids):
            device_str = f"cuda:{gpu_id}"
            p = mp.Process(
                target=run_sampling_worker,
                args=(
                    device_str, args.cfg, num_designs, num_cycle, num_timesteps,
                    write_trajectory, out_dir, use_partial_diffusion, batch_ranges[i]
                )
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print("All GPU processes completed")
    else:
        # Single GPU/CPU mode: use the same worker function
        device_str = str(device)  # Convert torch.device to string
        batch_range = (0, num_cycle)  # Process all cycles
        run_sampling_worker(
            device_str, args.cfg, num_designs, num_cycle, num_timesteps,
            write_trajectory, out_dir, use_partial_diffusion, batch_range
        )

if __name__ == "__main__":
    main()
