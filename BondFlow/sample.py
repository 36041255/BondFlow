import os
import argparse
import torch
from omegaconf import OmegaConf
from BondFlow.models.sampler import Sampler
import BondFlow.data.utils as iu

def main():
    parser = argparse.ArgumentParser(description="Cyclize a segment using only YAML configuration")
    parser.add_argument("--cfg", default="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/cyclize.yaml", help="Config yaml path")
    parser.add_argument("--device", default="auto",help="Device selection (e.g., 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1')")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Selected device: {device}")
    cfg = OmegaConf.load(args.cfg)
    
    num_designs = int(getattr(cfg.inference, 'num_designs', 1)) if hasattr(cfg, 'inference') else 1
    # num_cycle: run N batches, each of size num_designs (total samples = N * num_designs)
    # Backward compatible: default is 1 batch.
    num_cycle = cfg.inference.num_cycle
    num_timesteps = int(cfg.interpolant.sampling.num_timesteps)
    write_trajectory = cfg.inference.write_trajectory

    sampler = Sampler(cfg, device=device)
    
    assert cfg.inference.output_prefix is not None, "inference.output_prefix must be set in YAML"
    print("output_prefix",cfg.inference.output_prefix)
    out_prefix = cfg.inference.output_prefix
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    use_partial_diffusion = getattr(cfg.design_config, 'use_partial_diffusion', False)
    assert cfg.design_config.use_partial_diffusion is not None, "design_config.use_partial_diffusion must be set in YAML"
    if use_partial_diffusion:
        partial_t = cfg.design_config.partial_t  
        assert partial_t is not None, "design_config.partial_t must be set in YAML"
        assert cfg.design_config.input_pdb is not None, "design_config.input_pdb must be set in YAML"
        assert cfg.design_config.contigs is not None, "design_config.contigs must be set in YAML"

        # Build Target strictly from YAML
        pdb_parsed = iu.process_target(
            cfg.design_config.input_pdb,
            parse_hetatom=False,
            center=False,
            parse_link=True,
            link_csv_path=getattr(cfg.preprocess, 'link_config', None),
        )
        # 推理阶段：显式标记 inference=True，让 Target 把 YAML 中的 New_ 残基视作 body，
        # 而不是训练时的 padding 语义。
        target = iu.Target(cfg.design_config, pdb_parsed, nc_pos_prob=0.0, inference=True)

        # Derive inputs from Target (see BondFlow.data.utils.Target)
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
        
        # PLM / full-structure metadata (aligned with dataloader.PDB_dataset)
        # pdb_core_id: 基于 process_target 的 'pdb_id'，否则退回到 input 路径名
        if 'pdb_id' in pdb_parsed:
            pdb_core_id = [pdb_parsed['pdb_id']] * num_designs
        else:
            pdb_basename = os.path.splitext(os.path.basename(cfg.design_config.input_pdb))[0]
            pdb_core_id = [pdb_basename] * num_designs

        # pdb_seq_full / pdb_idx_full: 每个 batch 元素一份 full-structure 元数据
        #
        # IMPORTANT:
        # In inference mode, Target may augment "full" metadata to include pseudo indices
        # for New_ segments (so PLM windowing/scatter can map them). Therefore we must
        # use Target-provided fields rather than raw pdb_parsed.
        assert getattr(target, "pdb_seq_full", None) is not None, "Target.pdb_seq_full must be set in YAML"
        assert getattr(target, "pdb_idx_full", None) is not None, "Target.pdb_idx_full must be set in YAML"
        assert getattr(target, "full_origin_pdb_idx", None) is not None, "Target.full_origin_pdb_idx must be set in YAML"
        pdb_seq_full = [getattr(target, "pdb_seq_full", None)] * num_designs
        pdb_idx_full = [getattr(target, "pdb_idx_full", None)] * num_designs
        
        # origin_pdb_idx: 设计窗口到 full 结构的映射（Target 自带）
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

        print(f"Using partial diffusion starting from t={partial_t}")
        
        # 准备目标数据用于 partial diffusion
        # 从 target 中提取目标结构信息
        xyz_target = target.full_xyz[None, :, :3, :].to(device)  # [1, L, 3, 3] (N, CA, C)
        seq_target = target.full_seq[None, :].to(device)         # [1, L]
        ss_target = target.full_bond_matrix[None, :, :].to(device)  # [1, L, L]
        # Run N batches; outputs are numbered continuously: 0..(num_cycle*num_designs-1)
        total = num_cycle * num_designs
        print(f"Running partial diffusion in {num_cycle} batches x {num_designs} designs = {total} total")
        for batch_i in range(num_cycle):
            start_index = batch_i * num_designs
            sampler.sample_from_partial(
                xyz_target=xyz_target,
                seq_target=seq_target,
                ss_target=ss_target,
                num_batch=num_designs,          # batch size per call
                num_res=L,
                N_C_anchor=N_C_anchor.repeat(num_designs, 1, 1, 1),
                partial_t=partial_t,            # 从指定时间步开始
                num_timesteps=num_timesteps,
                rf_idx=rf_idx.repeat(num_designs, 1),              # 扩展到 batch 维度
                pdb_idx=pdb_idx * num_designs,                     # 复制 pdb_idx 到每个 batch
                res_mask=res_mask.repeat(num_designs, 1),          # 扩展到 batch 维度
                str_mask=str_mask.repeat(num_designs, 1),          # 扩展到 batch 维度
                seq_mask=seq_mask.repeat(num_designs, 1),          # 扩展到 batch 维度
                bond_mask=bond_mask.repeat(num_designs, 1, 1),     # 扩展到 batch 维度
                head_mask=head_mask.repeat(num_designs, 1),        # 扩展到 batch 维度
                tail_mask=tail_mask.repeat(num_designs, 1),        # 扩展到 batch 维度
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
        print("Using full diffusion from prior (but still honoring contigs/bond_condition if provided)")

        # If user provides contigs/bond_condition, build Target to derive masks + fixed initial seq/bond matrix.
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
                # Minimal placeholder; Target will still work for New_/explicit-seq contigs.
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

            # Fixed initial values to respect YAML constraints during prior sampling:
            seq_init = target.full_seq[None, :].to(device)
            ss_init = target.full_bond_matrix[None, :, :].to(device)
            xyz_init = target.full_xyz[None, :, :3, :].to(device)

            # Metadata consistent with partial-diffusion branch
            if pdb_parsed and "pdb_id" in pdb_parsed:
                pdb_core_id = [pdb_parsed["pdb_id"]] * num_designs
            else:
                pdb_core_id = ["prior"] * num_designs

            # PLM / full-structure metadata
            #
            # IMPORTANT:
            # Always prefer Target-provided `pdb_seq_full/pdb_idx_full` in inference mode,
            # because Target may augment "full" metadata to include pseudo indices for New_
            # residues (so PLM windowing/scatter can map them).
            if not getattr(cfg.design_config, "input_pdb", None):
                # No input structure: treat the design itself as the full structure.
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
            for batch_i in range(num_cycle):
                start_index = batch_i * num_designs
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
                    # --- respect YAML fixed conditions ---
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
            # Legacy pure-prior mode: use fixed length only
            print("No contigs/bond_condition provided, using fixed length only")
            assert cfg.design_config.length is not None, "design_config.length must be set in YAML"
            total = num_cycle * num_designs
            print(f"Running pure prior sampling in {num_cycle} batches x {num_designs} designs = {total} total")
            for batch_i in range(num_cycle):
                start_index = batch_i * num_designs
                sampler.sample_from_prior(
                    num_batch=num_designs,
                    num_res=cfg.design_config.length,
                    num_timesteps=num_timesteps,
                    record_trajectory=False,
                    out_pdb_dir=out_dir,
                    start_index=int(start_index),
                )

if __name__ == "__main__":
    main()