import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pickle
from collections import defaultdict
import os
import math
import BondFlow.data.utils as iu
import random

def load_or_process_target(pdb_file, cache_dir=None, **kwargs):
    """
    Loads a parsed PDB from cache if available, otherwise processes and caches it.
    The cache key is the basename of the pdb_file.

    Any extra **kwargs are forwarded to iu.process_target, so you can pass
    things like:
        - parse_hetatom / parse_link / center / parse_alpha
        - plm_encoder: optional callable to compute per-chain PLM embeddings
        - plm_max_chain_length: optional int length cap for PLM encoder
    """
    if cache_dir:
        pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
        cache_file = os.path.join(cache_dir, f"{pdb_id}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Warning: Could not load cached file {cache_file}, will re-process. Error: {e}")

    # If not in cache or cache is invalid, process the file
    pdb_parsed = iu.process_target(pdb_file, **kwargs)

    if cache_dir:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
        cache_file = os.path.join(cache_dir, f"{pdb_id}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(pdb_parsed, f)
    
    return pdb_parsed



def generate_crop_target_pdb(
    pdb_file,
    chain_id,
    crop_mode,
    conf,
    crop_length: int = 256,
    fixed_res=None,
    cache_dir=None,
    N_C_add: bool = True,
    fixed_bond=None,
    expand_preference: str = "auto",
    target_expand_bias: float = 1.0,
    target_len_ratio=(0.2, 0.4),
    nc_pos_prob=0.3,
    hotspot_prob=0.5,
    hotspot_num_range=(1, 6),
    plm_encoder=None,
    plm_max_chain_length: int | None = None,
):

    # NOTE:
    #   We no longer compute PLM embeddings inside process_target/load_or_process_target.
    #   PLM features are now computed lazily inside the model forward pass on full chains.
    #   Therefore, we intentionally do NOT pass plm_encoder / plm_max_chain_length here.
    pdb_parsed = load_or_process_target(
        pdb_file,
        cache_dir=cache_dir,
        parse_hetatom=True,
        parse_link=True,
        center=False,
    )

    if 'complex' in crop_mode and target_len_ratio is not None:
        # If target_len_ratio is a tuple/list, sample uniformly from it
        if isinstance(target_len_ratio, (tuple, list)) and len(target_len_ratio) == 2:
            _target_len_ratio = random.uniform(target_len_ratio[0], target_len_ratio[1])
        else:
            raise ValueError("target_len_ratio must be a tuple/list of length 2")
    else:
        _target_len_ratio = None
    if random.uniform(0,1) < hotspot_prob:
        print(hotspot_num_range)
        hotspot_k_range = hotspot_num_range
    else:
        hotspot_k_range = None
    
    if 'complex' in crop_mode:
        crop_length_res = crop_length - 4
    elif 'monomer' in crop_mode:
        crop_length_res = crop_length - 2
    else:
        raise ValueError("crop_mode must be 'complex' or 'monomer'")

    contig, res_mask, hotspots = iu.generate_crop_contigs(
        pdb_parsed,
        chain_id,
        mode=crop_mode,
        crop_length=crop_length_res,
        fixed_res=fixed_res,
        expand_preference=expand_preference,
        target_expand_bias=target_expand_bias,
        target_len_ratio=_target_len_ratio,
        hotspot_k_range=hotspot_k_range,
    )
    print("contig",contig)
    print("hotspots",hotspots)
    contig_new = conf
    # Bond condition is now handled by randomly_fix_bonds or should be managed by the config, not here.
    contig_new.design_config.bond_condition = None
    contig_new.design_config.contigs = contig
    contig_new.design_config.hotspots = hotspots
    contig_new.design_config.partial_t = 0.1
    target = iu.Target(contig_new.design_config, pdb_parsed, N_C_add=N_C_add, nc_pos_prob=nc_pos_prob)
      
    # Recompute res_mask to align with possibly inserted NTER/CTER and trailing padding
    # Mark non-padding (origin != ('?','-1')) as 1, padding as 0
    # if hasattr(target, 'full_origin_pdb_idx') and isinstance(target.full_origin_pdb_idx, list):
    #     res_mask_full = torch.ones(len(target.full_origin_pdb_idx))
    #     pad_positions = [i for i, origin in enumerate(target.full_origin_pdb_idx) if origin == ('?', '-1')]
    #     if pad_positions:
    #         res_mask_full[pad_positions] = 0
    #     target.res_mask = res_mask_full
    # else:
    #     target.res_mask = res_mask
        # Apply random bond fixing if configured
    #target.res_mask = res_mask
    if fixed_bond:
        iu.randomly_fix_bonds(target, fixed_bond_config=fixed_bond)

    return target, pdb_parsed, contig

class PDB_dataset(Dataset):
    def __init__(
        self,
        conf,
        pdb_dir,
        crop_length,
        clusters_list,
        cache_dir=None,
        mask_bond_prob=0.5,
        mask_seq_str_prob=0.5,
        nc_pos_prob=None,
        hotspot_prob=None,
        plm_encoder=None,
        plm_max_chain_length: int | None = None,
        dataset_configs=None,
    ):
        self.conf = conf
        self.pdb_dir = pdb_dir
        self.crop_length = crop_length
        self.clusters_list = clusters_list
        self.cache_dir = cache_dir
        self.mask_bond_prob = mask_bond_prob
        self.mask_seq_str_prob = mask_seq_str_prob
        # Optional PLM encoder hook used by process_target
        self.plm_encoder = plm_encoder
        self.plm_max_chain_length = plm_max_chain_length
        self.dataset_configs = dataset_configs
        self.nc_pos_prob = nc_pos_prob
        self.hotspot_prob = hotspot_prob
    def __len__(self):
        return len(self.clusters_list)

    def __getitem__(self, sample_info):
        cluster_idx, target_mode = sample_info
        
        cluster_data = self.clusters_list[cluster_idx]
        if isinstance(cluster_data, dict):
            member_pdbs = cluster_data['members']
            current_pdb_dir = cluster_data['pdb_dir']
        else:
            member_pdbs = cluster_data
            current_pdb_dir = self.pdb_dir
        
        # To avoid infinite loops, shuffle and iterate through members
        shuffled_members = random.sample(member_pdbs, len(member_pdbs))

        for chosen_pdb_id in shuffled_members:
            #chosen_pdb_id = random.choice(['AF-P38585-F1-model_v4_A','AF-Q66PG2-F1-model_v4_A', '8WTE_J', '1BH4_A'])
            #chosen_pdb_id = '6O9I_E__6O9I_C_C'#'5DVV_C__5DVV_A_C'
            #chosen_pdb_id = '1BH4_A' #disful + N-C
            #chosen_pdb_id = 'AF-C5A3G1-F1-model_v4_A' #N+GLU
            # chosen_pdb_id = 'AF-P38585-F1-model_v4_A'
            # chosen_pdb_id = 'AF-A3DIH0-F1-model_v4_A' 
            # print("WARNING: use chosen_pdb_id",chosen_pdb_id)
            try:
                pdb_id = '_'.join(chosen_pdb_id.split('_')[:-1])
                chain_id = chosen_pdb_id.split('_')[-1]
                pdb_file = os.path.join(current_pdb_dir, f"{pdb_id}.cif")
                if not os.path.exists(pdb_file):
                    pdb_file = os.path.join(current_pdb_dir, f"{pdb_id}.pdb")
                    if not os.path.exists(pdb_file):
                        continue

                if random.random() > self.mask_seq_str_prob:
                    fixed_res = None
                else:
                    segments = random.randint(1, 2)
                    proportion = random.uniform(0.1, 0.4)
                    fixed_res = {'proportion': proportion, 'segments': segments}

                if random.random() > self.mask_bond_prob:
                    fixed_bond = None
                else:
                    fixed_bond = {'ratio_min': 0, 'ratio_max': 1}

                target, pdb_parsed, contig = generate_crop_target_pdb(
                    pdb_file,
                    chain_id,
                    target_mode,
                    self.conf,
                    self.crop_length,
                    fixed_res,
                    cache_dir=self.cache_dir,
                    N_C_add=True,
                    fixed_bond=fixed_bond,
                    expand_preference="auto",
                    target_expand_bias=1.0,
                    target_len_ratio=(0.2, 0.4),
                    nc_pos_prob=self.nc_pos_prob,
                    hotspot_prob=self.hotspot_prob,
                    hotspot_num_range=(1, 6),
                    plm_encoder=self.plm_encoder,
                    plm_max_chain_length=self.plm_max_chain_length,
                )
                
                # Collect all per-target tensors with "full" or "mask" in the name,
                # but explicitly exclude any precomputed full_plm_emb (PLM is now
                # computed lazily inside the model forward on full chains).
                data = {
                    k: getattr(target, k)
                    for k in target.__dict__
                    if ('full' in k or "mask" in k or "pdb_" in k) 
                }
                data['pdb_id'] = chosen_pdb_id
                data['contig'] = contig

                return data

            except Exception as e:
                print(f"Error processing {chosen_pdb_id}: {e}")
                # This member failed for this mode, try the next one.
                continue
        
        if self.dataset_configs is not None:
            # 1. 提取所有数据集的权重
            weights = [cfg['weight'] for cfg in self.dataset_configs]
            
            # 2. 根据权重随机选择一个配置 (random.choices 返回列表，取 [0])
            chosen_cfg = random.choices(self.dataset_configs, weights=weights, k=1)[0]
            
            # 3. 在该配置的 range 区间内随机选一个索引
            # range 通常是 (start, end)，random.randint 是闭区间，所以要 end - 1
            start, end = chosen_cfg['range']
            new_idx = random.randint(start, end - 1)
        else:
            # 兼容旧逻辑：如果没有配置，就全局均匀随机
            new_idx = random.randint(0, len(self.clusters_list) - 1)

        return self.__getitem__((new_idx, target_mode))

class ClusterRatioSampler(Sampler):
    def __init__(self, dataset, ratios, dataset_configs=None, num_replicas=None, rank=None, shuffle=True, 
                 distrubuted=False,samples_num=None,seed=0):
        if num_replicas is None and distrubuted:
            num_replicas = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        if rank is None and distrubuted:
            rank = torch.distributed.get_rank() if torch.distributed.is_available() else 0

        self.dataset = dataset
        self.ratios = ratios
        self.dataset_configs = dataset_configs
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.distrubuted = distrubuted

        self.num_clusters = len(dataset)
        self.total_size = samples_num if samples_num is not None else self.num_clusters
        
        if self.dataset_configs is None:
            # Backward compatibility: treat as single dataset
            self.dataset_configs = [{
                'range': (0, self.num_clusters),
                'weight': 1.0,
                'mode_ratios': self.ratios
            }]
            all_ratios = sum(list(self.ratios.values()))
            assert all_ratios == 1, "Ratios must sum to 1."
        else:
             # Validate configs
             total_weight = sum(cfg['weight'] for cfg in self.dataset_configs)
             assert abs(total_weight - 1.0) < 1e-6, "Dataset weights must sum to 1."

        print("self.num_replicas",self.num_replicas)
        print("self.rank",self.rank)
        print("self.total_size",self.total_size)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # This list will hold tuples of (cluster_idx, target_mode)
        indices = []
        
        for ds_config in self.dataset_configs:
            ds_range = ds_config['range'] # (start, end)
            ds_weight = ds_config['weight']
            ds_modes = ds_config['mode_ratios']
            
            ds_start, ds_end = ds_range
            ds_len = ds_end - ds_start
            if ds_len == 0: continue

            # Number of samples allocated to this dataset
            ds_total_samples = int(math.ceil(self.total_size * ds_weight))
            
            for mode, ratio in ds_modes.items():
                if ratio <= 0: continue
                
                target_count = int(math.ceil(ds_total_samples * ratio))
                
                # Sample cluster indices within this dataset's range
                # We sample offsets [0, ds_len) and add ds_start
                sampled_offsets = torch.randint(high=ds_len, size=(target_count,), generator=g).tolist()
                
                for offset in sampled_offsets:
                    cluster_idx = ds_start + offset
                    indices.append((cluster_idx, mode))
        
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)

        if self.distrubuted:      
            indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        if self.distrubuted:
            # 返回每个副本的样本数（向上取整）
            return math.ceil(self.total_size / self.num_replicas)
        else:
            # 非分布式模式下返回总数
            return self.total_size

    def set_epoch(self, epoch):
        self.epoch = epoch

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    processed_data = defaultdict(list)
    for item in batch:
        for key, value in item.items():
            processed_data[key].append(value)
    final_batch = {}
    for key, value_list in processed_data.items():
        if value_list and isinstance(value_list[0], torch.Tensor):
            vl = [v.unsqueeze(0) if v.dim() == value_list[0].dim() else v for v in value_list]
            final_batch[key] = torch.cat(vl, dim=0)
        else:
            final_batch[key] = value_list
    return final_batch

def get_dataloader(
    conf,
    batch_size,
    pdb_list_path,
    pdb_dir,
    sampling_ratios,
    distributed: bool = False,
    num_workers: int = 4,
    crop_length: int = 256,
    device: str = 'cpu',
    rank=None,
    num_replicas=None,
    val_split: float = 0.1,
    seed: int = 0,
    cache_dir=None,
    mask_bond_prob: float = 0.5,
    mask_seq_str_prob: float = 0.5,
    nc_pos_prob=None,
    hotspot_prob=None,
    dataset_probs=None,
    samples_num=None,
    plm_encoder=None,
    plm_max_chain_length: int | None = None,
):
    
    # Normalize inputs to lists for multi-dataset support
    if isinstance(pdb_list_path, str):
        pdb_list_paths = [pdb_list_path]
    else:
        pdb_list_paths = pdb_list_path

    if isinstance(pdb_dir, str):
        pdb_dirs = [pdb_dir] * len(pdb_list_paths)
    else:
        pdb_dirs = pdb_dir
        if len(pdb_dirs) != len(pdb_list_paths):
            raise ValueError("Number of pdb_dirs must match pdb_list_paths")

    if dataset_probs is None:
        dataset_probs = [1.0 / len(pdb_list_paths)] * len(pdb_list_paths)
    
    if len(dataset_probs) != len(pdb_list_paths):
        raise ValueError("Number of dataset_probs must match pdb_list_paths")

    # Handle sampling_ratios: if list, map to datasets; if dict, apply to all
    if isinstance(sampling_ratios, list):
         mode_ratios_list = sampling_ratios
         if len(mode_ratios_list) != len(pdb_list_paths):
             raise ValueError("Number of sampling_ratios dicts must match pdb_list_paths")
    else:
         mode_ratios_list = [sampling_ratios] * len(pdb_list_paths)

    all_train_clusters = []
    all_val_clusters = []
    
    train_dataset_configs = []
    val_dataset_configs = []

    current_train_start = 0
    current_val_start = 0

    for i, (list_path, p_dir) in enumerate(zip(pdb_list_paths, pdb_dirs)):
        clusters_dict = defaultdict(list)
        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    center, member = parts[0], parts[1]
                    clusters_dict[center].append(member)
        
        clusters = list(clusters_dict.values())
        if not clusters:
            # raise ValueError(f"Parsing cluster file failed: No clusters found in {list_path}")
            print(f"Warning: No clusters found in {list_path}")
            continue

        # Attach pdb_dir info to clusters
        clusters_with_info = [{'members': m, 'pdb_dir': p_dir} for m in clusters]

        random.Random(seed).shuffle(clusters_with_info)
        split_idx = int(len(clusters_with_info) * (1 - val_split))
        
        train_part = clusters_with_info[:split_idx]
        val_part = clusters_with_info[split_idx:]
        
        all_train_clusters.extend(train_part)
        all_val_clusters.extend(val_part)
        
        weight = dataset_probs[i]
        mode_ratios = mode_ratios_list[i]
        
        train_len = len(train_part)
        if train_len > 0:
            train_dataset_configs.append({
                'range': (current_train_start, current_train_start + train_len),
                'weight': weight,
                'mode_ratios': mode_ratios
            })
            current_train_start += train_len
            
        val_len = len(val_part)
        if val_len > 0:
            val_dataset_configs.append({
                'range': (current_val_start, current_val_start + val_len),
                'weight': weight,
                'mode_ratios': mode_ratios
            })
            current_val_start += val_len

    def normalize_configs(configs):
        total = sum(c['weight'] for c in configs)
        if total > 0:
            for c in configs:
                c['weight'] /= total
        return configs

    train_dataset_configs = normalize_configs(train_dataset_configs)
    val_dataset_configs = normalize_configs(val_dataset_configs)

    if not all_train_clusters:
         raise ValueError("No training clusters found across all datasets.")

    # Create training dataset and sampler
    # Pass first pdb_dir as default, though clusters have their own
    default_pdb_dir = pdb_dirs[0]
    train_dataset = PDB_dataset(
        conf,
        default_pdb_dir,
        crop_length,
        all_train_clusters,
        cache_dir=cache_dir,
        mask_bond_prob=mask_bond_prob,
        mask_seq_str_prob=mask_seq_str_prob,
        nc_pos_prob=nc_pos_prob,
        hotspot_prob=hotspot_prob,
        plm_encoder=plm_encoder,
        plm_max_chain_length=plm_max_chain_length,
        dataset_configs=train_dataset_configs,
    )
    print(f"Training set: {len(all_train_clusters)} clusters (combined).")
    

    train_sampler = ClusterRatioSampler(train_dataset, ratios=mode_ratios_list[0], # unused if dataset_configs is set
                                        dataset_configs=train_dataset_configs,
                                        seed=seed,num_replicas=num_replicas, 
                                        rank=rank, shuffle=True,distrubuted=distributed,
                                        samples_num=samples_num)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if 'cuda' in device  else False
    )

    # Validation dataloader
    val_dataloader = None
    if all_val_clusters:
        val_dataset = PDB_dataset(
            conf,
            default_pdb_dir,
            crop_length,
            all_val_clusters,
            cache_dir=cache_dir,
            mask_bond_prob=mask_bond_prob,
            mask_seq_str_prob=mask_seq_str_prob,
            nc_pos_prob=nc_pos_prob,
            hotspot_prob=hotspot_prob,
            plm_encoder=plm_encoder,
            plm_max_chain_length=plm_max_chain_length,
            dataset_configs=val_dataset_configs,
        )
        print(f"Validation set: {len(all_val_clusters)} clusters (combined).")

        val_sampler = ClusterRatioSampler(val_dataset, ratios=mode_ratios_list[0],
                                        dataset_configs=val_dataset_configs,
                                        seed=seed,num_replicas=num_replicas, 
                                        rank=rank, shuffle=True,distrubuted=distributed)

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=(val_sampler is None),
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if 'cuda' in device else False
        )
    
    return train_dataloader, val_dataloader