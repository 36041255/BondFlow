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
):

    pdb_parsed = load_or_process_target(
        pdb_file,
        cache_dir=cache_dir,
        parse_hetatom=True,
        parse_link=True,
        center=False,
    )

    contig, res_mask = iu.generate_crop_contigs(
        pdb_parsed,
        chain_id,
        mode=crop_mode,
        crop_length=crop_length,
        fixed_res=fixed_res,
        expand_preference=expand_preference,
        target_expand_bias=target_expand_bias,
    )

    print(contig)
    contig_new = conf
    # Bond condition is now handled by randomly_fix_bonds or should be managed by the config, not here.
    contig_new.design_config.bond_condition = None
    contig_new.design_config.contigs = contig

    contig_new.design_config.partial_t = 0.1
    target = iu.Target(contig_new.design_config, pdb_parsed, N_C_add=N_C_add)
      
    # Recompute res_mask to align with possibly inserted NTER/CTER and trailing padding
    # Mark non-padding (origin != ('?','-1')) as 1, padding as 0
    if hasattr(target, 'full_origin_pdb_idx') and isinstance(target.full_origin_pdb_idx, list):
        res_mask_full = torch.ones(len(target.full_origin_pdb_idx))
        pad_positions = [i for i, origin in enumerate(target.full_origin_pdb_idx) if origin == ('?', '-1')]
        if pad_positions:
            res_mask_full[pad_positions] = 0
        target.res_mask = res_mask_full
    else:
        target.res_mask = res_mask
        # Apply random bond fixing if configured
        
    if fixed_bond:
        iu.randomly_fix_bonds(target, fixed_bond_config=fixed_bond)

    return target, pdb_parsed, contig

class PDB_dataset(Dataset):
    def __init__(self, conf, pdb_dir, crop_length, clusters_list, cache_dir=None, mask_bond_prob=0.5,mask_seq_str_prob=0.5):
        self.conf = conf
        self.pdb_dir = pdb_dir
        self.crop_length = crop_length
        self.clusters_list = clusters_list
        self.cache_dir = cache_dir
        self.mask_bond_prob = mask_bond_prob
        self.mask_seq_str_prob = mask_seq_str_prob

    def __len__(self):
        return len(self.clusters_list)

    def __getitem__(self, sample_info):
        cluster_idx, target_mode = sample_info
        
        member_pdbs = self.clusters_list[cluster_idx]
        
        # To avoid infinite loops, shuffle and iterate through members
        shuffled_members = random.sample(member_pdbs, len(member_pdbs))

        for chosen_pdb_id in shuffled_members:
            #chosen_pdb_id = random.choice(['AF-P38585-F1-model_v4_A','AF-Q66PG2-F1-model_v4_A', '8WTE_J', '1BH4_A'])
            #chosen_pdb_id = '1BH4_A'
            #chosen_pdb_id = '1BH4_A' #disful + N-C
            #chosen_pdb_id = 'AF-C5A3G1-F1-model_v4_A' #N+GLU
            chosen_pdb_id = 'AF-P38585-F1-model_v4_A'
            # chosen_pdb_id = 'AF-A3DIH0-F1-model_v4_A' 
            print("WARNING: use chosen_pdb_id",chosen_pdb_id)
            try:
                pdb_id = '_'.join(chosen_pdb_id.split('_')[:-1])
                chain_id = chosen_pdb_id.split('_')[-1]
                pdb_file = os.path.join(self.pdb_dir, f"{pdb_id}.cif")
                if not os.path.exists(pdb_file):
                    pdb_file = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
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
                    pdb_file, chain_id, target_mode, self.conf, self.crop_length, fixed_res, 
                    cache_dir=self.cache_dir,fixed_bond=fixed_bond
                )
                
                data = {k: getattr(target, k) for k in target.__dict__ if 'full' in k or "mask" in k}
                data['pdb_id'] = chosen_pdb_id
                data['contig'] = contig
                return data
            except Exception as e:
                print(f"Error processing {chosen_pdb_id}: {e}")
                # This member failed for this mode, try the next one.
                continue
        
        new_idx = random.randint(0, len(self.clusters_list) - 1)
        return self.__getitem__((new_idx, target_mode))

class ClusterRatioSampler(Sampler):
    def __init__(self, dataset, ratios, num_replicas=None, rank=None, shuffle=True, 
                 distrubuted=False,samples_num=None,seed=0):
        if num_replicas is None and distrubuted:
            num_replicas = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        if rank is None and distrubuted:
            rank = torch.distributed.get_rank() if torch.distributed.is_available() else 0

        self.dataset = dataset
        self.ratios = ratios
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.distrubuted = distrubuted

        self.num_clusters = len(dataset)
        self.total_size = samples_num if samples_num is not None else self.num_clusters
        all_ratios = sum(list(self.ratios.values()))
        assert all_ratios == 1, "Ratios must sum to 1."
        print("self.num_replicas",self.num_replicas)
        print("self.rank",self.rank)
        print("self.total_size",self.total_size)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # This list will hold tuples of (cluster_idx, target_mode)
        indices = []
        for mode, ratio in self.ratios.items():
            if ratio <= 0: continue
            
            target_count = int(math.ceil(self.total_size * ratio))
            
            # Sample cluster indices for this mode
            sampled_cluster_indices = torch.randint(high=self.num_clusters, size=(target_count,), generator=g).tolist()
            
            # Add the (cluster_idx, mode) tuple to the list
            for cluster_idx in sampled_cluster_indices:
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

def get_dataloader(conf, batch_size, pdb_list_path, pdb_dir, 
                   sampling_ratios,
                   distributed=False, num_workers=4,
                   crop_length=256, device='cpu', rank=None, num_replicas=None, 
                   val_split=0.1,seed=0, cache_dir=None, 
                   mask_bond_prob =0.5,mask_seq_str_prob=0.5):
    
    # Parse clusters from file
    clusters_dict = defaultdict(list)
    with open(pdb_list_path, 'r') as f:
        for line in f:
            center, member = line.strip().split()
            clusters_dict[center].append(member)
    
    all_clusters = list(clusters_dict.values())
    if not all_clusters:
        raise ValueError("Parsing cluster file failed: No clusters found.")

    random.Random(seed).shuffle(all_clusters)
    split_idx = int(len(all_clusters) * (1 - val_split))
    train_clusters = all_clusters[:split_idx]
    val_clusters = all_clusters[split_idx:]

    # Create training dataset and sampler
    train_dataset = PDB_dataset(conf, pdb_dir, crop_length, train_clusters, cache_dir=cache_dir, 
                                mask_bond_prob=mask_bond_prob,mask_seq_str_prob=mask_seq_str_prob)
    print(f"Training set: {len(train_clusters)} clusters.")
    

    train_sampler = ClusterRatioSampler(train_dataset, ratios=sampling_ratios, 
                                        seed=seed,num_replicas=num_replicas, 
                                        rank=rank, shuffle=True,distrubuted=distributed)
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
    if val_clusters:
        val_dataset = PDB_dataset(conf, pdb_dir, crop_length, val_clusters, cache_dir=cache_dir,
                                mask_bond_prob = mask_bond_prob,mask_seq_str_prob=mask_seq_str_prob)
        print(f"Validation set: {len(val_clusters)} clusters.")

        # For validation, create a list of all possible (cluster, mode) combinations to iterate through.
        
        val_sampler = ClusterRatioSampler(val_dataset, ratios=sampling_ratios, 
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