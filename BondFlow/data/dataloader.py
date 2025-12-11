import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pickle
from collections import defaultdict, Counter
import os
import math
import random
import multiprocessing as mp

from tqdm import tqdm
import BondFlow.data.utils as iu


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


def _compute_member_spans(args):
    """
    Helper for multiprocessing: given a member id and paths, compute all LINK spans
    on its design chain. Returns (member, spans_list or None).
    """
    member, pdb_dir, cache_dir = args
    try:
        pdb_id = "_".join(member.split("_")[:-1])
        chain_id = member.split("_")[-1]

        pdb_file = os.path.join(pdb_dir, f"{pdb_id}.cif")
        if not os.path.exists(pdb_file):
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            if not os.path.exists(pdb_file):
                return member, None

        pdb_parsed = load_or_process_target(
            pdb_file,
            cache_dir=cache_dir,
            parse_hetatom=False,
            center=False,
            parse_link=True,
            parse_alpha=False,
        )

        pdb_idx = pdb_parsed.get("pdb_idx", [])
        links = pdb_parsed.get("links", [])
        if not links or not pdb_idx:
            return member, None

        # Build (chain, resnum) -> local index for this chain
        chain_pos = {}
        local_idx = 0
        for ch, resnum in pdb_idx:
            if ch == chain_id:
                chain_pos[(ch, resnum)] = local_idx
                local_idx += 1

        if not chain_pos:
            return member, None

        spans = []
        for link in links:
            idx1 = link.get("idx1")
            idx2 = link.get("idx2")
            if not idx1 or not idx2:
                continue
            ch1, _ = idx1
            ch2, _ = idx2
            if ch1 != chain_id or ch2 != chain_id:
                continue
            if idx1 not in chain_pos or idx2 not in chain_pos:
                continue
            span = abs(chain_pos[idx1] - chain_pos[idx2])
            if span <= 0:
                continue
            spans.append(span)

        if not spans:
            return member, None

        return member, spans
    except Exception as e:
        print(f"Error computing LINK spans for {member}: {e}")
        return member, None


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
    target_len_ratio = (0.4, 0.4)  ,
    nc_training = True,
    nc_pos_prob = 0.75,
):

    pdb_parsed = load_or_process_target(
        pdb_file,
        cache_dir=cache_dir,
        parse_hetatom=True,
        parse_link=True,
        center=False,
    )

    if crop_mode == 'complex' and target_len_ratio is not None:
        # If target_len_ratio is a tuple/list, sample uniformly from it
        if isinstance(target_len_ratio, (tuple, list)) and len(target_len_ratio) == 2:
            _target_len_ratio = random.uniform(target_len_ratio[0], target_len_ratio[1])
        else:
            raise ValueError("target_len_ratio must be a tuple/list of length 2")
    else:
        _target_len_ratio = None

    contig, res_mask = iu.generate_crop_contigs(
        pdb_parsed,
        chain_id,
        mode=crop_mode,
        crop_length=crop_length,
        fixed_res=fixed_res,
        expand_preference=expand_preference,
        target_expand_bias=target_expand_bias,
        target_len_ratio=_target_len_ratio,
    )
    print(contig)
    contig_new = conf
    # Bond condition is now handled by randomly_fix_bonds or should be managed by the config, not here.
    contig_new.design_config.bond_condition = None
    contig_new.design_config.contigs = contig

    contig_new.design_config.partial_t = 0.1
    target = iu.Target(contig_new.design_config, pdb_parsed, N_C_add=N_C_add, nc_training=nc_training, nc_pos_prob=nc_pos_prob)
      
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
        """
        sample_info can be:
          - (cluster_idx, target_mode): legacy behavior, uniform over members in cluster
          - (cluster_idx, member_local_idx, target_mode): explicit member index (weighted sampling in Sampler)
        """
        if isinstance(sample_info, tuple) and len(sample_info) == 3:
            cluster_idx, member_local_idx, target_mode = sample_info
        else:
            cluster_idx, target_mode = sample_info
            member_local_idx = None

        member_pdbs = self.clusters_list[cluster_idx]

        # Build an ordered list of member indices to try:
        #   - if member_local_idx is provided, try that one first, then fall back to others
        #   - otherwise, shuffle all members (legacy uniform behavior)
        if member_local_idx is not None and 0 <= member_local_idx < len(member_pdbs):
            ordered_indices = [member_local_idx] + [
                i for i in range(len(member_pdbs)) if i != member_local_idx
            ]
        else:
            ordered_indices = list(range(len(member_pdbs)))
            random.shuffle(ordered_indices)

        for idx in ordered_indices:
            chosen_pdb_id = member_pdbs[idx]
            #chosen_pdb_id = random.choice(['AF-P38585-F1-model_v4_A','AF-Q66PG2-F1-model_v4_A', '8WTE_J', '1BH4_A'])
            #chosen_pdb_id = '1BH4_A'
            #chosen_pdb_id = '1BH4_A' #disful + N-C
            #chosen_pdb_id = 'AF-C5A3G1-F1-model_v4_A' #N+GLU
            #chosen_pdb_id = '2LS1_A' # lassopeptide N-siacechain
            #chosen_pdb_id = '9CDT_B' #头尾环肽复合物 B是环肽
            try:
                pdb_id = '_'.join(chosen_pdb_id.split('_')[:-1])
                chain_id = chosen_pdb_id.split('_')[-1]
                pdb_file = os.path.join(self.pdb_dir, f"{pdb_id}.cif")
                print("chosen_pdb_id",chosen_pdb_id)
                if not os.path.exists(pdb_file):
                    pdb_file = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
                    if not os.path.exists(pdb_file):
                        print(f"pdb_file not found: {pdb_file}")
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
                
 
                target_expand_bias = random.uniform(0, 0.5)
                target, pdb_parsed, contig = generate_crop_target_pdb(
                    pdb_file, chain_id, target_mode, self.conf, self.crop_length, fixed_res,
                    cache_dir=self.cache_dir, fixed_bond=fixed_bond,
                    target_expand_bias = target_expand_bias
                )

                data = {k: getattr(target, k) for k in target.__dict__ if 'full' in k or "mask" in k}
                data['pdb_id'] = chosen_pdb_id
                data['contig'] = contig
                return data
            except Exception as e:
                print(f"Error processing {chosen_pdb_id}: {e}")
                # This member failed for this mode, try the next one.
                continue

        # All members in this cluster failed, fall back to a random new cluster
        new_idx = random.randint(0, len(self.clusters_list) - 1)
        # Keep the same target_mode, but no explicit member index (let it re-sample inside)
        return self.__getitem__((new_idx, target_mode))

class ClusterRatioSampler(Sampler):
    def __init__(self, dataset, ratios, num_replicas=None, rank=None, shuffle=True,
                 distrubuted=False, samples_num=None, seed=0,
                 link_span_alpha=0.5, link_span_workers=1):
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
        print("self.num_replicas", self.num_replicas)
        print("self.rank", self.rank)
        print("self.total_size", self.total_size)

        # LINK-span-based member reweighting: if link_span_alpha is not None,
        # build a weighted sampling table over (cluster, member, mode)
        self.link_span_alpha = link_span_alpha
        self.link_span_workers = max(1, int(link_span_workers)) if link_span_workers is not None else 1
        self.use_member_sampling = link_span_alpha is not None
        self.sample_table = None  # list of (cluster_idx, member_idx, mode)
        self.sample_weights = None  # list of float

        if self.use_member_sampling:
            self._build_member_sampling_table()

    def _build_member_sampling_table(self):
        """
        Pre-compute LINK-span-based weights for each member structure, then
        build a global sampling table over (cluster_idx, member_local_idx, mode).

        权重逻辑：
          1. 对所有结构的所有 LINK span 统计全局分布 p_s。
          2. 对每个 span 做幂次重加权 r_s = p_s^(alpha-1)，alpha in (0,1)，默认 0.5（开根号）。
          3. 对每个结构，取其所有 LINK span 的 r_s 平均作为该结构的权重 w_member。
          4. 对每个 (cluster, member, mode) 条目，赋权 weight = ratio[mode] * w_member。
        """
        alpha = self.link_span_alpha
        if alpha is None:
            self.use_member_sampling = False
            return

        clusters_list = self.dataset.clusters_list
        pdb_dir = self.dataset.pdb_dir
        cache_dir = getattr(self.dataset, "cache_dir", None)

        # If we have a cache directory, ensure a (possibly unified) cache exists.
        # For standard training, get_dataloader() should already have called
        # _ensure_link_span_weight_cache over all clusters (train+val), so here
        # we just load. If someone constructs the sampler manually elsewhere,
        # we fall back to computing weights only over this dataset's clusters.
        weight_cache_path = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            weight_cache_path = os.path.join(
                cache_dir, f"link_span_member_weights_alpha{alpha:.2f}.pkl"
            )
            if not os.path.exists(weight_cache_path):
                # No global cache yet (e.g., manual construction), compute over this dataset only.
                _ensure_link_span_weight_cache(
                    all_clusters=clusters_list,
                    pdb_dir=pdb_dir,
                    cache_dir=cache_dir,
                    alpha=alpha,
                    link_span_workers=self.link_span_workers,
                )

        member_weight = None
        if weight_cache_path is not None and os.path.exists(weight_cache_path):
            try:
                with open(weight_cache_path, "rb") as f:
                    member_weight = pickle.load(f)
            except Exception as e:
                print(
                    f"ClusterRatioSampler: Failed to load LINK-span weight cache {weight_cache_path}: {e}"
                )

        if not isinstance(member_weight, dict) or not member_weight:
            print(
                "ClusterRatioSampler: No valid LINK-span member weights found; "
                "falling back to unweighted cluster sampling."
            )
            self.use_member_sampling = False
            return

        # Build per-dataset sampling table from the shared member weights.
        self._build_sample_table_from_member_weight(clusters_list, member_weight)

    def _build_sample_table_from_member_weight(self, clusters_list, member_weight):
        """
        Given per-member weights, build the (cluster, member, mode) sampling table.

        采样策略（满足你的要求）：
          - cluster 间：保持均匀（在每个 mode 下，每个 cluster 的总权重相同）；
          - cluster 内：按 LINK-based member_weight 做归一化后采样。
        """
        sample_table = []
        sample_weights = []
        for mode, ratio in self.ratios.items():
            if ratio <= 0:
                continue
            for c_idx, cluster in enumerate(clusters_list):
                # 先收集该 cluster 内所有有正权重的 member
                member_indices = []
                member_weights_in_cluster = []
                for m_idx, member in enumerate(cluster):
                    w_m = member_weight.get(member, 0.0)
                    if w_m > 0:
                        member_indices.append(m_idx)
                        member_weights_in_cluster.append(w_m)

                if not member_indices:
                    # 该 cluster 在当前 mode 下没有可用的带权 member，跳过
                    continue

                # cluster 内归一化，使得 sum_m w_norm = 1
                cluster_sum = float(sum(member_weights_in_cluster))
                if cluster_sum <= 0:
                    continue

                for m_idx, w_m in zip(member_indices, member_weights_in_cluster):
                    w_norm = w_m / cluster_sum  # 仅决定 cluster 内分配
                    # cluster 间保持均匀，每个 cluster 的总权重 ~ ratio
                    w_entry = ratio * w_norm
                    sample_table.append((c_idx, m_idx, mode))
                    sample_weights.append(w_entry)

        if not sample_table:
            print(
                "ClusterRatioSampler: Weighted sampling table is empty; falling back to unweighted cluster sampling."
            )
            self.use_member_sampling = False
            return

        self.sample_table = sample_table
        self.sample_weights = sample_weights

    def __iter__(self):
        # If LINK-span-based member sampling is enabled and successfully built, use it.
        if self.use_member_sampling and self.sample_table and self.sample_weights:
            # Deterministic RNG per epoch
            rng = random.Random(self.seed + self.epoch)
            total = self.total_size
            print("self.sample_weights",len(self.sample_weights ),self.sample_weights[0:10])
            # random.Random.choices 支持带权重采样（有放回），效率足够
            selected_indices = rng.choices(
                range(len(self.sample_table)),
                weights=self.sample_weights,
                k=total,
            )
            indices = [self.sample_table[i] for i in selected_indices]

            if self.distrubuted:
                indices = indices[self.rank:self.total_size:self.num_replicas]
            return iter(indices)

        # Fallback: 原始的基于 cluster 的均匀采样逻辑（只返回 (cluster_idx, mode)）
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # This list will hold tuples of (cluster_idx, target_mode)
        indices = []
        for mode, ratio in self.ratios.items():
            if ratio <= 0:
                continue

            target_count = int(math.ceil(self.total_size * ratio))

            # Sample cluster indices for this mode
            sampled_cluster_indices = torch.randint(
                high=self.num_clusters, size=(target_count,), generator=g
            ).tolist()

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


def _ensure_link_span_weight_cache(all_clusters, pdb_dir, cache_dir,
                                   alpha, link_span_workers):
    """
    Ensure a unified LINK-span member-weight cache exists, computed over
    *all* clusters (train + val), so that train/val share the same p_s and
    member weights.
    """
    if cache_dir is None or alpha is None:
        return

    os.makedirs(cache_dir, exist_ok=True)
    weight_cache_path = os.path.join(
        cache_dir, f"link_span_member_weights_alpha{alpha:.2f}.pkl"
    )
    if os.path.exists(weight_cache_path):
        # Cache already exists, nothing to do.
        return

    # Collect all unique members from all clusters
    unique_members = set()
    for cluster in all_clusters:
        for member in cluster:
            unique_members.add(member)

    member_to_spans = {}
    span_counts = Counter()
    args_list = [(m, pdb_dir, cache_dir) for m in unique_members]
    workers = max(1, int(link_span_workers)) if link_span_workers is not None else 1

    if workers > 1:
        with mp.Pool(workers) as pool:
            for member, spans in tqdm(
                pool.imap_unordered(_compute_member_spans, args_list),
                total=len(args_list),
                desc="Precomputing unified LINK-span weights (mp)",
                leave=False,
            ):
                if not spans:
                    continue
                member_to_spans[member] = spans
                for s in spans:
                    span_counts[s] += 1
    else:
        for args in tqdm(
            args_list,
            desc="Precomputing unified LINK-span weights",
            leave=False,
        ):
            member, spans = _compute_member_spans(args)
            if not spans:
                continue
            member_to_spans[member] = spans
            for s in spans:
                span_counts[s] += 1

    if not span_counts:
        print(
            "ClusterRatioSampler: No LINK spans found while precomputing cache; "
            "unified LINK-span weighting will be disabled."
        )
        return

    total_links = sum(span_counts.values())
    span_to_p = {s: c / total_links for s, c in span_counts.items() if c > 0}
    span_to_r = {s: (p ** (alpha - 1.0)) for s, p in span_to_p.items()}

    member_weight = {}
    for member, spans in member_to_spans.items():
        ws = [span_to_r[s] for s in spans if s in span_to_r]
        if not ws:
            continue
        
        member_weight[member] = sum(ws) / len(ws)

    if not member_weight:
        print(
            "ClusterRatioSampler: No member weights computed while precomputing cache; "
            "unified LINK-span weighting will be disabled."
        )
        return

    try:
        with open(weight_cache_path, "wb") as f:
            pickle.dump(member_weight, f)
        print(
            f"ClusterRatioSampler: Precomputed unified LINK-span member weights "
            f"and saved to cache: {weight_cache_path}"
        )
    except Exception as e:
        print(
            f"ClusterRatioSampler: Failed to save unified LINK-span weight cache "
            f"{weight_cache_path}: {e}"
        )

def get_dataloader(conf, batch_size, pdb_list_path, pdb_dir,
                   sampling_ratios,
                   distributed=False, num_workers=4,
                   crop_length=256, device='cpu', rank=None, num_replicas=None,
                   val_split=0.1, seed=0, cache_dir=None,
                   mask_bond_prob=0.5, mask_seq_str_prob=0.5,
                   link_span_alpha=None, link_span_workers=1):
    
    # Parse clusters from file
    clusters_dict = defaultdict(list)
    with open(pdb_list_path, 'r') as f:
        for line in f:
            center, member = line.strip().split()
            clusters_dict[center].append(member)
    
    all_clusters = list(clusters_dict.values())
    if not all_clusters:
        raise ValueError("Parsing cluster file failed: No clusters found.")

    # Precompute a unified LINK-span weight cache over all clusters so that
    # train/val share the same member weights and LINK-span distribution.
    _ensure_link_span_weight_cache(
        all_clusters=all_clusters,
        pdb_dir=pdb_dir,
        cache_dir=cache_dir,
        alpha=link_span_alpha,
        link_span_workers=link_span_workers,
    )

    random.Random(seed).shuffle(all_clusters)
    split_idx = int(len(all_clusters) * (1 - val_split))
    train_clusters = all_clusters[:split_idx]
    val_clusters = all_clusters[split_idx:]

    # Create training dataset and sampler
    train_dataset = PDB_dataset(conf, pdb_dir, crop_length, train_clusters, cache_dir=cache_dir, 
                                mask_bond_prob=mask_bond_prob,mask_seq_str_prob=mask_seq_str_prob)
    print(f"Training set: {len(train_clusters)} clusters.")
    

    train_sampler = ClusterRatioSampler(train_dataset, ratios=sampling_ratios,
                                        seed=seed, num_replicas=num_replicas,
                                        rank=rank, shuffle=True, distrubuted=distributed,
                                        link_span_alpha=link_span_alpha,
                                        link_span_workers=link_span_workers)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if 'cuda' in device  else False,
        drop_last=True, 
    )

    # Validation dataloader
    val_dataloader = None
    if val_clusters:
        val_dataset = PDB_dataset(conf, pdb_dir, crop_length, val_clusters, cache_dir=cache_dir,
                                mask_bond_prob = mask_bond_prob,mask_seq_str_prob=mask_seq_str_prob)
        print(f"Validation set: {len(val_clusters)} clusters.")

        # For validation, create a list of all possible (cluster, mode) combinations to iterate through.
        
        val_sampler = ClusterRatioSampler(val_dataset, ratios=sampling_ratios,
                                          seed=seed, num_replicas=num_replicas,
                                          rank=rank, shuffle=True, distrubuted=distributed,
                                          link_span_alpha=link_span_alpha,
                                          link_span_workers=link_span_workers)

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=(val_sampler is None),
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if 'cuda' in device else False,
            drop_last=True,  
        )
    
    return train_dataloader, val_dataloader