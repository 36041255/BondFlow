"""
Knottin采样脚本 - 支持每个batch使用不同的拓扑配置

该脚本实现了CSαβ拓扑的knottin设计，每个batch可以有不同的：
- 序列长度（25-35）
- 二硫键连接模式
- 螺旋位置和β折叠位置

拓扑规则：
1. 交叉锚定：1-4（或1-8）锁定N/C端，2-5和3-6是核心
2. 螺旋稳定性：i, i+3或i, i+4约束
3. 穿心约束：3-6穿过1-4和2-5形成的空腔
"""

import os
import argparse
import torch
import torch.multiprocessing as mp
import random
import numpy as np
from omegaconf import OmegaConf, DictConfig
from BondFlow.models.sampler import Sampler
import BondFlow.data.utils as iu
from copy import deepcopy


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


def generate_knottin_topology(length, seed=None):
    """
    生成一个符合CSαβ拓扑的knottin配置
    
    Args:
        length: 序列长度 (25-35)
        seed: 随机种子，用于生成不同的拓扑变体
    
    Returns:
        dict: 包含contigs和pairs的配置
        {
            'length': int,
            'contigs': list,
            'pairs': list of [i, j] pairs,
            'cys_positions': list of CYS位置（用于logits_bias排除）
        }
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 确保长度在合理范围内
    length = max(25, min(35, length))
    
    # CSαβ拓扑的典型布局：
    # - N端区域（可能包含螺旋起始）
    # - 螺旋区域（通常6-10个残基）
    # - 中间连接区域
    # - β折叠区域（通常6-10个残基）
    # - C端区域
    
    # 定义区域边界（基于长度自适应）
    n_term_len = 2  # N端固定区域
    c_term_len = 2  # C端固定区域
    min_helix_len = 6
    min_sheet_len = 6
    min_middle_len = 3
    
    # 计算可用长度
    available_len = length - n_term_len - c_term_len - min_middle_len
    
    # 分配螺旋和β折叠长度
    if available_len >= min_helix_len + min_sheet_len:
        helix_len = random.choice([6, 7, 8, 9])
        sheet_len = random.choice([6, 7, 8])
        remaining = available_len - helix_len - sheet_len
        middle_len = max(min_middle_len, remaining)
    else:
        # 如果长度太短，使用最小值
        helix_len = min(available_len // 2, min_helix_len)
        sheet_len = min(available_len - helix_len, min_sheet_len)
        middle_len = min_middle_len
    
    # 最终调整以确保总长度正确
    total = n_term_len + helix_len + middle_len + sheet_len + c_term_len
    if total < length:
        middle_len += (length - total)
    elif total > length:
        middle_len = max(0, middle_len - (total - length))
    
    # 计算各区域的起始位置
    helix_start = n_term_len
    helix_end = helix_start + helix_len
    middle_start = helix_end
    middle_end = middle_start + middle_len
    sheet_start = middle_end
    sheet_end = min(sheet_start + sheet_len, length - c_term_len)
    c_term_start = sheet_end
    
    # 生成二硫键对（遵循CSαβ拓扑）
    # CYS在序列中的顺序：C1 < C2 < C3 < C4 < C5 < C6
    # Cys 1: N端区域（位置0附近）
    # Cys 2, 3: 螺旋区域，遵循i, i+3或i, i+4
    # Cys 4: 中间区域（在β折叠之前，用于与C1形成大环）
    # Cys 5, 6: β折叠区域
    # 二硫键连接：C1-C4, C2-C5, C3-C6
    
    # 选择拓扑变体
    topology_variant = random.choice(['variant1', 'variant2', 'variant3'])
    
    if topology_variant == 'variant1':
        # 标准CSαβ: 1-4, 2-5, 3-6
        # 序列顺序：C1 < C2 < C3 < C4 < C5 < C6
        cys1 = random.randint(0, min(2, length-1))  # N端
        cys2 = helix_start + random.randint(0, max(0, helix_len-4))  # 螺旋起始
        cys3 = cys2 + random.choice([3, 4])  # 螺旋，i+3或i+4
        # C4在中间区域，在β折叠之前，确保 C3 < C4 < C5
        cys4_min = max(cys3 + 1, middle_start)
        cys4_max = min(sheet_start - 1, middle_end - 1)
        if cys4_max < cys4_min:
            cys4_max = min(cys3 + 3, length - 1)
            cys4_min = cys3 + 1
        cys4 = random.randint(cys4_min, cys4_max)  # 中间区域
        cys5 = sheet_start + random.randint(0, max(0, sheet_len-3))  # β折叠起始
        # 确保 C4 < C5
        if cys5 <= cys4:
            cys5 = cys4 + 1
        # 确保cys6在β折叠区域内，且 C5 < C6
        cys6 = cys5 + random.randint(2, min(4, sheet_len - (cys5 - sheet_start) - 1))  # β折叠
        
        pairs = [
            [cys1, cys4],  # 1-4: N端-中间区域（形成大环）
            [cys2, cys5],  # 2-5: 螺旋-β折叠
            [cys3, cys6],  # 3-6: 螺旋-β折叠（穿心）
        ]
        
    elif topology_variant == 'variant2':
        # 变体：1-4距离更长，2-5, 3-6
        # 序列顺序：C1 < C2 < C3 < C4 < C5 < C6
        cys1 = random.randint(0, min(3, length-1))
        cys2 = helix_start + random.randint(0, max(0, helix_len-4))
        cys3 = cys2 + random.choice([3, 4])
        # C4在中间区域，但可以更靠后（形成更长的C1-C4距离），确保 C3 < C4 < C5
        cys4_min = max(cys3 + 1, middle_start)
        cys4_max = min(sheet_start - 1, middle_end - 1)
        if cys4_max < cys4_min:
            cys4_max = min(cys3 + 5, length - 1)
            cys4_min = cys3 + 1
        cys4 = random.randint(cys4_min, cys4_max)  # 中间区域，可能更靠后
        cys5 = sheet_start + random.randint(0, max(0, sheet_len-3))
        # 确保 C4 < C5
        if cys5 <= cys4:
            cys5 = cys4 + 1
        cys6 = cys5 + random.randint(2, min(4, sheet_len - (cys5 - sheet_start) - 1))
        
        pairs = [
            [cys1, cys4],  # 1-4: N端-中间区域（更长距离的大环）
            [cys2, cys5],  # 2-5: 螺旋-β折叠
            [cys3, cys6],  # 3-6: 螺旋-β折叠（穿心）
        ]
        
    else:  # variant3
        # 变体：调整螺旋和β折叠的相对位置
        # 序列顺序：C1 < C2 < C3 < C4 < C5 < C6
        cys1 = random.randint(0, min(2, length-1))
        cys2 = helix_start + random.randint(1, max(1, helix_len-4))
        cys3 = cys2 + random.choice([3, 4])
        # C4在中间区域，在β折叠之前，确保 C3 < C4 < C5
        cys4_min = max(cys3 + 1, middle_start)
        cys4_max = min(sheet_start - 1, middle_end - 1)
        if cys4_max < cys4_min:
            cys4_max = min(cys3 + 4, length - 1)
            cys4_min = cys3 + 1
        cys4 = random.randint(cys4_min, cys4_max)  # 中间区域
        cys5 = sheet_start + random.randint(1, max(1, sheet_len-3))
        # 确保 C4 < C5
        if cys5 <= cys4:
            cys5 = cys4 + 1
        # 确保cys6在β折叠区域内，且 C5 < C6
        cys6_candidates = []
        for offset in range(2, min(5, sheet_len - (cys5 - sheet_start))):
            candidate = cys5 + offset
            if candidate < sheet_end:  # 确保在β折叠区域内
                cys6_candidates.append(candidate)
        if cys6_candidates:
            cys6 = random.choice(cys6_candidates)
        else:
            # 如果无法避免，使用最小偏移
            cys6 = min(cys5 + 2, sheet_end - 1)
        
        pairs = [
            [cys1, cys4],  # 1-4: N端-中间区域
            [cys2, cys5],  # 2-5: 螺旋-β折叠
            [cys3, cys6],  # 3-6: 螺旋-β折叠（穿心）
        ]
    
    # 确保所有位置都在有效范围内并修正
    for i, pair in enumerate(pairs):
        pair[0] = max(0, min(length-1, int(pair[0])))
        pair[1] = max(0, min(length-1, int(pair[1])))
        # 确保不是自环
        if pair[0] == pair[1]:
            if pair[1] < length - 1:
                pair[1] = pair[1] + 1
            elif pair[0] > 0:
                pair[0] = pair[0] - 1
    
    # 检查并修正重复的CYS位置
    # 收集所有CYS位置
    all_cys_positions = []
    for pair in pairs:
        all_cys_positions.extend([pair[0], pair[1]])
    
    # 检查是否有重复
    unique_positions = set(all_cys_positions)
    
    # 如果有重复，修正它们
    if len(all_cys_positions) != len(unique_positions):
        # 找出所有已使用的位置
        used_positions = set()
        # 按顺序处理每个pair，确保位置唯一
        for pair in pairs:
            # 如果pair[0]已被使用，尝试调整
            if pair[0] in used_positions:
                # 尝试找到最近未使用的位置
                for offset in range(1, length):
                    candidate1 = pair[0] + offset
                    candidate2 = pair[0] - offset
                    if candidate1 < length and candidate1 not in used_positions:
                        pair[0] = candidate1
                        break
                    elif candidate2 >= 0 and candidate2 not in used_positions:
                        pair[0] = candidate2
                        break
            # 如果pair[1]已被使用，尝试调整
            if pair[1] in used_positions or pair[1] == pair[0]:
                for offset in range(1, length):
                    candidate1 = pair[1] + offset
                    candidate2 = pair[1] - offset
                    if candidate1 < length and candidate1 not in used_positions and candidate1 != pair[0]:
                        pair[1] = candidate1
                        break
                    elif candidate2 >= 0 and candidate2 not in used_positions and candidate2 != pair[0]:
                        pair[1] = candidate2
                        break
            
            # 确保pair不是自环
            if pair[0] == pair[1]:
                if pair[1] < length - 1:
                    pair[1] = pair[1] + 1
                elif pair[0] > 0:
                    pair[0] = pair[0] - 1
            
            # 记录已使用的位置
            used_positions.add(pair[0])
            used_positions.add(pair[1])
    
    # 最终验证：确保所有CYS位置唯一，且顺序正确 C1 < C2 < C3 < C4 < C5 < C6
    final_cys_positions = []
    for pair in pairs:
        final_cys_positions.extend([pair[0], pair[1]])
    
    if len(final_cys_positions) != len(set(final_cys_positions)) or len(set(final_cys_positions)) != 6:
        # 如果还有重复或不是6个唯一位置，重新生成
        if seed is not None:
            return generate_knottin_topology(length, seed=seed + 10000)
        else:
            return generate_knottin_topology(length, seed=None)
    
    # 提取CYS位置并验证顺序
    cys1_pos = pairs[0][0]  # C1
    cys2_pos = pairs[1][0]  # C2
    cys3_pos = pairs[2][0]  # C3
    cys4_pos = pairs[0][1]  # C4
    cys5_pos = pairs[1][1]  # C5
    cys6_pos = pairs[2][1]  # C6
    
    # 验证顺序：C1 < C2 < C3 < C4 < C5 < C6
    if not (cys1_pos < cys2_pos < cys3_pos < cys4_pos < cys5_pos < cys6_pos):
        # 如果顺序不对，重新生成
        if seed is not None:
            return generate_knottin_topology(length, seed=seed + 20000)
        else:
            return generate_knottin_topology(length, seed=None)
    
    # 去重并排序pairs，确保每个pair都是有效的
    valid_pairs = []
    seen = set()
    for p in pairs:
        p_sorted = tuple(sorted([int(p[0]), int(p[1])]))
        if p_sorted[0] != p_sorted[1] and p_sorted not in seen:
            valid_pairs.append(list(p_sorted))
            seen.add(p_sorted)
    pairs_0based = sorted(valid_pairs)
    
    # 转换为1-based（用于显示和配置）
    pairs_1based = [[p[0] + 1, p[1] + 1] for p in pairs_0based]
    
    # 重新计算CYS位置（1-based）
    cys_positions_1based = sorted(set([p[0] for p in pairs_1based] + [p[1] for p in pairs_1based]))
    
    # 生成contigs（使用New_来设置长度）
    # 假设使用prior模式，只需要设置New_长度
    contigs = [[f"New_{length}-{length}"]]
    
    return {
        'length': length,
        'contigs': contigs,
        'pairs': pairs_1based,  # 1-based
        'pairs_0based': pairs_0based,  # 保留0-based用于内部计算
        'cys_positions': cys_positions_1based,  # 1-based
        'topology_variant': topology_variant,
    }


def update_config_for_batch(cfg, topology_config):
    """
    为当前batch更新配置
    
    Args:
        cfg: OmegaConf配置对象
        topology_config: generate_knottin_topology返回的配置
    
    Returns:
        更新后的配置对象（深拷贝）
    """
    # 深拷贝配置以避免修改原始配置
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_new = OmegaConf.create(cfg_dict)
    
    # 更新contigs
    # 如果原始配置中有contigs，需要保留格式（可能包含多个chain）
    original_contigs = getattr(cfg.design_config, 'contigs', None)
    if original_contigs is not None and len(original_contigs) > 0:
        # 保留原始contigs结构，只更新第一个chain中的New_长度
        new_contigs = []
        for chain_idx, chain in enumerate(original_contigs):
            if chain_idx == 0:
                # 第一个chain：更新New_长度
                new_chain = []
                for item in chain:
                    if isinstance(item, str) and item.startswith('New_'):
                        # 替换New_的长度
                        new_chain.append(f"New_{topology_config['length']}-{topology_config['length']}")
                    else:
                        # 保留其他内容
                        new_chain.append(item)
                new_contigs.append(new_chain)
            else:
                # 其他chain：完全保留
                new_contigs.append(list(chain) if isinstance(chain, (list, tuple)) else chain)
        cfg_new.design_config.contigs = new_contigs
    else:
        # 如果没有原始contigs，使用生成的contigs
        cfg_new.design_config.contigs = topology_config['contigs']
    
    # 更新length - 设置为null，让系统自动从contigs推断
    cfg_new.design_config.length = None
    
    # 更新guidance中的pairs
    # 配置文件中的pairs是1-based的，所以直接使用1-based的pairs
    if hasattr(cfg_new, 'guidance') and hasattr(cfg_new.guidance, 'list'):
        for guidance_item in cfg_new.guidance.list:
            if hasattr(guidance_item, 'name') and guidance_item.name == 'type_soft_bond_count':
                # 找到disulfide类型
                if hasattr(guidance_item, 'types'):
                    for type_item in guidance_item.types:
                        if hasattr(type_item, 'name') and type_item.name == 'disulfide':
                            # 直接使用1-based的pairs（与配置文件格式一致）
                            type_item.pairs = [[int(p[0]), int(p[1])] for p in topology_config['pairs']]
                            break
    
    # 更新logits_bias中的positions
    # positions_mode: exclude 表示排除positions中列出的位置，其他位置都生效
    # 所以positions应该直接设置为CYS位置（要排除的位置）
    if hasattr(cfg_new, 'guidance') and hasattr(cfg_new.guidance, 'list'):
        for guidance_item in cfg_new.guidance.list:
            if hasattr(guidance_item, 'name') and guidance_item.name == 'logits_bias':
                # positions_mode是exclude，所以positions应该直接设置为CYS位置
                # 这样除了这些CYS位置，其他位置都会应用bias（压低CYS生成）
                guidance_item.positions = topology_config['cys_positions']  # 1-based，直接使用CYS位置
    
    return cfg_new


def run_sampling_worker_knottin(device_str, base_cfg_path, num_designs, num_cycle, num_timesteps,
                                write_trajectory, out_dir, use_partial_diffusion, batch_range,
                                length_range=(25, 35), topology_seed_base=42):
    """
    单个设备上的knottin采样工作函数
    
    Args:
        device_str: 设备字符串
        base_cfg_path: 基础配置文件路径
        num_designs: 每个批次的样本数
        num_cycle: 总批次数
        num_timesteps: 时间步数
        write_trajectory: 是否写入轨迹
        out_dir: 输出目录
        use_partial_diffusion: 是否使用部分扩散
        batch_range: (start_batch, end_batch)
        length_range: (min_length, max_length) 长度范围
        topology_seed_base: 拓扑生成的种子基数
    """
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
    
    print(f"[{device_label}] Starting knottin sampling on device {device}")
    print(f"[{device_label}] Processing batches {batch_range[0]} to {batch_range[1]-1}")
    print(f"[{device_label}] Length range: {length_range[0]}-{length_range[1]}")
    
    # 加载基础配置
    base_cfg = OmegaConf.load(base_cfg_path)
    
    # 解析相对路径：将相对于配置文件所在目录的路径转换为绝对路径
    config_dir = os.path.dirname(os.path.abspath(base_cfg_path))
    resolve_config_paths(base_cfg, config_dir)
    
    # 从配置中获取 out_dir（已在 resolve_config_paths 中计算）
    out_dir = getattr(base_cfg.inference, 'out_dir', None)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    start_batch, end_batch = batch_range
    num_batches_this_gpu = end_batch - start_batch
    
    print(f"[{device_label}] Processing {num_batches_this_gpu} batches")
    
    for local_batch_i in range(num_batches_this_gpu):
        global_batch_i = start_batch + local_batch_i
        start_index = global_batch_i * num_designs
        
        # 为当前batch生成拓扑配置
        # 使用global_batch_i作为种子的一部分，确保每个batch不同
        topology_seed = topology_seed_base + global_batch_i * 1000
        length = random.randint(length_range[0], length_range[1])
        
        topology_config = generate_knottin_topology(length, seed=topology_seed)
        
        print(f"\n[{device_label}] Batch {global_batch_i}/{num_cycle-1}")
        print(f"  Length: {topology_config['length']}")
        print(f"  Topology variant: {topology_config['topology_variant']}")
        print(f"  Disulfide pairs (CSαβ topology):")
        if len(topology_config['pairs']) >= 3:
            print(f"    C1-C4 (N-C lock): {topology_config['pairs'][0]}")
            print(f"    C2-C5 (helix-sheet): {topology_config['pairs'][1]}")
            print(f"    C3-C6 (threading): {topology_config['pairs'][2]}")
        else:
            for i, pair in enumerate(topology_config['pairs']):
                print(f"    Pair {i+1}: {pair}")
        print(f"  CYS positions: {topology_config['cys_positions']}")
        print(f"    -> C1: {topology_config['pairs'][0][0]}, C2: {topology_config['pairs'][1][0]}, C3: {topology_config['pairs'][2][0]}")
        print(f"    -> C4: {topology_config['pairs'][0][1]}, C5: {topology_config['pairs'][1][1]}, C6: {topology_config['pairs'][2][1]}")
        print(f"  Start index: {start_index}")
        
        # 更新配置
        cfg = update_config_for_batch(base_cfg, topology_config)
        
        # 重新初始化sampler（因为配置改变了）
        sampler = Sampler(cfg, device=device)
        
        # 检查是否使用partial diffusion
        use_partial = getattr(cfg.design_config, 'use_partial_diffusion', False) and use_partial_diffusion
        
        if use_partial:
            # Partial diffusion模式
            partial_t = cfg.design_config.partial_t
            
            pdb_parsed = iu.process_target(
                cfg.design_config.input_pdb,
                parse_hetatom=False,
                center=False,
                parse_link=True,
                link_csv_path=getattr(cfg.preprocess, 'link_config', None),
            )
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
            
            if 'pdb_id' in pdb_parsed:
                pdb_core_id = [pdb_parsed['pdb_id']] * num_designs
            else:
                pdb_basename = os.path.splitext(os.path.basename(cfg.design_config.input_pdb))[0]
                pdb_core_id = [pdb_basename] * num_designs
            
            assert getattr(target, "pdb_seq_full", None) is not None
            assert getattr(target, "pdb_idx_full", None) is not None
            assert getattr(target, "full_origin_pdb_idx", None) is not None
            
            pdb_seq_full = [getattr(target, "pdb_seq_full", None)] * num_designs
            pdb_idx_full = [getattr(target, "pdb_idx_full", None)] * num_designs
            origin_pdb_idx = [target.full_origin_pdb_idx] * num_designs
            
            xyz_target = target.full_xyz[None, :, :3, :].to(device)
            seq_target = target.full_seq[None, :].to(device)
            ss_target = target.full_bond_matrix[None, :, :].to(device)
            
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
                assert cfg.design_config.length is not None
                
                sampler.sample_from_prior(
                    num_batch=num_designs,
                    num_res=cfg.design_config.length,
                    num_timesteps=num_timesteps,
                    record_trajectory=False,
                    out_pdb_dir=out_dir,
                    start_index=int(start_index),
                )
        
        # 清理sampler以释放内存（可选，如果内存充足可以保留）
        del sampler
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    print(f"[{device_label}] Finished all batches")


def preview_topologies(cfg_path, num_cycle, length_range=(25, 35), topology_seed_base=42, max_preview=10):
    """
    预览模式：只生成和打印拓扑配置，不实际运行采样
    
    Args:
        cfg_path: 基础配置文件路径
        num_cycle: 总批次数
        length_range: (min_length, max_length) 长度范围
        topology_seed_base: 拓扑生成的种子基数
        max_preview: 最多预览的batch数量（如果为None则预览所有）
    """
    # 加载基础配置
    base_cfg = OmegaConf.load(cfg_path)
    
    print("="*80)
    print("KNOTTIN TOPOLOGY PREVIEW MODE")
    print("="*80)
    print(f"Base config: {cfg_path}")
    print(f"Total batches: {num_cycle}")
    print(f"Length range: {length_range[0]}-{length_range[1]}")
    print(f"Topology seed base: {topology_seed_base}")
    print("="*80)
    print()
    
    preview_count = min(num_cycle, max_preview) if max_preview else num_cycle
    
    for batch_i in range(preview_count):
        topology_seed = topology_seed_base + batch_i * 1000
        length = random.randint(length_range[0], length_range[1])
        
        topology_config = generate_knottin_topology(length, seed=topology_seed)
        
        # 更新配置以显示实际使用的配置
        cfg = update_config_for_batch(base_cfg, topology_config)
        
        print(f"{'='*80}")
        print(f"Batch {batch_i}/{num_cycle-1} Configuration:")
        print(f"{'='*80}")
        print(f"  Length: {topology_config['length']}")
        print(f"  Topology variant: {topology_config['topology_variant']}")
        print(f"  Disulfide pairs (CSαβ topology):")
        if len(topology_config['pairs']) >= 3:
            print(f"    C1-C4 (N-C lock): {topology_config['pairs'][0]}")
            print(f"    C2-C5 (helix-sheet): {topology_config['pairs'][1]}")
            print(f"    C3-C6 (threading): {topology_config['pairs'][2]}")
        else:
            for i, pair in enumerate(topology_config['pairs']):
                print(f"    Pair {i+1}: {pair}")
        print(f"  CYS positions: {topology_config['cys_positions']}")
        print(f"    -> C1: {topology_config['pairs'][0][0]}, C2: {topology_config['pairs'][1][0]}, C3: {topology_config['pairs'][2][0]}")
        print(f"    -> C4: {topology_config['pairs'][0][1]}, C5: {topology_config['pairs'][1][1]}, C6: {topology_config['pairs'][2][1]}")
        
        # logits_bias的positions（positions_mode: exclude，所以positions是CYS位置）
        logits_bias_positions = topology_config['cys_positions']
        all_positions = list(range(1, topology_config['length'] + 1))
        bias_applied_positions = [p for p in all_positions if p not in set(logits_bias_positions)]
        print(f"  Logits_bias positions (exclude mode, CYS positions): {logits_bias_positions}")
        print(f"    -> Total positions: {len(all_positions)}, CYS excluded: {len(logits_bias_positions)}, Bias applied to: {len(bias_applied_positions)} positions")
        
        print(f"\n  Updated Config (key sections):")
        print(f"  design_config.contigs: {cfg.design_config.contigs}")
        print(f"  design_config.length: {cfg.design_config.length}")
        
        # 显示guidance配置
        if hasattr(cfg, 'guidance') and hasattr(cfg.guidance, 'list'):
            for guidance_item in cfg.guidance.list:
                if hasattr(guidance_item, 'name'):
                    if guidance_item.name == 'logits_bias':
                        print(f"\n  guidance.logits_bias:")
                        print(f"    positions: {guidance_item.positions}")
                        print(f"    positions_mode: {getattr(guidance_item, 'positions_mode', 'exclude')}")
                        print(f"    bias[CYS]: {getattr(guidance_item, 'bias', [0]*20)[4] if hasattr(guidance_item, 'bias') and len(getattr(guidance_item, 'bias', [])) > 4 else 'N/A'}")
                    elif guidance_item.name == 'type_soft_bond_count':
                        print(f"\n  guidance.type_soft_bond_count:")
                        if hasattr(guidance_item, 'types'):
                            for type_item in guidance_item.types:
                                if hasattr(type_item, 'name') and type_item.name == 'disulfide':
                                    print(f"    disulfide.pairs: {type_item.pairs}")
                                    print(f"    disulfide.mode: {getattr(type_item, 'mode', 'N/A')}")
                                    print(f"    disulfide.target_N: {getattr(type_item, 'target_N', 'N/A')}")
        
        print()
    
    if max_preview and num_cycle > max_preview:
        print(f"... (showing first {max_preview} batches, total {num_cycle} batches)")
        print()
    
    print("="*80)
    print("Preview complete. Use without --preview to run actual sampling.")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Sample knottin binders with varying topologies")
    parser.add_argument("--cfg", default="/home/fit/lulei/WORK/xjt/Protein_design/BondFlow/BondFlow/config/cyclize.yaml",
                       help="Base config yaml path")
    parser.add_argument("--device", default="auto",
                       help="Device selection: 'auto', 'cpu', 'cuda', 'cuda:0', or multiple GPUs like 'cuda:0,cuda:1'")
    parser.add_argument("--min_length", type=int, default=25, help="Minimum sequence length")
    parser.add_argument("--max_length", type=int, default=35, help="Maximum sequence length")
    parser.add_argument("--topology_seed", type=int, default=42, help="Base seed for topology generation")
    parser.add_argument("--preview", action="store_true", 
                       help="Preview mode: only print topologies without running sampling")
    parser.add_argument("--max_preview", type=int, default=10,
                       help="Maximum number of batches to preview (default: 10, use 0 for all)")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.cfg)
    
    # 解析相对路径：将相对于配置文件所在目录的路径转换为绝对路径
    config_dir = os.path.dirname(os.path.abspath(args.cfg))
    resolve_config_paths(cfg, config_dir)
    
    num_designs = int(getattr(cfg.inference, 'num_designs', 1)) if hasattr(cfg, 'inference') else 1
    num_cycle = cfg.inference.num_cycle
    num_timesteps = int(cfg.interpolant.sampling.num_timesteps)
    write_trajectory = cfg.inference.write_trajectory
    
    length_range = (args.min_length, args.max_length)
    
    # 预览模式：只打印拓扑配置，不运行采样
    if args.preview:
        max_preview = None if args.max_preview == 0 else args.max_preview
        preview_topologies(args.cfg, num_cycle, length_range, args.topology_seed, max_preview)
        return
    
    assert cfg.inference.output_prefix is not None, "inference.output_prefix must be set in YAML"
    
    # 从配置中获取 out_dir（已在 resolve_config_paths 中计算）
    out_dir = getattr(cfg.inference, 'out_dir', None)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    use_partial_diffusion = getattr(cfg.design_config, 'use_partial_diffusion', False)
    
    # 解析设备参数
    device_str = args.device.strip()
    
    if ',' in device_str:
        # Multi-GPU mode
        gpu_strs = [g.strip() for g in device_str.split(',')]
        gpu_ids = []
        for gpu_str in gpu_strs:
            if gpu_str.startswith('cuda:'):
                gpu_id = int(gpu_str.split(':')[1])
                gpu_ids.append(gpu_id)
            elif gpu_str == 'cuda':
                gpu_ids.append(0)
            else:
                raise ValueError(f"Invalid GPU specification: {gpu_str}")
        
        num_gpus = len(gpu_ids)
        available_gpus = torch.cuda.device_count()
        for gpu_id in gpu_ids:
            if gpu_id >= available_gpus:
                raise ValueError(f"GPU {gpu_id} not available. Only {available_gpus} GPUs detected.")
        
        print(f"Multi-GPU mode: Using {num_gpus} GPUs: {gpu_ids}")
        use_multiprocessing = True
        device = None
    elif device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Single device mode: Selected device: {device}")
        num_gpus = 1
        use_multiprocessing = False
        gpu_ids = None
    elif device_str.startswith("cuda:"):
        gpu_id = int(device_str.split(':')[1])
        available_gpus = torch.cuda.device_count()
        if gpu_id >= available_gpus:
            raise ValueError(f"GPU {gpu_id} not available. Only {available_gpus} GPUs detected.")
        device = torch.device(device_str)
        print(f"Single device mode: Selected device: {device}")
        num_gpus = 1
        use_multiprocessing = False
        gpu_ids = None
    else:
        device = torch.device(device_str)
        print(f"Single device mode: Selected device: {device}")
        num_gpus = 1
        use_multiprocessing = False
        gpu_ids = None
    
    if use_multiprocessing:
        # Multi-GPU parallel sampling
        batches_per_gpu = num_cycle // num_gpus
        remainder = num_cycle % num_gpus
        
        batch_ranges = []
        start = 0
        for i in range(num_gpus):
            end = start + batches_per_gpu + (1 if i < remainder else 0)
            batch_ranges.append((start, end))
            start = end
        
        print(f"\n{'='*60}")
        print(f"Multi-GPU Knottin Sampling Configuration:")
        print(f"  Total cycles: {num_cycle}")
        print(f"  Number of GPUs: {num_gpus}")
        print(f"  Length range: {length_range[0]}-{length_range[1]}")
        print(f"  Topology seed base: {args.topology_seed}")
        print(f"{'='*60}\n")
        
        mp.set_start_method('spawn', force=True)
        
        processes = []
        for i, gpu_id in enumerate(gpu_ids):
            device_str = f"cuda:{gpu_id}"
            p = mp.Process(
                target=run_sampling_worker_knottin,
                args=(
                    device_str, args.cfg, num_designs, num_cycle, num_timesteps,
                    write_trajectory, out_dir, use_partial_diffusion, batch_ranges[i],
                    length_range, args.topology_seed
                )
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        print("All GPU processes completed")
    else:
        # Single GPU/CPU mode
        device_str = str(device)
        batch_range = (0, num_cycle)
        run_sampling_worker_knottin(
            device_str, args.cfg, num_designs, num_cycle, num_timesteps,
            write_trajectory, out_dir, use_partial_diffusion, batch_range,
            length_range, args.topology_seed
        )


if __name__ == "__main__":
    main()

