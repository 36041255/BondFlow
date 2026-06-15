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
import re
from omegaconf import OmegaConf, DictConfig
from BondFlow.models.sampler import Sampler
import BondFlow.data.utils as iu
from copy import deepcopy


def parse_residue_range(range_str):
    """
    解析残基范围字符串，支持格式如 'B/79-B/82' 或 'B/79-82'
    
    Args:
        range_str: 残基范围字符串
    
    Returns:
        dict: 包含start_chain, start_res, end_chain, end_res
    """
    pattern = re.compile(
        r"^(?P<start_chain>[^/]+)/(?P<start_res>-?\d+[A-Za-z]*)"
        r"-"
        r"(?:(?P<end_chain>[^/]+)/)?"
        r"(?P<end_res>-?\d+[A-Za-z]*)$"
    )
    match = pattern.match(range_str.strip())
    
    if not match:
        raise ValueError(f"Invalid range format: {range_str}. Expected 'chain/res-res' or 'chain/res-chain/res'.")
    
    parts = match.groupdict()
    start_chain = parts['start_chain']
    end_chain = parts['end_chain'] if parts['end_chain'] else start_chain
    
    return {
        'start_chain': start_chain,
        'start_res': parts['start_res'],
        'end_chain': end_chain,
        'end_res': parts['end_res']
    }


def parse_motif_from_contig(contig_item):
    """
    从contig项中解析motif信息
    
    Args:
        contig_item: contig项，例如 'ETGE:seq_FIX:str_DNV' 或 'B/79-B/82:seq_FIX:str_FIX'
    
    Returns:
        dict: motif配置，如果无法解析则返回None
        {
            'type': 'sequence' 或 'pdb_fragment',
            'value': 'ETGE' 或 'B/79-B/82',
            'seq_fix_type': 'seq_FIX',
            'str_fix_type': 'str_DNV',
            'contig_str': 原始contig字符串
        }
    """
    if not isinstance(contig_item, str):
        return None
    
    # 检查是否包含 :seq_ 或 :str_，这是motif的标志
    if ':seq_' not in contig_item and ':str_' not in contig_item:
        return None
    
    # 解析格式：value:seq_FIX:str_DNV 或 value:seq_FIX:str_FIX
    parts = contig_item.split(':')
    if len(parts) < 3:
        return None
    
    value = parts[0]
    seq_fix_type = parts[1] if len(parts) > 1 else 'seq_FIX'
    str_fix_type = parts[2] if len(parts) > 2 else 'str_DNV'
    
    # 判断是序列motif还是PDB片段motif
    # 序列motif: 纯字母，如 'ETGE'
    # PDB片段motif: 包含 '/' 和 '-'，如 'B/79-B/82'
    if '/' in value and '-' in value:
        motif_type = 'pdb_fragment'
    elif value.isalpha():
        motif_type = 'sequence'
    else:
        # 可能是其他格式，尝试解析
        return None
    
    return {
        'type': motif_type,
        'value': value,
        'seq_fix_type': seq_fix_type,
        'str_fix_type': str_fix_type,
        'contig_str': contig_item
    }


def get_motif_length(motif_config, pdb_parsed=None):
    """
    计算motif的长度
    
    Args:
        motif_config: motif配置字典
        pdb_parsed: 解析后的PDB数据（PDB片段motif需要）
    
    Returns:
        motif长度（残基数）
    """
    if motif_config['type'] == 'sequence':
        # 序列motif：直接计算序列长度
        return len(motif_config['value'])
    
    elif motif_config['type'] == 'pdb_fragment':
        # PDB片段motif：从PDB中提取
        if pdb_parsed is None:
            raise ValueError("PDB fragment motif requires input_pdb to be set and parsed")
        
        # 解析片段范围
        fragment_str = motif_config['value']
        parts = parse_residue_range(fragment_str)
        
        start_chain = parts['start_chain']
        start_res = parts['start_res']
        end_chain = parts['end_chain']
        end_res = parts['end_res']
        
        # 在pdb_parsed中查找对应的索引
        try:
            start_idx = pdb_parsed['pdb_idx'].index((start_chain, start_res))
            end_idx = pdb_parsed['pdb_idx'].index((end_chain, end_res))
            return end_idx - start_idx + 1
        except (ValueError, KeyError) as e:
            raise ValueError(f"Cannot find PDB fragment {fragment_str} in parsed PDB: {e}")
    
    else:
        raise ValueError(f"Unknown motif type: {motif_config['type']}")


def extract_motif_from_contigs(contigs):
    """
    从contigs配置中提取motif信息（仅从第一个chain提取）
    
    Args:
        contigs: contigs配置列表
    
    Returns:
        tuple: (motif_config, new_contigs_without_motif)
        motif_config: motif配置，如果没有则返回None
        new_contigs: 移除motif后的contigs（用于计算New残基长度）
    """
    if not contigs or len(contigs) == 0:
        return None, contigs
    
    # 只从第一个chain提取motif
    first_chain = contigs[0]
    # 处理OmegaConf的ListConfig类型
    if not isinstance(first_chain, (list, tuple)):
        # 尝试转换为list（处理OmegaConf的ListConfig）
        try:
            first_chain = list(first_chain)
        except (TypeError, ValueError):
            return None, contigs
    
    # 查找motif（包含 :seq_ 或 :str_ 的项）
    motif_config = None
    motif_index = None
    
    for i, item in enumerate(first_chain):
        parsed = parse_motif_from_contig(item)
        if parsed is not None:
            motif_config = parsed
            motif_index = i
            break
    
    if motif_config is None:
        return None, contigs
    
    # 构建移除motif后的contigs（用于计算New残基总长度）
    new_chain = [item for i, item in enumerate(first_chain) if i != motif_index]
    new_contigs = [new_chain] + list(contigs[1:]) if len(contigs) > 1 else [new_chain]
    
    return motif_config, new_contigs


def build_contigs_with_motif(motif_config, motif_start, motif_end, total_length):
    """
    生成包含motif的contigs格式
    
    Args:
        motif_config: motif配置字典
        motif_start, motif_end: motif位置范围（0-based）
        total_length: 总序列长度
    
    Returns:
        contigs列表，例如：[['New_12', 'ETGE:seq_FIX:str_DNV', 'New_12']]
    """
    motif_length = motif_end - motif_start
    new_length_before = motif_start
    new_length_after = total_length - motif_end
    
    contigs_list = []
    
    # 前面的New残基
    if new_length_before > 0:
        contigs_list.append(f'New_{new_length_before}-{new_length_before}')
    
    # Motif部分
    motif_str = f"{motif_config['value']}:{motif_config['seq_fix_type']}:{motif_config['str_fix_type']}"
    contigs_list.append(motif_str)
    
    # 后面的New残基
    if new_length_after > 0:
        contigs_list.append(f'New_{new_length_after}-{new_length_after}')
    
    return [contigs_list]


def determine_motif_position(total_length, motif_length, insert_position, cys_positions=None):
    """
    确定motif在序列中的插入位置
    
    Args:
        total_length: 总序列长度（包括motif）
        motif_length: motif长度
        insert_position: 插入位置策略 ('start', 'middle', 'end', 'after_C2', 'before_C4')
        cys_positions: CYS位置列表（1-based，用于'after_C2'等策略）
    
    Returns:
        (motif_start, motif_end) - motif在序列中的位置范围（0-based）
    """
    new_length = total_length - motif_length  # 除去motif后的New残基长度
    
    if insert_position == 'start':
        # 插入到序列开头
        return (0, motif_length)
    
    elif insert_position == 'middle':
        # 插入到中间
        mid_point = new_length // 2
        return (mid_point, mid_point + motif_length)
    
    elif insert_position == 'end':
        # 插入到序列末尾
        return (new_length, total_length)
    
    elif insert_position == 'after_C2':
        # 插入到C2之后（需要CYS位置信息）
        if cys_positions is None or len(cys_positions) < 2:
            # 如果没有CYS位置，使用middle作为fallback
            mid_point = new_length // 2
            return (mid_point, mid_point + motif_length)
        cys2_pos = cys_positions[1] - 1  # 转换为0-based
        start = max(cys2_pos + 1, 0)  # C2之后
        if start + motif_length > total_length:
            start = max(0, total_length - motif_length)
        return (start, start + motif_length)
    
    elif insert_position == 'before_C4':
        # 插入到C4之前
        if cys_positions is None or len(cys_positions) < 4:
            # 如果没有CYS位置，使用middle作为fallback
            mid_point = new_length // 2
            return (mid_point, mid_point + motif_length)
        cys4_pos = cys_positions[3] - 1  # 转换为0-based
        end = min(cys4_pos, new_length)  # C4之前
        start = max(0, end - motif_length)
        return (start, end)
    
    elif insert_position == 'random':
        # 在允许的范围内随机插入，但尽量避开 N/C 端
        # 默认两端各保留至少 2 个残基
        min_terminal_gap = 2
        
        # 检查是否有足够的空间来避开两端
        if new_length >= 2 * min_terminal_gap:
            start_min = min_terminal_gap
            start_max = new_length - min_terminal_gap
            start = random.randint(start_min, start_max)
        else:
            # 如果空间不够（总长度太短），则在整个范围内随机
            start = random.randint(0, new_length)
            
        return (start, start + motif_length)
    
    else:
        # 默认使用middle
        mid_point = new_length // 2
        return (mid_point, mid_point + motif_length)


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
    
    # 解析 model.model_config_path（先解析这个，因为ckpt_path可能在model.yaml中）
    if hasattr(cfg.model, 'model_config_path') and cfg.model.model_config_path:
        original_model_config_path = cfg.model.model_config_path
        cfg.model.model_config_path = resolve_relative_path(cfg.model.model_config_path, config_dir)
        
        # 如果model.yaml存在，加载它并解析其中的ckpt_path
        if os.path.exists(cfg.model.model_config_path):
            model_config_dir = os.path.dirname(cfg.model.model_config_path)
            model_cfg = OmegaConf.load(cfg.model.model_config_path)
            
            # 如果model.yaml中有ckpt_path，从model.yaml所在目录解析
            if hasattr(model_cfg.model, 'ckpt_path') and model_cfg.model.ckpt_path:
                # 将解析后的路径更新到主配置中
                resolved_ckpt_path = resolve_relative_path(model_cfg.model.ckpt_path, model_config_dir)
                cfg.model.ckpt_path = resolved_ckpt_path
    
    # 如果主配置中也有ckpt_path且还是相对路径，从主配置目录解析
    if hasattr(cfg.model, 'ckpt_path') and cfg.model.ckpt_path:
        if not os.path.isabs(cfg.model.ckpt_path):
            cfg.model.ckpt_path = resolve_relative_path(cfg.model.ckpt_path, config_dir)
    
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


def _generate_knottin_topology_free(length, seed, max_retries, min_cys_gap=2, exclude_region=None, terminal_bias_prob=0.8):
    """
    不使用区域约束的自由拓扑生成（不强制螺旋/β折叠区域）
    只保持CSαβ拓扑的核心连接模式：C1-C4, C2-C5, C3-C6
    以及顺序约束：C1 < C2 < C3 < C4 < C5 < C6
    确保相邻CYS之间至少间隔min_cys_gap个位置（避免CYS相邻）
    
    Args:
        length: 序列长度
        seed: 随机种子
        max_retries: 最大重试次数
        min_cys_gap: 相邻CYS之间的最小间隔（位置数），默认2（即至少间隔1个残基）
        exclude_region: (start, end) 需要避开的区域（0-based，左闭右开），例如motif区域
        terminal_bias_prob: 将C1和C6偏置到末端的概率 (0-1)，默认0.8
    
    Returns:
        dict: 包含contigs和pairs的配置
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 选择拓扑变体
    topology_variant = random.choice(['variant1', 'variant2', 'variant3'])
    
    # 处理exclude_region（motif区域）
    exclude_start, exclude_end = exclude_region if exclude_region else (None, None)
    
    def is_in_exclude_region(pos):
        """检查位置是否在exclude区域内"""
        if exclude_start is None or exclude_end is None:
            return False
        return exclude_start <= pos < exclude_end
    
    def adjust_position_away_from_exclude(pos, prefer_forward=True):
        """调整位置，避开exclude区域"""
        if not is_in_exclude_region(pos):
            return pos
        
        if prefer_forward and exclude_end < length:
            # 优先向前移动（向右）
            return exclude_end
        elif exclude_start > 0:
            # 向后移动（向左）
            return exclude_start - 1
        else:
            # 如果无法调整，返回边界
            return 0 if exclude_start > 0 else min(length - 1, exclude_end)
    
    for attempt in range(max_retries):
        try:
            # C1: N端附近
            # 引入概率控制：默认80% 概率极短 N 端 (0-1)，20% 概率稍长 (0-3)
            if random.random() < terminal_bias_prob:
                cys1_max = 1
            else:
                cys1_max = min(3, length // 4)
            
            cys1 = random.randint(0, max(0, cys1_max))
            cys1 = adjust_position_away_from_exclude(cys1, prefer_forward=True)
            
            # C2: 序列前半部分，确保与C1至少间隔min_cys_gap
            cys2_min = max(cys1 + min_cys_gap, 1)
            cys2_max = length // 2
            if cys2_max < cys2_min:
                cys2_max = min(cys2_min + 3, length - 5)
            cys2 = random.randint(cys2_min, cys2_max)
            cys2 = adjust_position_away_from_exclude(cys2, prefer_forward=True)
            
            # C3: C2之后，保持i+3或i+4的间隔（模拟螺旋稳定性，已经满足最小间隔）
            cys3_offset = random.choice([3, 4])
            cys3 = cys2 + cys3_offset
            # 确保C3在合理范围内
            cys3_max = min(length // 2 + 3, length - 4)
            if cys3 > cys3_max:
                cys3 = min(cys3_max, cys2 + 3)
            # 确保C3与C2的间隔至少为min_cys_gap
            if cys3 - cys2 < min_cys_gap:
                cys3 = cys2 + min_cys_gap
            cys3 = adjust_position_away_from_exclude(cys3, prefer_forward=True)
            
            # C4: 在C3和C5之间，序列中后部，确保与C3至少间隔min_cys_gap
            cys4_min = max(cys3 + min_cys_gap, length // 3)
            cys4_max = min(length * 2 // 3, length - 3)
            if cys4_max < cys4_min:
                cys4_max = min(cys4_min + 5, length - 3)
            cys4 = random.randint(cys4_min, cys4_max)
            cys4 = adjust_position_away_from_exclude(cys4, prefer_forward=True)
            
            # C5: 序列后半部分，确保与C4至少间隔min_cys_gap
            cys5_min = max(cys4 + min_cys_gap, length * 2 // 3)
            cys5_max = length - 2
            if cys5_max < cys5_min:
                cys5_max = min(cys5_min + 3, length - 2)
            cys5 = random.randint(cys5_min, cys5_max)
            cys5 = adjust_position_away_from_exclude(cys5, prefer_forward=True)
            
            # C6: C5之后
            # 引入概率控制：默认80% 概率尽量延伸到C端 (减少tail)，20% 概率自然延伸 (保留多样性)
            if random.random() < terminal_bias_prob:
                # 紧凑模式：尽量让C6在最后
                cys6_target_min = max(cys5 + min_cys_gap, length - 2)
                cys6_target_max = length - 1
                
                if cys6_target_min <= cys6_target_max:
                    cys6 = random.randint(cys6_target_min, cys6_target_max)
                else:
                    cys6 = cys5 + min_cys_gap
            else:
                # 宽松模式：允许自然的间隔分布 (可能产生C端尾巴)
                max_offset = min(4, length - cys5 - 1)
                min_offset = max(2, min_cys_gap)
                if max_offset < min_offset:
                    cys6_offset = min_cys_gap
                else:
                    cys6_offset = random.randint(min_offset, max_offset)
                cys6 = cys5 + cys6_offset
            
            # 确保C6在有效范围内
            cys6 = min(length - 1, cys6)
            cys6 = adjust_position_away_from_exclude(cys6, prefer_forward=True)
            
            # 验证顺序：C1 < C2 < C3 < C4 < C5 < C6
            if not (cys1 < cys2 < cys3 < cys4 < cys5 < cys6):
                if attempt < max_retries - 1:
                    continue
                else:
                    raise ValueError("Failed to generate valid CYS order")
            
            # 验证相邻CYS之间的间隔（至少min_cys_gap）
            cys_positions = [cys1, cys2, cys3, cys4, cys5, cys6]
            for i in range(len(cys_positions) - 1):
                gap = cys_positions[i + 1] - cys_positions[i]
                if gap < min_cys_gap:
                    if attempt < max_retries - 1:
                        break  # 重试
                    else:
                        raise ValueError(f"CYS {i+1} and {i+2} are too close (gap={gap}, min={min_cys_gap})")
            else:
                # 如果所有间隔都满足，继续处理
                pass
            
            # 确保所有位置都在有效范围内
            cys1 = max(0, min(length - 1, cys1))
            cys2 = max(0, min(length - 1, cys2))
            cys3 = max(0, min(length - 1, cys3))
            cys4 = max(0, min(length - 1, cys4))
            cys5 = max(0, min(length - 1, cys5))
            cys6 = max(0, min(length - 1, cys6))
            
            # 再次验证顺序和间隔（调整后可能改变）
            cys_positions = [cys1, cys2, cys3, cys4, cys5, cys6]
            if not (cys1 < cys2 < cys3 < cys4 < cys5 < cys6):
                if attempt < max_retries - 1:
                    continue
                else:
                    raise ValueError("Failed to generate valid CYS order after adjustment")
            
            # 再次验证间隔
            for i in range(len(cys_positions) - 1):
                gap = cys_positions[i + 1] - cys_positions[i]
                if gap < min_cys_gap:
                    if attempt < max_retries - 1:
                        break  # 重试
                    else:
                        raise ValueError(f"CYS {i+1} and {i+2} are too close after adjustment (gap={gap}, min={min_cys_gap})")
            else:
                # 如果所有间隔都满足，继续处理
                pass
            
            pairs_0based = [
                [cys1, cys4],  # 1-4: N端-中间区域
                [cys2, cys5],  # 2-5: 前半-后半
                [cys3, cys6],  # 3-6: 前半-后半（穿心）
            ]
            
            # 去重并排序pairs
            valid_pairs = []
            seen = set()
            for p in pairs_0based:
                p_sorted = tuple(sorted([int(p[0]), int(p[1])]))
                if p_sorted[0] != p_sorted[1] and p_sorted not in seen:
                    valid_pairs.append(list(p_sorted))
                    seen.add(p_sorted)
            pairs_0based = sorted(valid_pairs)
            
            # 转换为1-based
            pairs_1based = [[p[0] + 1, p[1] + 1] for p in pairs_0based]
            
            # 重新计算CYS位置（1-based）
            cys_positions_1based = sorted(set([p[0] for p in pairs_1based] + [p[1] for p in pairs_1based]))
            
            # 验证：应该有6个唯一的CYS位置
            if len(cys_positions_1based) != 6:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise ValueError(f"Expected 6 unique CYS positions, got {len(cys_positions_1based)}")
            
            # 生成contigs（_generate_knottin_topology_free不处理motif，motif在generate_knottin_topology中处理）
            contigs = [[f"New_{length}-{length}"]]
            
            return {
                'length': length,
                'contigs': contigs,
                'pairs': pairs_1based,  # 1-based
                'pairs_0based': pairs_0based,  # 保留0-based用于内部计算
                'cys_positions': cys_positions_1based,  # 1-based
                'topology_variant': topology_variant,
            }
            
        except (ValueError, IndexError) as e:
            if attempt < max_retries - 1:
                # 更新种子重试
                if seed is not None:
                    random.seed(seed + (attempt + 1) * 10000)
                    np.random.seed(seed + (attempt + 1) * 10000)
                continue
            else:
                raise ValueError(f"Failed to generate valid knottin topology for length {length} after {max_retries} retries: {e}")
    
    raise ValueError(f"Failed to generate valid knottin topology for length {length} after {max_retries} retries")


def generate_knottin_topology(length, seed=None, max_retries=5, use_region_constraints=True, min_cys_gap=2, 
                             motif_config=None, pdb_parsed=None, motif_position=None, terminal_bias_prob=0.8):
    """
    生成一个符合CSαβ拓扑的knottin配置，支持motif插入
    
    Args:
        length: 序列长度 (18-35，推荐25-35，18-24可能不稳定)
        seed: 随机种子，用于生成不同的拓扑变体
        max_retries: 最大重试次数，避免无限递归
        use_region_constraints: 是否使用螺旋/β折叠区域约束（默认True）
        min_cys_gap: 相邻CYS之间的最小间隔（位置数），默认2（即至少间隔1个残基）
        motif_config: motif配置字典，如果为None则不使用motif
        pdb_parsed: 解析后的PDB数据（PDB片段motif需要）
        motif_position: motif插入位置策略 ('start', 'middle', 'end', 'after_C2', 'before_C4')
        terminal_bias_prob: 将C1和C6偏置到末端的概率 (0-1)，默认0.8 (仅当use_region_constraints=False或使用motif时有效)
    
    Returns:
        dict: 包含contigs和pairs的配置
        {
            'length': int,
            'contigs': list,
            'pairs': list of [i, j] pairs,
            'cys_positions': list of CYS位置（用于logits_bias排除）
            'motif_info': dict with motif信息（如果使用motif）
        }
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 处理motif配置
    motif_length = 0
    motif_start = None
    motif_end = None
    exclude_region = None
    
    if motif_config:
        try:
            motif_length = get_motif_length(motif_config, pdb_parsed)
            # 总长度应该包括motif，所以实际用于生成CYS的New残基长度是 length - motif_length
            new_length = length - motif_length
            if new_length < 14:
                raise ValueError(f"After subtracting motif length ({motif_length}), remaining length ({new_length}) is too short (minimum 14)")
        except Exception as e:
            raise ValueError(f"Failed to get motif length: {e}")
    
    # 确保长度在合理范围内（支持18-35，但18-24可能不稳定，推荐25-35）
    # 如果使用motif，这里的length是总长度（包括motif）
    effective_length = length - motif_length if motif_config else length
    effective_length = max(14, min(35, effective_length))
    
    # 如果不使用区域约束，或者使用motif时，使用自由拓扑生成（更简单且已支持motif）
    if not use_region_constraints or motif_config:
        # 先临时生成CYS位置（用于确定motif位置）
        temp_result = _generate_knottin_topology_free(length, seed, max_retries, min_cys_gap, exclude_region=None, terminal_bias_prob=terminal_bias_prob)
        temp_cys_positions = temp_result['cys_positions']  # 1-based
        
        # 确定motif位置
        if motif_config:
            insert_pos = motif_position if motif_position else 'middle'
            motif_start, motif_end = determine_motif_position(
                length, motif_length, insert_pos, temp_cys_positions
            )
            exclude_region = (motif_start, motif_end)
            
            # 重新生成CYS位置，避开motif区域
            result = _generate_knottin_topology_free(length, seed, max_retries, min_cys_gap, exclude_region, terminal_bias_prob=terminal_bias_prob)
            
            # 构建包含motif的contigs（替换_generate_knottin_topology_free生成的简单contigs）
            result['contigs'] = build_contigs_with_motif(motif_config, motif_start, motif_end, length)
            result['motif_info'] = {
                'start': motif_start,
                'end': motif_end,
                'length': motif_length,
                'config': motif_config
            }
        else:
            result = temp_result
        
        return result
    
    # 处理motif配置（在使用区域约束时）
    motif_start = None
    motif_end = None
    if motif_config:
        try:
            motif_length = get_motif_length(motif_config, pdb_parsed)
            new_length = length - motif_length
            if new_length < 14:
                raise ValueError(f"After subtracting motif length ({motif_length}), remaining length ({new_length}) is too short (minimum 14)")
            # 先临时生成CYS位置来确定motif位置
            # 使用effective_length生成CYS位置，然后确定motif位置
            effective_length = new_length
        except Exception as e:
            raise ValueError(f"Failed to get motif length: {e}")
    else:
        effective_length = length
        motif_length = 0
    
    # 警告：如果长度太短，可能难以满足所有约束
    if effective_length < 25:
        # 对于短序列，使用更保守的策略
        pass
    
    # CSαβ拓扑的典型布局：
    # - N端区域（可能包含螺旋起始）
    # - 螺旋区域（通常6-10个残基）
    # - 中间连接区域
    # - β折叠区域（通常6-10个残基）
    # - C端区域
    
    # 定义区域边界（基于长度自适应）
    # 对于极短序列（<20），使用更激进的减少策略
    # 注意：这里使用effective_length（不包括motif）
    if effective_length < 20:
        n_term_len = 1  # N端固定区域
        c_term_len = 1  # C端固定区域
        min_helix_len = 4  # 最小螺旋长度（极短序列时进一步减少）
        min_sheet_len = 4  # 最小β折叠长度（极短序列时进一步减少）
        min_middle_len = 1  # 最小中间区域长度
    elif length < 25:
        n_term_len = 1  # N端固定区域（短序列时减少）
        c_term_len = 1  # C端固定区域（短序列时减少）
        min_helix_len = 5  # 最小螺旋长度（短序列时减少）
        min_sheet_len = 5  # 最小β折叠长度（短序列时减少）
        min_middle_len = 1  # 最小中间区域长度（短序列时减少）
    else:
        n_term_len = 2  # N端固定区域
        c_term_len = 2  # C端固定区域
        min_helix_len = 6
        min_sheet_len = 6
        min_middle_len = 3
    
    # 计算可用长度（使用effective_length）
    available_len = effective_length - n_term_len - c_term_len - min_middle_len
    
    # 分配螺旋和β折叠长度
    if available_len >= min_helix_len + min_sheet_len:
        if effective_length < 20:
            # 极短序列时使用更小的范围
            helix_len = random.choice([4, 5, 6])
            sheet_len = random.choice([4, 5, 6])
        elif effective_length < 25:
            # 短序列时使用更小的范围
            helix_len = random.choice([5, 6, 7])
            sheet_len = random.choice([5, 6, 7])
        else:
            helix_len = random.choice([6, 7, 8, 9])
            sheet_len = random.choice([6, 7, 8])
        remaining = available_len - helix_len - sheet_len
        middle_len = max(min_middle_len, remaining)
    else:
        # 如果长度太短，使用最小值
        helix_len = min(available_len // 2, min_helix_len)
        sheet_len = min(available_len - helix_len, min_sheet_len)
        middle_len = min_middle_len
    
    # 最终调整以确保总长度正确（使用effective_length）
    total = n_term_len + helix_len + middle_len + sheet_len + c_term_len
    if total < effective_length:
        middle_len += (effective_length - total)
    elif total > effective_length:
        middle_len = max(0, middle_len - (total - effective_length))
    
    # 计算各区域的起始位置（基于effective_length，但需要考虑motif插入后的偏移）
    # 这里先按effective_length计算，后续再根据motif位置调整
    helix_start = n_term_len
    helix_end = helix_start + helix_len
    middle_start = helix_end
    middle_end = middle_start + middle_len
    sheet_start = middle_end
    sheet_end = min(sheet_start + sheet_len, effective_length - c_term_len)
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
        cys1_max = 2 if effective_length >= 25 else (1 if effective_length >= 20 else 0)
        cys1 = random.randint(0, min(cys1_max, length-1))  # N端
        cys2_offset_max = max(0, helix_len - (4 if effective_length >= 20 else 3))  # 极短序列时允许更小的偏移
        cys2 = helix_start + random.randint(0, cys2_offset_max)  # 螺旋起始
        # 确保C2与C1至少间隔min_cys_gap
        if cys2 - cys1 < min_cys_gap:
            cys2 = min(cys1 + min_cys_gap, helix_end - 1)
        cys3_offset = random.choice([3, 4]) if effective_length >= 20 else 3  # 极短序列时只用i+3
        cys3 = cys2 + cys3_offset  # 螺旋，i+3或i+4
        # 确保C3在螺旋区域内，且与C2至少间隔min_cys_gap
        if cys3 >= helix_end:
            cys3 = min(helix_end - 1, cys2 + max(3, min_cys_gap))
        # 确保C3与C2的间隔至少为min_cys_gap
        if cys3 - cys2 < min_cys_gap:
            cys3 = min(cys2 + min_cys_gap, helix_end - 1)
        # C4在中间区域，在β折叠之前，确保 C3 < C4 < C5
        # 确保C4与C3至少间隔min_cys_gap个位置（避免相邻）
        cys4_min = max(cys3 + min_cys_gap, middle_start)
        cys4_max = min(sheet_start - 1, middle_end - 1)
        if cys4_max < cys4_min:
            # 如果中间区域太小，允许C4稍微超出中间区域
            cys4_max_offset = max(3, min_cys_gap) if effective_length >= 25 else (max(2, min_cys_gap) if effective_length >= 20 else max(1, min_cys_gap))
            cys4_max = min(cys3 + cys4_max_offset, effective_length - 1)
            cys4_min = cys3 + min_cys_gap
        cys4 = random.randint(cys4_min, cys4_max)  # 中间区域
        cys5_offset_max = max(0, sheet_len - (3 if length >= 20 else 2))
        cys5 = sheet_start + random.randint(0, cys5_offset_max)  # β折叠起始
        # 确保 C4 < C5 且至少间隔min_cys_gap
        if cys5 <= cys4:
            cys5 = min(cys4 + min_cys_gap, sheet_end - 1)
        elif cys5 - cys4 < min_cys_gap:
            cys5 = min(cys4 + min_cys_gap, sheet_end - 1)
        # 确保cys6在β折叠区域内，且 C5 < C6，至少间隔min_cys_gap
        cys6_min_offset = max(2, min_cys_gap) if length >= 25 else max(1, min_cys_gap)  # 至少间隔min_cys_gap
        cys6_max_offset = min(4, sheet_len - (cys5 - sheet_start) - 1)
        if cys6_max_offset < cys6_min_offset:
            cys6_max_offset = cys6_min_offset
        cys6 = cys5 + random.randint(cys6_min_offset, cys6_max_offset)  # β折叠
        # 确保C6在β折叠区域内，且与C5至少间隔min_cys_gap
        if cys6 >= sheet_end:
            cys6 = min(sheet_end - 1, cys5 + cys6_min_offset)
        if cys6 - cys5 < min_cys_gap:
            cys6 = min(cys5 + min_cys_gap, sheet_end - 1)
        
        pairs = [
            [cys1, cys4],  # 1-4: N端-中间区域（形成大环）
            [cys2, cys5],  # 2-5: 螺旋-β折叠
            [cys3, cys6],  # 3-6: 螺旋-β折叠（穿心）
        ]
        
    elif topology_variant == 'variant2':
        # 变体：1-4距离更长，2-5, 3-6
        # 序列顺序：C1 < C2 < C3 < C4 < C5 < C6
        cys1_max = 3 if effective_length >= 25 else (2 if effective_length >= 20 else 1)
        cys1 = random.randint(0, min(cys1_max, length-1))
        cys2_offset_max = max(0, helix_len - (4 if effective_length >= 20 else 3))
        cys2 = helix_start + random.randint(0, cys2_offset_max)
        # 确保C2与C1至少间隔min_cys_gap
        if cys2 - cys1 < min_cys_gap:
            cys2 = min(cys1 + min_cys_gap, helix_end - 1)
        cys3_offset = random.choice([3, 4]) if length >= 20 else 3
        cys3 = cys2 + cys3_offset
        # 确保C3在螺旋区域内
        if cys3 >= helix_end:
            cys3 = min(helix_end - 1, cys2 + 3)
        # C4在中间区域，但可以更靠后（形成更长的C1-C4距离），确保 C3 < C4 < C5
        # 确保C4与C3至少间隔min_cys_gap个位置（避免相邻）
        cys4_min = max(cys3 + min_cys_gap, middle_start)
        cys4_max = min(sheet_start - 1, middle_end - 1)
        if cys4_max < cys4_min:
            cys4_max_offset = max(5, min_cys_gap) if effective_length >= 25 else (max(3, min_cys_gap) if effective_length >= 20 else max(2, min_cys_gap))
            cys4_max = min(cys3 + cys4_max_offset, effective_length - 1)
            cys4_min = cys3 + min_cys_gap
        cys4 = random.randint(cys4_min, cys4_max)  # 中间区域，可能更靠后
        cys5_offset_max = max(0, sheet_len - (3 if length >= 20 else 2))
        cys5 = sheet_start + random.randint(0, cys5_offset_max)
        # 确保 C4 < C5 且至少间隔min_cys_gap
        if cys5 <= cys4:
            cys5 = min(cys4 + min_cys_gap, sheet_end - 1)
        elif cys5 - cys4 < min_cys_gap:
            cys5 = min(cys4 + min_cys_gap, sheet_end - 1)
        cys6_min_offset = max(2, min_cys_gap) if length >= 25 else max(1, min_cys_gap)  # 至少间隔min_cys_gap
        cys6_max_offset = min(4, sheet_len - (cys5 - sheet_start) - 1)
        if cys6_max_offset < cys6_min_offset:
            cys6_max_offset = cys6_min_offset
        cys6 = cys5 + random.randint(cys6_min_offset, cys6_max_offset)
        if cys6 >= sheet_end:
            cys6 = min(sheet_end - 1, cys5 + cys6_min_offset)
        if cys6 - cys5 < min_cys_gap:
            cys6 = min(cys5 + min_cys_gap, sheet_end - 1)
        
        pairs = [
            [cys1, cys4],  # 1-4: N端-中间区域（更长距离的大环）
            [cys2, cys5],  # 2-5: 螺旋-β折叠
            [cys3, cys6],  # 3-6: 螺旋-β折叠（穿心）
        ]
        
    else:  # variant3
        # 变体：调整螺旋和β折叠的相对位置
        # 序列顺序：C1 < C2 < C3 < C4 < C5 < C6
        cys1_max = 2 if effective_length >= 25 else (1 if effective_length >= 20 else 0)
        cys1 = random.randint(0, min(cys1_max, length-1))
        cys2_start = 1 if effective_length >= 25 else 0
        cys2_offset_max = max(cys2_start, helix_len - (4 if effective_length >= 20 else 3))
        cys2 = helix_start + random.randint(cys2_start, cys2_offset_max)
        # 确保C2与C1至少间隔min_cys_gap
        if cys2 - cys1 < min_cys_gap:
            cys2 = min(cys1 + min_cys_gap, helix_end - 1)
        cys3_offset = random.choice([3, 4]) if length >= 20 else 3
        cys3 = cys2 + cys3_offset
        # 确保C3在螺旋区域内，且与C2至少间隔min_cys_gap
        if cys3 >= helix_end:
            cys3 = min(helix_end - 1, cys2 + max(3, min_cys_gap))
        # 确保C3与C2的间隔至少为min_cys_gap
        if cys3 - cys2 < min_cys_gap:
            cys3 = min(cys2 + min_cys_gap, helix_end - 1)
        # C4在中间区域，在β折叠之前，确保 C3 < C4 < C5
        # 确保C4与C3至少间隔min_cys_gap个位置（避免相邻）
        cys4_min = max(cys3 + min_cys_gap, middle_start)
        cys4_max = min(sheet_start - 1, middle_end - 1)
        if cys4_max < cys4_min:
            cys4_max_offset = max(4, min_cys_gap) if effective_length >= 25 else (max(2, min_cys_gap) if effective_length >= 20 else max(1, min_cys_gap))
            cys4_max = min(cys3 + cys4_max_offset, effective_length - 1)
            cys4_min = cys3 + min_cys_gap
        cys4 = random.randint(cys4_min, cys4_max)  # 中间区域
        cys5_start = 1 if effective_length >= 25 else 0
        cys5_offset_max = max(cys5_start, sheet_len - (3 if effective_length >= 20 else 2))
        cys5 = sheet_start + random.randint(cys5_start, cys5_offset_max)
        # 确保 C4 < C5 且至少间隔min_cys_gap
        if cys5 <= cys4:
            cys5 = min(cys4 + min_cys_gap, sheet_end - 1)
        elif cys5 - cys4 < min_cys_gap:
            cys5 = min(cys4 + min_cys_gap, sheet_end - 1)
        # 确保cys6在β折叠区域内，且 C5 < C6，至少间隔min_cys_gap
        cys6_min_offset = max(2, min_cys_gap) if length >= 25 else max(1, min_cys_gap)  # 至少间隔min_cys_gap
        cys6_max_offset = min(5, sheet_len - (cys5 - sheet_start))
        cys6_candidates = []
        for offset in range(cys6_min_offset, cys6_max_offset):
            candidate = cys5 + offset
            if candidate < sheet_end and candidate - cys5 >= min_cys_gap:  # 确保在β折叠区域内且满足间隔
                cys6_candidates.append(candidate)
        if cys6_candidates:
            cys6 = random.choice(cys6_candidates)
        else:
            # 如果无法避免，使用最小偏移（至少min_cys_gap）
            cys6 = min(cys5 + max(cys6_min_offset, min_cys_gap), sheet_end - 1)
        if cys6 - cys5 < min_cys_gap:
            cys6 = min(cys5 + min_cys_gap, sheet_end - 1)
        
        pairs = [
            [cys1, cys4],  # 1-4: N端-中间区域
            [cys2, cys5],  # 2-5: 螺旋-β折叠
            [cys3, cys6],  # 3-6: 螺旋-β折叠（穿心）
        ]
    
    # 确保所有位置都在有效范围内并修正（使用effective_length，但需要考虑motif）
    # 如果使用motif，CYS位置需要避开motif区域，并且需要考虑motif插入后的位置偏移
    for i, pair in enumerate(pairs):
        pair[0] = max(0, min(effective_length-1, int(pair[0])))
        pair[1] = max(0, min(effective_length-1, int(pair[1])))
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
    
    # 如果有重复，修正它们（同时考虑min_cys_gap约束）
    if len(all_cys_positions) != len(unique_positions):
        # 先收集所有CYS位置并排序，以便检查间隔
        all_cys_sorted = sorted(set(all_cys_positions))
        
        # 找出所有已使用的位置
        used_positions = set()
        # 按顺序处理每个pair，确保位置唯一且满足间隔要求
        for pair in pairs:
            # 如果pair[0]已被使用，尝试调整
            if pair[0] in used_positions:
                # 尝试找到最近未使用的位置，且满足间隔要求
                for offset in range(min_cys_gap, length):
                    candidate1 = pair[0] + offset
                    candidate2 = pair[0] - offset
                    # 检查候选位置是否满足间隔要求
                    valid1 = (candidate1 < length and candidate1 not in used_positions and
                             all(abs(candidate1 - pos) >= min_cys_gap or pos == pair[0] for pos in used_positions))
                    valid2 = (candidate2 >= 0 and candidate2 not in used_positions and
                             all(abs(candidate2 - pos) >= min_cys_gap or pos == pair[0] for pos in used_positions))
                    if valid1:
                        pair[0] = candidate1
                        break
                    elif valid2:
                        pair[0] = candidate2
                        break
            # 如果pair[1]已被使用，尝试调整
            if pair[1] in used_positions or pair[1] == pair[0]:
                for offset in range(min_cys_gap, length):
                    candidate1 = pair[1] + offset
                    candidate2 = pair[1] - offset
                    # 检查候选位置是否满足间隔要求
                    valid1 = (candidate1 < length and candidate1 not in used_positions and candidate1 != pair[0] and
                             all(abs(candidate1 - pos) >= min_cys_gap or pos == pair[1] for pos in used_positions))
                    valid2 = (candidate2 >= 0 and candidate2 not in used_positions and candidate2 != pair[0] and
                             all(abs(candidate2 - pos) >= min_cys_gap or pos == pair[1] for pos in used_positions))
                    if valid1:
                        pair[1] = candidate1
                        break
                    elif valid2:
                        pair[1] = candidate2
                        break
            
            # 确保pair不是自环，且满足最小间隔
            if pair[0] == pair[1]:
                if pair[1] < length - 1:
                    pair[1] = pair[1] + min_cys_gap
                elif pair[0] >= min_cys_gap:
                    pair[0] = pair[0] - min_cys_gap
            
            # 记录已使用的位置
            used_positions.add(pair[0])
            used_positions.add(pair[1])
    
    # 最终验证：确保所有CYS位置唯一，且顺序正确 C1 < C2 < C3 < C4 < C5 < C6
    final_cys_positions = []
    for pair in pairs:
        final_cys_positions.extend([pair[0], pair[1]])
    
    if len(final_cys_positions) != len(set(final_cys_positions)) or len(set(final_cys_positions)) != 6:
        # 如果还有重复或不是6个唯一位置，重新生成
        if max_retries > 0:
            if seed is not None:
                return generate_knottin_topology(length, seed=seed + 10000, max_retries=max_retries-1, use_region_constraints=use_region_constraints, min_cys_gap=min_cys_gap)
            else:
                return generate_knottin_topology(length, seed=None, max_retries=max_retries-1, use_region_constraints=use_region_constraints, min_cys_gap=min_cys_gap)
        else:
            # 如果重试次数用完，抛出异常或返回一个简化的配置
            raise ValueError(f"Failed to generate valid knottin topology for length {length} after {max_retries} retries. "
                           f"Try increasing the sequence length (minimum recommended: 25).")
    
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
        if max_retries > 0:
            if seed is not None:
                return generate_knottin_topology(length, seed=seed + 20000, max_retries=max_retries-1, use_region_constraints=use_region_constraints, min_cys_gap=min_cys_gap)
            else:
                return generate_knottin_topology(length, seed=None, max_retries=max_retries-1, use_region_constraints=use_region_constraints, min_cys_gap=min_cys_gap)
        else:
            # 如果重试次数用完，抛出异常
            raise ValueError(f"Failed to generate valid knottin topology with correct CYS order for length {length} "
                           f"after {max_retries} retries. Try increasing the sequence length (minimum recommended: 25).")
    
    # 验证相邻CYS之间的最小间隔
    cys_positions = [cys1_pos, cys2_pos, cys3_pos, cys4_pos, cys5_pos, cys6_pos]
    for i in range(len(cys_positions) - 1):
        gap = cys_positions[i + 1] - cys_positions[i]
        if gap < min_cys_gap:
            # 如果间隔太小，尝试调整
            # 优先向后调整（增加后面的位置）
            if i + 1 < len(cys_positions) - 1:  # 不是最后一个CYS
                # 尝试增加下一个CYS的位置
                needed_increase = min_cys_gap - gap
                if cys_positions[i + 1] + needed_increase < length:
                    cys_positions[i + 1] += needed_increase
                    # 更新对应的pair
                    if i + 1 == 0:  # C1
                        pairs[0][0] = cys_positions[0]
                    elif i + 1 == 1:  # C2
                        pairs[1][0] = cys_positions[1]
                    elif i + 1 == 2:  # C3
                        pairs[2][0] = cys_positions[2]
                    elif i + 1 == 3:  # C4
                        pairs[0][1] = cys_positions[3]
                    elif i + 1 == 4:  # C5
                        pairs[1][1] = cys_positions[4]
                    elif i + 1 == 5:  # C6
                        pairs[2][1] = cys_positions[5]
                else:
                    # 如果无法向后调整，尝试向前调整（减少前面的位置）
                    if i > 0 and cys_positions[i] - needed_increase >= 0:
                        cys_positions[i] -= needed_increase
                        # 更新对应的pair
                        if i == 0:  # C1
                            pairs[0][0] = cys_positions[0]
                        elif i == 1:  # C2
                            pairs[1][0] = cys_positions[1]
                        elif i == 2:  # C3
                            pairs[2][0] = cys_positions[2]
                        elif i == 3:  # C4
                            pairs[0][1] = cys_positions[3]
                        elif i == 4:  # C5
                            pairs[1][1] = cys_positions[4]
            # 如果调整后仍然不满足，重新生成
            gap_after = cys_positions[i + 1] - cys_positions[i]
            if gap_after < min_cys_gap:
                if max_retries > 0:
                    if seed is not None:
                        return generate_knottin_topology(length, seed=seed + 30000, max_retries=max_retries-1, use_region_constraints=use_region_constraints, min_cys_gap=min_cys_gap)
                    else:
                        return generate_knottin_topology(length, seed=None, max_retries=max_retries-1, use_region_constraints=use_region_constraints, min_cys_gap=min_cys_gap)
                else:
                    raise ValueError(f"CYS {i+1} and {i+2} are too close (gap={gap}, min={min_cys_gap}) for length {length}")
    
    # 重新提取调整后的位置
    cys1_pos = pairs[0][0]
    cys2_pos = pairs[1][0]
    cys3_pos = pairs[2][0]
    cys4_pos = pairs[0][1]
    cys5_pos = pairs[1][1]
    cys6_pos = pairs[2][1]
    
    # 最终验证：确保所有相邻CYS都满足最小间隔
    cys_positions_final = [cys1_pos, cys2_pos, cys3_pos, cys4_pos, cys5_pos, cys6_pos]
    for i in range(len(cys_positions_final) - 1):
        gap = cys_positions_final[i + 1] - cys_positions_final[i]
        if gap < min_cys_gap:
            # 如果仍然不满足，重新生成
            if max_retries > 0:
                if seed is not None:
                    return generate_knottin_topology(length, seed=seed + 40000, max_retries=max_retries-1, use_region_constraints=use_region_constraints, min_cys_gap=min_cys_gap)
                else:
                    return generate_knottin_topology(length, seed=None, max_retries=max_retries-1, use_region_constraints=use_region_constraints, min_cys_gap=min_cys_gap)
            else:
                raise ValueError(f"Final validation failed: CYS {i+1} and {i+2} are too close (gap={gap}, min={min_cys_gap}) for length {length}")
    
    # 计算头尾剩余空间，然后整体平移连接点模式
    # 这样可以避免最后空出一大段，让CYS分布更均匀
    head_space = cys1_pos  # N端到C1之间的空间
    tail_space = length - 1 - cys6_pos  # C6到C端之间的空间
    
    # 计算可以平移的范围
    # 可以向前平移（向左）：最多平移head_space格
    # 可以向后平移（向右）：最多平移tail_space格
    # 平移0格也是允许的（保持原位置）
    
    # 选择平移方向：如果尾部空间大，尽量向后平移；如果头部空间大，尽量向前平移
    # 但也可以随机选择，增加多样性
    if tail_space > head_space and tail_space > 0:
        # 尾部空间更大，允许向后平移
        max_shift_right = min(tail_space, 3)  # 最多向后平移3格，避免过度
        max_shift_left = min(head_space, 2)  # 也可以向前平移，但限制更严格
        # 随机选择平移量：-max_shift_left 到 +max_shift_right（包括0）
        shift = random.randint(-max_shift_left, max_shift_right)
    elif head_space > tail_space and head_space > 0:
        # 头部空间更大，允许向前平移
        max_shift_left = min(head_space, 3)  # 最多向前平移3格
        max_shift_right = min(tail_space, 2)  # 也可以向后平移，但限制更严格
        # 随机选择平移量：-max_shift_left 到 +max_shift_right（包括0）
        shift = random.randint(-max_shift_left, max_shift_right)
    else:
        # 空间差不多或都很小，允许小幅平移
        max_shift = min(min(head_space, tail_space), 2)
        shift = random.randint(-max_shift, max_shift)
    
    # 应用平移：所有CYS位置同时平移
    if shift != 0:
        # 计算平移后的位置
        new_cys1 = cys1_pos + shift
        new_cys2 = cys2_pos + shift
        new_cys3 = cys3_pos + shift
        new_cys4 = cys4_pos + shift
        new_cys5 = cys5_pos + shift
        new_cys6 = cys6_pos + shift
        
        # 检查平移后是否都在有效范围内，且满足间隔要求
        new_positions = [new_cys1, new_cys2, new_cys3, new_cys4, new_cys5, new_cys6]
        valid_shift = (new_cys1 >= 0 and new_cys6 < length and 
                      new_cys1 < new_cys2 < new_cys3 < new_cys4 < new_cys5 < new_cys6)
        # 验证所有相邻CYS的间隔
        if valid_shift:
            for i in range(len(new_positions) - 1):
                if new_positions[i + 1] - new_positions[i] < min_cys_gap:
                    valid_shift = False
                    break
        
        if valid_shift:
            # 平移有效，更新pairs
            pairs[0][0] = new_cys1  # C1
            pairs[1][0] = new_cys2  # C2
            pairs[2][0] = new_cys3  # C3
            pairs[0][1] = new_cys4  # C4
            pairs[1][1] = new_cys5  # C5
            pairs[2][1] = new_cys6  # C6
        # 如果平移无效，保持原位置（shift=0的效果）
    
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
    
    # 处理motif：如果使用motif，需要调整CYS位置避开motif区域
    motif_start = None
    motif_end = None
    if motif_config:
        # 先确定motif位置（使用临时CYS位置）
        temp_cys_positions = [cys1_pos, cys2_pos, cys3_pos, cys4_pos, cys5_pos, cys6_pos]
        insert_pos = motif_position if motif_position else 'middle'
        motif_start, motif_end = determine_motif_position(
            length, motif_length, insert_pos, [p + 1 for p in temp_cys_positions]  # 转换为1-based
        )
        exclude_region = (motif_start, motif_end)
        
        # 调整CYS位置，避开motif区域
        def adjust_cys_away_from_motif(pos_0based):
            if motif_start <= pos_0based < motif_end:
                # 在motif区域内，调整位置
                if motif_end < length:
                    return motif_end  # 优先移动到motif之后
                elif motif_start > 0:
                    return motif_start - 1  # 移动到motif之前
                else:
                    return 0
            return pos_0based
        
        # 调整所有CYS位置
        cys1_pos = adjust_cys_away_from_motif(cys1_pos)
        cys2_pos = adjust_cys_away_from_motif(cys2_pos)
        cys3_pos = adjust_cys_away_from_motif(cys3_pos)
        cys4_pos = adjust_cys_away_from_motif(cys4_pos)
        cys5_pos = adjust_cys_away_from_motif(cys5_pos)
        cys6_pos = adjust_cys_away_from_motif(cys6_pos)
        
        # 更新pairs
        pairs = [
            [cys1_pos, cys4_pos],
            [cys2_pos, cys5_pos],
            [cys3_pos, cys6_pos],
        ]
        
        # 重新验证顺序和间隔
        cys_positions_check = [cys1_pos, cys2_pos, cys3_pos, cys4_pos, cys5_pos, cys6_pos]
        if not (cys1_pos < cys2_pos < cys3_pos < cys4_pos < cys5_pos < cys6_pos):
            # 如果调整后顺序不对，可能需要重新生成或报错
            # 这里先尝试简单调整
            pass  # 暂时跳过，让后续验证处理
        
        # 重新计算pairs_0based和pairs_1based（使用调整后的pairs）
        # 去重并排序pairs
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
    # 如果使用motif，需要构建包含motif的contigs
    if motif_config:
        # 构建包含motif的contigs
        contigs = build_contigs_with_motif(motif_config, motif_start, motif_end, length)
        motif_info = {
            'start': motif_start,
            'end': motif_end,
            'length': motif_length,
            'config': motif_config
        }
    else:
        contigs = [[f"New_{length}-{length}"]]
        motif_info = None
    
    return {
        'length': length,
        'contigs': contigs,
        'pairs': pairs_1based,  # 1-based
        'pairs_0based': pairs_0based,  # 保留0-based用于内部计算
        'cys_positions': cys_positions_1based,  # 1-based
        'topology_variant': topology_variant,
        'motif_info': motif_info,
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
    # 如果topology_config中有生成的contigs（包含motif），直接使用
    if 'contigs' in topology_config and topology_config['contigs']:
        original_contigs = getattr(cfg.design_config, 'contigs', None)
        if original_contigs is not None and len(original_contigs) > 0:
            # 保留原始contigs结构，替换第一个chain
            new_contigs = []
            # 第一个chain使用生成的contigs（包含motif）
            new_contigs.append(topology_config['contigs'][0])
            # 其他chain完全保留
            for chain_idx in range(1, len(original_contigs)):
                new_contigs.append(list(original_contigs[chain_idx]) if isinstance(original_contigs[chain_idx], (list, tuple)) else original_contigs[chain_idx])
            cfg_new.design_config.contigs = new_contigs
        else:
            # 如果没有原始contigs，直接使用生成的contigs
            cfg_new.design_config.contigs = topology_config['contigs']
    else:
        # 如果没有生成的contigs，使用原有逻辑
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
                            # 保留其他内容（包括motif）
                            new_chain.append(item)
                    new_contigs.append(new_chain)
                else:
                    # 其他chain：完全保留
                    new_contigs.append(list(chain) if isinstance(chain, (list, tuple)) else chain)
            cfg_new.design_config.contigs = new_contigs
        else:
            # 如果没有原始contigs，使用生成的contigs
            cfg_new.design_config.contigs = topology_config.get('contigs', [[f"New_{topology_config['length']}-{topology_config['length']}"]])
    
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
    # 但如果设置了 auto_update_positions: false，则跳过更新
    if hasattr(cfg_new, 'guidance') and hasattr(cfg_new.guidance, 'list'):
        for guidance_item in cfg_new.guidance.list:
            if hasattr(guidance_item, 'name') and guidance_item.name == 'logits_bias':
                # 检查是否设置了 auto_update_positions: false
                auto_update = getattr(guidance_item, 'auto_update_positions', True)  # 默认为True，保持向后兼容
                if auto_update:
                    # positions_mode是exclude，所以positions应该直接设置为CYS位置
                    # 这样除了这些CYS位置，其他位置都会应用bias（压低CYS生成）
                    guidance_item.positions = topology_config['cys_positions']  # 1-based，直接使用CYS位置
                # 如果 auto_update_positions 为 false，则保持原配置不变
    
    return cfg_new


def generate_knottin_topology_with_resampling(initial_length, length_range, topology_seed,
                                              use_region_constraints=True, min_cys_gap=2,
                                              motif_config=None, pdb_parsed=None,
                                              motif_position='random', terminal_bias_prob=0.8,
                                              max_sampling_attempts=20, log_prefix=""):
    """
    为单个batch生成拓扑；遇到可恢复的拓扑采样失败时，自动重采样而不是直接抛错。

    Args:
        initial_length: 首次尝试的长度
        length_range: (min_length, max_length) 长度范围
        topology_seed: 当前batch的基础随机种子
        max_sampling_attempts: 单个batch的最大重采样次数
        log_prefix: 日志前缀，例如 "[GPU 0] Batch 12: "

    Returns:
        dict | None: 成功时返回拓扑配置，连续失败时返回None
    """
    resample_rng = random.Random(topology_seed + 1)
    last_error = None

    for attempt in range(max_sampling_attempts):
        length = initial_length if attempt == 0 else resample_rng.randint(length_range[0], length_range[1])
        attempt_seed = topology_seed + attempt * 10000

        try:
            topology_config = generate_knottin_topology(
                length,
                seed=attempt_seed,
                use_region_constraints=use_region_constraints,
                min_cys_gap=min_cys_gap,
                motif_config=motif_config,
                pdb_parsed=pdb_parsed,
                motif_position=motif_position,
                terminal_bias_prob=terminal_bias_prob,
            )

            if attempt > 0:
                print(
                    f"{log_prefix}Info: Topology generation succeeded after "
                    f"{attempt + 1} attempts (length={length}, seed={attempt_seed})."
                )

            return topology_config
        except ValueError as e:
            last_error = e
            if attempt < max_sampling_attempts - 1:
                print(
                    f"{log_prefix}Warning: Topology generation failed for "
                    f"length {length} (attempt {attempt + 1}/{max_sampling_attempts}, "
                    f"seed={attempt_seed}): {e}. Resampling..."
                )

    print(
        f"{log_prefix}Warning: Skipping this batch after "
        f"{max_sampling_attempts} failed topology attempts. Last error: {last_error}"
    )
    return None


def run_sampling_worker_knottin(device_str, base_cfg_path, num_designs, num_cycle, num_timesteps,
                                write_trajectory, out_dir, use_partial_diffusion, batch_range,
                                length_range=(18, 35), topology_seed_base=42, use_region_constraints=True, min_cys_gap=2, terminal_bias_prob=0.8):
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
        use_region_constraints: 是否使用螺旋/β折叠区域约束
        min_cys_gap: 相邻CYS之间的最小间隔（位置数），默认2
        terminal_bias_prob: 将C1和C6偏置到末端的概率 (0-1)，默认0.8
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
    print(f"[{device_label}] Use region constraints: {use_region_constraints}")
    print(f"[{device_label}] Min CYS gap: {min_cys_gap}")
    print(f"[{device_label}] Terminal bias probability: {terminal_bias_prob}")
    
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
    skipped_batches = 0

    print(f"[{device_label}] Processing {num_batches_this_gpu} batches")
    
    for local_batch_i in range(num_batches_this_gpu):
        global_batch_i = start_batch + local_batch_i
        start_index = global_batch_i * num_designs
        
        # 为当前batch生成拓扑配置
        # 使用global_batch_i作为种子的一部分，确保每个batch不同
        topology_seed = topology_seed_base + global_batch_i * 1000
        length = random.randint(length_range[0], length_range[1])
        
        # 从配置中提取motif信息（如果存在）
        motif_config = None
        pdb_parsed = None
        # 默认使用 random，让 motif 位置在每个 batch 中随机变化
        # 用户也可以在 design_config 中指定 motif_position: 'middle' / 'start' / 'end' / 'after_C2' 等
        motif_position = getattr(base_cfg.design_config, 'motif_position', 'random')
        
        original_contigs = getattr(base_cfg.design_config, 'contigs', None)
        if original_contigs and len(original_contigs) > 0:
            # 处理OmegaConf的ListConfig类型
            try:
                original_contigs = [list(chain) if not isinstance(chain, (list, tuple)) else chain for chain in original_contigs]
            except (TypeError, ValueError):
                pass
            first_chain = original_contigs[0]
            if isinstance(first_chain, (list, tuple)):
                motif_config, _ = extract_motif_from_contigs(original_contigs)
                
                # 如果找到motif，需要解析PDB（如果是PDB片段motif）
                if motif_config and motif_config['type'] == 'pdb_fragment':
                    input_pdb = getattr(base_cfg.design_config, 'input_pdb', None)
                    if input_pdb:
                        try:
                            pdb_parsed = iu.process_target(
                                input_pdb,
                                parse_hetatom=False,
                                center=False,
                                parse_link=True,
                                link_csv_path=getattr(base_cfg.preprocess, 'link_config', None),
                            )
                        except Exception as e:
                            print(f"Warning: Failed to parse PDB for motif: {e}")
                            motif_config = None
        
        # 生成拓扑配置；如果撞到短序列/边界条件，自动重采样而不是直接退出worker
        topology_config = generate_knottin_topology_with_resampling(
            initial_length=length,
            length_range=length_range,
            topology_seed=topology_seed,
            use_region_constraints=use_region_constraints,
            min_cys_gap=min_cys_gap,
            motif_config=motif_config,
            pdb_parsed=pdb_parsed,
            motif_position=motif_position,
            terminal_bias_prob=terminal_bias_prob,
            log_prefix=f"[{device_label}] Batch {global_batch_i}: ",
        )

        if topology_config is None:
            skipped_batches += 1
            continue
        
        print(f"\n[{device_label}] Batch {global_batch_i}/{num_cycle-1}")
        print(f"  Length: {topology_config['length']}")
        if 'topology_variant' in topology_config:
            print(f"  Topology variant: {topology_config['topology_variant']}")
        if topology_config.get('motif_info'):
            motif_info = topology_config['motif_info']
            print(f"  Motif: {motif_info['config']['value']} (length={motif_info['length']}, position={motif_info['start']}-{motif_info['end']})")
        print(f"  Disulfide pairs (CSαβ topology):")
        if len(topology_config['pairs']) >= 3:
            print(f"    C1-C4 (N-C lock): {topology_config['pairs'][0]}")
            print(f"    C2-C5 (helix-sheet): {topology_config['pairs'][1]}")
            print(f"    C3-C6 (threading): {topology_config['pairs'][2]}")
        else:
            for i, pair in enumerate(topology_config['pairs']):
                print(f"    Pair {i+1}: {pair}")
        print(f"  CYS positions: {topology_config['cys_positions']}")
        if len(topology_config['pairs']) >= 3:
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
                pdb_parsed=pdb_parsed,  # 传递原始PDB数据
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
                    pdb_parsed=pdb_parsed,  # 传递原始PDB数据，用于写入完整链
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
    
    if skipped_batches > 0:
        print(f"[{device_label}] Finished all batches with {skipped_batches} skipped batches")
    else:
        print(f"[{device_label}] Finished all batches")


def preview_topologies(cfg_path, num_cycle, length_range=(18, 35), topology_seed_base=42, max_preview=10, use_region_constraints=True, min_cys_gap=2, terminal_bias_prob=0.8):
    """
    预览模式：只生成和打印拓扑配置，不实际运行采样
    
    Args:
        cfg_path: 基础配置文件路径
        num_cycle: 总批次数
        length_range: (min_length, max_length) 长度范围
        topology_seed_base: 拓扑生成的种子基数
        max_preview: 最多预览的batch数量（如果为None则预览所有）
        use_region_constraints: 是否使用螺旋/β折叠区域约束
        min_cys_gap: 相邻CYS之间的最小间隔（位置数），默认2
        terminal_bias_prob: 将C1和C6偏置到末端的概率 (0-1)，默认0.8
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
    print(f"Use region constraints: {use_region_constraints}")
    print(f"Min CYS gap: {min_cys_gap}")
    print(f"Terminal bias probability: {terminal_bias_prob}")
    print("="*80)
    print()
    
    # 解析相对路径
    config_dir = os.path.dirname(os.path.abspath(cfg_path))
    resolve_config_paths(base_cfg, config_dir)
    
    # 从配置中提取motif信息（如果存在）
    motif_config = None
    pdb_parsed = None
    # 默认使用 random
    motif_position = getattr(base_cfg.design_config, 'motif_position', 'random')
    
    original_contigs = getattr(base_cfg.design_config, 'contigs', None)
    if original_contigs and len(original_contigs) > 0:
        # 处理OmegaConf的ListConfig类型
        try:
            original_contigs = [list(chain) if not isinstance(chain, (list, tuple)) else chain for chain in original_contigs]
        except (TypeError, ValueError):
            pass
        first_chain = original_contigs[0]
        if isinstance(first_chain, (list, tuple)):
            motif_config, _ = extract_motif_from_contigs(original_contigs)
            
            # 如果找到motif，需要解析PDB（如果是PDB片段motif）
            if motif_config and motif_config['type'] == 'pdb_fragment':
                input_pdb = getattr(base_cfg.design_config, 'input_pdb', None)
                if input_pdb:
                    try:
                        pdb_parsed = iu.process_target(
                            input_pdb,
                            parse_hetatom=False,
                            center=False,
                            parse_link=True,
                            link_csv_path=getattr(base_cfg.preprocess, 'link_config', None),
                        )
                    except Exception as e:
                        print(f"Warning: Failed to parse PDB for motif: {e}")
                        motif_config = None
    
    if motif_config:
        print(f"Motif detected: {motif_config['value']} (type: {motif_config['type']})")
        print(f"Motif position: {motif_position}")
        print()
    
    preview_count = min(num_cycle, max_preview) if max_preview else num_cycle
    skipped_batches = 0

    for batch_i in range(preview_count):
        topology_seed = topology_seed_base + batch_i * 1000
        length = random.randint(length_range[0], length_range[1])
        
        # 生成拓扑配置（包含motif支持）；失败时自动重采样并给出提示
        topology_config = generate_knottin_topology_with_resampling(
            initial_length=length,
            length_range=length_range,
            topology_seed=topology_seed,
            use_region_constraints=use_region_constraints,
            min_cys_gap=min_cys_gap,
            motif_config=motif_config,
            pdb_parsed=pdb_parsed,
            motif_position=motif_position,
            terminal_bias_prob=terminal_bias_prob,
            log_prefix=f"[Preview] Batch {batch_i}: ",
        )

        if topology_config is None:
            skipped_batches += 1
            continue
        
        # 更新配置以显示实际使用的配置
        cfg = update_config_for_batch(base_cfg, topology_config)
        
        print(f"{'='*80}")
        print(f"Batch {batch_i}/{num_cycle-1} Configuration:")
        print(f"{'='*80}")
        print(f"  Length: {topology_config['length']}")
        if 'topology_variant' in topology_config:
            print(f"  Topology variant: {topology_config['topology_variant']}")
        if topology_config.get('motif_info'):
            motif_info = topology_config['motif_info']
            print(f"  Motif: {motif_info['config']['value']} (length={motif_info['length']}, position={motif_info['start']}-{motif_info['end']})")
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
        if topology_config.get('motif_info'):
            motif_info = topology_config['motif_info']
            print(f"  Motif info: {motif_info['config']['value']} at positions {motif_info['start']}-{motif_info['end']} (length={motif_info['length']})")
        
        # 显示guidance配置
        if hasattr(cfg, 'guidance') and hasattr(cfg.guidance, 'list'):
            for guidance_item in cfg.guidance.list:
                if hasattr(guidance_item, 'name'):
                    if guidance_item.name == 'logits_bias':
                        print(f"\n  guidance.logits_bias:")
                        # 安全访问 positions 字段（可能不存在）
                        positions = getattr(guidance_item, 'positions', None)
                        if positions is not None:
                            print(f"    positions: {positions}")
                        else:
                            print(f"    positions: (not set)")
                        print(f"    positions_mode: {getattr(guidance_item, 'positions_mode', 'exclude')}")
                        print(f"    auto_update_positions: {getattr(guidance_item, 'auto_update_positions', True)}")
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

    if skipped_batches > 0:
        print(f"Warning: Skipped {skipped_batches} preview batches after repeated topology generation failures.")
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
    parser.add_argument("--min_length", type=int, default=25, help="Minimum sequence length (18-35, recommended: 25-35)")
    parser.add_argument("--max_length", type=int, default=35, help="Maximum sequence length (18-35)")
    parser.add_argument("--topology_seed", type=int, default=42, help="Base seed for topology generation")
    parser.add_argument("--preview", action="store_true", 
                       help="Preview mode: only print topologies without running sampling")
    parser.add_argument("--max_preview", type=int, default=10,
                       help="Maximum number of batches to preview (default: 10, use 0 for all)")
    parser.add_argument("--no_region_constraints", action="store_true",
                       help="Disable helix/sheet region constraints for CYS placement (more flexible topology)")
    parser.add_argument("--min_cys_gap", type=int, default=1,
                       help="Minimum gap between adjacent CYS positions (default: 2, i.e., at least 1 residue between CYS)")
    parser.add_argument("--terminal_bias_prob", type=float, default=0.8,
                       help="Probability (0.0-1.0) to bias C1/C6 towards termini in --no_region_constraints mode (default: 0.8)")
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
    use_region_constraints = not args.no_region_constraints
    min_cys_gap = args.min_cys_gap
    terminal_bias_prob = args.terminal_bias_prob
    
    # 预览模式：只打印拓扑配置，不运行采样
    if args.preview:
        max_preview = None if args.max_preview == 0 else args.max_preview
        preview_topologies(args.cfg, num_cycle, length_range, args.topology_seed, max_preview, use_region_constraints, min_cys_gap, terminal_bias_prob)
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
        print(f"  Use region constraints: {use_region_constraints}")
        print(f"  Min CYS gap: {min_cys_gap}")
        print(f"  Terminal bias probability: {terminal_bias_prob}")
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
                    length_range, args.topology_seed, use_region_constraints, min_cys_gap,
                    terminal_bias_prob
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
            length_range, args.topology_seed, use_region_constraints, min_cys_gap,
            terminal_bias_prob
        )


if __name__ == "__main__":
    main()
