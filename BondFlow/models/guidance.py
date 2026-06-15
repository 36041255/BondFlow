import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from BondFlow.models.Loss import OpenFoldClashLoss
import BondFlow.data.utils as iu

from BondFlow.data.link_utils import LinkInfo
from BondFlow.models.layers import DSMProjection
from rfdiff.chemical import aa2num


class Guidance:
    """Base class for guidance modules.

    Each hook receives and returns a dict to allow extensibility without changing signatures.
    Hooks are no-ops by default.
    """

    def __init__(self, cfg: Optional[Any] = None, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device

    def pre_model(self, model_raw: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        """Hook right after model forward, before building interpolant inputs.

        model_raw keys (typical):
          - logits: [B, L, C]
          - px0_bb: [B, L, 3, 3]
          - alpha_pred: [B, L, 10, 2]
          - bond_mat_pred: [B, L, L]
        """
        return model_raw

    def pre_interpolant(self, model_out: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        """Hook before calling interpolant.sample_step.

        model_out keys (typical):
          - pred_trans: [B, L, 3]
          - pred_rotmats: [B, L, 3, 3]
          - pred_aatypes: [B, L]
          - pred_logits: [B, L, C]
          - pred_ss: [B, L, L]
        """
        return model_out

    def post_step(self, step_out: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        """Hook after interpolant.sample_step, before building all-atom for next x.

        step_out keys (typical):
          - trans_t_2: [B, L, 3]
          - rotmats_t_2: [B, L, 3, 3]
          - aatypes_t_2: [B, L]
          - ss_t_2: [B, L, L]
        """
        return step_out


class GuidanceManager:
    def __init__(self, guidances: Optional[List[Guidance]] = None) -> None:
        self.guidances: List[Guidance] = guidances or []

    def pre_model(self, model_raw: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        updated = model_raw
        for g in self.guidances:
            updated = g.pre_model(updated, **context)
        return updated

    def pre_interpolant(self, model_out: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        updated = model_out
        for g in self.guidances:
            updated = g.pre_interpolant(updated, **context)
        return updated

    def post_step(self, step_out: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        updated = step_out
        for g in self.guidances:
            updated = g.post_step(updated, **context)
        return updated


def _compute_entropy_regularization(P: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    """
    计算熵正则化项，鼓励概率值接近 0 或 1（二值化）。
    
    熵公式：H = -sum(P * log(P) + (1-P) * log(1-P))
    - 当 P=0 或 P=1 时，熵为 0（完全确定）
    - 当 P=0.5 时，熵最大（最不确定）
    
    Args:
        P: 概率矩阵 [B, L, L] 或 [B, L, ...]
        mask: 可选的有效位置 mask [B, L, L] 或 [B, L, ...]，与 P 相同的形状
        eps: 数值稳定性参数
    
    Returns:
        熵值（标量），值越大表示熵越高（越不确定）
    """
    if mask is not None:
        # 只计算有效位置的熵；使用 masked_select 避免 nan * 0 问题
        P_valid = P.masked_select(mask.bool())
        if P_valid.numel() == 0:
            return P.new_tensor(0.0)
            
        P_safe = P_valid.clamp(min=eps, max=1.0 - eps)
        entropy = -(P_safe * torch.log(P_safe) + (1.0 - P_safe) * torch.log(1.0 - P_safe))
        return entropy.sum()
    else:
        # 计算所有位置的平均熵
        P_safe = P.clamp(min=eps, max=1.0 - eps)
        entropy = -(P_safe * torch.log(P_safe) + (1.0 - P_safe) * torch.log(1.0 - P_safe))
        return entropy.mean()

def _schedule_weight(t_1: torch.Tensor, base: float, schedule: str = "linear", power: float = 1.0) -> torch.Tensor:
    """Compute a scalar or per-batch weight based on t_1 in [0, 1].

    Schedules:
      - linear: base * (1 - t_1)
      - quadratic: base * (1 - t_1) ** 2
      - cosine: base * (1 - cos(pi * (1 - t_1)))/2 (ramps up as t decreases)
      - exp: base * (1 - t_1) ** power (power provided)
      - inverse: base / (1 - t_1) (ramps up as t increases)
    """
    schedule = (schedule or "linear").lower()
    if schedule == "linear":
        return base * (1.0 - t_1)
    if schedule == "quadratic":
        return base * (1.0 - t_1) ** 2
    if schedule == "cosine":
        return base * (1.0 - torch.cos(math.pi * (1.0 - t_1)) / 2.0)
    if schedule == "exp":
        return base * (1.0 - t_1) ** power
    if schedule == "inverse":
        return base / (1.0 - t_1 + 1e-3)
    return base * (1.0 - t_1)


def _resolve_full_pdb_idx_from_context(context: Dict[str, Any], L: int) -> Optional[List]:
    """Resolve one sample's `full_pdb_idx` from guidance context."""

    def _looks_like_single(sample: Any) -> bool:
        return isinstance(sample, (list, tuple)) and len(sample) == L

    def _pick(candidate: Any) -> Optional[List]:
        if _looks_like_single(candidate):
            return list(candidate)
        if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
            first = candidate[0]
            if _looks_like_single(first):
                return list(first)
        return None

    resolved = _pick(context.get("full_pdb_idx", None))
    if resolved is not None:
        return resolved

    fixed_batch_data = context.get("fixed_batch_data", None)
    if isinstance(fixed_batch_data, dict):
        for key in ("pdb_idx", "pdb_idx_full", "origin_pdb_idx"):
            resolved = _pick(fixed_batch_data.get(key, None))
            if resolved is not None:
                return resolved
    return None


def _resolve_origin_pdb_idx_from_context(context: Dict[str, Any], L: int) -> Optional[List]:
    """Resolve one sample's `full_origin_pdb_idx` from guidance context."""

    def _looks_like_single(sample: Any) -> bool:
        return isinstance(sample, (list, tuple)) and len(sample) == L

    def _pick(candidate: Any) -> Optional[List]:
        if _looks_like_single(candidate):
            return list(candidate)
        if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
            first = candidate[0]
            if _looks_like_single(first):
                return list(first)
        return None

    for key in ("origin_pdb_idx", "full_origin_pdb_idx"):
        resolved = _pick(context.get(key, None))
        if resolved is not None:
            return resolved

    fixed_batch_data = context.get("fixed_batch_data", None)
    if isinstance(fixed_batch_data, dict):
        for key in ("origin_pdb_idx", "full_origin_pdb_idx"):
            resolved = _pick(fixed_batch_data.get(key, None))
            if resolved is not None:
                return resolved
    return None


class LogitsBiasGuidance(Guidance):
    """Add a bias vector to the first 20 amino-acid logits with a t-dependent weight."""

    def __init__(self, cfg: Optional[Any] = None, device: str = "cpu") -> None:
        super().__init__(cfg, device)
        bias_list = getattr(cfg, "bias", None) if cfg is not None else None
        self.bias = None
        if bias_list is not None:
            self.bias = torch.tensor(bias_list, dtype=torch.float32, device=device)
        self.weight = float(getattr(cfg, "weight", 1.0)) if cfg is not None else 1.0
        self.schedule = str(getattr(cfg, "schedule", "linear")) if cfg is not None else "linear"
        self.power = float(getattr(cfg, "power", 1.0)) if cfg is not None else 1.0
        # Optional position control:
        # - positions: list[int|str] residue indices; str supports "start-end" inclusive
        # - positions_mode: "include" (only these positions) or "exclude" (all except these positions)
        # - index_base: 0 (0-based indices) or 1 (1-based indices)
        # - ignore_seq_mask: if True, do not additionally gate by masks["seq_mask"]
        self.positions = getattr(cfg, "positions", None) if cfg is not None else None
        # Backward/alternative keys
        if self.positions is None and cfg is not None:
            inc = getattr(cfg, "include_positions", None)
            exc = getattr(cfg, "exclude_positions", None)
            if inc is not None:
                self.positions = inc
                self.positions_mode = "include"
            elif exc is not None:
                self.positions = exc
                self.positions_mode = "exclude"
            else:
                self.positions_mode = str(getattr(cfg, "positions_mode", "include")).lower()
        else:
            self.positions_mode = str(getattr(cfg, "positions_mode", "include")).lower() if cfg is not None else "include"
        self.ignore_seq_mask = bool(getattr(cfg, "ignore_seq_mask", False)) if cfg is not None else False

    @staticmethod
    def _parse_positions(raw: Any, L: int, *, index_base: int = 0) -> Optional[torch.Tensor]:
        """Parse positions config into a boolean mask of shape [L]."""
        if raw is None:
            return None
        if L <= 0:
            return torch.zeros((0,), dtype=torch.bool)

        items = raw
        if isinstance(items, (int, float, str)):
            items = [items]
        try:
            items = list(items)
        except Exception:
            return None

        mask = torch.zeros((L,), dtype=torch.bool)

        def norm_idx(x: int) -> Optional[int]:
            ii = int(x) - int(index_base)
            if ii < 0:
                ii = L + ii  # allow negative indexing from end
            if 0 <= ii < L:
                return ii
            return None

        for it in items:
            if it is None:
                continue
            # Numeric index
            if isinstance(it, (int, float)):
                ii = norm_idx(int(it))
                if ii is not None:
                    mask[ii] = True
                continue
            # String index or range
            if isinstance(it, str):
                s = it.strip()
                if not s:
                    continue
                # Allow "a-b" or "a:b" ranges (inclusive)
                sep = "-" if ("-" in s) else (":" if (":" in s) else None)
                if sep is not None:
                    parts = [p.strip() for p in s.split(sep) if p.strip()]
                    if len(parts) == 2:
                        try:
                            a = int(parts[0])
                            b = int(parts[1])
                        except Exception:
                            continue
                        ia = norm_idx(a)
                        ib = norm_idx(b)
                        if ia is None or ib is None:
                            continue
                        lo, hi = (ia, ib) if ia <= ib else (ib, ia)
                        mask[lo : hi + 1] = True
                        continue
                # Single int string
                try:
                    ii = norm_idx(int(s))
                except Exception:
                    ii = None
                if ii is not None:
                    mask[ii] = True
                continue
        return mask

    def pre_model(self, model_raw: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        if self.bias is None:
            return model_raw
        logits: torch.Tensor = model_raw.get("logits")
        if logits is None:
            return model_raw
        t_1: torch.Tensor = context.get("t_1")
        if t_1 is None:
            return model_raw
            
        # Get seq_mask if available
        masks = context.get("masks", {})
        seq_mask = masks.get("seq_mask", None) # [B, L]

        # ensure bias length matches first 20 channels or broadcast
        bias = self.bias
        if bias.numel() not in (1, 20):
            return model_raw
        # compute weight per-batch
        w = _schedule_weight(t_1, self.weight, self.schedule, self.power)  # [B]
        while w.dim() < logits.dim():
            w = w.unsqueeze(-1)

        # apply to first 20 logits
        bias_term = None
        if bias.numel() == 1:
            bias_term = w * bias
        else:
            bias_term = w * bias.unsqueeze(0).unsqueeze(0)

        # Optional positional mask
        B, L, _ = logits.shape
        pos_mask_1d = self._parse_positions(self.positions, L, index_base=0)
        pos_mask = None
        if pos_mask_1d is not None:
            pos_mask = pos_mask_1d.to(device=logits.device).view(1, L).expand(B, -1)  # [B,L]
            mode = (self.positions_mode or "include").lower()
            if mode in ("exclude", "outside", "except"):
                pos_mask = ~pos_mask
            elif mode in ("include", "inside", "only"):
                pass
            else:
                # unknown mode -> no-op
                pos_mask = None

        # Apply masks (seq_mask and/or pos_mask)
        combined_mask = None
        if (seq_mask is not None) and (not self.ignore_seq_mask):
            combined_mask = seq_mask.bool()
        if pos_mask is not None:
            combined_mask = pos_mask if combined_mask is None else (combined_mask & pos_mask)
        if combined_mask is not None:
            bias_term = bias_term * combined_mask.unsqueeze(-1).float()

        model_raw["logits"] = logits.clone()
        model_raw["logits"][..., :20] = logits[..., :20] + bias_term
        
        return model_raw


class TransAnchorGuidance(Guidance):
    """Softly pull predicted translations toward anchors (e.g., trans_1)."""

    def __init__(self, cfg: Optional[Any] = None, device: str = "cpu") -> None:
        super().__init__(cfg, device)
        self.weight = float(getattr(cfg, "weight", 1.0)) if cfg is not None else 1.0
        self.schedule = str(getattr(cfg, "schedule", "linear")) if cfg is not None else "linear"
        self.power = float(getattr(cfg, "power", 1.0)) if cfg is not None else 1.0
        self.anchor_key = str(getattr(cfg, "anchor_key", "trans_1")) if cfg is not None else "trans_1"

    def pre_interpolant(self, model_out: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        target: Optional[torch.Tensor] = context.get(self.anchor_key)
        if target is None:
            return model_out
        pred_trans: Optional[torch.Tensor] = model_out.get("pred_trans")
        t_1: Optional[torch.Tensor] = context.get("t_1")
        if pred_trans is None or t_1 is None:
            return model_out
            
        # Get str_mask if available
        masks = context.get("masks", {})
        str_mask = masks.get("str_mask", None) # [B, L]
        
        # compute blending weight
        w = _schedule_weight(t_1, self.weight, self.schedule, self.power).view(-1, 1, 1)
        
        # blend toward anchor
        updated_trans = (1.0 - w) * pred_trans + w * target
        
        # Apply mask: keep original pred_trans where mask is 0
        if str_mask is not None:
            mask_f = str_mask.unsqueeze(-1).float()
            updated_trans = updated_trans * mask_f + pred_trans * (1.0 - mask_f)
            
        updated = model_out.copy()
        updated["pred_trans"] = updated_trans
        return updated


def build_guidances(cfg: Optional[Any], device: str = "cpu") -> List[Guidance]:
    """Build a list of guidance instances from config.

    Expected schemas:
      - None: returns []
      - Dict with key "list": a list of {name: str, ...params}
      - List[Dict]: same as above
    """
    if cfg is None:
        return []

    guidances_cfg: List[Any]
    if isinstance(cfg, list):
        guidances_cfg = cfg
    else:
        list_attr = getattr(cfg, "list", None)
        if list_attr is None:
            # single entry dict {name:..., ...}
            guidances_cfg = [cfg]
        else:
            guidances_cfg = list(list_attr)

    built: List[Guidance] = []
    for gcfg in guidances_cfg:
        name = str(getattr(gcfg, "name", None) or getattr(gcfg, "type", None) or "").lower()
        cls = _GUIDANCE_REGISTRY.get(name)
        if cls is None:
            continue
        built.append(cls(gcfg, device=device))
    return built


class SingleBondGuidance(Guidance):
    """
    Guidance on the ss / bond matrix to enforce that there is
    at least (or exactly) one off-diagonal pair with value ~1.

    Config schema (all optional, with defaults):
      name: single_bond
      mode: "at_least_one" | "exactly_one"
      threshold: 0.5      # minimum value to consider as a "bond"
      target_value: 1.0   # value to set for the selected bond
      self_weight: 1.0    # diagonal weight multiplier (0 to clear diagonal)
      schedule: "linear"  # how strongly this guidance is applied over time
      weight: 1.0
      power: 1.0
    """

    def __init__(self, cfg: Optional[Any] = None, device: str = "cpu") -> None:
        super().__init__(cfg, device)
        self.mode = str(getattr(cfg, "mode", "at_least_one")).lower() if cfg is not None else "at_least_one"
        self.threshold = float(getattr(cfg, "threshold", 0.5)) if cfg is not None else 0.5
        self.target_value = float(getattr(cfg, "target_value", 1.0)) if cfg is not None else 1.0
        self.self_weight = float(getattr(cfg, "self_weight", 1.0)) if cfg is not None else 1.0
        self.weight = float(getattr(cfg, "weight", 1.0)) if cfg is not None else 1.0
        self.schedule = str(getattr(cfg, "schedule", "linear")) if cfg is not None else "linear"
        self.power = float(getattr(cfg, "power", 1.0)) if cfg is not None else 1.0

        # DSM 参数
        self.sinkhorn_iters = 30
        self.eps = 1e-8

    def post_step(self, step_out: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        """
        Modify ss_t_2 (shape [B, L, L]) after interpolant.sample_step.
        We:
          1) optionally down-weight diagonal,
          2) ensure at least/only one off-diagonal pair is strongly "on".
        """
        ss_t_2: Optional[torch.Tensor] = step_out.get("ss_t_2", None)
        t_1: Optional[torch.Tensor] = context.get("t_1", None)
        if ss_t_2 is None or t_1 is None:
            return step_out
            
        masks = context.get("masks", {})
        bond_mask = masks.get("bond_mask", None) # [B, L, L]

        B, L, _ = ss_t_2.shape
        M = ss_t_2.clone()

        # apply time-dependent global weight so that guidance is stronger at low t
        w = _schedule_weight(t_1, self.weight, self.schedule, self.power)  # [B]
        # 这里做一个截断，避免 w > 1 或 w < 0 的极端情况
        w = w.view(B, 1, 1).clamp(0.0, 1.0)

        # 1) down-/up-weight diagonal to discourage trivial self-bonds
        if self.self_weight != 1.0:
            eye = torch.eye(L, device=M.device, dtype=M.dtype).unsqueeze(0)
            M = M * (1.0 - eye) + M * self.self_weight * eye

        # 2) enforce a strong off-diagonal (i,j) per batch, with hard zeroing of
        #    row i and column j (except at (i,j) and (j,i)), before Sinkhorn.
        #    Other positions stay unchanged.
        eye_bool = torch.eye(L, device=M.device, dtype=torch.bool).unsqueeze(0)  # [1, L, L]
        # mask out diagonal when searching max off-diagonal
        masked = M.masked_fill(eye_bool, float("-inf"))
        
        # 如果提供了 bond_mask，还要屏蔽掉不允许改变的区域，避免选到固定的键（或者非键）
        if bond_mask is not None:
            masked = masked.masked_fill(~bond_mask.bool(), float("-inf"))
            
        vals, flat_idx = masked.view(B, -1).max(dim=1)  # [B], [B]
        i = flat_idx // L  # [B]
        j = flat_idx % L   # [B]
        batch_ids = torch.arange(B, device=M.device)

        guided = M.clone()

        if self.mode == "exactly_one":
            # 对所有 batch：将第 i 行和第 j 列除 (i,j)/(j,i) 外全部置 0，然后把 (i,j)/(j,i) 设为 target_value
            guided[batch_ids, i, :] = 0.0
            guided[batch_ids, :, j] = 0.0
            guided[batch_ids, i, j] = self.target_value
            guided[batch_ids, j, i] = self.target_value
        elif self.mode == "at_least_one":
            # 只对当前最强键小于 threshold 的 batch 做上述操作
            need = vals < self.threshold  # [B] bool
            if need.any():
                b_ids = torch.nonzero(need, as_tuple=False).view(-1)
                guided[b_ids, i[b_ids], :] = 0.0
                guided[b_ids, :, j[b_ids]] = 0.0
                guided[b_ids, i[b_ids], j[b_ids]] = self.target_value
                guided[b_ids, j[b_ids], i[b_ids]] = self.target_value

        # 3) 使用 DSMProjection 做 Sinkhorn 投影
        # 设置 base_tau=1.0，因为我们将手动传入 log(guided)
        dsm = DSMProjection(base_tau=1, max_iter=self.sinkhorn_iters, eps=self.eps)
        # 将非负矩阵转为伪 logits：log(x + eps)。DSM 内部会做 exp(logits/tau)，当 tau=1 时还原为 x + eps
        logits = torch.log(guided.clamp(min=1e-12))
        
        guided = dsm(logits, mask_2d=bond_mask, mat_true=ss_t_2)

        # 4) blend with original using time-dependent weight
        blended = (1.0 - w) * ss_t_2 + w * guided
        
        step_out["ss_t_2"] = blended
        return step_out


class SoftBondCountGuidance(Guidance):
    """
    Differentiable bond-guidance on the ss / bond matrix via a soft count of bonds.

    The idea:
      - Take current ss_t_2 as a continuous variable.
      - Project to a (near) symmetric doubly-stochastic matrix with Sinkhorn.
      - Build a soft "bond count" C from a smooth indicator q_ij = sigmoid(alpha*(P_ij - tau)).
      - Define an energy on C to encourage exactly / at-least N bonds.
      - Do a few steps of gradient descent on ss_t_2 w.r.t. this energy (generation-time only).

    Config schema (all optional, with defaults):
      name: soft_bond_count
      mode: "exact_N" | "at_least_N"
        # exact_N: encourage soft count C ≈ target_N (MSE/KL loss)
        # at_least_N: encourage C >= target_N
      target_N: 1          # desired number of bonds (soft)
      alpha: 20.0          # sigmoid sharpness around tau
      tau: 0.5             # threshold around which "bond" is counted
      eta: 0.1             # gradient step size
      n_steps: 1           # how many inner GD steps per sampling step
      sinkhorn_iters: 5    # Sinkhorn iterations per projection
      eps: 1e-8            # numerical epsilon for normalisation
      top_k_soft: 0        # if > 0, use top-k soft counting instead of sum
      power_sum_beta: 0    # if > 1, use power-sum counting: C = sum(P_ij^beta)
                           # priority: top_k_soft > power_sum_beta > sum
      region: null         # Optional region specification to restrict guidance application
                           # Supports formats similar to bond_condition:
                           #   - List of residue indices: [3, 9, 13, 21, 26, 28]
                           #   - Range string: "3-28" or "3:28"
                           #   - Multiple ranges: ["3-10", "20-28"]
                           #   - bond_condition-like format: "A|A", "A|B", "A100-A200|B100-B200"
                           #     (resolved against design/full_pdb_idx by default; set
                           #      reference_space: origin to resolve against full_origin_pdb_idx)
      reference_space: "design"  # "design" -> full_pdb_idx, "origin" -> full_origin_pdb_idx
      region_mode: "include"  # "include" (only apply to region) or "exclude" (apply everywhere except region)
      min_bond_distance: 0    # Minimum distance (in residues) for bonds. Bonds between residues with 
                              # distance <= min_bond_distance will be suppressed. 0 = no suppression.
                              # For example, min_bond_distance=3 suppresses bonds between residues 
                              # that are within 3 positions of each other (|i-j| <= 3).
      suppress_adjacent_weight: 1.0  # Weight for suppressing adjacent bonds in energy function
    """

    def __init__(self, cfg: Optional[Any] = None, device: str = "cpu") -> None:
        super().__init__(cfg, device)
        self.mode = str(getattr(cfg, "mode", "exact_N")).lower() if cfg is not None else "exact_N"
        self.target_N = float(getattr(cfg, "target_N", 1.0)) if cfg is not None else 1.0
        self.alpha = float(getattr(cfg, "alpha", 20.0)) if cfg is not None else 20.0
        self.tau = float(getattr(cfg, "tau", 0.5)) if cfg is not None else 0.5
        
        # 新增 top_k_soft 参数
        self.top_k_soft = int(getattr(cfg, "top_k_soft", 0)) if cfg is not None else 0
        # Power-Sum 参数：如果 > 1，使用 C = sum(P_ij^beta) 进行软计数
        # 优先级：top_k_soft > power_sum_beta > sum
        self.power_sum_beta = float(getattr(cfg, "power_sum_beta", 0.0)) if cfg is not None else 0.0
        # 熵正则化权重：鼓励概率值接近 0 或 1（二值化）
        self.entropy_weight = float(getattr(cfg, "entropy_weight", 0.0)) if cfg is not None else 0.0

        # eta -> weight
        self.weight = float(getattr(cfg, "weight",)) if cfg is not None else 0.1
        
        # 时间调度方式：linear / quadratic / cosine / exp / inverse
        self.schedule = str(getattr(cfg, "schedule", "linear")).lower() if cfg is not None else "linear"
        self.schedule_power = float(getattr(cfg, "power", 1.0)) if cfg is not None else 1.0
        
        # 最小时间阈值：当 t < min_t 时，禁用 guidance 以避免采样后期的不稳定性
        self.min_t = float(getattr(cfg, "min_t", 0.0)) if cfg is not None else 0.0
        
        self.n_steps = int(getattr(cfg, "n_steps", 1)) if cfg is not None else 1
        self.sinkhorn_iters = int(getattr(cfg, "sinkhorn_iters", 30)) if cfg is not None else 30
        self.eps = float(getattr(cfg, "eps", 1e-6)) if cfg is not None else 1e-6
        
        # 实例化 DSM 模块，tau=1.0 配合 log 输入
        self.dsm = DSMProjection(base_tau = self.tau, max_iter=self.sinkhorn_iters, eps=self.eps)
        
        # 区域配置：用于限制 guidance 的应用范围
        self.region = getattr(cfg, "region", None) if cfg is not None else None
        self.reference_space = str(
            getattr(cfg, "reference_space", getattr(cfg, "region_reference_space", "design"))
        ).lower() if cfg is not None else "design"
        self.region_mode = str(getattr(cfg, "region_mode", "include")).lower() if cfg is not None else "include"
        
        # 相邻残基抑制配置：抑制距离过近的残基之间的键
        self.min_bond_distance = int(getattr(cfg, "min_bond_distance", 0)) if cfg is not None else 0
        self.suppress_adjacent_weight = float(getattr(cfg, "suppress_adjacent_weight", 1.0)) if cfg is not None else 1.0

    @staticmethod
    def _parse_region(
        region_spec: Any,
        L: int,
        *,
        ref_idx: Optional[List] = None,
        full_pdb_idx: Optional[List] = None,
        index_base: int = 0
    ) -> Optional[torch.Tensor]:
        """
        Parse region specification into a 2D boolean mask [L, L] for bond matrix.
        
        Supports:
          - List of residue indices: [3, 9, 13] -> mask[i,j] = True if i in list and j in list
          - Range string: "3-28" or "3:28" -> mask[i,j] = True if i,j in range
          - Multiple ranges: ["3-10", "20-28"] -> union of ranges
          - bond_condition-like: "A|A", "A|B", "A100-A200|B100-B200" (requires a reference index list)
        
        Args:
            region_spec: Region specification (see formats above)
            L: Sequence length
            ref_idx: Optional list of (chain_id, res_num) tuples for bond_condition-like parsing
            full_pdb_idx: Backward-compatible alias for ref_idx
            index_base: 0-based (0) or 1-based (1) indexing
        
        Returns:
            Boolean mask [L, L] where True indicates positions to apply guidance, or None if invalid
        """
        if region_spec is None:
            return None
        
        if L <= 0:
            return torch.zeros((0, 0), dtype=torch.bool)
        
        # Initialize mask
        region_mask_1d = torch.zeros((L,), dtype=torch.bool)
        region_mask_2d_accum: Optional[torch.Tensor] = None
        ref_lookup = ref_idx if ref_idx is not None else full_pdb_idx
        
        def norm_idx(x: int) -> Optional[int]:
            """Normalize index to 0-based."""
            ii = int(x) - int(index_base)
            if ii < 0:
                ii = L + ii  # allow negative indexing from end
            if 0 <= ii < L:
                return ii
            return None
        
        def parse_bond_condition_like(spec_str: str) -> Optional[torch.Tensor]:
            """Parse bond_condition-like format: "A|A", "A|B", "A100-A200|B100-B200"."""
            if ref_lookup is None or len(ref_lookup) != L:
                return None
            
            try:
                parts = spec_str.split(':')
                res_parts = parts[0].split('|')
                if len(res_parts) != 2:
                    return None
                
                res1_spec, res2_spec = res_parts[0].strip(), res_parts[1].strip()
                
                def get_indices(spec):
                    """Get residue indices from spec like 'A', 'A100-A200', 'A/100-A/200'."""
                    if len(spec) == 1:  # Chain, e.g. 'A'
                        return [i for i, p_idx in enumerate(ref_lookup) if p_idx[0] == spec]
                    
                    range_parts = spec.split('-')
                    if len(range_parts) != 2:
                        return []
                    
                    def get_res_idx(res_spec):
                        # Support "A/2" and "A2"
                        if '/' in res_spec:
                            chain_id, token = res_spec.split('/', 1)
                        else:
                            chain_id, token = res_spec[0], res_spec[1:]
                        
                        if token == 'start':
                            chain_indices = [i for i, p_idx in enumerate(ref_lookup) if p_idx[0] == chain_id]
                            return min(chain_indices) if chain_indices else -1
                        elif token == 'end':
                            chain_indices = [i for i, p_idx in enumerate(ref_lookup) if p_idx[0] == chain_id]
                            return max(chain_indices) if chain_indices else -1
                        else:
                            try:
                                res_num = int(token)
                                try:
                                    return ref_lookup.index((chain_id, res_num))
                                except ValueError:
                                    return -1
                            except ValueError:
                                return -1
                    
                    start_idx = get_res_idx(range_parts[0])
                    end_idx = get_res_idx(range_parts[1])
                    
                    if start_idx == -1 or end_idx == -1:
                        return []
                    
                    return list(range(start_idx, end_idx + 1))
                
                indices1 = get_indices(res1_spec)
                indices2 = get_indices(res2_spec)
                
                # Create 2D mask: True for pairs (i,j) where i in indices1 and j in indices2
                mask_2d = torch.zeros((L, L), dtype=torch.bool)
                for i in indices1:
                    for j in indices2:
                        if 0 <= i < L and 0 <= j < L:
                            mask_2d[i, j] = True
                            mask_2d[j, i] = True  # Symmetric
                
                return mask_2d
            except Exception:
                return None
        
        # Handle different input types
        if isinstance(region_spec, str):
            # Try bond_condition-like format first
            if '|' in region_spec:
                mask_2d = parse_bond_condition_like(region_spec)
                if mask_2d is not None:
                    return mask_2d
            
            # Try range format: "3-28" or "3:28"
            sep = "-" if ("-" in region_spec) else (":" if (":" in region_spec) else None)
            if sep is not None:
                parts = [p.strip() for p in region_spec.split(sep) if p.strip()]
                if len(parts) == 2:
                    try:
                        a = int(parts[0])
                        b = int(parts[1])
                        ia = norm_idx(a)
                        ib = norm_idx(b)
                        if ia is not None and ib is not None:
                            lo, hi = (ia, ib) if ia <= ib else (ib, ia)
                            region_mask_1d[lo : hi + 1] = True
                    except Exception:
                        pass
            else:
                # Single index string
                try:
                    ii = norm_idx(int(region_spec))
                    if ii is not None:
                        region_mask_1d[ii] = True
                except Exception:
                    pass
        
        elif isinstance(region_spec, (list, tuple)):
            # List of indices or range strings
            for item in region_spec:
                if isinstance(item, (int, float)):
                    ii = norm_idx(int(item))
                    if ii is not None:
                        region_mask_1d[ii] = True
                elif isinstance(item, str):
                    # Try bond_condition-like format first for strings such as
                    # "A/113-A/113|B/start-B/end", which also contain '-'.
                    if '|' in item:
                        mask_2d = parse_bond_condition_like(item)
                        if mask_2d is not None:
                            region_mask_2d_accum = mask_2d if region_mask_2d_accum is None else (region_mask_2d_accum | mask_2d)
                        continue
                    # Try range format
                    sep = "-" if ("-" in item) else (":" if (":" in item) else None)
                    if sep is not None:
                        parts = [p.strip() for p in item.split(sep) if p.strip()]
                        if len(parts) == 2:
                            try:
                                a = int(parts[0])
                                b = int(parts[1])
                                ia = norm_idx(a)
                                ib = norm_idx(b)
                                if ia is not None and ib is not None:
                                    lo, hi = (ia, ib) if ia <= ib else (ib, ia)
                                    region_mask_1d[lo : hi + 1] = True
                            except Exception:
                                pass
        # Convert 1D mask to 2D: mask[i,j] = True if both i and j are in region
        region_mask_2d = None
        if region_mask_1d.any():
            region_mask_2d = region_mask_1d.unsqueeze(0) & region_mask_1d.unsqueeze(1)

        if region_mask_2d_accum is not None:
            return region_mask_2d_accum if region_mask_2d is None else (region_mask_2d_accum | region_mask_2d)
        if region_mask_2d is not None:
            return region_mask_2d

        return None

    def post_step(self, step_out: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        """
        Generation-time guidance:
          - does NOT update model parameters,
          - only refines ss_t_2 by a few steps of gradient descent on a soft bond-count energy.
        """
        ss_t_2: Optional[torch.Tensor] = step_out.get("ss_t_2", None)
        t_1: Optional[torch.Tensor] = context.get("t_1", None)
        if ss_t_2 is None or t_1 is None:
            return step_out
            
        masks = context.get("masks", {})
        bond_mask = masks.get("bond_mask", None) # [B, L, L]

        B, L, _ = ss_t_2.shape
        device = ss_t_2.device
        
        # Parse region mask if specified
        region_mask_2d = None
        if self.region is not None:
            # Resolve region references against design/full_pdb_idx by default, or
            # origin/full_origin_pdb_idx when reference_space=origin.
            full_pdb_idx = _resolve_full_pdb_idx_from_context(context, L)
            origin_pdb_idx = _resolve_origin_pdb_idx_from_context(context, L)
            region_ref_idx = origin_pdb_idx if self.reference_space == "origin" else full_pdb_idx
            region_mask_2d = self._parse_region(
                self.region,
                L,
                ref_idx=region_ref_idx,
                full_pdb_idx=full_pdb_idx,
                index_base=0
            )
            if region_mask_2d is not None:
                region_mask_2d = region_mask_2d.to(device=device)
                # Apply region_mode: "exclude" means invert the mask
                if self.region_mode in ("exclude", "outside", "except"):
                    region_mask_2d = ~region_mask_2d
                # Expand to batch dimension: [L, L] -> [B, L, L]
                region_mask_2d = region_mask_2d.unsqueeze(0).expand(B, -1, -1)
        
        # Create adjacent residue suppression mask if specified
        adjacent_mask_2d = None
        if self.min_bond_distance > 0:
            # Create mask for residues within min_bond_distance
            # adjacent_mask[i, j] = True if |i - j| <= min_bond_distance AND i != j (exclude diagonal)
            indices = torch.arange(L, device=device)
            i_grid, j_grid = torch.meshgrid(indices, indices, indexing='ij')
            distance = torch.abs(i_grid - j_grid)
            # Exclude diagonal: only include positions where |i - j| > 0
            eye_2d = torch.eye(L, device=device, dtype=torch.bool)
            adjacent_mask_2d = (distance <= self.min_bond_distance) & (~eye_2d)
            adjacent_mask_2d = adjacent_mask_2d.to(device=device)
            # Expand to batch dimension: [L, L] -> [B, L, L]
            adjacent_mask_2d = adjacent_mask_2d.unsqueeze(0).expand(B, -1, -1)

        # 时间调度：当 1-t < min_t 时，用 min_t 替代 1-t 来计算权重
        # 这样可以确保在采样早期（t接近1，1-t接近0）时，guidance仍然有一个最小权重
        if self.min_t > 0:
            # 计算 1-t_1，并确保它不小于 min_t
            one_minus_t = 1.0 - t_1  # [B]
            one_minus_t_clamped = torch.clamp(one_minus_t, min=self.min_t)  # [B]
            # 调整 t_1，使得 1-t_1 = one_minus_t_clamped
            t_1_adjusted = 1.0 - one_minus_t_clamped  # [B]
        else:
            t_1_adjusted = t_1

        # Time-dependent step size: weight_t = schedule(t_1) * weight
        # 以前叫 eta_t，现在改叫 weight_t，逻辑不变
        # 使用配置中的 schedule 参数，而不是硬编码 "linear"
        weight_t = _schedule_weight(t_1_adjusted, self.weight, schedule=self.schedule, power=self.schedule_power)  # [B]
        weight_t = weight_t.view(B, 1, 1)

        # Start from current matrix; we will refine it locally.
        
        # 1. 初始化切空间变量 ss_logits
        # ss_t_2 是概率值 [0, 1]，我们需要将其映射到无约束的 logits 空间。
        # 逆 Sinkhorn 并不简单，但作为初始值，我们可以用 tau * log(ss + eps) 近似。
        # 注意：这里我们优化的变量是 ss_logits，而不是 ss。
        ss_val = ss_t_2.detach()
        # 避免 log(0)
        ss_val = ss_val.clamp(min=self.eps)
        # 初始化 logits。tau 乘因子是为了配合 DSMProjection 内部的 /tau 操作，使其量级合理。
        ss_logits = (self.tau * torch.log(ss_val)).clone().detach().requires_grad_(True)
        
        ss_orig = ss_val.clone() # Keep original for masking restore
        
        for _ in range(max(self.n_steps, 0)):
            with torch.enable_grad():
                # 2. 前向计算 P = Sinkhorn(ss_logits)
                # 这样得到的 P 天然满足 DSM 约束
                P = self.dsm(ss_logits, mask_2d=bond_mask, mat_true=ss_orig)

                # Remove diagonal when counting bonds
                eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
                P_off = P.masked_fill(eye, 0.0)
                
                # Apply region mask if specified: only count bonds within the specified region
                if region_mask_2d is not None:
                    P_off = P_off * region_mask_2d.float()
                
                # Note: Energy calculation now respects region_mask if specified.
                # If region_mask is None, behavior is global (counts all bonds).
                
                # Add energy penalty for adjacent bonds if suppression is enabled
                adjacent_penalty = None
                if adjacent_mask_2d is not None and self.suppress_adjacent_weight > 0:
                    # Penalize high probabilities in adjacent positions
                    # Use log penalty: -log(1 - P) to strongly suppress adjacent bonds
                    # When P is close to 0, penalty is small; when P is close to 1, penalty is large
                    # Note: adjacent_mask_2d already excludes diagonal, so we can use it directly
                    P_adjacent = P_off * adjacent_mask_2d.float()
                    # Clamp to avoid log(0) and log(1)
                    P_adjacent_safe = P_adjacent.clamp(min=self.eps, max=1.0 - self.eps)
                    # Penalty: -log(1 - P) encourages P to be close to 0
                    # Sum over all adjacent positions (excluding diagonal); use sum (not mean) for classifier guidance sampling
                    adjacent_penalty = (-torch.log(1.0 - P_adjacent_safe)).sum(dim=(1, 2)).sum() * self.suppress_adjacent_weight
                
                # 计数方法优先级：top_k_soft > power_sum_beta > sum
                if self.top_k_soft > 0:
                    # 如果启用了 top_k_soft，只取前 k 个最大的概率值来计算 C
                    flat_probs = P_off.reshape(P_off.shape[0], -1)
                    # topk values: [B, k]
                    top_vals, _ = torch.topk(flat_probs, k=min(self.top_k_soft, flat_probs.shape[1]), dim=1)
                    C = 0.5 * top_vals.sum(dim=1) # [B]
                elif self.power_sum_beta > 1.0:
                    # Power-Sum 方法：C = sum(P_ij^beta)，通过幂运算抑制噪音，放大强信号
                    # beta > 1 时，小概率值会被进一步压缩，大概率值会被放大
                    P_powered = torch.pow(P_off, self.power_sum_beta)
                    C = 0.5 * P_powered.sum(dim=(1, 2))  # [B]
                else:
                    # 默认使用简单的求和
                    C = 0.5 * P_off.sum(dim=(1, 2))  # [B]

                if self.mode == "exact_n":
                    # Encourage C ≈ target_N; use sum (not mean) for classifier guidance sampling
                    energy = ((C - self.target_N) ** 2).sum()
                    diff = None
                elif self.mode == "at_least_n":
                    # Encourage C >= target_N (no penalty when already above); use sum (not mean) for classifier guidance sampling
                    diff = torch.relu(self.target_N - C)
                    energy = (diff ** 2).sum()
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                    return step_out

                # Add adjacent bond suppression penalty if enabled
                if adjacent_penalty is not None:
                    energy = energy + adjacent_penalty

                # 公共：熵正则化与统计（exact_n 和 at_least_n 逻辑一致）
                # 创建 mask：与计算 C 时保持一致，排除对角线，可选的 bond_mask 过滤
                eye_bool = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
                valid_mask_stats = (~eye_bool)
                valid_mask_entropy = (~eye_bool)
                if bond_mask is not None:
                    valid_mask_entropy = valid_mask_entropy & bond_mask.bool()

                entropy_reg = None
                entropy_stats = None
                if self.entropy_weight > 0:
                    entropy_reg = _compute_entropy_regularization(P_off, mask=valid_mask_entropy, eps=self.eps)
                    print("entropy_reg: ", entropy_reg.item())
                    energy = energy + self.entropy_weight * entropy_reg

                    # 统计不接近 0 或 1 的元素（全局统计，与 C 计算一致）
                    P_masked = P_off * valid_mask_stats.float()  # [B, L, L]
                    not_binary_mask = (P_masked > 0.1) & (P_masked < 0.9)  # [B, L, L]
                    n_not_binary = not_binary_mask.float().sum(dim=(1, 2))  # [B]
                    near_zero_mask = (P_masked <= 0.1) & valid_mask_stats  # [B, L, L]
                    n_near_zero = near_zero_mask.float().sum(dim=(1, 2))  # [B]
                    near_one_mask = (P_masked >= 0.9) & valid_mask_stats  # [B, L, L]
                    n_near_one = near_one_mask.float().sum(dim=(1, 2))  # [B]
                    mid_range_mask = (P_masked > 0.2) & (P_masked < 0.8) & valid_mask_stats  # [B, L, L]
                    n_mid_range = mid_range_mask.float().sum(dim=(1, 2))  # [B]
                    significant_mask = (P_masked > 0.01) & valid_mask_stats  # [B, L, L]
                    n_significant = significant_mask.float().sum(dim=(1, 2))  # [B]
                    significant_probs = P_masked[significant_mask.bool()] if significant_mask.any() else P_masked.new_tensor([])
                    if significant_probs.numel() > 0:
                        significant_mean = significant_probs.mean().item()
                        significant_sum = significant_probs.sum().item() / B  # 每个 batch 的平均和
                    else:
                        significant_mean = 0.0
                        significant_sum = 0.0

                    entropy_stats = {
                        'n_not_binary': n_not_binary.mean().item(),
                        'n_near_zero': n_near_zero.mean().item(),
                        'n_near_one': n_near_one.mean().item(),
                        'n_mid_range': n_mid_range.mean().item(),
                        'n_significant': n_significant.mean().item(),
                        'significant_mean': significant_mean,
                        'significant_sum': significant_sum,
                    }

                # 打印 loss 信息
                C_mean = C.mean().item()
                C_std = C.std().item() if B > 1 else 0.0
                if self.top_k_soft > 0:
                    count_method = f"topk(k={self.top_k_soft})"
                elif self.power_sum_beta > 1.0:
                    count_method = f"power_sum(beta={self.power_sum_beta:.2f})"
                else:
                    count_method = "sum"
                if self.entropy_weight > 0 and entropy_reg is not None:
                    entropy_info = f", entropy_reg={entropy_reg.item():.6f}"
                    if entropy_stats is not None:
                        entropy_info += f" [not_binary(0.1-0.9):{entropy_stats['n_not_binary']:.1f}, near_zero(<0.1):{entropy_stats['n_near_zero']:.1f}, near_one(>0.9):{entropy_stats['n_near_one']:.1f}, mid_range(0.2-0.8):{entropy_stats['n_mid_range']:.1f}]"
                        entropy_info += f" | significant(>0.01): n={entropy_stats['n_significant']:.1f}, mean={entropy_stats['significant_mean']:.3f}, sum={entropy_stats['significant_sum']:.3f}"
                else:
                    entropy_info = ""
                if self.mode == "exact_n":
                    print(f"[SoftBondCountGuidance exact_N] count_method={count_method}, C_mean={C_mean:.4f}, C_std={C_std:.4f}, target_N={self.target_N:.4f}, energy={energy.item():.6f}{entropy_info}")
                else:
                    diff_mean = diff.mean().item()
                    print(f"[SoftBondCountGuidance at_least_N] count_method={count_method}, C_mean={C_mean:.4f}, C_std={C_std:.4f}, target_N={self.target_N:.4f}, diff_mean={diff_mean:.4f}, energy={energy.item():.6f}{entropy_info}")

                # 3. 对 ss_logits 求导
                if torch.isnan(energy).any():
                    print("energy is nan")
                grad_logits, = torch.autograd.grad(
                    energy, ss_logits, retain_graph=False, create_graph=False
                )
                # 判断 grad_logits 是否全为 0
                if torch.all(grad_logits == 0):
                    print("[SoftBondCountGuidance Debug] grad_logits is all zero.")
            # 4. 在切空间更新 ss_logits
            # 注意：如果 bond_mask 或 region_mask 存在，虽然 DSM 会处理前向的屏蔽，但更新 logits 时也可以
            # 显式屏蔽梯度以避免无关区域的 logits 漂移（虽然它们在前向时会被 mask 掉）。
            if bond_mask is not None:
                grad_logits = grad_logits * bond_mask.float()
            # Also apply region_mask to gradients
            if region_mask_2d is not None:
                grad_logits = grad_logits * region_mask_2d.float()
            # Suppress gradients for adjacent positions if suppression is enabled
            if adjacent_mask_2d is not None and self.suppress_adjacent_weight > 0:
                # Strongly suppress gradients for adjacent positions
                grad_logits = grad_logits * (~adjacent_mask_2d.bool()).float()

            # 使用 weight_t 作为步长
            # 注意：这里的步长尺度可能需要根据 logits 的性质微调，但通常 weight_t 即可。
            # 检查 grad_logits 是否包含 NaN
            if torch.isnan(grad_logits).any():
                print("[SoftBondCountGuidance Debug] grad_logits contains NaN!")
            ss_logits = ss_logits - weight_t * grad_logits

        # Final projection to get valid probability matrix
        with torch.no_grad():
            ss_projected = self.dsm(ss_logits, mask_2d=bond_mask, mat_true=ss_orig)
            
        step_out["ss_t_2"] = ss_projected.detach()
        return step_out


class TypeAwareSoftBondCountGuidance(Guidance):
    """
    Type-aware soft bond-count guidance on the model's *predicted* bond matrix and sequence logits.

    目标：针对不同键类型（如二硫键 / 异肽键 / 内酯键 / 泛共价 covalent）约束“键的数量”，而不指定具体位置，
    并通过同一个能量函数同时对 bond 矩阵和序列 logits 做小步梯度更新。
    计数能量使用 Poisson-KL 形式：
      - exact_N:   对所有 batch 最小化 KL(Pois(N) || Pois(C_tau))
      - at_least_N: 仅对 C_tau < N 的 batch 最小化 KL，C_tau >= N 时 loss=0

    配置示例（OmegaConf）：

      guidance:
        - name: type_soft_bond_count
          link_csv_path: /path/to/link.csv
          bond_step: 0.1   # base step size for bond matrix (time-scheduled)
          seq_step: 0.05   # base step size for logits guidance (time-scheduled)
          n_steps: 1
          sinkhorn_iters: 5
          schedule: linear # "linear" | "quadratic" | "cosine" | "exp"
          power: 1.0       # only used when schedule == "exp"
          types:
            - name: disulfide
              mode: exact_N        # "exact_N" | "at_least_N" | "range_N" | "fixed_pairs" | "only_fixed_pairs"
              target_N: 1.0       # exact_N / at_least_N 使用
              min_N: 1.0          # range_N 使用：软计数下界
              max_N: 2.0          # range_N 使用：软计数上界
              weight: 1.0
              # 计数方法选项：
              top_k_soft: 0        # 如果 > 0，使用 top-k 软计数而不是全部求和
              power_sum_beta: 0    # 如果 > 1，使用 C = sum(P_ij^beta) 进行软计数（优先级：top_k_soft > power_sum_beta > sum）
          region: null             # Optional region specification (same format as SoftBondCountGuidance)
          reference_space: "design" # "design" -> full_pdb_idx, "origin" -> full_origin_pdb_idx
          region_mode: "include"   # "include" (only apply to region) or "exclude" (apply everywhere except region)
          power_sum_beta: 0        # Global power_sum_beta (can be overridden per type)
          min_bond_distance: 0     # Minimum bond distance (suppress bonds between residues with |i-j| <= min_bond_distance)
          suppress_adjacent_weight: 1.0  # Weight for suppressing adjacent bonds
          debug: false             # When true, print per-type C_tau/target/E_tau, gradients, bond matrix stats
          debug_print_every: 1     # Print every N guidance calls (1 = every call)

            - name: isopeptide
              mode: at_least_N
              target_N: 0.0
              weight: 0.5

            - name: lactone
              mode: at_least_N
              target_N: 0.0
              weight: 0.5

            - name: covalent     # 泛共价：包括所有侧链-侧链规则 + 端基相关规则（仅在真实端基位置上生效）
              # fixed_pairs: 只“偏好”指定 pairs，有键也允许在其他位点出现
              # only_fixed_pairs: 尽量把键全部压到指定 pairs，其它位置的该类型键强烈压制为 0
              mode: fixed_pairs
              target_N: 1.0
              weight: 1.0
              # pairs 以 (i,j) 的 0-based 残基下标指定“希望成键”的位置，例如序列第 0 与第 15 个残基闭环：
              pairs:
                - [0, 15]

    注意：
      - 该模块在 pre_model 阶段工作，直接修改 model_raw["bond_mat_pred"] 和 model_raw["logits"]。
      - 不对模型参数求梯度，仅在采样时对中间变量做几步能量下降（generation-time guidance）。
      - 键类型由 link.csv 中的 bond_spec 规则自动解析：
          * disulfide / isopeptide / lactone 等通过侧链原子模式分类；
          * covalent 统计所有“非 N/C 端基”的侧链共价规则 + 端基相关规则，
            其中端基相关规则只在 head_mask/tail_mask 指示的真实 N/C 端位置上起作用，
            不会在序列中部误当作闭环键。
    """

    _SUPPORTED_TYPES = ("disulfide", "isopeptide", "lactone", "covalent")

    def __init__(self, cfg: Optional[Any] = None, device: str = "cpu") -> None:
        super().__init__(cfg, device)
        
        # 阶段选择: "pre_model", "post_step", "both"
        self.stage = str(getattr(cfg, "stage", "pre_model")).lower() if cfg is not None else "pre_model"

        # 超参数
        # bond_step: when None (yaml null), we switch to "hard bond matrix" mode (no gradient update on bonds).
        if cfg is not None:
            _bond_step_raw = getattr(cfg, "bond_step", getattr(cfg, "eta", 0.1))
            self.bond_step: Optional[float] = None if _bond_step_raw is None else float(_bond_step_raw)
        else:
            self.bond_step = 0.1
        self.n_steps = int(getattr(cfg, "n_steps", 1)) if cfg is not None else 1
        self.sinkhorn_iters = int(getattr(cfg, "sinkhorn_iters", 5)) if cfg is not None else 5
        self.eps = float(getattr(cfg, "eps", 1e-4)) if cfg is not None else 1e-4
        # seq_step: when None (yaml null), we switch to "no-grad direct assignment" for fixed-pair modes.
        if cfg is not None:
            _seq_step_raw = getattr(cfg, "seq_step", 0.05)
            self.seq_step: Optional[float] = None if _seq_step_raw is None else float(_seq_step_raw)
        else:
            self.seq_step = 0.05
        self.schedule = str(getattr(cfg, "schedule", "linear")) if cfg is not None else "linear"
        self.power = float(getattr(cfg, "power", 1.0)) if cfg is not None else 1.0
        self.tau = float(getattr(cfg, "tau", 0.5)) if cfg is not None else 0.5
        # 熵正则化权重：鼓励概率值接近 0 或 1（二值化）
        self.entropy_weight = float(getattr(cfg, "entropy_weight", 0.0)) if cfg is not None else 0.0
        self.entropy_mode = str(getattr(cfg, "entropy_mode", "effective")).lower() if cfg is not None else "effective"
        
        # 区域配置：用于限制 guidance 的应用范围
        self.region = getattr(cfg, "region", None) if cfg is not None else None
        self.reference_space = str(
            getattr(cfg, "reference_space", getattr(cfg, "region_reference_space", "design"))
        ).lower() if cfg is not None else "design"
        self.region_mode = str(getattr(cfg, "region_mode", "include")).lower() if cfg is not None else "include"
        
        # Power-Sum 参数：如果 > 1，使用 C = sum(P_ij^beta) 进行软计数
        self.power_sum_beta = float(getattr(cfg, "power_sum_beta", 1.0)) if cfg is not None else 1.0
        
        # 相邻残基抑制配置：抑制距离过近的残基之间的键
        self.min_bond_distance = int(getattr(cfg, "min_bond_distance", 0)) if cfg is not None else 0
        self.suppress_adjacent_weight = float(getattr(cfg, "suppress_adjacent_weight", 1.0)) if cfg is not None else 1.0

        # Debug: 为 true 时打印每个类型的 C_tau、target、能量贡献、梯度统计等
        self.debug = bool(getattr(cfg, "debug", False)) if cfg is not None else False
        self.debug_print_every = int(getattr(cfg, "debug_print_every", 1)) if cfg is not None else 1
        self._debug_call_count = 0

        # 实例化 DSM 模块
        self.dsm = DSMProjection(base_tau=self.tau, max_iter=self.sinkhorn_iters, eps=self.eps)

        # link.csv
        link_csv_path = None
        if cfg is not None:
            link_csv_path = getattr(cfg, "link_csv_path", None)
        self.link_info: Optional[LinkInfo] = None
        if link_csv_path is not None:
            try:
                self.link_info = LinkInfo(link_csv_path, device=device)
            except Exception as e:
                print(f"[TypeAwareSoftBondCountGuidance] Failed to load LinkInfo from {link_csv_path}: {e}")
                self.link_info = None

        # 键类型配置
        self.type_cfgs: List[SimpleNamespace] = []
        configured_types = []
        if cfg is not None:
            configured_types = getattr(cfg, "types", None)
        if configured_types is None:
            default_type = SimpleNamespace(
                name="disulfide",
                mode="exact_N",
                target_N=1.0,
                weight=1.0,
            )
            self.type_cfgs = [default_type]
        else:
            for t_cfg in configured_types:
                name = str(getattr(t_cfg, "name", None) or getattr(t_cfg, "type", None) or "disulfide").lower()
                mode = str(getattr(t_cfg, "mode", "exact_N")).lower()
                loss_type = str(getattr(t_cfg, "loss_type", "mse")).lower()
                target_N = float(getattr(t_cfg, "target_N", 1.0))
                min_N = float(getattr(t_cfg, "min_N", getattr(t_cfg, "min_target_N", target_N)))
                max_N = float(getattr(t_cfg, "max_N", getattr(t_cfg, "max_target_N", target_N)))
                weight = float(getattr(t_cfg, "weight", 1.0))
                top_k_soft = int(getattr(t_cfg, "top_k_soft", 0))  # 新增
                # Power-Sum 参数：如果 > 1，使用 C = sum(P_ij^beta) 进行软计数
                power_sum_beta = float(getattr(t_cfg, "power_sum_beta", self.power_sum_beta))
                raw_pairs = getattr(t_cfg, "pairs", None)
                pairs: List[tuple] = []
                if raw_pairs is not None:
                    try:
                        for p in raw_pairs:
                            if p is None:
                                continue
                            if len(p) != 2:
                                continue
                            i, j = int(p[0]), int(p[1])
                            if i >= 0 and j >= 0:
                                pairs.append((i, j))
                    except Exception as e:
                        print(f"[TypeAwareSoftBondCountGuidance] Failed to parse pairs for type '{name}': {e}")
                if name not in self._SUPPORTED_TYPES:
                    print(f"[TypeAwareSoftBondCountGuidance] Unsupported type '{name}', ignoring.")
                    continue
                self.type_cfgs.append(
                    SimpleNamespace(
                        name=name,
                        mode=mode,
                        loss_type=loss_type,
                        target_N=target_N,
                        min_N=min_N,
                        max_N=max_N,
                        top_k_soft=top_k_soft,  # 新增
                        power_sum_beta=power_sum_beta,  # 新增
                        weight=weight,
                        pairs=pairs,
                    )
                )

        # 预计算
        if self.link_info is not None and getattr(self.link_info, "compat_matrix", None) is not None:
            K = int(self.link_info.compat_matrix.shape[0])
        else:
            K = 21
        self.num_aatypes = K

        self.type_pair_mats: Dict[str, torch.Tensor] = {}
        self.covalent_terminal_pairs: Optional[torch.Tensor] = None
        if self.link_info is not None and getattr(self.link_info, "bond_spec", None):
            self._build_type_pair_mats(K)
        else:
            if self.link_info is None:
                print("[TypeAwareSoftBondCountGuidance] No LinkInfo available; guidance will be a no-op.")
            else:
                print("[TypeAwareSoftBondCountGuidance] link_info.bond_spec is empty; guidance will be a no-op.")

    @staticmethod
    def _classify_rules(rules: List[Dict[str, Any]]) -> Optional[str]:
        # (保持原样)
        def norm(name: Any) -> str:
            return (str(name).strip().upper()) if name is not None else ""

        for r in rules:
            a1, a2 = norm(r.get("atom1")), norm(r.get("atom2"))
            if a1 == "SG" and a2 == "SG":
                return "disulfide"
        for r in rules:
            a1, a2 = norm(r.get("atom1")), norm(r.get("atom2"))
            if (a1 in ("OG", "OG1", "OH") and a2 in ("CG", "CD")) or (
                a2 in ("OG", "OG1", "OH") and a1 in ("CG", "CD")
            ):
                return "lactone"
        for r in rules:
            a1, a2 = norm(r.get("atom1")), norm(r.get("atom2"))
            if (a1 == "NZ" and a2 in ("CG", "CD")) or (a2 == "NZ" and a1 in ("CG", "CD")):
                return "isopeptide"
        return None

    def _build_type_pair_mats(self, K: int) -> None:
        # (保持原样)
        device = self.device
        type_mats = {
            "disulfide": torch.zeros((K, K), dtype=torch.bool, device=device),
            "isopeptide": torch.zeros((K, K), dtype=torch.bool, device=device),
            "lactone": torch.zeros((K, K), dtype=torch.bool, device=device),
            "covalent": torch.zeros((K, K), dtype=torch.bool, device=device),
        }
        covalent_terminal = torch.zeros((K, K), dtype=torch.bool, device=device)

        for (r1, r2), rules in self.link_info.bond_spec.items():
            if r1 >= K or r2 >= K:
                continue
            has_sidechain_rule = False
            has_terminal_rule = False
            for r in rules:
                a1 = (r.get("atom1") or "").strip().upper()
                a2 = (r.get("atom2") or "").strip().upper()
                if a1 not in ("N", "C") and a2 not in ("N", "C"):
                    has_sidechain_rule = True
                if (a1 in ("N", "C")) or (a2 in ("N", "C")):
                    has_terminal_rule = True

            if has_sidechain_rule:
                type_mats["covalent"][r1, r2] = True
                type_mats["covalent"][r2, r1] = True
            if has_terminal_rule:
                covalent_terminal[r1, r2] = True
                covalent_terminal[r2, r1] = True

            bond_type = self._classify_rules(rules)
            if bond_type is None:
                continue
            if bond_type not in type_mats:
                continue
            type_mats[bond_type][r1, r2] = True
            type_mats[bond_type][r2, r1] = True

        self.type_pair_mats = type_mats
        self.covalent_terminal_pairs = covalent_terminal

    @staticmethod
    def _gather_fixed_pairs(type_cfgs: List[SimpleNamespace]) -> List[tuple]:
        """Collect union of (i,j) pairs from type configs with fixed/only_fixed modes."""
        pairs: List[tuple] = []
        for t_cfg in type_cfgs or []:
            mode = str(getattr(t_cfg, "mode", "")).lower()
            if mode not in ("fixed_pairs", "only_fixed_pairs"):
                continue
            for p in (getattr(t_cfg, "pairs", None) or []):
                if p is None:
                    continue
                try:
                    ii, jj = int(p[0]), int(p[1])  # support tuple/list
                except Exception:
                    continue
                pairs.append((ii, jj))
        return pairs

    @staticmethod
    def _build_fixed_pair_matrix(
        L: int,
        res_mask: Optional[torch.Tensor],
        pairs: List[tuple],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build a hard doubly-stochastic / permutation-like matrix P_fixed [1, L, L]:
          - For unpaired valid residues: P[i,i] = 1
          - For paired residues (i,j): P[i,j] = P[j,i] = 1 and P[i,i] = P[j,j] = 0
          - All other entries are 0
        Invalid residues (res_mask=False) get all-zeros rows/cols.
        """
        if L <= 0:
            return torch.zeros((1, 0, 0), device=device, dtype=dtype)

        P = torch.zeros((L, L), device=device, dtype=dtype)
        if res_mask is None:
            valid = torch.ones((L,), device=device, dtype=torch.bool)
        else:
            # res_mask may be [B,L]; we assume consistent across batch, so use the first item.
            valid = res_mask[0].to(device=device, dtype=torch.bool) if res_mask.dim() == 2 else res_mask.to(device=device, dtype=torch.bool)
            if valid.numel() != L:
                valid = torch.ones((L,), device=device, dtype=torch.bool)

        # Start as identity on valid residues
        diag_idx = torch.arange(L, device=device)
        P[diag_idx, diag_idx] = valid.to(dtype)

        # Enforce specified disjoint pairs
        partner = torch.full((L,), -1, device=device, dtype=torch.long)
        for (ii, jj) in pairs or []:
            if ii is None or jj is None:
                continue
            i, j = int(ii), int(jj)
            if i == j:
                continue
            if not (0 <= i < L and 0 <= j < L):
                continue
            if not (bool(valid[i].item()) and bool(valid[j].item())):
                continue

            # Ensure disjoint pairing; if conflict, skip (but keep previous constraints)
            if (partner[i] not in (-1, j)) or (partner[j] not in (-1, i)):
                print(f"[TypeAwareSoftBondCountGuidance] Warning: conflicting fixed pairs for residues {i},{j}; skipping this pair.")
                continue
            partner[i] = j
            partner[j] = i

            # Clear rows/cols by overwriting with the hard assignment
            P[i, :] = 0.0
            P[:, i] = 0.0
            P[j, :] = 0.0
            P[:, j] = 0.0
            P[i, j] = 1.0
            P[j, i] = 1.0

        # Zero out invalid rows/cols (if any)
        if res_mask is not None:
            inv = ~valid
            if inv.any():
                P[inv, :] = 0.0
                P[:, inv] = 0.0

        return P.unsqueeze(0)  # [1, L, L]

    @staticmethod
    def _force_logits_onehot(
        logits: torch.Tensor,  # [B,L,C]
        pos: int,
        aa_idx: int,
        *,
        K: int,
        hi: float = 10.0,
        lo: float = -10.0,
    ) -> torch.Tensor:
        """Overwrite logits at one position to be near one-hot on first K channels."""
        if not (0 <= pos < logits.shape[1]):
            return logits
        if not (0 <= aa_idx < K):
            return logits
        logits[:, pos, :K] = lo
        logits[:, pos, aa_idx] = hi
        return logits

    def _choose_aa_pair(
        self,
        type_name: str,
        probs_i: Optional[torch.Tensor],  # [K] or None
        probs_j: Optional[torch.Tensor],  # [K] or None
        fixed_i: Optional[int] = None,
        fixed_j: Optional[int] = None,
    ) -> Optional[tuple]:
        """
        Choose (aa_i, aa_j) for a fixed residue pair.
        - If probs_i/probs_j are provided: maximize probs_i[a] * probs_j[b] over allowed type pairs.
        - Else: maximize link_info.compat_matrix[a,b] over allowed type pairs (fallback).
        Supports fixing one side (fixed_i/fixed_j).
        """
        type_mat = self.type_pair_mats.get(type_name)
        if type_mat is None or not type_mat.any():
            return None

        K = type_mat.shape[0]
        allowed = type_mat
        device = allowed.device

        # Special-case: disulfide -> force CYS/CYS when possible.
        if type_name == "disulfide":
            cys = aa2num.get("CYS", None)
            if cys is not None and 0 <= int(cys) < K:
                return (int(cys), int(cys))

        if probs_i is not None and probs_j is not None:
            pi = probs_i[:K].to(device=device)
            pj = probs_j[:K].to(device=device)
            score = pi.view(K, 1) * pj.view(1, K)
        else:
            if self.link_info is not None and getattr(self.link_info, "compat_matrix", None) is not None:
                score = self.link_info.compat_matrix[:K, :K].to(device=device)
            else:
                score = torch.ones((K, K), device=device, dtype=torch.float32)

        score = score.masked_fill(~allowed, float("-inf"))

        if fixed_i is not None:
            if not (0 <= int(fixed_i) < K):
                return None
            score = score[int(fixed_i), :].view(1, K)
            flat = score.view(-1)
            b = int(torch.argmax(flat).item())
            if not torch.isfinite(flat[b]):
                return None
            return (int(fixed_i), b)

        if fixed_j is not None:
            if not (0 <= int(fixed_j) < K):
                return None
            score = score[:, int(fixed_j)].view(K, 1)
            flat = score.view(-1)
            a = int(torch.argmax(flat).item())
            if not torch.isfinite(flat[a]):
                return None
            return (a, int(fixed_j))

        flat = score.view(-1)
        idx = int(torch.argmax(flat).item())
        if not torch.isfinite(flat[idx]):
            return None
        a = idx // K
        b = idx % K
        return (int(a), int(b))

    def _apply_direct_seq_for_fixed_pairs_pre_model(
        self,
        logits_work: torch.Tensor,  # [B,L,C]
        seq_mask: Optional[torch.Tensor],  # [B,L]
    ) -> torch.Tensor:
        """Directly assign residue types on fixed pairs when seq_step is None (no-grad mode)."""
        if self.seq_step is not None:
            return logits_work

        # Only activate if there are any fixed-pair modes configured.
        fixed_types = [t for t in (self.type_cfgs or []) if str(getattr(t, "mode", "")).lower() in ("fixed_pairs", "only_fixed_pairs")]
        if not fixed_types:
            return logits_work

        B, L, C = logits_work.shape
        K = min(self.num_aatypes, C)
        probs = F.softmax(logits_work[:, :, :K], dim=-1)

        for t_cfg in fixed_types:
            type_name = str(getattr(t_cfg, "name", "")).lower()
            pairs = getattr(t_cfg, "pairs", None) or []
            for (ii, jj) in pairs:
                i, j = int(ii), int(jj)
                if not (0 <= i < L and 0 <= j < L):
                    continue
                # respect seq_mask: only modify designable positions
                allow_i = True if seq_mask is None else bool(seq_mask[:, i].any().item())
                allow_j = True if seq_mask is None else bool(seq_mask[:, j].any().item())

                # If neither is designable, skip.
                if (not allow_i) and (not allow_j):
                    continue

                # Determine fixed side from current argmax if that side is not designable.
                fixed_i = None
                fixed_j = None
                if not allow_i:
                    fixed_i = int(torch.argmax(logits_work[0, i, :K]).item())
                if not allow_j:
                    fixed_j = int(torch.argmax(logits_work[0, j, :K]).item())

                chosen = self._choose_aa_pair(type_name, probs_i=probs[0, i], probs_j=probs[0, j], fixed_i=fixed_i, fixed_j=fixed_j)
                if chosen is None:
                    continue
                aa_i, aa_j = chosen
                if allow_i:
                    logits_work = self._force_logits_onehot(logits_work, i, aa_i, K=K)
                if allow_j:
                    logits_work = self._force_logits_onehot(logits_work, j, aa_j, K=K)

        return logits_work

    def _apply_direct_seq_for_fixed_pairs_post_step(
        self,
        aatypes: torch.Tensor,  # [B,L]
        seq_mask: Optional[torch.Tensor],  # [B,L]
    ) -> torch.Tensor:
        """Directly assign residue types on fixed pairs when seq_step is None (no-grad mode)."""
        if self.seq_step is not None:
            return aatypes

        fixed_types = [t for t in (self.type_cfgs or []) if str(getattr(t, "mode", "")).lower() in ("fixed_pairs", "only_fixed_pairs")]
        if not fixed_types:
            return aatypes

        B, L = aatypes.shape
        K = self.num_aatypes

        for t_cfg in fixed_types:
            type_name = str(getattr(t_cfg, "name", "")).lower()
            pairs = getattr(t_cfg, "pairs", None) or []
            for (ii, jj) in pairs:
                i, j = int(ii), int(jj)
                if not (0 <= i < L and 0 <= j < L):
                    continue

                allow_i = True if seq_mask is None else bool(seq_mask[:, i].any().item())
                allow_j = True if seq_mask is None else bool(seq_mask[:, j].any().item())
                if (not allow_i) and (not allow_j):
                    continue

                fixed_i = None if allow_i else int(aatypes[0, i].item())
                fixed_j = None if allow_j else int(aatypes[0, j].item())

                chosen = self._choose_aa_pair(type_name, probs_i=None, probs_j=None, fixed_i=fixed_i, fixed_j=fixed_j)
                if chosen is None:
                    continue
                aa_i, aa_j = chosen
                if 0 <= aa_i < K and allow_i:
                    aatypes[:, i] = aa_i
                if 0 <= aa_j < K and allow_j:
                    aatypes[:, j] = aa_j

        return aatypes

    def _compute_energy(
        self,
        P: torch.Tensor,
        logits: torch.Tensor,
        res_mask: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor],
        tail_mask: Optional[torch.Tensor],
        region_mask_2d: Optional[torch.Tensor] = None,
        adjacent_mask_2d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # (保持原样)
        if not self.type_cfgs:
            return P.new_tensor(0.0)

        B, L, _ = P.shape
        device = P.device

        if res_mask is None:
            res_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        pair_mask = (res_mask.unsqueeze(1) & res_mask.unsqueeze(2))
        eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & (~eye)

        P_off = P * pair_mask.float()
        K = min(self.num_aatypes, logits.shape[-1])
        probs = F.softmax(logits[..., :K], dim=-1)

        # Hard fixed-pair bond matrix is only enabled when bond_step is None.
        # In that case energy becomes independent of P/ss_logits (bond grads are None).
        fixed_pairs_all = self._gather_fixed_pairs(self.type_cfgs) if self.bond_step is None else []
        P_fixed_off: Optional[torch.Tensor] = None
        if fixed_pairs_all and self.bond_step is None:
            P_fixed = self._build_fixed_pair_matrix(
                L=L,
                res_mask=res_mask,
                pairs=fixed_pairs_all,
                device=device,
                dtype=P.dtype,
            ).expand(B, -1, -1)  # [B,L,L]
            P_fixed_off = P_fixed * pair_mask.float()

        total_energy = P.new_tensor(0.0)
        entropy_reg_accum = P.new_tensor(0.0)
        debug_info_list: List[Dict[str, Any]] = []
        for t_cfg in self.type_cfgs:
            type_name = t_cfg.name
            type_mat = self.type_pair_mats.get(type_name)
            if type_mat is None or not type_mat.any():
                continue

            type_mat_dev = type_mat.to(device=device, dtype=probs.dtype)
            Cp = torch.matmul(probs, type_mat_dev)
            compat = torch.einsum("bik,bjk->bij", Cp, probs)

            if type_name == "covalent" and self.covalent_terminal_pairs is not None:
                term_mat = self.covalent_terminal_pairs.to(device=device, dtype=probs.dtype)
                Cp_term = torch.matmul(probs, term_mat)
                compat_term = torch.einsum("bik,bjk->bij", Cp_term, probs)
                if head_mask is not None or tail_mask is not None:
                    if head_mask is None: head_mask = torch.zeros_like(res_mask, dtype=torch.bool, device=device)
                    if tail_mask is None: tail_mask = torch.zeros_like(res_mask, dtype=torch.bool, device=device)
                    term_gate = (
                        head_mask.unsqueeze(1) | head_mask.unsqueeze(2) |
                        tail_mask.unsqueeze(1) | tail_mask.unsqueeze(2)
                    )
                    compat_term = compat_term * term_gate.float()
                compat = compat + compat_term

            compat = compat * pair_mask.float()
            # Use hard fixed pairing matrix when requested; otherwise use current P.
            if P_fixed_off is not None:
                P_eff = P_fixed_off * compat
            else:
                P_eff = P_off * compat
            
            # Apply region mask if specified
            if region_mask_2d is not None:
                P_eff = P_eff * region_mask_2d.float()
            
            # Apply adjacent mask suppression if specified (exclude adjacent positions)
            if adjacent_mask_2d is not None:
                P_eff = P_eff * (~adjacent_mask_2d.bool()).float()

            if self.entropy_weight > 0 and self.entropy_mode in ("effective", "p_eff"):
                # Accumulate entropy for each type's P_eff
                curr_entropy = _compute_entropy_regularization(P_eff, mask=None, eps=self.eps)
                entropy_reg_accum = entropy_reg_accum + curr_entropy

            top_k = int(getattr(t_cfg, "top_k_soft", 0))
            # Power-Sum 参数：如果 > 1，使用 C = sum(P_ij^beta) 进行软计数
            power_sum_beta = float(getattr(t_cfg, "power_sum_beta", self.power_sum_beta))

            target = float(t_cfg.target_N)
            min_esp = 1e-8
            mode = str(getattr(t_cfg, "mode", "exact_n")).lower()
            loss_type = str(getattr(t_cfg, "loss_type", "mse")).lower()

            if mode in ("fixed_pairs", "only_fixed_pairs"):
                if not getattr(t_cfg, "pairs", None):
                    E_tau = P.new_tensor(0.0)
                else:
                    pair_mask_type = torch.zeros((L, L), dtype=P_eff.dtype, device=device)
                    for (ii, jj) in t_cfg.pairs:
                        if 0 <= ii < L and 0 <= jj < L:
                            pair_mask_type[ii, jj] = 1.0
                            pair_mask_type[jj, ii] = 1.0
                    pair_mask_type = pair_mask_type.unsqueeze(0) * pair_mask.float()

                    P_target = P_eff * pair_mask_type
                    
                    # 计数方法优先级：top_k > power_sum_beta > sum
                    if top_k > 0:
                        flat_target = P_target.reshape(B, -1)
                        top_vals, _ = torch.topk(flat_target, k=min(top_k, flat_target.shape[1]), dim=1)
                        C_target = 0.5 * top_vals.sum(dim=1)
                    elif power_sum_beta > 1.0:
                        # Power-Sum 方法：C = sum(P_ij^beta)
                        P_target_powered = torch.pow(P_target, power_sum_beta)
                        C_target = 0.5 * P_target_powered.sum(dim=(1, 2))
                    else:
                        C_target = 0.5 * P_target.sum(dim=(1, 2))

                    if loss_type == "mse":
                        E_target = (C_target - target) ** 2
                    else:
                        C_safe = torch.where(C_target < min_esp, C_target/C_target.detach()*min_esp, C_target)
                        E_target = (C_safe - target + target * torch.log(target / C_safe))

                    if mode == "fixed_pairs":
                        E_tau = E_target.sum()
                    else:
                        # New semantics: only_fixed_pairs uses a hard fixed pairing matrix (handled above),
                        # and we do NOT add extra diagonal regularization here.
                        E_tau = E_target.sum()
                    if self.debug:
                        C_mean = C_target.mean().item()
                        C_std = C_target.std().item() if C_target.numel() > 1 else 0.0
                        debug_info_list.append({"type": type_name, "mode": mode, "C_mean": C_mean, "C_std": C_std, "target": target, "E_tau": E_tau.item(), "weight": float(t_cfg.weight)})
            elif mode == "exact_n":
                # 计数方法优先级：top_k > power_sum_beta > sum
                if top_k > 0:
                    flat_eff = P_eff.reshape(B, -1)
                    top_vals, _ = torch.topk(flat_eff, k=min(top_k, flat_eff.shape[1]), dim=1)
                    C_tau = 0.5 * top_vals.sum(dim=1)
                elif power_sum_beta > 1.0:
                    # Power-Sum 方法：C = sum(P_ij^beta)
                    P_eff_powered = torch.pow(P_eff, power_sum_beta)
                    C_tau = 0.5 * P_eff_powered.sum(dim=(1, 2))
                else:
                    C_tau = 0.5 * P_eff.sum(dim=(1, 2))

                if loss_type == "mse":
                    E_tau = ((C_tau - target) ** 2).sum()
                else:
                    if target < min_esp:
                         C_safe = torch.where(C_tau < min_esp, C_tau/C_tau.detach()*min_esp, C_tau)
                         E_tau = C_safe.sum()
                    else:
                        C_safe = torch.where(C_tau < min_esp, C_tau/C_tau.detach()*min_esp, C_tau)
                        E_tau = (C_safe - target + target * torch.log(target / C_safe)).sum()
                if self.debug:
                    C_mean = C_tau.mean().item()
                    C_std = C_tau.std().item() if C_tau.numel() > 1 else 0.0
                    debug_info_list.append({"type": type_name, "mode": mode, "C_mean": C_mean, "C_std": C_std, "target": target, "E_tau": E_tau.item(), "weight": float(t_cfg.weight)})
            elif mode == "at_least_n":
                # 计数方法优先级：top_k > power_sum_beta > sum
                if top_k > 0:
                    flat_eff = P_eff.reshape(B, -1)
                    top_vals, _ = torch.topk(flat_eff, k=min(top_k, flat_eff.shape[1]), dim=1)
                    C_tau = 0.5 * top_vals.sum(dim=1)
                elif power_sum_beta > 1.0:
                    # Power-Sum 方法：C = sum(P_ij^beta)
                    P_eff_powered = torch.pow(P_eff, power_sum_beta)
                    C_tau = 0.5 * P_eff_powered.sum(dim=(1, 2))
                else:
                    C_tau = 0.5 * P_eff.sum(dim=(1, 2))

                need = C_tau < target
                if need.any():
                    if loss_type == "mse":
                        E_vals = (target - C_tau[need]) ** 2
                    else:
                        C_need = torch.where(C_tau[need] < min_esp, C_tau[need]/C_tau[need].detach()*min_esp, C_tau[need])
                        if target  < min_esp:
                            E_vals = C_need
                        else:
                            E_vals = C_need - target + target * torch.log(target / C_need)
                    E_tau = E_vals.sum()
                else:
                    E_tau = P.new_tensor(0.0)
                if self.debug:
                    C_mean = C_tau.mean().item()
                    C_std = C_tau.std().item() if C_tau.numel() > 1 else 0.0
                    debug_info_list.append({"type": type_name, "mode": mode, "C_mean": C_mean, "C_std": C_std, "target": target, "E_tau": E_tau.item(), "weight": float(t_cfg.weight)})
            elif mode in ("range_n", "range", "between_n", "between"):
                # Penalize only when the soft count is outside [min_N, max_N].
                if top_k > 0:
                    flat_eff = P_eff.reshape(B, -1)
                    top_vals, _ = torch.topk(flat_eff, k=min(top_k, flat_eff.shape[1]), dim=1)
                    C_tau = 0.5 * top_vals.sum(dim=1)
                elif power_sum_beta > 1.0:
                    P_eff_powered = torch.pow(P_eff, power_sum_beta)
                    C_tau = 0.5 * P_eff_powered.sum(dim=(1, 2))
                else:
                    C_tau = 0.5 * P_eff.sum(dim=(1, 2))

                lower = float(getattr(t_cfg, "min_N", target))
                upper = float(getattr(t_cfg, "max_N", target))
                if upper < lower:
                    lower, upper = upper, lower
                below = torch.relu(C_tau.new_tensor(lower) - C_tau)
                above = torch.relu(C_tau - C_tau.new_tensor(upper))
                E_tau = (below.pow(2) + above.pow(2)).sum()
                if self.debug:
                    C_mean = C_tau.mean().item()
                    C_std = C_tau.std().item() if C_tau.numel() > 1 else 0.0
                    debug_info_list.append({
                        "type": type_name,
                        "mode": mode,
                        "C_mean": C_mean,
                        "C_std": C_std,
                        "target": 0.5 * (lower + upper),
                        "target_min": lower,
                        "target_max": upper,
                        "E_tau": E_tau.item(),
                        "weight": float(t_cfg.weight),
                    })
            else:
                raise ValueError(f"Unknown mode: {mode}")
            total_energy = total_energy + float(t_cfg.weight) * E_tau
        
        if self.debug:
            self._last_debug_info = debug_info_list
        else:
            self._last_debug_info = None

        # Add adjacent bond suppression penalty if enabled
        if adjacent_mask_2d is not None and self.suppress_adjacent_weight > 0:
            # Penalize high probabilities in adjacent positions using log penalty
            P_adjacent = P_off * adjacent_mask_2d.float()
            # Clamp to avoid log(0) and log(1)
            P_adjacent_safe = P_adjacent.clamp(min=self.eps, max=1.0 - self.eps)
            # Penalty: -log(1 - P) encourages P to be close to 0; use sum (not mean) for classifier guidance sampling
            adjacent_penalty = (-torch.log(1.0 - P_adjacent_safe)).sum(dim=(1, 2)).sum() * self.suppress_adjacent_weight
            total_energy = total_energy + adjacent_penalty
        
        # 添加熵正则化：鼓励 P_off 或 P_eff 接近 0 或 1（二值化）
        entropy_reg = None
        entropy_stats = None
        if self.entropy_weight > 0:
            if self.entropy_mode in ("effective", "p_eff"):
                # Use accumulated entropy from P_eff
                entropy_reg = entropy_reg_accum
                total_energy = total_energy + self.entropy_weight * entropy_reg
            else:
                # Fallback to original behavior: entropy on P_off
                # 计算有效位置的熵（排除对角线）
                eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
                valid_mask_entropy = pair_mask & (~eye)
                entropy_reg = _compute_entropy_regularization(P_off, mask=valid_mask_entropy, eps=self.eps)
                total_energy = total_energy + self.entropy_weight * entropy_reg
                
                # 统计不接近0或1的元素
                P_masked = P_off * valid_mask_entropy.float()  # [B, L, L]
                not_binary_mask = (P_masked > 0.1) & (P_masked < 0.9)  # [B, L, L]
                n_not_binary = not_binary_mask.float().sum(dim=(1, 2))  # [B]
                near_zero_mask = (P_masked <= 0.1) & valid_mask_entropy  # [B, L, L]
                n_near_zero = near_zero_mask.float().sum(dim=(1, 2))  # [B]
                near_one_mask = (P_masked >= 0.9) & valid_mask_entropy  # [B, L, L]
                n_near_one = near_one_mask.float().sum(dim=(1, 2))  # [B]
                mid_range_mask = (P_masked > 0.2) & (P_masked < 0.8) & valid_mask_entropy  # [B, L, L]
                n_mid_range = mid_range_mask.float().sum(dim=(1, 2))  # [B]
                entropy_stats = {
                    'n_not_binary': n_not_binary.mean().item(),
                    'n_near_zero': n_near_zero.mean().item(),
                    'n_near_one': n_near_one.mean().item(),
                    'n_mid_range': n_mid_range.mean().item(),
                }
        
        # 将熵统计信息存储为类属性，供外部访问
        self._last_entropy_reg = entropy_reg
        self._last_entropy_stats = entropy_stats
        
        return total_energy

    def _should_print_debug(self) -> bool:
        """按 debug_print_every 控制打印频率。"""
        if not self.debug:
            return False
        self._debug_call_count += 1
        return (self._debug_call_count % self.debug_print_every) == 0

    def _print_debug(
        self,
        stage: str,
        energy: torch.Tensor,
        t_1: Optional[torch.Tensor] = None,
        eta_t: Optional[torch.Tensor] = None,
        seq_step_t: Optional[torch.Tensor] = None,
        grad_ss: Optional[torch.Tensor] = None,
        grad_seq: Optional[torch.Tensor] = None,
        P: Optional[torch.Tensor] = None,
        iter_idx: int = 0,
    ) -> None:
        """打印 TypeAwareSoftBondCountGuidance 的 debug 信息。"""
        if not self._should_print_debug():
            return
        parts = [f"[TypeAwareSoftBondCountGuidance {stage}] iter={iter_idx}, energy={energy.item():.6f}"]
        if t_1 is not None:
            parts.append(f"t_1_mean={t_1.mean().item():.4f}")
        if eta_t is not None:
            parts.append(f"eta_t={eta_t.mean().item():.4e}")
        if seq_step_t is not None:
            parts.append(f"seq_step_t={seq_step_t.mean().item():.4e}")
        print(" ".join(parts))
        if getattr(self, "_last_debug_info", None):
            for info in self._last_debug_info:
                if "target_min" in info and "target_max" in info:
                    target_desc = f"range=[{info['target_min']:.2f},{info['target_max']:.2f}]"
                else:
                    target_desc = f"target={info['target']:.2f}"
                print(f"  [{info['type']}] mode={info['mode']}, C_mean={info['C_mean']:.4f}, C_std={info['C_std']:.4f}, "
                      f"{target_desc}, E_tau={info['E_tau']:.6f}, weight={info['weight']:.2f}")
        if grad_ss is not None:
            gs = grad_ss.detach()
            print(f"  grad_ss: mean={gs.abs().mean().item():.3e}, max={gs.abs().max().item():.3e}")
        if grad_seq is not None:
            gq = grad_seq.detach()
            print(f"  grad_seq: mean={gq.abs().mean().item():.3e}, max={gq.abs().max().item():.3e}")
        if P is not None:
            p = P.detach()
            print(f"  P(bond): mean={p.mean().item():.4f}, max={p.max().item():.4f}, sum={p.sum().item():.2f}")
        if hasattr(self, "_last_entropy_reg") and self._last_entropy_reg is not None:
            print(f"  entropy_reg={self._last_entropy_reg.item():.6f}")

    def pre_model(self, model_raw: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        if self.stage not in ("pre_model", "both"):
            return model_raw

        if self.link_info is None or not self.type_pair_mats:
            return model_raw

        bond_mat_pred: Optional[torch.Tensor] = model_raw.get("bond_mat_pred", None)
        logits: Optional[torch.Tensor] = model_raw.get("logits", None)
        if bond_mat_pred is None:
            return model_raw

        t_1: Optional[torch.Tensor] = context.get("t_1", None)
        masks: Dict[str, torch.Tensor] = context.get("masks", {}) or {}
        res_mask: Optional[torch.Tensor] = masks.get("res_mask", None)
        head_mask: Optional[torch.Tensor] = masks.get("head_mask", None)
        tail_mask: Optional[torch.Tensor] = masks.get("tail_mask", None)
        bond_mask: Optional[torch.Tensor] = masks.get("bond_mask", None) # [B, L, L]
        seq_mask: Optional[torch.Tensor] = masks.get("seq_mask", None)   # [B, L]

        B, L, _ = bond_mat_pred.shape
        device = bond_mat_pred.device
        
        # Parse region mask if specified
        region_mask_2d = None
        if self.region is not None:
            # Resolve region references against design/full_pdb_idx by default, or
            # origin/full_origin_pdb_idx when reference_space=origin.
            full_pdb_idx = _resolve_full_pdb_idx_from_context(context, L)
            origin_pdb_idx = _resolve_origin_pdb_idx_from_context(context, L)
            region_ref_idx = origin_pdb_idx if self.reference_space == "origin" else full_pdb_idx
            region_mask_2d = SoftBondCountGuidance._parse_region(
                self.region,
                L,
                ref_idx=region_ref_idx,
                full_pdb_idx=full_pdb_idx,
                index_base=0
            )
            if region_mask_2d is not None:
                region_mask_2d = region_mask_2d.to(device=device)
                # Apply region_mode: "exclude" means invert the mask
                if self.region_mode in ("exclude", "outside", "except"):
                    region_mask_2d = ~region_mask_2d
                # Expand to batch dimension: [L, L] -> [B, L, L]
                region_mask_2d = region_mask_2d.unsqueeze(0).expand(B, -1, -1)
        
        # Create adjacent residue suppression mask if specified
        adjacent_mask_2d = None
        if self.min_bond_distance > 0:
            # Create mask for residues within min_bond_distance
            # adjacent_mask[i, j] = True if |i - j| <= min_bond_distance AND i != j (exclude diagonal)
            indices = torch.arange(L, device=device)
            i_grid, j_grid = torch.meshgrid(indices, indices, indexing='ij')
            distance = torch.abs(i_grid - j_grid)
            # Exclude diagonal: only include positions where |i - j| > 0
            eye_2d = torch.eye(L, device=device, dtype=torch.bool)
            adjacent_mask_2d = (distance <= self.min_bond_distance) & (~eye_2d)
            adjacent_mask_2d = adjacent_mask_2d.to(device=device)
            # Expand to batch dimension: [L, L] -> [B, L, L]
            adjacent_mask_2d = adjacent_mask_2d.unsqueeze(0).expand(B, -1, -1)

        if logits is None:
            return model_raw

        if t_1 is None:
            if self.bond_step is not None:
                eta_t = torch.full((B, 1, 1), self.bond_step, device=device, dtype=bond_mat_pred.dtype)
            else:
                eta_t = None
            if self.seq_step is not None:
                seq_step_t = torch.full((B, 1, 1), self.seq_step, device=device, dtype=bond_mat_pred.dtype)
            else:
                seq_step_t = None
        else:
            eta_t = (
                _schedule_weight(t_1, self.bond_step, schedule=self.schedule, power=self.power).view(B, 1, 1)
                if self.bond_step is not None
                else None
            )
            if self.seq_step is not None:
                seq_step_t = _schedule_weight(t_1, self.seq_step, schedule=self.schedule, power=self.power).view(B, 1, 1)
            else:
                seq_step_t = None

        ss = bond_mat_pred.detach()
        ss_orig = ss.clone()
        logits_work = logits.detach()

        # If seq_step is None, use direct assignment for fixed-pair modes (no gradient updates).
        logits_work = self._apply_direct_seq_for_fixed_pairs_pre_model(logits_work, seq_mask=seq_mask)
        
        # 1. 变量变换：ss -> ss_logits
        ss_val = ss.clamp(min=self.eps)
        ss_logits = (self.tau * torch.log(ss_val)).clone().detach().requires_grad_(True)

        for _ in range(max(self.n_steps, 0)):
            with torch.enable_grad():
                logits_var = logits_work.clone().requires_grad_(True)

                # 2. 前向 Sinkhorn
                P = self.dsm(ss_logits, mask_2d=bond_mask, mat_true=ss_orig)

                energy = self._compute_energy(P, logits_var, res_mask, head_mask, tail_mask,
                                             region_mask_2d=region_mask_2d, adjacent_mask_2d=adjacent_mask_2d)

                grads = torch.autograd.grad(
                    energy,
                    (ss_logits, logits_var),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_ss_logits, grad_seq_logits = grads

                # Debug 或熵正则化统计（仅在第一次迭代打印）
                if _ >= 0:
                    if self.debug:
                        self._print_debug(
                            "pre_model", energy, t_1=t_1, eta_t=eta_t, seq_step_t=seq_step_t,
                            grad_ss=grad_ss_logits, grad_seq=grad_seq_logits, P=P, iter_idx=_,
                        )
                    elif self.entropy_weight > 0 and hasattr(self, '_last_entropy_reg') and self._last_entropy_reg is not None:
                        entropy_info = f", entropy_reg={self._last_entropy_reg.item():.6f}"
                        if hasattr(self, '_last_entropy_stats') and self._last_entropy_stats is not None:
                            stats = self._last_entropy_stats
                            entropy_info += f" [not_binary(0.1-0.9):{stats['n_not_binary']:.1f}, near_zero(<0.1):{stats['n_near_zero']:.1f}, near_one(>0.9):{stats['n_near_one']:.1f}, mid_range(0.2-0.8):{stats['n_mid_range']:.1f}]"
                        print(f"[TypeAwareSoftBondCountGuidance pre_model] energy={energy.item():.6f}{entropy_info}")

            if grad_ss_logits is None and grad_seq_logits is None:
                break

            if grad_ss_logits is not None:
                # Only update bonds when bond_step is provided.
                if eta_t is not None:
                    if bond_mask is not None:
                        grad_ss_logits = grad_ss_logits * bond_mask.float()
                    # Apply region_mask to gradients
                    if region_mask_2d is not None:
                        grad_ss_logits = grad_ss_logits * region_mask_2d.float()
                    # Suppress gradients for adjacent positions if suppression is enabled
                    if adjacent_mask_2d is not None and self.suppress_adjacent_weight > 0:
                        grad_ss_logits = grad_ss_logits * (~adjacent_mask_2d.bool()).float()
                    # 3. 更新 ss_logits
                    # 打印 ss 梯度和目标的数量级
                    # print(
                    #     f"[ss] grad_ss_logits mean: {grad_ss_logits.abs().mean().item():.3e}, "
                    #     f"max: {grad_ss_logits.abs().max().item():.3e}, "
                    #     f"ss_logits mean: {ss_logits.abs().mean().item():.3e}, "
                    #     f"max: {ss_logits.abs().max().item():.3e}, "
                    #     f"eta_t: {eta_t.mean().item():.3e}"
                    # )
                    ss_logits = ss_logits - eta_t * grad_ss_logits

            if grad_seq_logits is not None and (self.seq_step is not None) and self.seq_step > 0 and (seq_step_t is not None):
                if seq_mask is not None:
                    # print(f"[seq] grad_seq_logits mean: {grad_seq_logits.abs().mean().item():.3e}, "
                    # f"max: {grad_seq_logits.abs().max().item():.3e}, "
                    # f"seq_logits mean: {logits_work.abs().mean().item():.3e}, "
                    # f"max: {logits_work.abs().max().item():.3e}, "
                    # f"seq_step_t: {seq_step_t.mean().item():.3e}")
                    grad_seq_logits = grad_seq_logits * seq_mask.unsqueeze(-1).float()
                logits_work = logits_work - seq_step_t * grad_seq_logits * (self.num_aatypes) 

        # Final projection
        with torch.no_grad():
            ss_projected = self.dsm(ss_logits, mask_2d=bond_mask, mat_true=ss_orig)

        # Only override bond matrix with hard fixed pairs when bond_step is None.
        fixed_pairs_all = self._gather_fixed_pairs(self.type_cfgs) if self.bond_step is None else []
        if fixed_pairs_all and self.bond_step is None:
            P_fixed = self._build_fixed_pair_matrix(
                L=L,
                res_mask=res_mask,
                pairs=fixed_pairs_all,
                device=device,
                dtype=bond_mat_pred.dtype,
            ).expand(B, -1, -1)
            model_raw["bond_mat_pred"] = P_fixed.detach()
        else:
            model_raw["bond_mat_pred"] = ss_projected.detach()
        model_raw["logits"] = logits_work

        return model_raw

    def post_step(self, step_out: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        if self.stage not in ("post_step", "both"):
            return step_out

        if self.link_info is None or not self.type_pair_mats:
            return step_out

        ss_t_2: Optional[torch.Tensor] = step_out.get("ss_t_2", None)
        aatypes_t_2: Optional[torch.Tensor] = step_out.get("aatypes_t_2", None)
        
        if ss_t_2 is None:
            return step_out

        t_1: Optional[torch.Tensor] = context.get("t_1", None)
        masks: Dict[str, torch.Tensor] = context.get("masks", {}) or {}
        res_mask: Optional[torch.Tensor] = masks.get("res_mask", None)
        head_mask: Optional[torch.Tensor] = masks.get("head_mask", None)
        tail_mask: Optional[torch.Tensor] = masks.get("tail_mask", None)
        bond_mask: Optional[torch.Tensor] = masks.get("bond_mask", None) # [B, L, L]
        seq_mask: Optional[torch.Tensor] = masks.get("seq_mask", None)   # [B, L]

        B, L, _ = ss_t_2.shape
        device = ss_t_2.device
        
        # Parse region mask if specified
        region_mask_2d = None
        if self.region is not None:
            # Resolve region references against design/full_pdb_idx by default, or
            # origin/full_origin_pdb_idx when reference_space=origin.
            full_pdb_idx = _resolve_full_pdb_idx_from_context(context, L)
            origin_pdb_idx = _resolve_origin_pdb_idx_from_context(context, L)
            region_ref_idx = origin_pdb_idx if self.reference_space == "origin" else full_pdb_idx
            region_mask_2d = SoftBondCountGuidance._parse_region(
                self.region,
                L,
                ref_idx=region_ref_idx,
                full_pdb_idx=full_pdb_idx,
                index_base=0
            )
            if region_mask_2d is not None:
                region_mask_2d = region_mask_2d.to(device=device)
                # Apply region_mode: "exclude" means invert the mask
                if self.region_mode in ("exclude", "outside", "except"):
                    region_mask_2d = ~region_mask_2d
                # Expand to batch dimension: [L, L] -> [B, L, L]
                region_mask_2d = region_mask_2d.unsqueeze(0).expand(B, -1, -1)
        
        # Create adjacent residue suppression mask if specified
        adjacent_mask_2d = None
        if self.min_bond_distance > 0:
            # Create mask for residues within min_bond_distance
            # adjacent_mask[i, j] = True if |i - j| <= min_bond_distance AND i != j (exclude diagonal)
            indices = torch.arange(L, device=device)
            i_grid, j_grid = torch.meshgrid(indices, indices, indexing='ij')
            distance = torch.abs(i_grid - j_grid)
            # Exclude diagonal: only include positions where |i - j| > 0
            eye_2d = torch.eye(L, device=device, dtype=torch.bool)
            adjacent_mask_2d = (distance <= self.min_bond_distance) & (~eye_2d)
            adjacent_mask_2d = adjacent_mask_2d.to(device=device)
            # Expand to batch dimension: [L, L] -> [B, L, L]
            adjacent_mask_2d = adjacent_mask_2d.unsqueeze(0).expand(B, -1, -1)

        if aatypes_t_2 is None:
            return step_out

        if t_1 is None:
            if self.bond_step is not None:
                eta_t = torch.full((B, 1, 1), self.bond_step, device=device, dtype=ss_t_2.dtype)
            else:
                eta_t = None
            if self.seq_step is not None:
                seq_step_t = torch.full((B, 1, 1), self.seq_step, device=device, dtype=ss_t_2.dtype)
            else:
                seq_step_t = None
        else:
            eta_t = (
                _schedule_weight(t_1, self.bond_step, schedule=self.schedule, power=self.power).view(B, 1, 1)
                if self.bond_step is not None
                else None
            )
            if self.seq_step is not None:
                seq_step_t = _schedule_weight(t_1, self.seq_step, schedule=self.schedule, power=self.power).view(B, 1, 1)
            else:
                seq_step_t = None

        ss = ss_t_2.detach()
        ss_orig = ss.clone()

        # If seq_step is None, apply direct assignment on aatypes for fixed-pair modes.
        aatypes_t_2 = self._apply_direct_seq_for_fixed_pairs_post_step(aatypes_t_2, seq_mask=seq_mask)
        
        # 1. 变量变换：ss -> ss_logits
        # Prevent saturation by clamping away from 0 and 1
        ss_val = ss.clamp(min=self.eps, max=1.0-self.eps)
        ss_logits = (self.tau * torch.log(ss_val)).clone().detach().requires_grad_(True)

        # 2. 准备 seq_logits (从离散 aatypes_t_2 转换)
        K = self.num_aatypes
        max_idx = aatypes_t_2.max().item()
        vocab_size = max(K, int(max_idx) + 1)
        
        seq_one_hot = F.one_hot(aatypes_t_2.long(), num_classes=vocab_size).float()
        # 映射到 log 空间，加 eps 避免 log(0)，同时也避免 log(1)
        seq_probs = seq_one_hot.clamp(min=self.eps, max=1.0-self.eps)
        seq_logits = torch.log(seq_probs).detach().requires_grad_(True)

        for _ in range(max(self.n_steps, 0)):
            with torch.enable_grad():
                logits_var = seq_logits
                
                # 2. 前向 Sinkhorn
                P = self.dsm(ss_logits, mask_2d=bond_mask, mat_true=ss_orig)

                energy = self._compute_energy(P, logits_var, res_mask, head_mask, tail_mask,
                                             region_mask_2d=region_mask_2d, adjacent_mask_2d=adjacent_mask_2d)

                if not torch.isfinite(energy):
                    print("error in bond type guidence (post_step)")
                    break
                grads = torch.autograd.grad(
                    energy,
                    (ss_logits, logits_var),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_ss_logits, grad_seq_logits = grads

                # Debug 或熵正则化统计（post_step，仅在第一次迭代打印）
                if _ >= 0:
                    if self.debug:
                        self._print_debug(
                            "post_step", energy, t_1=t_1, eta_t=eta_t, seq_step_t=seq_step_t,
                            grad_ss=grad_ss_logits, grad_seq=grad_seq_logits, P=P, iter_idx=_,
                        )
                    if self.entropy_weight > 0 and hasattr(self, '_last_entropy_reg') and self._last_entropy_reg is not None:
                        entropy_info = f", entropy_reg={self._last_entropy_reg.item():.6f}"
                        if hasattr(self, '_last_entropy_stats') and self._last_entropy_stats is not None:
                            stats = self._last_entropy_stats
                            entropy_info += f" [not_binary(0.1-0.9):{stats['n_not_binary']:.1f}, near_zero(<0.1):{stats['n_near_zero']:.1f}, near_one(>0.9):{stats['n_near_one']:.1f}, mid_range(0.2-0.8):{stats['n_mid_range']:.1f}]"
                        print(f"[TypeAwareSoftBondCountGuidance post_step] energy={energy.item():.6f}{entropy_info}")

            if grad_ss_logits is None and grad_seq_logits is None:
                break

            if grad_ss_logits is not None:
                # Only update bonds when bond_step is provided.
                if eta_t is not None:
                    if bond_mask is not None:
                        grad_ss_logits = grad_ss_logits * bond_mask.float()
                    # Apply region_mask to gradients
                    if region_mask_2d is not None:
                        grad_ss_logits = grad_ss_logits * region_mask_2d.float()
                    # Suppress gradients for adjacent positions if suppression is enabled
                    if adjacent_mask_2d is not None and self.suppress_adjacent_weight > 0:
                        grad_ss_logits = grad_ss_logits * (~adjacent_mask_2d.bool()).float()
                    # 打印 ss 梯度和目标的数量级
                    # print(
                    #     f"[ss] grad_ss_logits mean: {grad_ss_logits.abs().mean().item():.3e}, "
                    #     f"max: {grad_ss_logits.abs().max().item():.3e}, "
                    #     f"ss_logits mean: {ss_logits.abs().mean().item():.3e}, "
                    #     f"max: {ss_logits.abs().max().item():.3e}, "
                    #     f"eta_t: {eta_t.mean().item():.3e}"
                    # )
                    ss_logits = ss_logits - eta_t * grad_ss_logits

            if grad_seq_logits is not None and (self.seq_step is not None) and self.seq_step > 0 and (seq_step_t is not None):
                if seq_mask is not None:
                    grad_seq_logits = grad_seq_logits * seq_mask.unsqueeze(-1).float()
                # 打印 seq 梯度和目标的数量级
                # print(f"[seq] grad_seq_logits mean: {grad_seq_logits.abs().mean().item():.3e}, "
                #       f"max: {grad_seq_logits.abs().max().item():.3e}, "
                #       f"seq_logits mean: {seq_logits.abs().mean().item():.3e}, "
                #       f"max: {seq_logits.abs().max().item():.3e}, "
                #       f"seq_step_t: {seq_step_t.mean().item():.3e}")
                seq_logits = seq_logits - seq_step_t * grad_seq_logits * (self.num_aatypes)

        # Final projection and conversion back
        with torch.no_grad():
            ss_projected = self.dsm(ss_logits, mask_2d=bond_mask, mat_true=ss_orig)
            new_aatypes = torch.argmax(seq_logits, dim=-1)

        fixed_pairs_all = self._gather_fixed_pairs(self.type_cfgs) if self.bond_step is None else []
        if fixed_pairs_all and self.bond_step is None:
            P_fixed = self._build_fixed_pair_matrix(
                L=L,
                res_mask=res_mask,
                pairs=fixed_pairs_all,
                device=device,
                dtype=ss_t_2.dtype,
            ).expand(B, -1, -1)
            step_out["ss_t_2"] = P_fixed.detach()
        else:
            step_out["ss_t_2"] = ss_projected.detach()
        step_out["aatypes_t_2"] = new_aatypes.detach()

        return step_out


class ClashGuidance(Guidance):
    """
    Guidance to reduce clashes using OpenFoldClashLoss.
    Applies gradient descent on translation of the next step.
    Strength increases from 0 at start_t to weight at t=1.
    
    Supports both pre_model and post_step stages:
    - pre_model: operates on px0_bb (backbone coordinates) from model_raw
    - post_step: operates on trans_t_2/rotmats_t_2 from step_out
    """

    def __init__(self, cfg: Optional[Any] = None, device: str = "cpu") -> None:
        super().__init__(cfg, device)
        self.start_t = float(getattr(cfg, "start_t", 0.5))
        self.weight = float(getattr(cfg, "weight", 1.0))
        self.n_steps = int(getattr(cfg, "n_steps", 1))
        self.link_csv_path = getattr(cfg, "link_csv_path", None)
        # Stage selection: "pre_model", "post_step", or "both"
        self.stage = str(getattr(cfg, "stage", "post_step")).lower() if cfg is not None else "post_step"

        self.clash_loss = OpenFoldClashLoss(
            link_csv_path=self.link_csv_path,
            device=device,
            log_raw=False,
            include_within=True,
            treat_adjacent_as_bonded=True,
            reduction='sum',
        )

    def pre_model(self, model_raw: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        """
        Pre-model clash guidance: operates on px0_bb (backbone coordinates).
        Extracts trans/rotmats from px0_bb, builds all-atom structure, computes clash loss,
        and updates px0_bb via gradient descent on translations.
        """
        if self.stage not in ("pre_model", "both"):
            return model_raw

        t_1 = context.get("t_1")  # [B]
        if t_1 is None:
            return model_raw

        # Check if any batch item is in active time range
        w_t = torch.zeros_like(t_1)
        mask_active = (t_1 >= self.start_t)
        if not mask_active.any():
            return model_raw

        denom = 1.0 - self.start_t
        if abs(denom) < 1e-6:
            denom = 1.0  # avoid div 0

        # Linear ramp from 0 at start_t to weight at 1.0
        w_t[mask_active] = self.weight * (t_1[mask_active] - self.start_t) / denom
        w_t = w_t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting

        # Inputs from model_raw
        px0_bb = model_raw.get("px0_bb")  # [B, L, 3, 3] (N, CA, C)
        logits = model_raw.get("logits")  # [B, L, C]
        alpha_pred = model_raw.get("alpha_pred")  # [B, L, 10, 2]
        bond_mat_pred = model_raw.get("bond_mat_pred")  # [B, L, L]

        if px0_bb is None:
            return model_raw

        # Get context
        allatom = context.get("allatom")
        if allatom is None:
            return model_raw

        masks = context.get("masks", {})
        res_mask = masks.get("res_mask")
        head_mask = masks.get("head_mask")
        tail_mask = masks.get("tail_mask")
        str_mask = masks.get("str_mask")
        nc_anchor = masks.get("N_C_anchor")

        # Extract trans (CA coordinates) and rotmats from px0_bb
        # px0_bb shape: [B, L, 3, 3] where last dim is [N, CA, C]
        trans = px0_bb[:, :, 1, :].detach().clone()  # [B, L, 3] (CA coordinates)
        rotmats = iu.get_R_from_xyz(px0_bb.detach())  # [B, L, 3, 3]

        # Get sequence prediction from logits
        if logits is not None:
            aatypes = torch.argmax(logits[..., :20], dim=-1).long()  # [B, L]
        else:
            return model_raw

        # Optimization loop
        for step_idx in range(self.n_steps):
            # Ensure trans requires grad for this iteration
            trans = trans.detach().clone().requires_grad_(True)
            
            with torch.enable_grad():
                # Rebuild backbone from updated trans and fixed rotmats
                backbone = iu.get_xyz_from_RT(rotmats, trans)  # [B, L, 3, 3]

                # Build all-atom structure
                _, coords_14 = allatom(
                    aatypes,
                    backbone,
                    alpha_pred,
                    bond_mat=bond_mat_pred,
                    res_mask=res_mask,
                    head_mask=head_mask,
                    tail_mask=tail_mask,
                    N_C_anchor=nc_anchor,
                    use_H=False
                )
                # Update virtual node coordinates
                coords_14 = iu.update_nc_node_coordinates(coords_14, nc_anchor, head_mask, tail_mask, apply_offset=False)

                # Compute loss
                if head_mask is not None or tail_mask is not None:
                    final_res_mask = res_mask.float() * (~head_mask).float() * (~tail_mask).float()
                else:
                    final_res_mask = res_mask

                loss = self.clash_loss(
                    nc_anchor,
                    coords_14,
                    aatypes,
                    final_res_mask,
                    bond_mat=bond_mat_pred,
                    head_mask=head_mask,
                    tail_mask=tail_mask
                )

                if step_idx == 0:
                    print(f"[ClashGuidance pre_model] loss_clash (before optimization): {loss.item():.4f}")

                # Gradient
                grad = torch.autograd.grad(loss, trans, retain_graph=False, create_graph=False)[0]

            # Update
            # Minimize loss -> subtract gradient
            # Apply mask: Only update where str_mask is 1 (designable/diffusing)
            if str_mask is not None:
                grad = grad * str_mask.unsqueeze(-1).float()

            trans = trans - w_t * grad

        # Rebuild px0_bb from updated trans and rotmats
        with torch.no_grad():
            px0_bb_updated = iu.get_xyz_from_RT(rotmats, trans.detach())

        model_raw["px0_bb"] = px0_bb_updated
        return model_raw

    def post_step(self, step_out: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        t_1 = context.get("t_1")  # [B]
        if t_1 is None:
            return step_out

        # Check if any batch item is in active time range
        # t_1 is [B]. We compute per-batch weight.
        w_t = torch.zeros_like(t_1)
        mask_active = (t_1 >= self.start_t)
        if not mask_active.any():
            return step_out

        denom = 1.0 - self.start_t
        if abs(denom) < 1e-6: denom = 1.0  # avoid div 0

        # Linear ramp from 0 at start_t to weight at 1.0
        w_t[mask_active] = self.weight * (t_1[mask_active] - self.start_t) / denom
        w_t = w_t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting

        # Inputs
        trans_t_2 = step_out.get("trans_t_2")  # [B, L, 3]
        rotmats_t_2 = step_out.get("rotmats_t_2")  # [B, L, 3, 3]
        aatypes_t_2 = step_out.get("aatypes_t_2")  # [B, L]
        alpha_pred = context.get("alpha_pred")  # [B, L, 10, 2]
        allatom = context.get("allatom")

        if trans_t_2 is None or rotmats_t_2 is None or allatom is None:
            return step_out

        # Prepare for optimization
        # Only optimize trans for now as it's safer/easier than rotmats
        trans_var = trans_t_2.detach().clone().requires_grad_(True)
        rotmats_fixed = rotmats_t_2.detach()

        masks = context.get("masks", {})
        res_mask = masks.get("res_mask")
        head_mask = masks.get("head_mask")
        tail_mask = masks.get("tail_mask")
        str_mask = masks.get("str_mask")
        nc_anchor = masks.get("N_C_anchor")
        bond_mat = step_out.get("ss_t_2") # Use current bond mat

        # Optimization loop
        for _ in range(self.n_steps):
            with torch.enable_grad():
                # Build backbone
                backbone = iu.get_xyz_from_RT(rotmats_fixed, trans_var)  # [B, L, 3, 3]

                # Build allatom
                # Note: aatypes_t_2 must be long for embedding
                # We reuse alpha_pred from t_1 (best guess for sidechain angles)
                _, coords_14 = allatom(
                    aatypes_t_2.long(),
                    backbone,
                    alpha_pred,
                    bond_mat=bond_mat,
                    res_mask=res_mask,
                    head_mask=head_mask,
                    tail_mask=tail_mask,
                    N_C_anchor=nc_anchor,
                    use_H=False
                )
                # IMPORTANT: Update virtual node coordinates to match sampling completion behavior
                coords_14 = iu.update_nc_node_coordinates(coords_14, nc_anchor, head_mask, tail_mask, apply_offset=False)

                # Compute loss
                # IMPORTANT: Use final_res_mask (excluding head/tail virtual nodes) to match refine_sidechain_by_gd behavior.
                # This ensures consistent clash loss calculation between guidance and refinement stages.
                # OpenFoldClashLoss will still use head_mask/tail_mask for termini exclusion logic internally.
                if head_mask is not None or tail_mask is not None:
                    final_res_mask = res_mask.float() * (~head_mask).float() * (~tail_mask).float()
                else:
                    final_res_mask = res_mask
                loss = self.clash_loss(
                    nc_anchor,
                    coords_14,
                    aatypes_t_2.long(),
                    final_res_mask,
                    bond_mat=bond_mat,
                    head_mask=head_mask,
                    tail_mask=tail_mask
                )
                # print(f"[clash loss] loss_clash: {loss.item():.4f}")
                # if _ == 0:
                #     print(f"loss_clash (before optimization): {loss.item():.4f}")
                # Gradient
                grad = torch.autograd.grad(loss, trans_var)[0]

            # Update
            # Minimize loss -> subtract gradient
            # Apply mask: Only update where str_mask is 1 (designable/diffusing)
            if str_mask is not None:
                grad = grad * str_mask.unsqueeze(-1).float()

            trans_var = trans_var - w_t * grad

        step_out["trans_t_2"] = trans_var.detach()
        return step_out


class RadiusOfGyrationGuidance(Guidance):
    """
    Guidance to penalize radius of gyration (Rg) to encourage compact structures.
    Applies gradient descent on translation of the next step.
    Uses time scheduling (schedule and power) to control guidance strength over time.
    
    The radius of gyration is defined as:
    Rg² = (1/N) * Σᵢ (rᵢ - r_cm)²
    where r_cm is the center of mass and N is the number of atoms.
    
    Supports both pre_model and post_step stages:
    - pre_model: operates on px0_bb (backbone coordinates) from model_raw
    - post_step: operates on trans_t_2/rotmats_t_2 from step_out
    
    Config parameters:
      - weight: base weight for guidance (default: 1.0)
      - schedule: time scheduling mode - "linear", "quadratic", "cosine", "exp", "inverse" (default: "linear")
      - power: power parameter for "exp" schedule (default: 1.0)
      - target_Rg: target radius of gyration (default: 10.0)
      - n_steps: number of optimization steps (default: 1)
      - use_ca_only: use only CA atoms for Rg calculation (default: True)
      - loss_mode: "threshold" (only penalize if Rg > target) or "mse" (always penalize deviation) (default: "threshold")
      - stage: "pre_model", "post_step", or "both" (default: "post_step")
    """

    def __init__(self, cfg: Optional[Any] = None, device: str = "cpu") -> None:
        super().__init__(cfg, device)
        self.weight = float(getattr(cfg, "weight", 1.0)) if cfg is not None else 1.0
        self.target_Rg = float(getattr(cfg, "target_Rg", 10.0)) if cfg is not None else 10.0
        self.n_steps = int(getattr(cfg, "n_steps", 1)) if cfg is not None else 1
        # Use CA atoms only (True) or all heavy atoms (False)
        self.use_ca_only = bool(getattr(cfg, "use_ca_only", True)) if cfg is not None else True
        # Loss mode: "mse" (penalize if Rg > target) or "threshold" (only penalize if Rg > target)
        self.loss_mode = str(getattr(cfg, "loss_mode", "threshold")).lower() if cfg is not None else "threshold"
        # Stage selection: "pre_model", "post_step", or "both"
        self.stage = str(getattr(cfg, "stage", "post_step")).lower() if cfg is not None else "post_step"
        # Time scheduling: linear / quadratic / cosine / exp / inverse
        self.schedule = str(getattr(cfg, "schedule", "linear")).lower() if cfg is not None else "linear"
        self.power = float(getattr(cfg, "power", 1.0)) if cfg is not None else 1.0

    @staticmethod
    def _compute_radius_of_gyration(
        coords: torch.Tensor,  # [B, N, 3] or [B, L, 3, 3] for backbone
        mask: Optional[torch.Tensor] = None,  # [B, N] or [B, L]
        use_ca_only: bool = True
    ) -> torch.Tensor:
        """
        Compute radius of gyration from coordinates.
        
        Args:
            coords: Coordinates tensor. If shape is [B, L, 3, 3], it's backbone (N, CA, C).
                   If shape is [B, N, 3], it's already flattened coordinates.
            mask: Optional mask for valid positions [B, N] or [B, L]
            use_ca_only: If True and coords is backbone, use only CA atoms
        
        Returns:
            Rg values [B]
        """
        B = coords.shape[0]
        device = coords.device
        
        # Handle backbone coordinates [B, L, 3, 3]
        if coords.dim() == 4 and coords.shape[-1] == 3 and coords.shape[-2] == 3:
            if use_ca_only:
                # Extract CA coordinates (index 1)
                coords_flat = coords[:, :, 1, :]  # [B, L, 3]
                if mask is not None:
                    # mask is [B, L]
                    mask_flat = mask
                else:
                    mask_flat = torch.ones(B, coords.shape[1], dtype=torch.bool, device=device)
            else:
                # Flatten all backbone atoms
                coords_flat = coords.view(B, -1, 3)  # [B, L*3, 3]
                if mask is not None:
                    # Expand mask for all atoms
                    mask_flat = mask.unsqueeze(-1).expand(-1, -1, 3).reshape(B, -1)  # [B, L*3]
                else:
                    mask_flat = torch.ones(B, coords.shape[1] * 3, dtype=torch.bool, device=device)
        else:
            # Already flattened [B, N, 3]
            coords_flat = coords
            mask_flat = mask if mask is not None else torch.ones(B, coords.shape[1], dtype=torch.bool, device=device)
        
        # Compute center of mass
        mask_float = mask_flat.float().unsqueeze(-1)  # [B, N, 1]
        coords_masked = coords_flat * mask_float  # [B, N, 3]
        n_valid = mask_float.sum(dim=1, keepdim=True)  # [B, 1, 1]
        n_valid = torch.clamp(n_valid, min=1.0)  # Avoid division by zero
        center_of_mass = coords_masked.sum(dim=1, keepdim=True) / n_valid  # [B, 1, 3]
        
        # Compute squared distances from center of mass
        diff = coords_flat - center_of_mass  # [B, N, 3]
        dist_sq = (diff ** 2).sum(dim=-1)  # [B, N]
        
        # Apply mask and compute mean
        dist_sq_masked = dist_sq * mask_flat.float()  # [B, N]
        mean_dist_sq = dist_sq_masked.sum(dim=1) / n_valid.squeeze(-1).squeeze(-1)  # [B]
        
        # Rg = sqrt(mean_dist_sq)
        Rg = torch.sqrt(torch.clamp(mean_dist_sq, min=1e-8))  # [B]
        
        return Rg

    def pre_model(self, model_raw: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        """
        Pre-model Rg guidance: operates on px0_bb (backbone coordinates).
        Computes Rg and applies gradient descent to reduce it if above target.
        """
        if self.stage not in ("pre_model", "both"):
            return model_raw

        t_1 = context.get("t_1")  # [B]
        if t_1 is None:
            return model_raw

        # Compute time-dependent weight using schedule
        w_t = _schedule_weight(t_1, self.weight, schedule=self.schedule, power=self.power)  # [B]
        w_t = w_t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting

        # Inputs from model_raw
        px0_bb = model_raw.get("px0_bb")  # [B, L, 3, 3] (N, CA, C)
        if px0_bb is None:
            return model_raw

        masks = context.get("masks", {})
        res_mask = masks.get("res_mask")
        str_mask = masks.get("str_mask")

        # Extract trans (CA coordinates) and rotmats from px0_bb
        trans = px0_bb[:, :, 1, :].detach().clone()  # [B, L, 3] (CA coordinates)
        rotmats = iu.get_R_from_xyz(px0_bb.detach())  # [B, L, 3, 3]

        # Optimization loop
        for step_idx in range(self.n_steps):
            # Ensure trans requires grad for this iteration
            trans = trans.detach().clone().requires_grad_(True)
            
            with torch.enable_grad():
                # Rebuild backbone from updated trans and fixed rotmats
                backbone = iu.get_xyz_from_RT(rotmats, trans)  # [B, L, 3, 3]

                # Compute Rg
                Rg = self._compute_radius_of_gyration(backbone, mask=res_mask, use_ca_only=self.use_ca_only)

                # Compute loss; use sum (not mean) for classifier guidance sampling
                if self.loss_mode == "mse":
                    # Always penalize deviation from target
                    loss = ((Rg - self.target_Rg) ** 2).sum()
                else:  # "threshold"
                    # Only penalize if Rg > target_Rg
                    excess = torch.clamp(Rg - self.target_Rg, min=0.0)
                    loss = (excess ** 2).sum()

                if step_idx == 0:
                    print(f"[RadiusOfGyrationGuidance pre_model] Rg (before optimization): {Rg.mean().item():.4f}, target: {self.target_Rg:.4f}, loss: {loss.item():.6f}")

                # Gradient
                grad = torch.autograd.grad(loss, trans, retain_graph=False, create_graph=False)[0]

            # Update
            # Minimize loss -> subtract gradient
            # Apply mask: Only update where str_mask is 1 (designable/diffusing)
            if str_mask is not None:
                grad = grad * str_mask.unsqueeze(-1).float()

            trans = trans - w_t * grad

        # Rebuild px0_bb from updated trans and rotmats
        with torch.no_grad():
            px0_bb_updated = iu.get_xyz_from_RT(rotmats, trans.detach())

        model_raw["px0_bb"] = px0_bb_updated
        return model_raw

    def post_step(self, step_out: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        """
        Post-step Rg guidance: operates on trans_t_2/rotmats_t_2.
        Computes Rg and applies gradient descent to reduce it if above target.
        """
        if self.stage not in ("post_step", "both"):
            return step_out

        t_1 = context.get("t_1")  # [B]
        if t_1 is None:
            return step_out

        # Compute time-dependent weight using schedule
        w_t = _schedule_weight(t_1, self.weight, schedule=self.schedule, power=self.power)  # [B]
        w_t = w_t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting

        # Inputs
        trans_t_2 = step_out.get("trans_t_2")  # [B, L, 3]
        rotmats_t_2 = step_out.get("rotmats_t_2")  # [B, L, 3, 3]

        if trans_t_2 is None or rotmats_t_2 is None:
            return step_out

        masks = context.get("masks", {})
        res_mask = masks.get("res_mask")
        str_mask = masks.get("str_mask")

        # Prepare for optimization
        trans_var = trans_t_2.detach().clone().requires_grad_(True)
        rotmats_fixed = rotmats_t_2.detach()

        # Optimization loop
        for step_idx in range(self.n_steps):
            with torch.enable_grad():
                # Build backbone
                backbone = iu.get_xyz_from_RT(rotmats_fixed, trans_var)  # [B, L, 3, 3]

                # Compute Rg
                Rg = self._compute_radius_of_gyration(backbone, mask=res_mask, use_ca_only=self.use_ca_only)

                # Compute loss; use sum (not mean) for classifier guidance sampling
                if self.loss_mode == "mse":
                    # Always penalize deviation from target
                    loss = ((Rg - self.target_Rg) ** 2).sum()
                else:  # "threshold"
                    # Only penalize if Rg > target_Rg
                    excess = torch.clamp(Rg - self.target_Rg, min=0.0)
                    loss = (excess ** 2).sum()

                if step_idx == 0:
                    print(f"[RadiusOfGyrationGuidance post_step] Rg (before optimization): {Rg.mean().item():.4f}, target: {self.target_Rg:.4f}, loss: {loss.item():.6f}")

                # Gradient
                grad = torch.autograd.grad(loss, trans_var, retain_graph=False, create_graph=False)[0]

            # Update
            # Minimize loss -> subtract gradient
            # Apply mask: Only update where str_mask is 1 (designable/diffusing)
            if str_mask is not None:
                grad = grad * str_mask.unsqueeze(-1).float()

            trans_var = trans_var - w_t * grad

        step_out["trans_t_2"] = trans_var.detach()
        return step_out


class ResidueBondGuidance(Guidance):
    """
    Residue-centric bond guidance on ss_t_2.

    Supports two residue-level modes:
      - at_least_one:
          encourage one anchor residue to have degree >= target_degree
          within a candidate set (or among all valid residues if candidates is omitted).
      - unique_in_candidates:
          encourage one anchor residue to form a unique bond with exactly one
          residue from a candidate set, while suppressing the anchor's other bonds.

    Optional candidate focusing:
      - top_k:
          if > 0, only keep the current top-k scoring candidates for each anchor
          update. This is useful when omitting `candidates` would otherwise let the
          chosen partner jump across too many residues during sampling.

    Example:
      guidance:
        list:
          - name: residue_bond
            stage: post_step
            weight: 1.0
            schedule: linear
            n_steps: 1
            step_size: 1.0
            debug: false
            debug_print_every: 1
            entries:
              - anchor: 10
                mode: at_least_one
                target_degree: 1.0
              - anchor: 15
                mode: unique_in_candidates
                candidates: [3, 8, 20]
                top_k: 3
                unique_target: 1.0
                off_target: 0.0

    Notes:
      - This guidance acts directly on the bond matrix and preserves symmetry.
      - Indices are 0-based by default; set index_base: 1 to use 1-based indices.
      - `anchor` / `candidates` may also use PDB-style references such as `A/42`.
      - `reference_space: origin` interprets those references against `full_origin_pdb_idx`;
        `reference_space: design` interprets them against `full_pdb_idx`.
      - The "unique" mode is unique for the anchor residue, i.e. the anchor is
        encouraged to keep only one bond inside the candidate set and suppress
        its other incident bonds.
    """

    _UNIQUE_SELECTED_BETA = 4.0
    _UNIQUE_OUTSIDE_WEIGHT = 0.25

    def __init__(self, cfg: Optional[Any] = None, device: str = "cpu") -> None:
        super().__init__(cfg, device)
        self.stage = str(getattr(cfg, "stage", "post_step")).lower() if cfg is not None else "post_step"
        self.weight = float(getattr(cfg, "weight", 1.0)) if cfg is not None else 1.0
        self.schedule = str(getattr(cfg, "schedule", "linear")).lower() if cfg is not None else "linear"
        self.power = float(getattr(cfg, "power", 1.0)) if cfg is not None else 1.0
        self.n_steps = int(getattr(cfg, "n_steps", 1)) if cfg is not None else 1
        self.step_size = float(getattr(cfg, "step_size", 1.0)) if cfg is not None else 1.0
        self.debug = bool(getattr(cfg, "debug", False)) if cfg is not None else False
        self.debug_print_every = int(getattr(cfg, "debug_print_every", 1)) if cfg is not None else 1
        self._debug_call_count = 0
        self.index_base = int(getattr(cfg, "index_base", 0)) if cfg is not None else 0
        self.reference_space = str(getattr(cfg, "reference_space", "design")).lower() if cfg is not None else "design"
        self.min_bond_distance = int(getattr(cfg, "min_bond_distance", 0)) if cfg is not None else 0
        self.top_k = max(int(getattr(cfg, "top_k", 0)), 0) if cfg is not None else 0
        self.entries = list(getattr(cfg, "entries", []) or []) if cfg is not None else []

        # Backward-compatible single-entry form.
        if not self.entries and cfg is not None and getattr(cfg, "anchor", None) is not None:
            self.entries = [cfg]

    def _normalize_index(self, idx: Any, L: int) -> Optional[int]:
        try:
            ii = int(idx) - self.index_base
        except Exception:
            return None
        if ii < 0:
            ii = L + ii
        if 0 <= ii < L:
            return ii
        return None

    @staticmethod
    def _split_pdb_ref(raw: Any) -> Optional[tuple]:
        """Parse PDB-style residue reference like 'A/42' or ('A', 42)."""
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            chain = str(raw[0]).strip()
            res = str(raw[1]).strip()
            if chain and res:
                return chain, res
            return None
        if not isinstance(raw, str):
            return None
        s = raw.strip()
        if not s:
            return None
        if "/" in s:
            chain, res = s.split("/", 1)
            chain = chain.strip()
            res = res.strip()
            if chain and res:
                return chain, res
            return None
        if len(s) >= 2 and s[0].isalpha():
            chain = s[0].strip()
            res = s[1:].strip()
            if chain and res:
                return chain, res
        return None

    @classmethod
    def _same_pdb_ref(cls, lhs: Any, rhs: Any) -> bool:
        left = cls._split_pdb_ref(lhs)
        right = cls._split_pdb_ref(rhs)
        if left is None or right is None:
            return False
        if left[0] != right[0]:
            return False
        if left[1] == right[1]:
            return True
        try:
            return int(left[1]) == int(right[1])
        except Exception:
            return False

    def _resolve_position(self, ref: Any, L: int, ref_idx: Optional[List]) -> Optional[int]:
        idx = self._normalize_index(ref, L)
        if idx is not None:
            return idx

        ref_parsed = self._split_pdb_ref(ref)
        if ref_parsed is None or ref_idx is None or len(ref_idx) != L:
            return None

        for ii, item in enumerate(ref_idx):
            if self._same_pdb_ref(item, ref_parsed):
                return ii
        return None

    def _parse_candidates(
        self,
        raw: Any,
        L: int,
        anchor: int,
        ref_idx: Optional[List],
    ) -> Optional[List[int]]:
        if raw is None:
            return None
        items = raw
        if isinstance(items, (int, float, str)):
            items = [items]
        try:
            items = list(items)
        except Exception:
            return None

        out: List[int] = []
        seen = set()
        for item in items:
            if isinstance(item, str):
                s = item.strip()
                if not s:
                    continue
                if ("-" in s) or (":" in s):
                    sep = "-" if ("-" in s) else ":"
                    parts = [p.strip() for p in s.split(sep) if p.strip()]
                    if len(parts) == 2:
                        a = self._resolve_position(parts[0], L, ref_idx)
                        b = self._resolve_position(parts[1], L, ref_idx)
                        if a is None or b is None:
                            continue
                        lo, hi = (a, b) if a <= b else (b, a)
                        for ii in range(lo, hi + 1):
                            if ii != anchor and ii not in seen:
                                out.append(ii)
                                seen.add(ii)
                        continue
            ii = self._resolve_position(item, L, ref_idx)
            if ii is None or ii == anchor or ii in seen:
                continue
            out.append(ii)
            seen.add(ii)
        return out

    def _build_allowed_mask(
        self,
        B: int,
        L: int,
        anchor: int,
        candidates: Optional[List[int]],
        device: torch.device,
        masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        allowed = torch.zeros((B, L), dtype=torch.bool, device=device)
        if candidates is None:
            allowed[:] = True
            allowed[:, anchor] = False
        else:
            valid_candidates = [j for j in candidates if 0 <= j < L and j != anchor]
            if valid_candidates:
                allowed[:, valid_candidates] = True

        res_mask = masks.get("res_mask", None)
        if res_mask is not None:
            allowed = allowed & res_mask.bool()
            if anchor < res_mask.shape[1]:
                allowed = allowed & res_mask[:, anchor].bool().unsqueeze(1)

        bond_mask = masks.get("bond_mask", None)
        if bond_mask is not None and anchor < bond_mask.shape[1]:
            allowed = allowed & bond_mask[:, anchor, :].bool()

        if self.min_bond_distance > 0:
            indices = torch.arange(L, device=device)
            allowed = allowed & ((indices - anchor).abs() > self.min_bond_distance).unsqueeze(0)

        allowed[:, anchor] = False
        return allowed

    @staticmethod
    def _restrict_allowed_to_top_k(
        scores: torch.Tensor,
        allowed: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Keep only the top-k currently scored candidates per batch row."""
        if top_k <= 0 or allowed.shape[-1] <= 0:
            return allowed

        max_k = min(int(top_k), allowed.shape[-1])
        if max_k <= 0:
            return allowed

        masked_scores = scores.masked_fill(~allowed, float("-inf"))
        top_vals, top_idx = torch.topk(masked_scores, k=max_k, dim=-1)
        valid = torch.isfinite(top_vals)
        focused = torch.zeros_like(allowed)
        focused.scatter_(1, top_idx, valid)
        return focused

    @staticmethod
    def _masked_row_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=values.dtype)
        denom = mask_f.sum(dim=-1)
        numer = (values * mask_f).sum(dim=-1)
        return torch.where(denom > 1e-8, numer / (denom + 1e-8), torch.zeros_like(numer))

    @classmethod
    def _compute_unique_soft_count(
        cls,
        row_values: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        row_selected = torch.clamp(row_values, min=0.0) * selected.float()
        return torch.pow(row_selected, cls._UNIQUE_SELECTED_BETA).sum(dim=-1)

    @staticmethod
    def _format_ref(item: Any) -> str:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            return f"{item[0]}/{item[1]}"
        return str(item)

    def _should_print_debug(self) -> bool:
        if not self.debug:
            return False
        self._debug_call_count += 1
        return (self._debug_call_count % max(self.debug_print_every, 1)) == 0

    def _print_debug(
        self,
        debug_rows: List[Dict[str, Any]],
        *,
        iter_idx: int,
        t_1: Optional[torch.Tensor],
        alpha: torch.Tensor,
    ) -> None:
        if not debug_rows:
            return
        parts = [f"[ResidueBondGuidance] iter={iter_idx}"]
        if t_1 is not None:
            parts.append(f"t_1_mean={t_1.mean().item():.4f}")
        parts.append(f"alpha={alpha.mean().item():.4f}")
        print(" ".join(parts))
        for row in debug_rows:
            if row["mode"] == "at_least_one":
                print(
                    f"  [{row['anchor']}] mode=at_least_one "
                    f"degree={row['degree']:.4f} target={row['target']:.4f} "
                    f"loss={row['loss']:.6f}"
                )
            else:
                partner = row.get("partner", None)
                partner_info = f" partner={partner}" if partner is not None else ""
                count_info = ""
                if "count" in row:
                    count_method = row.get("count_method", "count")
                    count_info = f" {count_method}={row['count']:.4f}"
                tail_info = f" tail_loss={row['tail_loss']:.6f}" if "tail_loss" in row else ""
                outside_loss_info = f" outside_loss={row['outside_loss']:.6f}" if "outside_loss" in row else ""
                print(
                    f"  [{row['anchor']}] mode=unique_in_candidates "
                    f"chosen={row['chosen']:.4f} other_allowed={row['other_allowed']:.4f} "
                    f"outside={row['outside']:.4f}{count_info}{tail_info}{outside_loss_info} "
                    f"loss={row['loss']:.6f}{partner_info}"
                )

    def post_step(self, step_out: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        if self.stage not in ("post_step", "both"):
            return step_out

        ss_t_2: Optional[torch.Tensor] = step_out.get("ss_t_2", None)
        t_1: Optional[torch.Tensor] = context.get("t_1", None)
        if ss_t_2 is None or t_1 is None or not self.entries:
            return step_out

        masks: Dict[str, torch.Tensor] = context.get("masks", {}) or {}
        B, L, _ = ss_t_2.shape
        device = ss_t_2.device
        full_pdb_idx = _resolve_full_pdb_idx_from_context(context, L)
        origin_pdb_idx = _resolve_origin_pdb_idx_from_context(context, L)
        if self.n_steps <= 0:
            return step_out

        default_ref_idx = origin_pdb_idx if self.reference_space == "origin" else full_pdb_idx

        w = _schedule_weight(t_1, self.weight, self.schedule, self.power).view(B, 1)
        alpha = (w * self.step_size).clamp(min=0.0, max=1.0)
        should_debug = self._should_print_debug()

        updated = ss_t_2.clone()

        for iter_idx in range(self.n_steps):
            debug_rows: List[Dict[str, Any]] = []
            for entry in self.entries:
                entry_ref_space = str(getattr(entry, "reference_space", self.reference_space)).lower()
                ref_idx = origin_pdb_idx if entry_ref_space == "origin" else full_pdb_idx
                if ref_idx is None:
                    ref_idx = default_ref_idx

                anchor = self._resolve_position(getattr(entry, "anchor", None), L, ref_idx)
                if anchor is None:
                    continue

                mode = str(getattr(entry, "mode", "at_least_one")).lower()
                candidates = self._parse_candidates(
                    getattr(entry, "candidates", None),
                    L,
                    anchor,
                    ref_idx,
                )
                allowed = self._build_allowed_mask(B, L, anchor, candidates, device, masks)
                entry_top_k = max(int(getattr(entry, "top_k", self.top_k)), 0)
                if entry_top_k > 0:
                    allowed = self._restrict_allowed_to_top_k(updated[:, anchor, :], allowed, entry_top_k)
                if not allowed.any():
                    continue

                row_current = updated[:, anchor, :]
                row_allowed = row_current * allowed.float()
                allowed_degree = row_allowed.sum(dim=-1, keepdim=True)

                if mode in ("at_least_one", "degree_at_least_one", "degree_gte_one"):
                    target_degree = float(getattr(entry, "target_degree", 1.0))
                    target_degree_t = torch.full_like(allowed_degree, target_degree)
                    deficit = torch.clamp(target_degree_t - allowed_degree, min=0.0)

                    score = row_allowed
                    fallback = allowed.float()
                    denom = score.sum(dim=-1, keepdim=True)
                    share = torch.where(
                        denom > 1e-8,
                        score / (denom + 1e-8),
                        fallback / (fallback.sum(dim=-1, keepdim=True) + 1e-8),
                    )
                    row_target = row_current + deficit * share
                    row_new = (1.0 - alpha) * row_current + alpha * row_target
                    if should_debug:
                        new_allowed_degree = (row_new * allowed.float()).sum(dim=-1, keepdim=True)
                        loss_after = torch.clamp(target_degree_t - new_allowed_degree, min=0.0).pow(2).mean()
                        anchor_label = self._format_ref(ref_idx[anchor]) if ref_idx is not None and 0 <= anchor < len(ref_idx) else str(anchor)
                        debug_rows.append(
                            {
                                "anchor": anchor_label,
                                "mode": "at_least_one",
                                "degree": new_allowed_degree.mean().item(),
                                "target": target_degree,
                                "loss": loss_after.item(),
                            }
                        )

                elif mode in ("unique_in_candidates", "unique_candidate", "one_of_candidates"):
                    unique_target = float(getattr(entry, "unique_target", 1.0))
                    off_target = float(getattr(entry, "off_target", 0.0))
                    outside_target = float(getattr(entry, "outside_target", off_target))
                    score_masked = row_allowed.masked_fill(~allowed, float("-inf"))
                    best_idx = torch.argmax(score_masked, dim=-1)
                    onehot = F.one_hot(best_idx, num_classes=L).to(dtype=updated.dtype) * allowed.float()

                    valid_update_mask = torch.ones_like(allowed)
                    res_mask = masks.get("res_mask", None)
                    if res_mask is not None:
                        valid_update_mask = valid_update_mask & res_mask.bool()
                        if anchor < res_mask.shape[1]:
                            valid_update_mask = valid_update_mask & res_mask[:, anchor].bool().unsqueeze(1)
                    bond_mask_local = masks.get("bond_mask", None)
                    if bond_mask_local is not None and anchor < bond_mask_local.shape[1]:
                        valid_update_mask = valid_update_mask & bond_mask_local[:, anchor, :].bool()
                    valid_update_mask[:, anchor] = False

                    allowed_non_selected_mask = allowed & (~onehot.bool())
                    outside_mask_bool = valid_update_mask & (~allowed)

                    with torch.enable_grad():
                        row_var = row_current.detach().clone().requires_grad_(True)
                        count_now = self._compute_unique_soft_count(row_var, onehot.bool())
                        count_loss = (count_now - unique_target).pow(2)
                        tail_loss = self._masked_row_mean((row_var - off_target).pow(2), allowed_non_selected_mask)
                        outside_loss = self._masked_row_mean((row_var - outside_target).pow(2), outside_mask_bool)
                        energy = (count_loss + tail_loss + self._UNIQUE_OUTSIDE_WEIGHT * outside_loss).sum()
                        grad_row, = torch.autograd.grad(
                            energy,
                            row_var,
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=False,
                        )

                    grad_row = torch.nan_to_num(grad_row, nan=0.0, posinf=0.0, neginf=0.0)
                    grad_row = grad_row * valid_update_mask.float()
                    row_new = row_current - alpha * grad_row
                    row_new[:, anchor] = row_current[:, anchor]

                    if should_debug:
                        row_new_dbg = torch.clamp(row_new, min=0.0, max=1.0)
                        row_new_allowed = row_new_dbg * allowed.float()
                        best_idx_new = torch.argmax(row_new_allowed.masked_fill(~allowed, float("-inf")), dim=-1)
                        onehot_new = F.one_hot(best_idx_new, num_classes=L).to(dtype=row_new_dbg.dtype) * allowed.float()
                        allowed_non_selected_new = allowed & (~onehot_new.bool())
                        count_after = self._compute_unique_soft_count(row_new_dbg, onehot_new.bool())
                        tail_after = self._masked_row_mean((row_new_dbg - off_target).pow(2), allowed_non_selected_new)
                        outside_after_loss = self._masked_row_mean((row_new_dbg - outside_target).pow(2), outside_mask_bool)
                        loss_after = (count_after - unique_target).pow(2) + tail_after + self._UNIQUE_OUTSIDE_WEIGHT * outside_after_loss
                        chosen_after = (row_new_dbg * onehot_new).sum(dim=-1).mean()
                        other_allowed_after = (row_new_dbg * allowed_non_selected_new.float()).sum(dim=-1).mean()
                        outside_after = (row_new_dbg * outside_mask_bool.float()).sum(dim=-1).mean()
                        if ref_idx is not None and len(best_idx_new) > 0:
                            best_partner_idx = int(best_idx_new[0].item())
                            if 0 <= best_partner_idx < len(ref_idx):
                                partner_label = self._format_ref(ref_idx[best_partner_idx])
                            else:
                                partner_label = str(best_partner_idx)
                        else:
                            partner_label = None
                        anchor_label = self._format_ref(ref_idx[anchor]) if ref_idx is not None and 0 <= anchor < len(ref_idx) else str(anchor)
                        debug_rows.append(
                            {
                                "anchor": anchor_label,
                                "mode": "unique_in_candidates",
                                "count_method": f"selected^{int(self._UNIQUE_SELECTED_BETA)}",
                                "count": count_after.mean().item(),
                                "chosen": chosen_after.item(),
                                "other_allowed": other_allowed_after.item(),
                                "outside": outside_after.item(),
                                "tail_loss": tail_after.mean().item(),
                                "outside_loss": outside_after_loss.mean().item(),
                                "loss": loss_after.mean().item(),
                                "partner": partner_label,
                            }
                        )

                else:
                    continue

                row_new = torch.clamp(row_new, min=0.0, max=1.0)

                bond_mask = masks.get("bond_mask", None)
                if bond_mask is not None:
                    allowed_row_update = bond_mask[:, anchor, :].bool()
                    row_new = torch.where(allowed_row_update, row_new, row_current)

                updated[:, anchor, :] = row_new
                updated[:, :, anchor] = row_new
                updated[:, anchor, anchor] = ss_t_2[:, anchor, anchor]

            if should_debug:
                self._print_debug(debug_rows, iter_idx=iter_idx, t_1=t_1, alpha=alpha)

        step_out["ss_t_2"] = updated
        return step_out


_GUIDANCE_REGISTRY = {
    "logits_bias": LogitsBiasGuidance,
    "trans_anchor": TransAnchorGuidance,
    "single_bond": SingleBondGuidance,
    "soft_bond_count": SoftBondCountGuidance,
    "type_soft_bond_count": TypeAwareSoftBondCountGuidance,
    "residue_bond": ResidueBondGuidance,
    "clash": ClashGuidance,
    "radius_of_gyration": RadiusOfGyrationGuidance,
}
