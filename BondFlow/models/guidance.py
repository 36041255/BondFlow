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
        self.index_base = int(getattr(cfg, "index_base", 0)) if cfg is not None else 0
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
        pos_mask_1d = self._parse_positions(self.positions, L, index_base=self.index_base)
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
      target_N: 1          # desired number of bonds (soft)
      alpha: 20.0          # sigmoid sharpness around tau
      tau: 0.5             # threshold around which "bond" is counted
      eta: 0.1             # gradient step size
      n_steps: 1           # how many inner GD steps per sampling step
      sinkhorn_iters: 5    # Sinkhorn iterations per projection
      eps: 1e-8            # numerical epsilon for normalisation
    """

    def __init__(self, cfg: Optional[Any] = None, device: str = "cpu") -> None:
        super().__init__(cfg, device)
        self.mode = str(getattr(cfg, "mode", "exact_N")).lower() if cfg is not None else "exact_N"
        self.target_N = float(getattr(cfg, "target_N", 1.0)) if cfg is not None else 1.0
        self.alpha = float(getattr(cfg, "alpha", 20.0)) if cfg is not None else 20.0
        self.tau = float(getattr(cfg, "tau", 0.5)) if cfg is not None else 0.5
        
        # 新增 top_k_soft 参数
        self.top_k_soft = int(getattr(cfg, "top_k_soft", 0)) if cfg is not None else 0

        # eta -> weight
        self.weight = float(getattr(cfg, "weight",)) if cfg is not None else 0.1
        
        self.n_steps = int(getattr(cfg, "n_steps", 1)) if cfg is not None else 1
        self.sinkhorn_iters = int(getattr(cfg, "sinkhorn_iters", 30)) if cfg is not None else 30
        self.eps = float(getattr(cfg, "eps", 1e-8)) if cfg is not None else 1e-8
        
        # 实例化 DSM 模块，tau=1.0 配合 log 输入
        self.dsm = DSMProjection(base_tau = self.tau, max_iter=self.sinkhorn_iters, eps=self.eps)

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

        # Time-dependent step size: weight_t = schedule(t_1) * weight
        # 以前叫 eta_t，现在改叫 weight_t，逻辑不变
        weight_t = _schedule_weight(t_1, self.weight, schedule="linear", power=1.0)  # [B]
        weight_t = weight_t.view(B, 1, 1)

        # Start from current matrix; we will refine it locally.
        
        # 1. 初始化切空间变量 ss_logits
        # ss_t_2 是概率值 [0, 1]，我们需要将其映射到无约束的 logits 空间。
        # 逆 Sinkhorn 并不简单，但作为初始值，我们可以用 tau * log(ss + eps) 近似。
        # 注意：这里我们优化的变量是 ss_logits，而不是 ss。
        ss_val = ss_t_2.detach()
        # 避免 log(0)
        ss_val = ss_val.clamp(min=1e-12)
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
                
                # Note: Energy calculation is global (counts all bonds), including those in masked-out regions.
                # This ensures we count existing fixed bonds.
                
                if self.top_k_soft > 0:
                    # 如果启用了 top_k_soft，只取前 k 个最大的概率值来计算 C
                    flat_probs = P_off.reshape(P_off.shape[0], -1)
                    # topk values: [B, k]
                    top_vals, _ = torch.topk(flat_probs, k=min(self.top_k_soft, flat_probs.shape[1]), dim=1)
                    C = 0.5 * top_vals.sum(dim=1) # [B]
                else:
                    C = 0.5 * P_off.sum(dim=(1, 2))  # [B]

                if self.mode == "exact_n":
                    # Encourage C ≈ target_N
                    energy = ((C - self.target_N) ** 2).mean()
                elif self.mode == "at_least_n":
                    # Encourage C >= target_N (no penalty when already above)
                    diff = torch.relu(self.target_N - C)
                    energy = (diff ** 2).mean()
                else:
                    # Unknown mode: no-op
                    return step_out

                # 3. 对 ss_logits 求导
                grad_logits, = torch.autograd.grad(
                    energy, ss_logits, retain_graph=False, create_graph=False
                )
                print("C ------------------------",C )
                print("energy ------------------------",energy )
            # 4. 在切空间更新 ss_logits
            # 注意：如果 bond_mask 存在，虽然 DSM 会处理前向的屏蔽，但更新 logits 时也可以
            # 显式屏蔽梯度以避免无关区域的 logits 漂移（虽然它们在前向时会被 mask 掉）。
            if bond_mask is not None:
                grad_logits = grad_logits * bond_mask.float()

            # 使用 weight_t 作为步长
            # 注意：这里的步长尺度可能需要根据 logits 的性质微调，但通常 weight_t 即可。
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
              mode: exact_N        # "exact_N" | "at_least_N" | "fixed_pairs" | "only_fixed_pairs"
              target_N: 1.0
              weight: 1.0

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
        
        # 顶层模式
        self.global_mode = str(getattr(cfg, "mode", "type")).lower() if cfg is not None else "type"
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
        self.eps = float(getattr(cfg, "eps", 1e-8)) if cfg is not None else 1e-8
        # seq_step: when None (yaml null), we switch to "no-grad direct assignment" for fixed-pair modes.
        if cfg is not None:
            _seq_step_raw = getattr(cfg, "seq_step", 0.05)
            self.seq_step: Optional[float] = None if _seq_step_raw is None else float(_seq_step_raw)
        else:
            self.seq_step = 0.05
        self.schedule = str(getattr(cfg, "schedule", "linear")) if cfg is not None else "linear"
        self.power = float(getattr(cfg, "power", 1.0)) if cfg is not None else 1.0
        self.tau = float(getattr(cfg, "tau", 0.5)) if cfg is not None else 0.5
        # "all" 模式参数
        self.all_count_mode = str(getattr(cfg, "all_count_mode", "exact_N")).lower() if cfg is not None else "exact_n"
        self.all_target_N = float(getattr(cfg, "all_target_N", getattr(cfg, "target_N", 1.0))) if cfg is not None else 1.0

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
                loss_type = str(getattr(t_cfg, "loss_type", "kl")).lower()
                target_N = float(getattr(t_cfg, "target_N", 1.0))
                weight = float(getattr(t_cfg, "weight", 1.0))
                top_k_soft = int(getattr(t_cfg, "top_k_soft", 0))  # 新增
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
                        top_k_soft=top_k_soft,  # 新增
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

            top_k = int(getattr(t_cfg, "top_k_soft", 0))

            target = float(t_cfg.target_N)
            min_esp = 1e-8
            mode = str(getattr(t_cfg, "mode", "exact_n")).lower()
            loss_type = str(getattr(t_cfg, "loss_type", "kl")).lower()

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
                    
                    if top_k > 0:
                        flat_target = P_target.reshape(B, -1)
                        top_vals, _ = torch.topk(flat_target, k=min(top_k, flat_target.shape[1]), dim=1)
                        C_target = 0.5 * top_vals.sum(dim=1)
                    else:
                        C_target = 0.5 * P_target.sum(dim=(1, 2))

                    if loss_type == "mse":
                        E_target = (C_target - target) ** 2
                    else:
                        C_safe = torch.where(C_target < min_esp, C_target/C_target.detach()*min_esp, C_target)
                        E_target = (C_safe - target + target * torch.log(target / C_safe))

                    if mode == "fixed_pairs":
                        E_tau = E_target.mean()
                    else:
                        # New semantics: only_fixed_pairs uses a hard fixed pairing matrix (handled above),
                        # and we do NOT add extra diagonal regularization here.
                        E_tau = E_target.mean()
            elif mode == "exact_n":
                if top_k > 0:
                    flat_eff = P_eff.reshape(B, -1)
                    top_vals, _ = torch.topk(flat_eff, k=min(top_k, flat_eff.shape[1]), dim=1)
                    C_tau = 0.5 * top_vals.sum(dim=1)
                else:
                    C_tau = 0.5 * P_eff.sum(dim=(1, 2))

                if loss_type == "mse":
                    E_tau = ((C_tau - target) ** 2).mean()
                else:
                    if target < min_esp:
                         C_safe = torch.where(C_tau < min_esp, C_tau/C_tau.detach()*min_esp, C_tau)
                         E_tau = C_safe.mean()
                    else:
                        C_safe = torch.where(C_tau < min_esp, C_tau/C_tau.detach()*min_esp, C_tau)
                        E_tau = (C_safe - target + target * torch.log(target / C_safe)).mean()
            elif mode == "at_least_n":
                if top_k > 0:
                    flat_eff = P_eff.reshape(B, -1)
                    top_vals, _ = torch.topk(flat_eff, k=min(top_k, flat_eff.shape[1]), dim=1)
                    C_tau = 0.5 * top_vals.sum(dim=1)
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
                    E_tau = E_vals.mean()
                else:
                    E_tau = P.new_tensor(0.0)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            print("E_tau",E_tau,"type_name",type_name)
            total_energy = total_energy + float(t_cfg.weight) * E_tau
        # print("total_energy",total_energy)
        return total_energy

    def _pre_model_all(
        self,
        bond_mat_pred: torch.Tensor,
        t_1: Optional[torch.Tensor],
        res_mask: Optional[torch.Tensor],
        bond_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        "all" 模式：忽略键类型与 logits，仅对整体 bond_mat 做 SoftBondCount 风格的软计数约束。
        """
        ss = bond_mat_pred.detach()
        ss_orig = ss.clone()
        B, L, _ = ss.shape
        device = ss.device

        if t_1 is None:
            eta_t = self.bond_step
            eta_t = torch.full((B, 1, 1), eta_t, device=device, dtype=ss.dtype)
        else:
            eta_t = _schedule_weight(t_1, self.bond_step, schedule=self.schedule, power=self.power).view(B, 1, 1)

        # 1. 变量变换：ss -> ss_logits
        ss_val = ss.clamp(min=1e-12)
        ss_logits = (self.tau * torch.log(ss_val)).clone().detach().requires_grad_(True)

        for _ in range(max(self.n_steps, 0)):
            with torch.enable_grad():
                # 2. 前向 Sinkhorn
                P = self.dsm(ss_logits, mask_2d=bond_mask, mat_true=ss_orig)

                if res_mask is None:
                    pair_mask = torch.ones(B, L, L, dtype=torch.bool, device=device)
                else:
                    pair_mask = (res_mask.unsqueeze(1) & res_mask.unsqueeze(2))
                eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
                pair_mask = pair_mask & (~eye)

                P_eff = P * pair_mask.float()
                C = 0.5 * P_eff.sum(dim=(1, 2))

                min_esp = 1e-8
                mode = self.all_count_mode.lower()

                if mode in ["exact_n", "exact n"]:
                    target = float(self.all_target_N)
                    C_safe = torch.where(C < min_esp, C / C.detach() * min_esp, C)
                    if target < min_esp:
                        energy = C_safe.mean()
                    else:
                        energy = (C_safe - target + target * torch.log(target / C_safe)).mean()
                elif mode == "at_least_n":
                    target = float(self.all_target_N)
                    need = C < target
                    if need.any():
                        C_need = torch.where(C[need] < min_esp, C[need] / C[need].detach() * min_esp, C[need])
                        if target < min_esp:
                            E_vals = C_need
                        else:
                            E_vals = C_need - target + target * torch.log(target / C_need)
                        energy = E_vals.mean()
                    else:
                        energy = ss_logits.new_tensor(0.0)
                else:
                    energy = ss_logits.new_tensor(0.0)

                grad_logits, = torch.autograd.grad(
                    energy, ss_logits, retain_graph=False, create_graph=False
                )

            if bond_mask is not None:
                grad_logits = grad_logits * bond_mask.float()

            # 3. 更新 ss_logits
            ss_logits = ss_logits - eta_t * grad_logits

        # Final projection
        with torch.no_grad():
            ss_projected = self.dsm(ss_logits, mask_2d=bond_mask, mat_true=ss_orig)

        return ss_projected.detach()

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

        if self.global_mode == "all":
            ss_projected = self._pre_model_all(
                bond_mat_pred=bond_mat_pred,
                t_1=t_1,
                res_mask=res_mask,
                bond_mask=bond_mask,
            )
            model_raw["bond_mat_pred"] = ss_projected
            return model_raw

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
        ss_val = ss.clamp(min=1e-12)
        ss_logits = (self.tau * torch.log(ss_val)).clone().detach().requires_grad_(True)

        for _ in range(max(self.n_steps, 0)):
            with torch.enable_grad():
                logits_var = logits_work.clone().requires_grad_(True)

                # 2. 前向 Sinkhorn
                P = self.dsm(ss_logits, mask_2d=bond_mask, mat_true=ss_orig)

                energy = self._compute_energy(P, logits_var, res_mask, head_mask, tail_mask)

                if not torch.isfinite(energy):
                    print("error in bond type guidence")
                    break
                print("energy----------------------",energy)
                grads = torch.autograd.grad(
                    energy,
                    (ss_logits, logits_var),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_ss_logits, grad_seq_logits = grads

            if grad_ss_logits is None and grad_seq_logits is None:
                break

            if grad_ss_logits is not None:
                # Only update bonds when bond_step is provided.
                if eta_t is not None:
                    if bond_mask is not None:
                        grad_ss_logits = grad_ss_logits * bond_mask.float()
                    # 3. 更新 ss_logits
                    # 打印 ss 梯度和目标的数量级
                    print(
                        f"[ss] grad_ss_logits mean: {grad_ss_logits.abs().mean().item():.3e}, "
                        f"max: {grad_ss_logits.abs().max().item():.3e}, "
                        f"ss_logits mean: {ss_logits.abs().mean().item():.3e}, "
                        f"max: {ss_logits.abs().max().item():.3e}, "
                        f"eta_t: {eta_t.mean().item():.3e}"
                    )
                    ss_logits = ss_logits - eta_t * grad_ss_logits

            if grad_seq_logits is not None and (self.seq_step is not None) and self.seq_step > 0 and (seq_step_t is not None):
                if seq_mask is not None:
                    print(f"[seq] grad_seq_logits mean: {grad_seq_logits.abs().mean().item():.3e}, "
                    f"max: {grad_seq_logits.abs().max().item():.3e}, "
                    f"seq_logits mean: {logits_work.abs().mean().item():.3e}, "
                    f"max: {logits_work.abs().max().item():.3e}, "
                    f"seq_step_t: {seq_step_t.mean().item():.3e}")
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

        if self.global_mode == "all":
            ss_projected = self._pre_model_all(
                bond_mat_pred=ss_t_2,
                t_1=t_1,
                res_mask=res_mask,
                bond_mask=bond_mask,
            )
            step_out["ss_t_2"] = ss_projected
            return step_out

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
        ss_val = ss.clamp(min=1e-12)
        ss_logits = (self.tau * torch.log(ss_val)).clone().detach().requires_grad_(True)

        # 2. 准备 seq_logits (从离散 aatypes_t_2 转换)
        K = self.num_aatypes
        max_idx = aatypes_t_2.max().item()
        vocab_size = max(K, int(max_idx) + 1)
        
        seq_one_hot = F.one_hot(aatypes_t_2.long(), num_classes=vocab_size).float()
        # 映射到 log 空间，加 eps 避免 log(0)
        seq_logits = torch.log(seq_one_hot + 1e-6).detach().requires_grad_(True)

        for _ in range(max(self.n_steps, 0)):
            with torch.enable_grad():
                logits_var = seq_logits
                
                # 2. 前向 Sinkhorn
                P = self.dsm(ss_logits, mask_2d=bond_mask, mat_true=ss_orig)

                energy = self._compute_energy(P, logits_var, res_mask, head_mask, tail_mask)

                if not torch.isfinite(energy):
                    print("error in bond type guidence (post_step)")
                    break
                # print("energy(post)----------------------",energy)
                grads = torch.autograd.grad(
                    energy,
                    (ss_logits, logits_var),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_ss_logits, grad_seq_logits = grads

            if grad_ss_logits is None and grad_seq_logits is None:
                break

            if grad_ss_logits is not None:
                # Only update bonds when bond_step is provided.
                if eta_t is not None:
                    if bond_mask is not None:
                        grad_ss_logits = grad_ss_logits * bond_mask.float()
                    # 打印 ss 梯度和目标的数量级
                    print(
                        f"[ss] grad_ss_logits mean: {grad_ss_logits.abs().mean().item():.3e}, "
                        f"max: {grad_ss_logits.abs().max().item():.3e}, "
                        f"ss_logits mean: {ss_logits.abs().mean().item():.3e}, "
                        f"max: {ss_logits.abs().max().item():.3e}, "
                        f"eta_t: {eta_t.mean().item():.3e}"
                    )
                    ss_logits = ss_logits - eta_t * grad_ss_logits

            if grad_seq_logits is not None and (self.seq_step is not None) and self.seq_step > 0 and (seq_step_t is not None):
                if seq_mask is not None:
                    grad_seq_logits = grad_seq_logits * seq_mask.unsqueeze(-1).float()
                # 打印 seq 梯度和目标的数量级
                print(f"[seq] grad_seq_logits mean: {grad_seq_logits.abs().mean().item():.3e}, "
                      f"max: {grad_seq_logits.abs().max().item():.3e}, "
                      f"seq_logits mean: {seq_logits.abs().mean().item():.3e}, "
                      f"max: {seq_logits.abs().max().item():.3e}, "
                      f"seq_step_t: {seq_step_t.mean().item():.3e}")
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
    """

    def __init__(self, cfg: Optional[Any] = None, device: str = "cpu") -> None:
        super().__init__(cfg, device)
        self.start_t = float(getattr(cfg, "start_t", 0.5))
        self.weight = float(getattr(cfg, "weight", 1.0))
        self.n_steps = int(getattr(cfg, "n_steps", 1))
        self.link_csv_path = getattr(cfg, "link_csv_path", None)

        self.clash_loss = OpenFoldClashLoss(
            link_csv_path=self.link_csv_path,
            device=device,
            log_raw=False,
            include_within=True,
            treat_adjacent_as_bonded=True
        )

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

                # Compute loss
                loss = self.clash_loss(
                    nc_anchor,
                    coords_14,
                    aatypes_t_2.long(),
                    res_mask,
                    bond_mat=bond_mat,
                    head_mask=head_mask,
                    tail_mask=tail_mask
                )
                print("loss_clash:", loss.item())
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


_GUIDANCE_REGISTRY = {
    "logits_bias": LogitsBiasGuidance,
    "trans_anchor": TransAnchorGuidance,
    "single_bond": SingleBondGuidance,
    "soft_bond_count": SoftBondCountGuidance,
    "type_soft_bond_count": TypeAwareSoftBondCountGuidance,
    "clash": ClashGuidance,
}
