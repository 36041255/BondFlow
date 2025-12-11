import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from BondFlow.data.link_utils import LinkInfo


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

    def pre_model(self, model_raw: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        if self.bias is None:
            return model_raw
        logits: torch.Tensor = model_raw.get("logits")
        if logits is None:
            return model_raw
        t_1: torch.Tensor = context.get("t_1")
        if t_1 is None:
            return model_raw
        # ensure bias length matches first 20 channels or broadcast
        bias = self.bias
        if bias.numel() not in (1, 20):
            return model_raw
        # compute weight per-batch
        w = _schedule_weight(t_1, self.weight, self.schedule, self.power)  # [B]
        while w.dim() < logits.dim():
            w = w.unsqueeze(-1)

        # apply to first 20 logits
        if bias.numel() == 1:
            model_raw["logits"] = logits.clone()
            model_raw["logits"][..., :20] = logits[..., :20] + w * bias
        else:
            model_raw["logits"] = logits.clone()
            model_raw["logits"][..., :20] = logits[..., :20] + w * bias.unsqueeze(0).unsqueeze(0)
        
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
        # compute blending weight
        w = _schedule_weight(t_1, self.weight, self.schedule, self.power).view(-1, 1, 1)
        # blend toward anchor
        updated = model_out.copy()
        updated["pred_trans"] = (1.0 - w) * pred_trans + w * target
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

    @staticmethod
    def _symmetric_sinkhorn(M: torch.Tensor, n_iters: int = 5, eps: float = 1e-8) -> torch.Tensor:
        """
        Approximate projection of M onto the set of symmetric doubly-stochastic matrices
        using alternating row/column normalization plus symmetrization.
        """
        for _ in range(n_iters):
            # Row normalize
            row_sum = M.sum(dim=-1, keepdim=True).clamp_min(eps)
            M = M / row_sum
            # Column normalize
            col_sum = M.sum(dim=-2, keepdim=True).clamp_min(eps)
            M = M / col_sum
            # Symmetrize
            M = 0.5 * (M + M.transpose(-1, -2))
        return M

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

        # 3) 对 guided 做对称 Sinkhorn 投影，尽量恢复对称双随机
        guided = self._symmetric_sinkhorn(guided)

        # 4) blend with original using time-dependent weight
        step_out["ss_t_2"] = (1.0 - w) * ss_t_2 + w * guided
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
        self.eta = float(getattr(cfg, "eta", 0.1)) if cfg is not None else 0.1
        self.n_steps = int(getattr(cfg, "n_steps", 1)) if cfg is not None else 1
        self.sinkhorn_iters = int(getattr(cfg, "sinkhorn_iters", 5)) if cfg is not None else 20
        self.eps = float(getattr(cfg, "eps", 1e-8)) if cfg is not None else 1e-8

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

        B, L, _ = ss_t_2.shape
        device = ss_t_2.device

        # Time-dependent step size: eta_t = schedule(t_1) * eta
        eta_t = _schedule_weight(t_1, self.eta, schedule="linear", power=1.0)  # [B]
        eta_t = eta_t.view(B, 1, 1)

        # Start from current matrix; we will refine it locally.
        
        ss = ss_t_2.detach()
        for _ in range(max(self.n_steps, 0)):
            with torch.enable_grad():
                ss_var = ss.clone().requires_grad_(True)  # [B, L, L]

                # Non-negative projection then symmetric Sinkhorn to stay near doubly stochastic
                M = ss_var.clamp_min(0.0)
                P = SingleBondGuidance._symmetric_sinkhorn(M, n_iters=self.sinkhorn_iters, eps=self.eps)

                # Remove diagonal when counting bonds
                eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
                P_off = P.masked_fill(eye, 0.0)

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

                grad_ss, = torch.autograd.grad(
                    energy, ss_var, retain_graph=False, create_graph=False
                )

            # Gradient descent step on ss, with time-dependent step size

            ss = ss - eta_t * grad_ss

            # Keep non-negative after each step
            # INSERT_YOUR_CODE
            # Shift ss so its minimum value in each batch is 0 (if there are negatives)
            min_ss = ss.view(B, -1).min(dim=1, keepdim=True)[0].clamp_max(0.0).view(B, 1, 1)
            ss = ss - min_ss


        # Final projection to (approx) symmetric doubly-stochastic
        ss_projected = SingleBondGuidance._symmetric_sinkhorn(ss, n_iters=self.sinkhorn_iters, eps=self.eps)
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

        # 超参数（随时间通过 schedule/power 加权）
        # bond_step: 每步对键合矩阵更新的基础步长（再乘以时间 schedule）
        self.bond_step = float(getattr(cfg, "bond_step", getattr(cfg, "eta", 0.1))) if cfg is not None else 0.1
        self.n_steps = int(getattr(cfg, "n_steps", 1)) if cfg is not None else 1
        self.sinkhorn_iters = int(getattr(cfg, "sinkhorn_iters", 5)) if cfg is not None else 5
        self.eps = float(getattr(cfg, "eps", 1e-8)) if cfg is not None else 1e-8
        # 序列 logits guidance 步长
        self.seq_step = float(getattr(cfg, "seq_step", 0.05)) if cfg is not None else 0.05
        # 时间调度
        self.schedule = str(getattr(cfg, "schedule", "linear")) if cfg is not None else "linear"
        self.power = float(getattr(cfg, "power", 1.0)) if cfg is not None else 1.0

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
            # 向后兼容：如果没有显式 types，就假设只做一个二硫键
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
                target_N = float(getattr(t_cfg, "target_N", 1.0))
                weight = float(getattr(t_cfg, "weight", 1.0))
                # 可选：针对该类型的“固定闭环位点”，以 0-based (i,j) 残基对列表给出
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
                        target_N=target_N,
                        weight=weight,
                        pairs=pairs,
                    )
                )

        # 预计算：对每种键类型，构建 (K,K) 的残基对允许矩阵
        # K 取自 compat_matrix 或回退为 21
        if self.link_info is not None and getattr(self.link_info, "compat_matrix", None) is not None:
            K = int(self.link_info.compat_matrix.shape[0])
        else:
            K = 21
        self.num_aatypes = K

        self.type_pair_mats: Dict[str, torch.Tensor] = {}
        # 额外记录：哪些 (res1,res2) 具有“端基相关”的共价规则（atom1 或 atom2 为 N/C）
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
        """
        根据 link.csv 规则粗略分类：
          - SG-SG             -> disulfide
          - OG/OG1/OH - CG/CD -> lactone (简化)
          - NZ - CG/CD        -> isopeptide (简化)
        """

        def norm(name: Any) -> str:
            return (str(name).strip().upper()) if name is not None else ""

        # 先查找二硫键
        for r in rules:
            a1, a2 = norm(r.get("atom1")), norm(r.get("atom2"))
            if a1 == "SG" and a2 == "SG":
                return "disulfide"

        # 内酯（非常简化的判断，仅作为引导）
        for r in rules:
            a1, a2 = norm(r.get("atom1")), norm(r.get("atom2"))
            if (a1 in ("OG", "OG1", "OH") and a2 in ("CG", "CD")) or (
                a2 in ("OG", "OG1", "OH") and a1 in ("CG", "CD")
            ):
                return "lactone"

        # 异肽键（Lys NZ 与酸性侧链 CG/CD）
        for r in rules:
            a1, a2 = norm(r.get("atom1")), norm(r.get("atom2"))
            if (a1 == "NZ" and a2 in ("CG", "CD")) or (a2 == "NZ" and a1 in ("CG", "CD")):
                return "isopeptide"

        return None

    def _build_type_pair_mats(self, K: int) -> None:
        """从 link_info.bond_spec 构建每个类型的 (K,K) 残基对允许矩阵。"""
        device = self.device
        type_mats = {
            "disulfide": torch.zeros((K, K), dtype=torch.bool, device=device),
            "isopeptide": torch.zeros((K, K), dtype=torch.bool, device=device),
            "lactone": torch.zeros((K, K), dtype=torch.bool, device=device),
            # "covalent": 所有在 bond_spec 中出现过、存在至少一条“非 N/C 端基规则”的残基对
            "covalent": torch.zeros((K, K), dtype=torch.bool, device=device),
        }
        covalent_terminal = torch.zeros((K, K), dtype=torch.bool, device=device)

        for (r1, r2), rules in self.link_info.bond_spec.items():
            if r1 >= K or r2 >= K:
                continue

            # 1) covalent: 统计“非端基”的共价规则（atom1/atom2 都不是 N/C）
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

            # 2) 具体类型：基于侧链原子模式分类（不把端基 C/N 规则误算进来）
            bond_type = self._classify_rules(rules)
            if bond_type is None:
                continue
            if bond_type not in type_mats:
                continue
            type_mats[bond_type][r1, r2] = True
            type_mats[bond_type][r2, r1] = True

        self.type_pair_mats = type_mats
        self.covalent_terminal_pairs = covalent_terminal

    def _compute_energy(
        self,
        P: torch.Tensor,
        logits: torch.Tensor,
        res_mask: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor],
        tail_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        给定 Sinkhorn 后的 bond 矩阵 P 和 logits，按类型计算软计数能量。
        同一个能量同时用于：
          - 对 ss/bond_mat_pred 做梯度下降（通过 P 依赖于 ss）
          - 对 logits 做梯度下降（通过类型兼容度对 aa 分布的依赖）
        """
        if not self.type_cfgs:
            return P.new_tensor(0.0)

        B, L, _ = P.shape
        device = P.device

        # pair mask: 有效残基 & 非对角
        if res_mask is None:
            res_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        pair_mask = (res_mask.unsqueeze(1) & res_mask.unsqueeze(2))
        eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & (~eye)

        P_off = P * pair_mask.float()  # [B,L,L]

        K = min(self.num_aatypes, logits.shape[-1])
        # 概率分布 [B,L,K]
        probs = F.softmax(logits[..., :K], dim=-1)

        total_energy = P.new_tensor(0.0)
        for t_cfg in self.type_cfgs:
            type_name = t_cfg.name
            type_mat = self.type_pair_mats.get(type_name)
            if type_mat is None:
                continue
            if not type_mat.any():
                continue

            type_mat_dev = type_mat.to(device=device, dtype=probs.dtype)  # [K,K]

            # 对每个残基对 (i,j)，计算它成为该类型键的“软概率”
            # compat[b,i,j] = probs[b,i,:] @ type_mat @ probs[b,j,:]^T
            Cp = torch.matmul(probs, type_mat_dev)  # [B,L,K]
            compat = torch.einsum("bik,bjk->bij", Cp, probs)  # [B,L,L]

            # covalent 类型：额外加入“端基相关”的共价规则，但只对真实端基位置开放
            if type_name == "covalent" and self.covalent_terminal_pairs is not None:
                term_mat = self.covalent_terminal_pairs.to(device=device, dtype=probs.dtype)
                Cp_term = torch.matmul(probs, term_mat)  # [B,L,K]
                compat_term = torch.einsum("bik,bjk->bij", Cp_term, probs)  # [B,L,L]
                # 仅当 (i,j) 至少有一方是真正的 N/C 端（由 head_mask/tail_mask 指示）时，才允许端基规则贡献
                if head_mask is not None or tail_mask is not None:
                    if head_mask is None:
                        head_mask = torch.zeros_like(res_mask, dtype=torch.bool, device=device)
                    if tail_mask is None:
                        tail_mask = torch.zeros_like(res_mask, dtype=torch.bool, device=device)
                    term_gate = (
                        head_mask.unsqueeze(1) | head_mask.unsqueeze(2) |
                        tail_mask.unsqueeze(1) | tail_mask.unsqueeze(2)
                    )  # [B,L,L]
                    compat_term = compat_term * term_gate.float()
                compat = compat + compat_term

            compat = compat * pair_mask.float()

            # 使用 compat 作为类型权重，对 P_off 做软计数
            P_eff = P_off * compat  # [B,L,L]

            target = float(t_cfg.target_N)
            min_esp = 1e-8
            mode = str(getattr(t_cfg, "mode", "exact_n")).lower()

            if mode in ("fixed_pairs", "only_fixed_pairs"):
                # 两种模式共用同一套“目标对”统计逻辑：
                #   - fixed_pairs: 只在指定 pairs 上用 Poisson-KL 约束，不额外惩罚其他位置；
                #   - only_fixed_pairs: 在此基础上再强烈压制 off-target 的同类型键。
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
                    C_target = 0.5 * P_target.sum(dim=(1, 2))  # [B]

                    # 目标：指定位点上接近 target_N（通常是 1）
                    C_safe = torch.where(C_target < min_esp, C_target/C_target.detach()*min_esp, C_target)
                    E_target = (C_safe - target + target * torch.log(target / C_safe))

                    if mode == "fixed_pairs":
                        # 只对目标 pairs 做约束，不过不禁止其它位置也成键
                        E_tau = E_target.mean()
                    else:  # "only_fixed_pairs"
                        # 对“理论上不参与该类型键”的残基，在对角线位置做目标为 1 的交叉熵惩罚，
                        # 鼓励这些残基把概率质量留在 self（不与其他残基形成该类型键）。
                        #
                        # 1) 找出哪些残基出现在任意 fixed_pairs 中 → 这些残基“允许成键”
                        residue_need_bond = torch.zeros(L, dtype=torch.bool, device=device)
                        for (ii, jj) in t_cfg.pairs:
                            if 0 <= ii < L and 0 <= jj < L:
                                residue_need_bond[ii] = True
                                residue_need_bond[jj] = True
                        # 其余残基视为“不需要该类型键”，但仍受 res_mask 约束
                        valid_res_mask = res_mask.bool()  # [B,L]
                        residue_no_bond = (~residue_need_bond).unsqueeze(0) & valid_res_mask  # [B,L]

                        # 2) 取出 Sinkhorn 后的对角线概率 P[b,i,i]（该类型的“self”概率）
                        diag_idx = torch.arange(L, device=device)
                        P_diag = P[:, diag_idx, diag_idx]  # [B,L]

                        # 3) 对 no_bond 残基做目标为 1 的 Bernoulli 交叉熵：-log(P_diag)
                        eps = 1e-6
                        P_diag_clamped = torch.where(P_diag < eps, P_diag/P_diag.detach()*eps, P_diag)
                        neglog = -torch.log(P_diag_clamped)  # [B,L]
                        mask_f = residue_no_bond.float()
                        # 每个 batch 内对“应当不成键”的残基做平均
                        per_batch_loss = (neglog * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
                        E_diag = per_batch_loss  # [B]

                        # 4) 总能量：目标对上的 Poisson-KL + 非目标残基的对角线交叉熵
                        E_tau = (E_target + E_diag).mean()

            elif mode == "exact_n":
                C_tau = 0.5 * P_eff.sum(dim=(1, 2))  # [B]
                # Poisson KL: KL(Pois(target) || Pois(C_tau))
                # E = C_tau - target + target * log(target / C_tau)
                C_safe = torch.where(C_tau < min_esp, C_tau/C_tau.detach()*min_esp, C_tau)
                E_tau = (C_safe - target + target * torch.log(target / C_safe)).mean()
            elif mode == "at_least_n":
                C_tau = 0.5 * P_eff.sum(dim=(1, 2))  # [B]
                # 只惩罚 C_tau < target 的 batch；C_tau >= target 时 loss = 0
                need = C_tau < target  # [B]
                if need.any():
                    C_need = torch.where(C_tau[need] < min_esp, C_tau[need]/C_tau[need].detach()*min_esp, C_tau[need])
                    # 同样使用 Poisson KL 形式，但只在未达标的 batch 上计算
                    E_vals = C_need - target + target * torch.log(target / C_need)
                    E_tau = E_vals.mean()
                else:
                    E_tau = P.new_tensor(0.0)
            else:
                # 未知模式：不贡献能量
                raise ValueError(f"Unknown mode: {mode}")
                E_tau = P.new_tensor(0.0)

            total_energy = total_energy + float(t_cfg.weight) * E_tau
        print("total_energy",total_energy)
        return total_energy

    def pre_model(self, model_raw: Dict[str, torch.Tensor], **context: Any) -> Dict[str, torch.Tensor]:
        """
        在模型 forward 之后、Interpolant 之前，对 bond_mat_pred 和 logits 做一次
        type-aware soft bond-count guidance：
          - 先对 bond_mat_pred 做几步梯度下降，使得各类型的软计数接近目标 N_tau；
          - 同时把能量梯度回传到 logits 上，做一小步 logits guidance。
        """
        if self.link_info is None or not self.type_pair_mats:
            return model_raw

        bond_mat_pred: Optional[torch.Tensor] = model_raw.get("bond_mat_pred", None)
        logits: Optional[torch.Tensor] = model_raw.get("logits", None)
        if bond_mat_pred is None or logits is None:
            return model_raw

        t_1: Optional[torch.Tensor] = context.get("t_1", None)
        masks: Dict[str, torch.Tensor] = context.get("masks", {}) or {}
        res_mask: Optional[torch.Tensor] = masks.get("res_mask", None)
        head_mask: Optional[torch.Tensor] = masks.get("head_mask", None)
        tail_mask: Optional[torch.Tensor] = masks.get("tail_mask", None)

        B, L, _ = bond_mat_pred.shape
        device = bond_mat_pred.device

        # 时间相关步长：
        #   - eta_t       = schedule(t_1) * bond_step  （bond 矩阵更新）
        #   - seq_step_t  = schedule(t_1) * seq_step   （logits guidance 更新）
        if t_1 is None:
            eta_t = self.bond_step
            eta_t = torch.full((B, 1, 1), eta_t, device=device, dtype=bond_mat_pred.dtype)
            seq_step_t = self.seq_step
            seq_step_t = torch.full((B, 1, 1), seq_step_t, device=device, dtype=bond_mat_pred.dtype)
        else:
            eta_t = _schedule_weight(t_1, self.bond_step, schedule=self.schedule, power=self.power).view(B, 1, 1)
            seq_step_t = _schedule_weight(t_1, self.seq_step, schedule=self.schedule, power=self.power).view(B, 1, 1)

        # 开始于当前预测矩阵
        ss = bond_mat_pred.detach()
        logits_work = logits.detach()
        for _ in range(max(self.n_steps, 0)):
            with torch.enable_grad():
                ss_var = ss.clone().requires_grad_(True)          # [B,L,L]
                logits_var = logits_work.clone().requires_grad_(True)  # [B,L,C]

                # 非负 + Sinkhorn
                M = ss_var.clamp_min(0.0)
                P = SingleBondGuidance._symmetric_sinkhorn(
                    M, n_iters=self.sinkhorn_iters, eps=self.eps
                )  # [B,L,L]

                energy = self._compute_energy(P, logits_var, res_mask, head_mask, tail_mask)

                if not torch.isfinite(energy):
                    print("error in bond type guidence")
                    break

                grads = torch.autograd.grad(
                    energy,
                    (ss_var, logits_var),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_ss, grad_logits = grads

            if grad_ss is None and grad_logits is None:
                break

            # 对 ss 做梯度下降
            if grad_ss is not None:
                ss = ss - eta_t * grad_ss
                # Shift ss so its minimum value in each batch is 0 (if there are negatives)
                min_ss = ss.view(B, -1).min(dim=1, keepdim=True)[0].clamp_max(0.0).view(B, 1, 1)
                ss = ss - min_ss

            # 对 logits 做一小步 guidance
            if grad_logits is not None and self.seq_step > 0:
                logits_work = logits_work - seq_step_t * grad_logits * (self.num_aatypes**2) 

        # 最终对 ss 做一次 Sinkhorn 投影
        ss_projected = SingleBondGuidance._symmetric_sinkhorn(
            ss, n_iters=self.sinkhorn_iters, eps=self.eps
        )
        model_raw["bond_mat_pred"] = ss_projected.detach()
        model_raw["logits"] = logits_work

        return model_raw

_GUIDANCE_REGISTRY = {
    "logits_bias": LogitsBiasGuidance,
    "trans_anchor": TransAnchorGuidance,
    "single_bond": SingleBondGuidance,
    "soft_bond_count": SoftBondCountGuidance,
    "type_soft_bond_count": TypeAwareSoftBondCountGuidance,
}


