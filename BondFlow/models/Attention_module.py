# Full set of attention modules with optional scaled_dot_product_attention (SDPA) backend
# - Supports mixed precision (AMP)
# - Keeps original API/signatures and mask semantics (mask: 1=valid, 0=invalid)
# - Use `use_sdpa=True` to enable PyTorch F.scaled_dot_product_attention path when available

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import einsum
# from contextlib import nullcontext
# from rfdiffusion.util_module import init_lecun_normal
# _HAS_SDPA = hasattr(F, "scaled_dot_product_attention")

# #-------------------- Helpers --------------------
# def _mask_to_bool(mask):
#     """Convert input mask to boolean mask where True means INVALID/padding.
#     Accepts None, bool mask, or 0/1 int mask (user convention: 1 valid, 0 invalid).
#     Returns None or boolean tensor with same leading dims as provided.
#     """
#     if mask is None:
#         return None
#     return (mask == 0) if mask.dtype != torch.bool else mask


# def build_2d_mask_from_1d(mask_bool, Q, K):
#     """Build a (B, Q, K) boolean mask (True=invalid) from a (B, L) 1D mask_bool.
#     This makes broadcasting explicit and avoids Q/K order mistakes.
#     If mask_bool is None returns None.
#     """
#     if mask_bool is None:
#         return None
#     # slice to required lengths then expand in correct dims
#     mask_q = mask_bool[:, :Q].unsqueeze(2)  # (B, Q, 1)
#     mask_k = mask_bool[:, :K].unsqueeze(1)  # (B, 1, K)
#     return (mask_q | mask_k)                # (B, Q, K)


# def _stable_softmax_on_last_dim(logits, dim, attn_mask_bool=None):
#     """Apply softmax stably in float32 then cast back to logits.dtype.
#     logits: can be any dtype (fp16, bfloat16, fp32). attn_mask_bool is boolean mask (True=invalid)
#     that can be broadcast to logits shape; masked positions get large negative number before softmax.
#     """
#     logits_f32 = logits.float()
#     if attn_mask_bool is not None:
#         logits_f32 = logits_f32.masked_fill(attn_mask_bool, -1e9)
#     probs_f32 = F.softmax(logits_f32, dim=dim)
#     return probs_f32.to(logits.dtype)


# def _sdpa_forward(q, k, v, mask_bool=None, bias_add=None, dropout_p=0.0, is_causal=False):
#     """
#     Unified wrapper around F.scaled_dot_product_attention.
#     q: (B, Lq, H, D)
#     k: (B, Lk, H, D)
#     v: (B, Lk, H, D)

#     mask_bool: None or boolean mask. If 2-D (B, L) assumed per-position mask; if 3-D (B, Lq, Lk) used directly.
#     bias_add: None or tensor that can be broadcast/converted into additive attn_mask for SDPA.
#               Common accepted shapes: (B, Lk, H), (B, Lq, Lk, H), (B, H, Lq, Lk) etc.

#     Returns: out (B, Lq, H, D)
#     """
#     if not _HAS_SDPA:
#         raise RuntimeError("scaled_dot_product_attention not available in this PyTorch build")

#     # Permute to SDPA expected layout: (B, H, L, D)
#     q_sd = q.permute(0, 2, 1, 3).contiguous()  # (B, H, Lq, D)
#     k_sd = k.permute(0, 2, 1, 3).contiguous()  # (B, H, Lk, D)
#     v_sd = v.permute(0, 2, 1, 3).contiguous()  # (B, H, Lk, D)

#     B, H, Lq, D = q_sd.shape
#     _, _, Lk, _ = k_sd.shape

#     attn_mask = None
#     # build additive float mask from boolean mask
#     if mask_bool is not None:
#         # mask_bool True = invalid
#         if mask_bool.dim() == 2:
#             # per-position mask (B, L) -> build (B, Lq, Lk)
#             mask_q = mask_bool[:, :Lq]
#             mask_k = mask_bool[:, :Lk]
#             mask_2d = mask_q.unsqueeze(2) | mask_k.unsqueeze(1)  # (B, Lq, Lk)
#             mask_float = mask_2d.to(q_sd.dtype) * (-1e9)
#         elif mask_bool.dim() == 3:
#             # already (B, Lq, Lk)
#             mask_float = mask_bool.to(q_sd.dtype) * (-1e9)
#         else:
#             raise RuntimeError("Unsupported mask_bool dim in SDPA wrapper: %d" % mask_bool.dim())
#         # expand head dimension to (B, 1, Lq, Lk) for broadcasting across heads
#         attn_mask = mask_float.unsqueeze(1)

#     # incorporate bias_add into attn_mask
#     if bias_add is not None:
#         b = bias_add
#         # common case: (B, Lk, H) -> convert to (B, H, 1, Lk)
#         if b.dim() == 3 and b.shape[-1] == H and b.shape[1] == Lk:
#             b = b.permute(0, 2, 1).unsqueeze(2)  # (B, H, 1, Lk)
#         # common case: (B, Lq, Lk, H) -> (B, H, Lq, Lk)
#         elif b.dim() == 4 and b.shape[-1] == H:
#             b = b.permute(0, 3, 1, 2)
#         # common case: already (B, H, Lq, Lk) -> use as-is
#         # else hope it broadcasts
#         b = b.to(q_sd.dtype)
#         if attn_mask is None:
#             attn_mask = b
#         else:
#             attn_mask = attn_mask + b

#     out_sd = F.scaled_dot_product_attention(q_sd, k_sd, v_sd, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
#     # back to (B, Lq, H, D)
#     out = out_sd.permute(0, 2, 1, 3).contiguous()
#     return out


# # Safe fallback for init_lecun_normal if not provided externally
# def _maybe_init_lecun(layer):
#     try:
#         # If user provides a helper init_lecun_normal that accepts layer, use it
#         init_lecun_normal(layer)
#         return
#     except Exception:
#         # fallback: simple lecun-normal style init for weights
#         if hasattr(layer, 'weight'):
#             fan_in = layer.in_features if hasattr(layer, 'in_features') else layer.weight.size(1)
#             std = 1.0 / math.sqrt(fan_in)
#             nn.init.normal_(layer.weight, mean=0.0, std=std)
#         if hasattr(layer, 'bias') and layer.bias is not None:
#             nn.init.zeros_(layer.bias)


# # -------------------- Attention --------------------
# class Attention(nn.Module):
#     def __init__(self, d_query, d_key, n_head, d_hidden, d_out, use_sdpa=True):
#         super(Attention, self).__init__()
#         self.h = n_head
#         self.dim = d_hidden
#         self.to_q = nn.Linear(d_query, n_head * d_hidden, bias=False)
#         self.to_k = nn.Linear(d_key, n_head * d_hidden, bias=False)
#         self.to_v = nn.Linear(d_key, n_head * d_hidden, bias=False)
#         self.to_out = nn.Linear(n_head * d_hidden, d_out)
#         self.scaling = 1.0 / math.sqrt(d_hidden)
#         self.use_sdpa = bool(use_sdpa) and _HAS_SDPA
#         # init
#         nn.init.xavier_uniform_(self.to_q.weight)
#         nn.init.xavier_uniform_(self.to_k.weight)
#         nn.init.xavier_uniform_(self.to_v.weight)

#     def forward(self, query, key, value, mask=None):
#         """mask: None or (B, L) with 1=valid, 0=invalid (original semantics)
#         returns: (B, Q, d_out)
#         """
#         mask_bool = _mask_to_bool(mask)  # True = invalid
#         B, Q = query.shape[:2]
#         _, K = key.shape[:2]

#         q_full = self.to_q(query).reshape(B, Q, self.h, self.dim)
#         k = self.to_k(key).reshape(B, K, self.h, self.dim)
#         v = self.to_v(value).reshape(B, K, self.h, self.dim)

#         # SDPA path: do NOT apply self.scaling to q (SDPA applies scaling internally)
#         if self.use_sdpa:
#             # build 2D mask (B, Q, K) if provided using helper
#             mask_2d = build_2d_mask_from_1d(mask_bool, Q, K)
#             out = _sdpa_forward(q_full, k, v, mask_bool=mask_2d, bias_add=None)
#             out = out.reshape(B, Q, self.h * self.dim)
#             out = self.to_out(out)
#             if mask_bool is not None:
#                 out = out.masked_fill(mask_bool[:, :Q].unsqueeze(-1), 0.0)
#             return out

#         # fallback stable-softmax path (mixed-precision friendly)
#         q = q_full * self.scaling
#         attn_logits = einsum('bqhd,bkhd->bhqk', q, k)  # (B, H, Q, K)

#         attn_for_mask = None
#         if mask_bool is not None:
#             mask_2d = build_2d_mask_from_1d(mask_bool, Q, K)  # (B, Q, K)
#             attn_for_mask = mask_2d.unsqueeze(1)  # (B,1,Q,K) to match (B,H,Q,K)

#         attn = _stable_softmax_on_last_dim(attn_logits, dim=-1, attn_mask_bool=attn_for_mask)
#         out = einsum('bhqk,bkhd->bqhd', attn, v)
#         out = out.reshape(B, Q, self.h * self.dim)
#         out = self.to_out(out)
#         if mask_bool is not None:
#             out = out.masked_fill(mask_bool[:, :Q].unsqueeze(-1), 0.0)
#         return out


# # -------------------- AttentionWithBias --------------------
# class AttentionWithBias(nn.Module):
#     def __init__(self, d_in=256, d_bias=128, n_head=8, d_hidden=32, use_sdpa=True):
#         super(AttentionWithBias, self).__init__()
#         self.norm_in = nn.LayerNorm(d_in)
#         self.norm_bias = nn.LayerNorm(d_bias)
#         self.to_q = nn.Linear(d_in, n_head * d_hidden, bias=False)
#         self.to_k = nn.Linear(d_in, n_head * d_hidden, bias=False)
#         self.to_v = nn.Linear(d_in, n_head * d_hidden, bias=False)
#         self.to_b = nn.Linear(d_bias, n_head, bias=False)
#         self.to_g = nn.Linear(d_in, n_head * d_hidden)
#         self.to_out = nn.Linear(n_head * d_hidden, d_in)
#         self.scaling = 1.0 / math.sqrt(d_hidden)
#         self.h = n_head
#         self.dim = d_hidden
#         self.use_sdpa = bool(use_sdpa) and _HAS_SDPA
#         # init
#         nn.init.xavier_uniform_(self.to_q.weight)
#         nn.init.xavier_uniform_(self.to_k.weight)
#         nn.init.xavier_uniform_(self.to_v.weight)
#         _maybe_init_lecun(self.to_b)
#         if hasattr(self.to_g, 'bias') and self.to_g.bias is not None:
#             nn.init.ones_(self.to_g.bias)

#     def forward(self, x, bias, mask=None):
#         # x: (B, L, d_in); bias: (B, L, d_bias)
#         B, L = x.shape[:2]
#         mask_bool = _mask_to_bool(mask)  # True = invalid

#         x = torch.nan_to_num(x)
#         bias = torch.nan_to_num(bias)

#         x_norm = self.norm_in(x)
#         bias_norm = self.norm_bias(bias)

#         query = self.to_q(x_norm).reshape(B, L, self.h, self.dim)
#         key = self.to_k(x_norm).reshape(B, L, self.h, self.dim)
#         value = self.to_v(x_norm).reshape(B, L, self.h, self.dim)
#         bias_h = self.to_b(bias_norm)  # (B, L, H)
#         gate = torch.sigmoid(self.to_g(x_norm))

#         # SDPA path
#         if self.use_sdpa:
#             # Build 2D mask (B, L, L) from 1D mask
#             mask_2d = build_2d_mask_from_1d(mask_bool, L, L)
#             out_sd = _sdpa_forward(query, key, value, mask_bool=mask_2d, bias_add=bias_h)
#             out = out_sd.reshape(B, L, -1)
#             out = gate * out
#             out = self.to_out(out)
#             if mask_bool is not None:
#                 out = out.masked_fill(mask_bool.unsqueeze(-1), 0.0)
#             return out

#         # fallback path (stable softmax)
#         key = key * self.scaling
#         attn_logits = einsum('bqhd,bkhd->bqkh', query, key)  # (B, Q, K, H)
#         attn_logits = attn_logits + bias_h.unsqueeze(1)

#         mask_for_logits = None
#         if mask_bool is not None and mask_bool.any():
#             mask_2d = build_2d_mask_from_1d(mask_bool, L, L)
#             mask_for_logits = mask_2d.unsqueeze(-1)  # (B, L, L, 1) -> broadcast to (B, Q, K, H)

#         attn = _stable_softmax_on_last_dim(attn_logits, dim=2, attn_mask_bool=mask_for_logits)
#         out = einsum('bqkh,bkhd->bqhd', attn, value).reshape(B, L, -1)
#         out = gate * out
#         out = self.to_out(out)
#         if mask_bool is not None and mask_bool.any():
#             out = out.masked_fill(mask_bool.unsqueeze(-1), 0.0)
#         return out


# # -------------------- SequenceWeight --------------------
# # class SequenceWeight(nn.Module):
# #     def __init__(self, d_msa, n_head, d_hidden, p_drop=0.1):
# #         super(SequenceWeight, self).__init__()
# #         self.h = n_head
# #         self.dim = d_hidden
# #         self.scale = 1.0 / math.sqrt(self.dim)
# #         self.to_query = nn.Linear(d_msa, n_head * d_hidden)
# #         self.to_key = nn.Linear(d_msa, n_head * d_hidden)
# #         self.dropout = nn.Dropout(p_drop)
# #         nn.init.xavier_uniform_(self.to_query.weight)
# #         nn.init.xavier_uniform_(self.to_key.weight)

# #     def forward(self, msa, mask=None):
# #         # msa: (B, N, L, feat)
# #         B, N, L = msa.shape[:3]
# #         mask_bool = _mask_to_bool(mask)  # True = invalid

# #         tar_seq = msa[:, 0]  # (B, L, feat)
# #         q = self.to_query(tar_seq).view(B, 1, L, self.h, self.dim)
# #         k = self.to_key(msa).view(B, N, L, self.h, self.dim)

# #         q = q * self.scale
# #         attn_logits = einsum('bqihd,bkihd->bkihq', q, k)  # keep original index order

# #         mask_expanded = None
# #         if mask_bool is not None and mask_bool.any():
# #             # expand (B, L) -> (B, 1, L, 1, 1) to broadcast to attn_logits
# #             mask_expanded = mask_bool.unsqueeze(1).unsqueeze(3).unsqueeze(4)
# #             attn_logits = attn_logits.masked_fill(mask_expanded, float('-1e9'))

# #         # softmax over N (dim=1 in attn_logits shape bkihq)
# #         attn = _stable_softmax_on_last_dim(attn_logits, dim=1, attn_mask_bool=None)

# #         if mask_expanded is not None:
# #             attn = attn.masked_fill(mask_expanded, 0.0)

# #         return self.dropout(attn)


# # -------------------- MSARowAttentionWithBias --------------------
# class MSARowAttentionWithBias(nn.Module):
#     def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32, use_sdpa=True):
#         super(MSARowAttentionWithBias, self).__init__()
#         self.norm_msa = nn.LayerNorm(d_msa)
#         self.norm_pair = nn.LayerNorm(d_pair)
#         #self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
#         self.to_q = nn.Linear(d_msa, n_head * d_hidden, bias=False)
#         self.to_k = nn.Linear(d_msa, n_head * d_hidden, bias=False)
#         self.to_v = nn.Linear(d_msa, n_head * d_hidden, bias=False)
#         self.to_b = nn.Linear(d_pair, n_head, bias=False)
#         self.to_g = nn.Linear(d_msa, n_head * d_hidden)
#         self.to_out = nn.Linear(n_head * d_hidden, d_msa)
#         self.scaling = 1.0 / math.sqrt(d_hidden)
#         self.h = n_head
#         self.dim = d_hidden
#         self.use_sdpa = bool(use_sdpa) and _HAS_SDPA
#         nn.init.xavier_uniform_(self.to_q.weight)
#         nn.init.xavier_uniform_(self.to_k.weight)
#         nn.init.xavier_uniform_(self.to_v.weight)
#         _maybe_init_lecun(self.to_b)
#         if hasattr(self.to_g, 'bias') and self.to_g.bias is not None:
#             nn.init.ones_(self.to_g.bias)

#     def forward(self, msa, pair, mask=None):
#         # msa: (B, N, L, d_msa); pair: (B, L, L, d_pair)
#         B, N, L = msa.shape[:3]
#         mask_bool = _mask_to_bool(mask)  # True = invalid

#         msa = torch.nan_to_num(msa)
#         pair = torch.nan_to_num(pair)

#         msa_norm = self.norm_msa(msa)
#         pair_norm = self.norm_pair(pair)

#         # seq_weight = self.seq_weight(msa, mask=mask)

#         query = self.to_q(msa_norm).reshape(B, N, L, self.h, self.dim)
#         key = self.to_k(msa_norm).reshape(B, N, L, self.h, self.dim)
#         value = self.to_v(msa_norm).reshape(B, N, L, self.h, self.dim)
#         bias = self.to_b(pair_norm)  # (B, L, L, H)
#         gate = torch.sigmoid(self.to_g(msa_norm))

#         # apply mask to q/k/v if provided
#         if mask_bool is not None and mask_bool.any():
#             mask_qkv = mask_bool.view(B, 1, L, 1, 1)
#             query = query.masked_fill(mask_qkv, 0.0)
#             key = key.masked_fill(mask_qkv, 0.0)
#             value = value.masked_fill(mask_qkv, 0.0)

#         # apply seq_weight (shape returned by SequenceWeight broadcasts)
#         #query = query * seq_weight.unsqueeze(-1)
#         key = key * self.scaling

#         # attn logits along the N dimension
#         attn_logits = einsum('bnqhd,bnkhd->bqkh', query, key)  # (B, Q, K, H)
#         attn_logits = attn_logits + bias

#         # mask along pair positions
#         mask_for_logits = None
#         if mask_bool is not None and mask_bool.any():
#             mask_2d = build_2d_mask_from_1d(mask_bool, L, L)
#             mask_for_logits = mask_2d.unsqueeze(-1)  # (B, L, L, 1)

#         # softmax on K dimension (dim=2 for bqkh)
#         attn = _stable_softmax_on_last_dim(attn_logits, dim=2, attn_mask_bool=mask_for_logits)

#         out = einsum('bqkh,bnkhd->bnqhd', attn, value).reshape(B, N, L, -1)
#         out = gate * out
#         out = self.to_out(out)

#         if mask_bool is not None and mask_bool.any():
#             out = out.masked_fill(mask_bool.view(B, 1, L, 1), 0.0)
#         return out


# # -------------------- BiasedAxialAttention --------------------
# class BiasedAxialAttention(nn.Module):
#     def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True, use_sdpa=True):
#         super(BiasedAxialAttention, self).__init__()
#         self.is_row = is_row
#         self.norm_pair = nn.LayerNorm(d_pair)
#         self.norm_bias = nn.LayerNorm(d_bias)
#         self.to_q = nn.Linear(d_pair, n_head * d_hidden, bias=False)
#         self.to_k = nn.Linear(d_pair, n_head * d_hidden, bias=False)
#         self.to_v = nn.Linear(d_pair, n_head * d_hidden, bias=False)
#         self.to_b = nn.Linear(d_bias, n_head, bias=False)
#         self.to_g = nn.Linear(d_pair, n_head * d_hidden)
#         self.to_out = nn.Linear(n_head * d_hidden, d_pair)
#         self.scaling = 1.0 / math.sqrt(d_hidden)
#         self.h = n_head
#         self.dim = d_hidden
#         self.use_sdpa = bool(use_sdpa) and _HAS_SDPA
#         nn.init.xavier_uniform_(self.to_q.weight)
#         nn.init.xavier_uniform_(self.to_k.weight)
#         nn.init.xavier_uniform_(self.to_v.weight)
#         _maybe_init_lecun(self.to_b)
#         if hasattr(self.to_g, 'bias') and self.to_g.bias is not None:
#             nn.init.ones_(self.to_g.bias)

#     def forward(self, pair, bias, mask=None):
#         # pair: (B, L, L, C) ; bias: same shape-ish ; mask: (B, L) or (B, L, L)
#         B, L, _, _ = pair.shape
#         mask_bool = _mask_to_bool(mask)  # True = invalid

#         safe_pair = torch.nan_to_num(pair)
#         safe_bias = torch.nan_to_num(bias)

#         if self.is_row:
#             # 仅对 pair/bias 交换轴，不再改动 mask_bool 形状
#             safe_pair = safe_pair.permute(0, 2, 1, 3).contiguous()
#             safe_bias = safe_bias.permute(0, 2, 1, 3).contiguous()
#             # 若用户传入 (B,L,L) 掩码且希望同样行列互换，可在此按需处理
#             if mask_bool is not None and mask_bool.dim() == 3 and mask_bool.shape[-2:] == (L, L):
#                 # 行/列轴同样互换以与 pair 对齐
#                 mask_bool = mask_bool.permute(0, 2, 1).contiguous()

#         pair_norm = self.norm_pair(safe_pair)
#         bias_norm = self.norm_bias(safe_bias)

#         query = self.to_q(pair_norm).reshape(B, L, L, self.h, self.dim).contiguous()
#         key = self.to_k(pair_norm).reshape(B, L, L, self.h, self.dim).contiguous()
#         value = self.to_v(pair_norm).reshape(B, L, L, self.h, self.dim).contiguous()
#         bias_h = self.to_b(bias_norm)  # (B, L, L, H)
#         gate = torch.sigmoid(self.to_g(pair_norm))

#         # SDPA path: attention along the second axis (j) for each i
#         if self.use_sdpa:
#             q = query.permute(0, 2, 1, 3, 4).reshape(B * L, L, self.h, self.dim)
#             k = key.permute(0, 2, 1, 3, 4).reshape(B * L, L, self.h, self.dim)
#             v = value.permute(0, 2, 1, 3, 4).reshape(B * L, L, self.h, self.dim)
#             bias_for_sdpa = bias_h.permute(0, 2, 1, 3).reshape(B * L, L, self.h)
            
#             mask_for_sdpa = None
#             if mask_bool is not None:
#                 if mask_bool.dim() == 2:
#                     # (B,L) -> (B,L,L); 无效若 i 或 j 任一无效
#                     mask_2d = mask_bool.unsqueeze(1) | mask_bool.unsqueeze(2)  # (B,L,L)
#                 elif mask_bool.dim() == 3 and mask_bool.shape[-2:] == (L, L):
#                     mask_2d = mask_bool
#                 else:
#                     raise RuntimeError(f"Unexpected mask_bool shape {mask_bool.shape} for SDPA path")
#                 # 展开为 (B*L, L) 作为 per-position mask (True=invalid)
#                 mask_for_sdpa = mask_2d.reshape(B * L, L)

#             out_sd = _sdpa_forward(q, k, v, mask_bool=mask_for_sdpa, bias_add=bias_for_sdpa)
#             out = out_sd.reshape(B, L, L, self.h * self.dim).permute(0, 2, 1, 3).contiguous()
#             out = gate * out
#             out = self.to_out(out)

#             if self.is_row:
#                 out = out.permute(0, 2, 1, 3).contiguous()

#             # 依据最原始的 1D mask 归零 (行或列任意无效位置整行/列置零)
#             if mask_bool is not None:
#                 if mask_bool.dim() == 2:
#                     mrow = mask_bool.unsqueeze(2)  # (B,L,1)
#                     mcol = mask_bool.unsqueeze(1)  # (B,1,L)
#                     out = out.masked_fill((mrow | mcol).unsqueeze(-1), 0.0)
#                 else:
#                     out = out.masked_fill(mask_bool.unsqueeze(-1), 0.0)
#             return out
#         # fallback path
#         query = query * self.scaling
#         key = key / math.sqrt(max(1, L))
#         attn_logits = einsum('bnihk,bnjhk->bijh', query, key)  # (B, i, j, H)
#         attn_logits = attn_logits + bias_h

#         mask_for_logits = None
#         if mask_bool is not None and mask_bool.any():
#             # mask_bool should be (B, L, L) or (B, L) convertible -> use helper
#             if mask_bool.dim() == 2:
#                 mask2d = build_2d_mask_from_1d(mask_bool, L, L)
#             else:
#                 mask2d = mask_bool
#             mask_for_logits = mask2d.unsqueeze(-1)

#         attn = _stable_softmax_on_last_dim(attn_logits, dim=2, attn_mask_bool=mask_for_logits)
#         out = einsum('bijh,bkjhd->bikhd', attn, value).reshape(B, L, L, -1)
#         out = gate * out
#         out = self.to_out(out)

#         if self.is_row:
#             out = out.permute(0, 2, 1, 3).contiguous()
#         if mask_bool is not None and mask_bool.any():
#             # 规范化 mask 到 (B,L,L), 兼容 1D / 2D 且长度不足时裁剪
#             if mask_bool.dim() == 2:
#                 mb = mask_bool[:, :L]          # (B,L')
#                 mask2d = build_2d_mask_from_1d(mb, L, L)
#             elif mask_bool.dim() == 3:
#                 mask2d = mask_bool[:, :L, :L]
#             else:
#                 raise RuntimeError(f"Unsupported mask dim {mask_bool.dim()} in fallback")
#             out = out.masked_fill(mask2d.unsqueeze(-1), 0.0)
#         return out

# End of file

# class FeedForwardLayer(nn.Module):
#     def __init__(self, d_model, r_ff, p_drop=0.1):
#         super(FeedForwardLayer, self).__init__()
#         self.norm = nn.LayerNorm(d_model)
#         self.linear1 = nn.Linear(d_model, d_model*r_ff)
#         self.dropout = nn.Dropout(p_drop)
#         self.linear2 = nn.Linear(d_model*r_ff, d_model)
    
#     def forward(self, src, mask=None):
#         # If mask not provided, maintain original NaN behavior
        
#         if mask is None:
#             mask = torch.isnan(src).any(dim=-1)
#         else:
#             # Ensure mask is boolean: True for invalid positions
#             mask = (mask == 0)
        
#         out = self.norm(src)
#         out = self.linear2(self.dropout(F.relu_(self.linear1(out))))
        
#         # Apply mask to output
#         if mask.any():
#             out = out.masked_fill(mask.unsqueeze(-1), torch.nan)
#         return out



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from opt_einsum import contract as einsum
from rfdiff.util_module import init_lecun_normal
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, d_query, d_key, n_head, d_hidden, d_out):
        super(Attention, self).__init__()
        self.h = n_head
        self.dim = d_hidden
        self.to_q = nn.Linear(d_query, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_key, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_key, n_head*d_hidden, bias=False)
        self.to_out = nn.Linear(n_head*d_hidden, d_out)
        self.scaling = 1/math.sqrt(d_hidden)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        # nn.init.zeros_(self.to_out.weight)
        # nn.init.zeros_(self.to_out.bias)

    def forward(self, query, key, value, mask=None):
        # batch_mask B,L 1 is valid, 0 is invalid
        mask = (mask == 0) 
        B, Q = query.shape[:2]
        B, K = key.shape[:2]

        q = self.to_q(query).reshape(B, Q, self.h, self.dim)
        k = self.to_k(key).reshape(B, K, self.h, self.dim)
        v = self.to_v(value).reshape(B, K, self.h, self.dim)
        
        q = q * self.scaling
        attn = einsum('bqhd,bkhd->bhqk', q, k)
        mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
        attn = attn.masked_fill(mask_2d.unsqueeze(1), -1e9)

        attn = F.softmax(attn, dim=-1)

        out = einsum('bhqk,bkhd->bqhd', attn, v)
        
        out = out.reshape(B, Q, self.h*self.dim)
        out = self.to_out(out)
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)

        return out 

# # Optimized version of Attention using SDPA
# class Attention(nn.Module):
#     def __init__(self, d_query, d_key, n_head, d_hidden, d_out):
#         super().__init__()
#         self.h, self.d_hidden = n_head, d_hidden
#         # Optimization: Merge K and V linear layers
#         self.to_q = nn.Linear(d_query, n_head * d_hidden, bias=False)
#         self.to_kv = nn.Linear(d_key, 2 * n_head * d_hidden, bias=False)
#         self.to_out = nn.Linear(n_head * d_hidden, d_out)
#         self.reset_parameter()
#     def reset_parameter(self):
#         nn.init.xavier_uniform_(self.to_q.weight)
#         nn.init.xavier_uniform_(self.to_kv.weight)
#     def forward(self, query, key, value, mask=None):
#         q = self.to_q(query)
#         k, v = self.to_kv(key).chunk(2, dim=-1)
        
#         q = rearrange(q, 'b q (h d) -> b h q d', h=self.h)
#         k = rearrange(k, 'b k (h d) -> b h k d', h=self.h)
#         v = rearrange(v, 'b k (h d) -> b h k d', h=self.h)
        
#         attn_mask = None
#         if mask is not None:
#             bool_mask = (mask == 0)
#             # Create a 2D mask and then add a dimension for the heads to enable broadcasting
#             mask_2d = bool_mask.unsqueeze(1) | bool_mask.unsqueeze(2)
#             attn_mask = mask_2d.unsqueeze(1) # Shape: [B, 1, Q, K]
            
#         out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
#         out = rearrange(out, 'b h q d -> b q (h d)')
#         out = self.to_out(out)
#         if mask is not None: out = out.masked_fill((mask == 0).unsqueeze(-1), 0.0)
#         return out
    
class AttentionWithBias(nn.Module):
    def __init__(self, d_in=256, d_bias=128, n_head=8, d_hidden=32):
        super(AttentionWithBias, self).__init__()
        self.norm_in = nn.LayerNorm(d_in)
        self.norm_bias = nn.LayerNorm(d_bias)
        self.to_q = nn.Linear(d_in, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_in, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_in, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False)
        self.to_g = nn.Linear(d_in, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_in)
        self.scaling = 1 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        self.to_b = init_lecun_normal(self.to_b)
        nn.init.ones_(self.to_g.bias)

    def forward(self, x, bias, mask=None):
        B,L = x.shape[:2]
        # Handle mask input
        mask = (mask == 0)  # Convert to boolean mask: True=invalid
        
        safe_x = torch.nan_to_num(x)
        safe_bias = torch.nan_to_num(bias)
        
        # Create 2D attention mask
        mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
        
        x_norm = self.norm_in(safe_x)
        bias_norm = self.norm_bias(safe_bias)
        
        query = self.to_q(x_norm).reshape(B, L, self.h, self.dim)
        key = self.to_k(x_norm).reshape(B, L, self.h, self.dim)
        value = self.to_v(x_norm).reshape(B, L, self.h, self.dim)
        bias_h = self.to_b(bias_norm)
        gate = torch.sigmoid(self.to_g(x_norm))
        
        key = key * self.scaling
        attn = einsum('bqhd,bkhd->bqkh', query, key)
        attn = attn + bias_h
        if mask.any():
            mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
            attn.masked_fill_(mask_2d.unsqueeze(-1), -1e9)
            
        attn = F.softmax(attn, dim=-2)
        
        out = einsum('bqkh,bkhd->bqhd', attn, value).reshape(B, L, -1)
        out = gate * out
        out = self.to_out(out)
        
        if mask.any():
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)
            
        return out


# optimized version of AttentionWithBias using SDPA

# class AttentionWithBias(nn.Module):
#     def __init__(self, d_in=256, d_bias=128, n_head=8, d_hidden=32):
#         super().__init__()
#         self.norm_in, self.norm_bias = nn.LayerNorm(d_in), nn.LayerNorm(d_bias)
#         self.to_qkv = nn.Linear(d_in, 3 * n_head * d_hidden, bias=False) # Merged
#         self.to_b, self.to_g, self.to_out = nn.Linear(d_bias, n_head, bias=False), nn.Linear(d_in, n_head*d_hidden), nn.Linear(n_head*d_hidden, d_in)
#         self.h = n_head
#         self.reset_parameter()
#     def reset_parameter(self):
#         nn.init.xavier_uniform_(self.to_qkv.weight)
#         self.to_b = init_lecun_normal(self.to_b)
#         nn.init.ones_(self.to_g.bias)
#     def forward(self, x, bias, mask=None):
#         x_norm, bias_norm = self.norm_in(torch.nan_to_num(x)), self.norm_bias(torch.nan_to_num(bias))
#         q, k, v = self.to_qkv(x_norm).chunk(3, dim=-1)
#         q, k, v = [rearrange(t, 'b l (h d) -> b h l d', h=self.h) for t in (q, k, v)]
        
#         # Optimization: Pass the bias tensor as a float mask to SDPA
#         float_mask = self.to_b(bias_norm)
#         float_mask = rearrange(float_mask, 'b q k h -> b h q k')

#         if mask is not None:
#             padding_mask = (mask == 0)
#             mask_2d = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
#             float_mask = float_mask.masked_fill(mask_2d.unsqueeze(1), -torch.inf)

#         out = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
#         out = rearrange(out, 'b h l d -> b l (h d)')
        
#         gate = torch.sigmoid(self.to_g(x_norm))
#         out = gate * out
#         out = self.to_out(out)
#         if mask is not None: out = out.masked_fill((mask == 0).unsqueeze(-1), 0.0)
#         return out

# class SequenceWeight(nn.Module):
#     def __init__(self, d_msa, n_head, d_hidden, p_drop=0.1):
#         super(SequenceWeight, self).__init__()
#         self.h = n_head
#         self.dim = d_hidden
#         self.scale = 1.0 / math.sqrt(self.dim)
#         self.to_query = nn.Linear(d_msa, n_head*d_hidden)
#         self.to_key = nn.Linear(d_msa, n_head*d_hidden)
#         self.dropout = nn.Dropout(p_drop)
#         self.reset_parameter()
    
#     def reset_parameter(self):
#         nn.init.xavier_uniform_(self.to_query.weight)
#         nn.init.xavier_uniform_(self.to_key.weight)

#     def forward(self, msa, mask=None):  # Added mask parameter
#         B, N, L = msa.shape[:3]
        
#         mask_L = (mask == 0)  # Assuming mask is provided for target sequence (B, L)
        
#         tar_seq = msa[:,0]
        
#         q = self.to_query(tar_seq).view(B, 1, L, self.h, self.dim)
#         k = self.to_key(msa).view(B, N, L, self.h, self.dim)
        
#         q = q * self.scale
#         attn = einsum('bqihd,bkihd->bkihq', q, k)
#         mask_expanded = mask_L.unsqueeze(1).unsqueeze(3).unsqueeze(4)
#         # Apply mask if provided
#         if mask_L.any():
#             attn = attn.masked_fill(mask_expanded, -1e9)

#         attn = F.softmax(attn, dim=1)
       
#         if mask.any():
#             attn = attn.masked_fill(mask_expanded, 0.0)

#         return self.dropout(attn)


class MSARowAttentionWithBias(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32):
        super(MSARowAttentionWithBias, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        #self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        self.to_b = init_lecun_normal(self.to_b)
        nn.init.ones_(self.to_g.bias)
        
    def forward(self, msa, pair, mask=None):  # Added mask parameter
        B, N, L = msa.shape[:3]
        mask = (mask == 0)  # Assuming mask is (B, L) for entire MSA
        
        safe_msa = torch.nan_to_num(msa)
        safe_pair = torch.nan_to_num(pair)
        
        msa_norm = self.norm_msa(safe_msa)
        pair_norm = self.norm_pair(safe_pair)
        
        # Pass mask to SequenceWeight
        #seq_weight = self.seq_weight(msa, mask=mask)
        
        query = self.to_q(msa_norm).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa_norm).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa_norm).reshape(B, N, L, self.h, self.dim)
        bias = self.to_b(pair_norm)
        gate = torch.sigmoid(self.to_g(msa_norm))
        
        query = query.masked_fill(mask.view(B, 1, L, 1, 1), 0.0)  # Apply mask to query
        value = value.masked_fill(mask.view(B, 1, L, 1, 1), 0.0)  # Apply mask to value
        key = key.masked_fill(mask.view(B, 1, L, 1, 1), 0.0)  # Apply mask to key

        #query = query * seq_weight.expand(-1, -1, -1, -1, self.dim)
        key = key * self.scaling
        attn = einsum('bnqhd,bnkhd->bqkh', query, key)
    
        attn = attn + bias
        
        # Create 2D mask from 1D mask
        mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
        if mask_2d.any():
            attn = attn.masked_fill(mask_2d.view(B, L, L,1), -1e9)

        attn = F.softmax(attn, dim=-2)

        out = einsum('bqkh,bnkhd->bnqhd', attn, value).reshape(B, N, L, -1)
        out = gate * out
        out = self.to_out(out)
        
        # Apply mask to output
        if mask.any():
            out = out.masked_fill(mask.view(B, 1, L, 1), 0.0)
            
        return out

# class MSAColAttention(nn.Module):
#     def __init__(self, d_msa=256, n_head=8, d_hidden=32):
#         super(MSAColAttention, self).__init__()
#         self.norm_msa = nn.LayerNorm(d_msa)
#         self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
#         self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
#         self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
#         self.to_g = nn.Linear(d_msa, n_head*d_hidden)
#         self.to_out = nn.Linear(n_head*d_hidden, d_msa)
#         self.scaling = 1/math.sqrt(d_hidden)
#         self.h = n_head
#         self.dim = d_hidden
#         self.reset_parameter()

#     def reset_parameter(self):
#         nn.init.xavier_uniform_(self.to_q.weight)
#         nn.init.xavier_uniform_(self.to_k.weight)
#         nn.init.xavier_uniform_(self.to_v.weight)
#         nn.init.ones_(self.to_g.bias)

#     def forward(self, msa, mask=None):  # Added mask parameter
#         B, N, L = msa.shape[:3]
        
#         mask = (mask == 0) # (B, L)
        
#         msa_norm = self.norm_msa(msa)
        
#         query = self.to_q(msa_norm).reshape(B, N, L, self.h, self.dim)
#         key = self.to_k(msa_norm).reshape(B, N, L, self.h, self.dim)
#         value = self.to_v(msa_norm).reshape(B, N, L, self.h, self.dim)
#         gate = torch.sigmoid(self.to_g(msa_norm))
        
#         query = query.permute(0, 2, 3, 1, 4)  # B, L, h, N, d
#         key = key.permute(0, 2, 3, 4, 1)       # B, L, h, d, N
        
#         attn = torch.matmul(query * self.scaling, key)  # B, L, h, N, N

#         # if mask.any():
#         #     print(mask.shape)
#         #     print(attn.shape)
#         #     attn = attn.masked_fill(mask.view(B, L, self.h, N, N), -1e9)

#         attn = F.softmax(attn, dim=-1)
        
#         value = value.permute(0, 2, 3, 1, 4)  # B, L, h, N, d
#         out = torch.matmul(attn, value)        # B, L, h, N, d
#         out = out.permute(0, 3, 1, 2, 4).reshape(B, N, L, -1)
#         out = gate * out
#         out = self.to_out(out)

#         if mask.any():
#             out = out.masked_fill(mask.view(B, 1, L, 1), 0.0)

#         return out


# class MSAColGlobalAttention(nn.Module):
#     def __init__(self, d_msa=64, n_head=8, d_hidden=8):
#         super(MSAColGlobalAttention, self).__init__()
#         self.norm_msa = nn.LayerNorm(d_msa)
#         self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
#         self.to_k = nn.Linear(d_msa, d_hidden, bias=False)
#         self.to_v = nn.Linear(d_msa, d_hidden, bias=False)
#         self.to_g = nn.Linear(d_msa, n_head*d_hidden)
#         self.to_out = nn.Linear(n_head*d_hidden, d_msa)
#         self.scaling = 1/math.sqrt(d_hidden)
#         self.h = n_head
#         self.dim = d_hidden
#         self.reset_parameter()

#     def reset_parameter(self):
#         nn.init.xavier_uniform_(self.to_q.weight)
#         nn.init.xavier_uniform_(self.to_k.weight)
#         nn.init.xavier_uniform_(self.to_v.weight)
#         nn.init.ones_(self.to_g.bias)

#     def forward(self, msa, mask=None):  # Added mask parameter
#         B, N, L = msa.shape[:3]

#         mask = (mask == 0)
#         msa_norm = self.norm_msa(msa)
        
#         # Masked average for query
#         query = self.to_q(msa_norm).reshape(B,N, L, self.h, self.dim)
#         if mask.any():
#             query= query.masked_fill(mask.view(B, 1, L, 1, 1), 0.)
#         query = query.mean(dim=1)
        
#         key = self.to_k(msa_norm)
#         value = self.to_v(msa_norm)
#         gate = torch.sigmoid(self.to_g(msa_norm))
        
#         query = query * self.scaling
#         attn = einsum('blhd,bnld->blhn', query, key)  # B, L, h, N
        
#         if mask.any():
#             attn.masked_fill_(mask.view(B, L, 1, 1), -1e9)

#         attn = F.softmax(attn, dim=-1)
        
#         out = einsum('blhn,bnld->blhd', attn, value).reshape(B, L, -1)
#         out = out.unsqueeze(1).expand(-1, N, -1, -1)
#         out = gate * out
#         out = self.to_out(out)

#         if mask.any():
#             out = out.masked_fill(mask.view(B, 1, L, 1), 0.0)

#         return out


class BiasedAxialAttention(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super(BiasedAxialAttention, self).__init__()
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)
        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False)
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_pair)
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        self.to_b = init_lecun_normal(self.to_b)
        nn.init.ones_(self.to_g.bias)
    def forward(self, pair, bias, mask=None):  # Added mask parameter
        B, L, _, _ = pair.shape

            # Convert to boolean: True for invalid positions
            # Assuming mask is (B, L, L) for pair representation
        mask = (mask == 0)  # Convert to boolean mask: True=invalid       
            # Create 2D attention mask
        mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)        
        safe_pair = torch.nan_to_num(pair)
        safe_bias = torch.nan_to_num(bias)
        
        if self.is_row:
            safe_pair = safe_pair.permute(0, 2, 1, 3)
            safe_bias = safe_bias.permute(0, 2, 1, 3)
            mask_2d = mask_2d.permute(0, 2, 1)
            
        pair_norm = self.norm_pair(safe_pair)
        bias_norm = self.norm_bias(safe_bias)
        
        query = self.to_q(pair_norm).reshape(B, L, L, self.h, self.dim)
        key = self.to_k(pair_norm).reshape(B, L, L, self.h, self.dim)
        value = self.to_v(pair_norm).reshape(B, L, L, self.h, self.dim)
        bias = self.to_b(bias_norm)
        gate = torch.sigmoid(self.to_g(pair_norm))
        
        query = query * self.scaling
        key = key / math.sqrt(L)
        attn = einsum('bnihk,bnjhk->bijh', query, key)
        attn = attn + bias
        
        if mask_2d.any():
            attn = attn.masked_fill(mask_2d.unsqueeze(-1), -1e9)
            
        attn = F.softmax(attn, dim=-2)
        
        out = einsum('bijh,bkjhd->bikhd', attn, value).reshape(B, L, L, -1)
        out = gate * out
        out = self.to_out(out)
        
        if self.is_row:
            out = out.permute(0, 2, 1, 3)
            
        # Apply mask to output
        if mask_2d.any():
            out = out.masked_fill(mask_2d.unsqueeze(-1), 0.0)
            
        return out