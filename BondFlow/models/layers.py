import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.amp import autocast

def stats_nan(tensor, name=''):
    return
    shape = tensor.shape
    total_elements = tensor.numel()
    nan_count = torch.isnan(tensor).sum().item()
    nan_percentage = (nan_count / total_elements) * 100 if total_elements > 0 else 0
    print(name,shape,nan_percentage)
# 1. 辅助函数：将字节转换为更易读的 MB
def print_gpu_memory(step_name="",device='cuda'):
    return
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"{step_name:<30} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

class BondingNetwork(nn.Module):
    def __init__(self, d_msa, d_state=None, d_pair=None, p_drop=0):
        super(BondingNetwork, self).__init__()
        #
        # self.proj_factor = nn.Sequential(
        #     nn.Linear(d_msa, d_msa),
        #     nn.ReLU(),
        #     nn.Dropout(p_drop),
        #     nn.Linear(d_msa, 1),
        #     nn.Sigmoid(),
        # ) 
        # d_state_pair = d_state + d_pair
        self.proj_bonding = nn.Sequential(
            nn.Linear(d_pair ,d_pair),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_pair ,d_pair),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_pair, 1),
        )
        self.dsc_proj_layer = DSMProjection()

    def forward(self, msa, state, pair,mask_2d=None,mat_true=None):
        # input: pair info (B, L, L, C0)
        # msa: (B, L, C1)
        # state: (B, L, C2)

        # left_msa = msa.unsqueeze(2)
        # right_msa = msa.unsqueeze(1)
        # pairwise_msa = left_msa + right_msa  
        # facotr = self.proj_factor(pairwise_msa).squeeze(-1)  

        # pair_state = torch.cat((pair , state.unsqueeze(2) + state.unsqueeze(1)), dim=-1)
        # pair_state =self.proj_bonding(pair_state).squeeze(-1) # (B, L, L)

        # pair_state = pair_state + torch.eye(pair_state.size(1)).to(pair_state.device) * -1e9
        # bonding = self.symmetric_matrix_transform(pair_state)
        # bonding_matrix = torch.mul(pair_state, facotr) 
        # L = state.size(1)
        # state_i = state.unsqueeze(2).expand(-1, L, L, -1)  # [B, N, N, C]
        # state_j = state.unsqueeze(1).expand(-1, L, L, -1)  # [B, N, N, C]
        # state_pair = torch.cat([state_i, state_j], dim=-1)  # [B, N, N, 2C]
        # print(state_pair.shape)
        pair_in = pair #torch.cat((pair , state_pair), dim=-1)
        bonding_matrix_tmp = self.proj_bonding(pair_in).squeeze(-1)  # (B, L, L)
        bonding_matrix = self.dsc_proj_layer(bonding_matrix_tmp,mask_2d,mat_true)  # (B, L, L)

        return bonding_matrix # (B, L, L)

class DSMProjection(nn.Module):
    
    """
    带有掩码功能的稳定双随机矩阵投影。
    
    通过Sinkhorn-Knopp算法将输入的logits矩阵转换为一个双随机矩阵。
    可以接受一个2D掩码来忽略矩阵中的特定元素。
    """
    def __init__(self, base_tau=0.25, max_iter=30, eps=1e-8):
        super().__init__()
        self.base_tau = base_tau
        self.max_iter = max_iter
        self.eps = eps

    def forward(self, logits: torch.Tensor, mask_2d: torch.Tensor=None, mat_true=None) -> torch.Tensor:
        """
        前向传播。

        Args:
            logits (torch.Tensor): 输入的 logits 矩阵，形状为 (B, L, L)。
            mask_2d (torch.Tensor, optional): 
                一个布尔类型的2D掩码，形状为 (B, L, L)。
                `True` 表示保留，`False` 表示遮蔽。默认为 None。

        Returns:
            torch.Tensor: 经过投影后的双随机矩阵。
        """
    # def forward(self, logits: torch.Tensor, mask_2d: torch.Tensor=None) -> torch.Tensor:
        """
        前向传播。
        """
        # 使用 logits.device.type 而不是 str(logits.device)
        # with autocast(device_type=logits.device.type, enabled=False):
        orig_dtype = logits.dtype

        # 若是半精度, 临时升到 float32 以避免溢出
        if logits.dtype in (torch.float16, torch.bfloat16):
            work_logits = logits.float()
        else:
            work_logits = logits

        B, L, _ = work_logits.shape
        tau = self.base_tau

        if mask_2d is None:
            mask_2d = torch.ones((B, L, L), dtype=torch.bool, device=work_logits.device)
        else:
            mask_2d = mask_2d.bool()

        # 数值稳定: 先减去每行最大值
        logits_centered = work_logits - work_logits.max(dim=-1, keepdim=True).values
        sym_logits = (logits_centered + logits_centered.transpose(-1, -2)) / 2

        # 用 -inf 而不是 -1e9, 可表示且 exp(-inf)=0
        sym_logits.masked_fill_(~mask_2d, float('-inf'))

        # 指数
        matrix = torch.exp(sym_logits / tau)  # 被遮蔽位置 exp(-inf)=0

        for _ in range(self.max_iter):
            # Normalize columns (sum over rows)
            row_sum = matrix.sum(dim=-2, keepdim=True)
            matrix = matrix / row_sum.clamp(min=self.eps)
            # Re-apply mask to avoid numerical drift on masked entries
            matrix = matrix * mask_2d.float()

            # Normalize rows (sum over columns)
            col_sum = matrix.sum(dim=-1, keepdim=True)
            matrix = matrix / col_sum.clamp(min=self.eps)
            # Re-apply mask to avoid numerical drift on masked entries
            matrix = matrix * mask_2d.float()

        matrix = (matrix + matrix.transpose(-1, -2)) / 2

        # 再次应用掩码 (保持 0)
        matrix = matrix * mask_2d.float()

        # 如果需要与外部混精兼容, 转回原 dtype
        if matrix.dtype != orig_dtype:
            matrix = matrix.to(orig_dtype)
        
        if mat_true is not None:
            matrix = mask_2d.float() * matrix + (1 - mask_2d.float()) * mat_true

        return matrix

    
class TimeEmbedding(nn.Module):
    """
    PyTorch implementation of the time step embedding used in AlphaFold 3.

    This module converts a scalar time variable 't' (ranging from 0 to 1)
    into a high-dimensional conditioning vector using a noise schedule
    and Fourier features, as described in AF3's supplementary information.
    """
    def __init__(self, d_embed: int, s_max=160.0, s_min=0.0004, p=7.0, sigma_data=16.0):
        """
        Initializes the TimeStepEmbedding module.

        Args:
            d_embed (int): The dimension of the high-dimensional embedding.
            s_max (float): Maximum noise level for the schedule.
            s_min (float): Minimum noise level for the schedule.
            p (float): Exponent for the noise schedule.
            sigma_data (float): Data variance constant used in scaling.
        """
        super().__init__()
        self.d_embed = d_embed
        self.s_max = s_max
        self.s_min = s_min
        self.p = p
        self.sigma_data = sigma_data

        # Randomly generate weight and bias once before training, then keep them fixed.
        # considered trainable parameters.
        self.register_buffer('w', torch.randn(self.d_embed))
        self.register_buffer('b', torch.randn(self.d_embed))

        self.time_projection_layer = nn.Sequential(
            nn.LayerNorm(d_embed),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time step embedding.

        Args:
            t (torch.Tensor): A scalar or a batch of scalars representing the
                              normalized time, with values between 0 and 1.
                              Shape: (B, ...)

        Returns:
            torch.Tensor: The high-dimensional time embedding.
                          Shape: (B, ..., d_embed).
        """
        # Ensure t is a tensor and on the same device as the model weights
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.w.device, dtype=self.w.dtype)
        
        # Add a new dimension for embedding
        t_expanded = t.unsqueeze(-1) # Shape: (B, ..., 1)

        # 1. Calculate the actual noise level `t_hat` from the normalized time `t`
        s_max_p = self.s_max ** (1 / self.p)
        s_min_p = self.s_min ** (1 / self.p)
        
        t_hat = self.sigma_data * ((s_max_p + t_expanded * (s_min_p - s_max_p)) ** self.p)
        
        # 2. Apply logarithmic scaling to `t_hat`.
        # We add a small epsilon to prevent log(0).
        log_t_hat_scaled = 0.25 * torch.log(t_hat / self.sigma_data + 1e-9)
        # 3. Apply Fourier Embedding.
        # `cos(2π(t̂w+b))` where t̂ is the log-scaled value.
        embedding = torch.cos(2 * math.pi * (log_t_hat_scaled * self.w + self.b))
        embedding = self.time_projection_layer(embedding)
        return embedding

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.linear1(x) * F.silu(self.linear2(x))
class Transition(nn.Module):
    """
    AlphaFold 3 中 Transition 层的 PyTorch 实现。
    该实现基于补充材料中的 Algorithm 11。
    它使用 SwiGLU 门控机制来代替传统的 ReLU 激活。
    """
    def __init__(self, dim: int, expansion_factor: int, d_condition: int = None, p_drop: float = 0.15):
        """
        初始化 Transition 模块。

        Args:
            dim (int): 输入和输出特征的维度 (对应算法中的 'c')。
            expansion_factor (int): 隐藏层维度的扩展因子 (对应算法中的 'n')。
            d_condition (int, optional): 条件向量的维度。如果为 None，则不使用条件。默认为 None。
            p_drop (float): Dropout 概率。
        """
        super().__init__()
        self.use_condition = d_condition is not None
        hidden_dim = dim * expansion_factor
        
        if not self.use_condition:
            self.layer_norm = nn.LayerNorm(dim)
        else:
            self.layer_norm = AdaLayerNorm(dim, d_condition, gate=True)
            
        self.swiglu = SwiGLU(dim, hidden_dim)
        self.dropout = nn.Dropout(p_drop)
        self.linear_output = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (..., dim)。
            condition (torch.Tensor, optional): 条件张量。如果模块被初始化为使用条件，则此项必须提供。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        gate = None
        if not self.use_condition:
            norm_x = self.layer_norm(x)
        else:
            if condition is None:
                raise ValueError("Conditioning tensor must be provided for a conditional Transition block.")
            norm_x, gate = self.layer_norm(x, condition)

        gated_activation = self.swiglu(norm_x)
        output = self.dropout(gated_activation)
        output = self.linear_output(output)

        
        if gate is not None:
            output = output * gate

        return output
    
class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size, condition_embed_dim, gate=True):
        super().__init__()
        # 禁用LayerNorm的默认仿射参数
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, bias=False)
        self.gate = gate
        self.condition_embed_dim = condition_embed_dim
        expand_mul = 3 if gate else 2 
        # 条件嵌入到仿射参数的映射
        self.mlp = nn.Sequential(
            nn.Linear(condition_embed_dim, condition_embed_dim),
            nn.SiLU(),
            nn.Linear(condition_embed_dim, expand_mul * hidden_size)  # 同时生成scale和shift
        )
        # nn.init.zeros_(self.mlp[0].weight)
        # nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x, cond_emb):

        """
        输入: 
          x - 特征 [batch, ..., hidden_size]
          cond_emb - 时间嵌入 [batch, ..., time_embed_dim]
        输出: 条件归一化后的特征
        """
        # 1. 标准层归一化(无参数)
        x_norm = self.norm(x)
        # 2. 从时间嵌入生成仿射参数
        params = self.mlp(cond_emb) 
        if self.gate:
            scale, shift, gate = params.chunk(3, dim=-1)  # 各 [batch, ..., hidden_size]
            return x_norm * (1 + scale) + shift, F.sigmoid(gate)
        else:
            scale, shift = params.chunk(2, dim=-1)
            return x_norm * (1 + scale) + shift