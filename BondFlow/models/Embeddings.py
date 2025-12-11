import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
import torch.utils.checkpoint as checkpoint
from rfdiff.util import get_tips
from rfdiff.util_module import Dropout, create_custom_forward, rbf, init_lecun_normal
from Attention_module import Attention,  AttentionWithBias
from BondFlow.models.Track_module import PairStr2Pair
from BondFlow.models.layers import *

# Module contains classes and functions to generate initial embeddings

class PositionalEncoding2D(nn.Module):
    # Add relative positional encoding to pair features
    def __init__(self, d_model, minpos=-32, maxpos=32, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos)+maxpos+1
        self.emb = nn.Embedding(self.nbin, d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x, idx):
        bins = torch.arange(self.minpos, self.maxpos, device=x.device)
        seqsep = idx[:,None,:] - idx[:,:,None] # (B, L, L)
        #
        ib = torch.bucketize(seqsep, bins).long() # (B, L, L)
        emb = self.emb(ib) #(B, L, L, d_model)
        x = x + emb # add relative positional encoding
        return self.drop(x)

class MSA_emb(nn.Module):
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_pair=128, d_state=32, d_init=21,
                 minpos=-32, maxpos=32, p_drop=0.1, max_atom_dist = 32+1, input_seq_onehot=False):
        super(MSA_emb, self).__init__()
        #self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(d_init, d_msa) # embedding for query sequence -- used for MSA embedding
        self.emb_left = nn.Embedding(d_init, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_right = nn.Embedding(d_init, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_state = nn.Embedding(d_init, d_state)
        self.drop = nn.Dropout(p_drop)
        self.pos = PositionalEncoding2D(d_pair, minpos=minpos, maxpos=maxpos, p_drop=p_drop)
        self.CA_emb = nn.Embedding(max_atom_dist, d_pair) # embedding for CA distance matrix
        self.input_seq_onehot=input_seq_onehot

        self.reset_parameter()

    def reset_parameter(self):
        #self.emb = init_lecun_normal(self.emb)
        self.emb_q = init_lecun_normal(self.emb_q)
        self.emb_left = init_lecun_normal(self.emb_left)
        self.emb_right = init_lecun_normal(self.emb_right)
        self.emb_state = init_lecun_normal(self.emb_state)

        #nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx, CA_dist_matrix):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        #   - CA_dist_matrix: CA distance matrix (B, L, L) -- used for pair embedding
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        #   - pair: Initial Pair embedding (B, L, L, d_pair)

        N = msa.shape[1] # number of sequenes in MSA
        # msa embedding
        #msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding
        # Sergey's one hot trick
        #tmp = (seq @ self.emb_q.weight).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        #msa = msa + tmp.expand(-1, N, -1, -1) # adding query embedding to MSA

        #msa = self.drop(msa)

        msa= self.drop(seq @ self.emb_q.weight).unsqueeze(1).expand(-1, N, -1, -1)
        # pair embedding.unsqueeze(1)
        # Sergey's one hot trick
        left  = (seq @ self.emb_left.weight)[:,None] # (B, 1, L, d_pair)
        right = (seq @ self.emb_right.weight)[:,:,None] # (B, L, 1, d_pair)

        pair = left + right # (B, L, L, d_pair)
        pair = self.pos(pair, idx) # add relative position

        CA_matrix_emb =  self.CA_emb(CA_dist_matrix)
        pair = pair + CA_matrix_emb
        # state embedding
        # Sergey's one hot trick
        state = self.drop(seq @ self.emb_state.weight)
        return msa, pair, state

# class Extra_emb(nn.Module):
#     # Get initial seed MSA embedding
#     def __init__(self, d_msa=256, d_init=22+1+2, p_drop=0.1, input_seq_onehot=False):
#         super(Extra_emb, self).__init__()
#         self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
#         self.emb_q = nn.Embedding(22, d_msa) # embedding for query sequence
#         self.drop = nn.Dropout(p_drop)

#         self.input_seq_onehot=input_seq_onehot

#         self.reset_parameter()

#     def reset_parameter(self):
#         self.emb = init_lecun_normal(self.emb)
#         nn.init.zeros_(self.emb.bias)

#     def forward(self, msa, seq, idx):
#         # Inputs:
#         #   - msa: Input MSA (B, N, L, d_init)
#         #   - seq: Input Sequence (B, L)
#         #   - idx: Residue index
#         # Outputs:
#         #   - msa: Initial MSA embedding (B, N, L, d_msa)
#         N = msa.shape[1] # number of sequenes in MSA
#         msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding

#         # Sergey's one hot trick
#         seq = (seq @ self.emb_q.weight).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
#         msa = msa + seq.expand(-1, N, -1, -1) # adding query embedding to MSA
#         return self.drop(msa)

class TemplatePairStack(nn.Module):
    # process template pairwise features
    # use structure-biased attention
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, p_drop=0.25):
        super(TemplatePairStack, self).__init__()
        self.n_block = n_block
        proc_s = [PairStr2Pair(d_pair=d_templ, n_head=n_head, d_hidden=d_hidden, p_drop=p_drop) for i in range(n_block)]
        self.block = nn.ModuleList(proc_s)
        self.norm = nn.LayerNorm(d_templ)
    def forward(self, templ, rbf_feat, mask=None, use_checkpoint=False):
        B, L = templ.shape[:2]
        templ = templ.reshape(B, L, L, -1)

        for i_block in range(self.n_block):
            if use_checkpoint:
                templ = checkpoint.checkpoint(create_custom_forward(self.block[i_block],mask=mask), templ, rbf_feat, None)
            else:
                templ = self.block[i_block](templ, rbf_feat,None,mask=mask)
        return self.norm(templ).reshape(B, L, L, -1)

class TemplateTorsionStack(nn.Module):
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, p_drop=0.15):
        super(TemplateTorsionStack, self).__init__()
        self.n_block=n_block
        self.proj_pair = nn.Linear(d_templ, d_templ)
        proc_s = [AttentionWithBias(d_in=d_templ, d_bias=d_templ,
                                    n_head=n_head, d_hidden=d_hidden) for i in range(n_block)]
        self.row_attn = nn.ModuleList(proc_s)
        proc_s = [Transition(d_templ, 4, p_drop=p_drop) for i in range(n_block)]
        self.ff = nn.ModuleList(proc_s)
        self.norm = nn.LayerNorm(d_templ)

    def reset_parameter(self):
        self.proj_pair = init_lecun_normal(self.proj_pair)
        nn.init.zeros_(self.proj_pair.bias)

    def forward(self, tors, pair, mask=None, use_checkpoint=False):
        B, L = tors.shape[:2]
        tors = tors.reshape(B, L, -1)
        pair = pair.reshape(B, L, L, -1)
        #pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = self.proj_pair(pair)

        for i_block in range(self.n_block):
            if use_checkpoint:
                tors = tors + checkpoint.checkpoint(create_custom_forward(self.row_attn[i_block],mask=mask), tors, pair)
            else:
                tors = tors + self.row_attn[i_block](tors, pair,mask=mask)
            tors = tors + self.ff[i_block](tors)
        return self.norm(tors).reshape(B, L, -1)

class Templ_emb(nn.Module):
    # Get template embedding
    # Features are
    #   t2d:
    #   - 37 distogram bins + 6 orientations (43)
    #   - Mask (missing/unaligned) (1)
    #   t1d:
    #   - tiled AA sequence (20 standard aa + gap)
    #   - confidence (1)
    #   - contacting or note (1). NB this is added for diffusion model. Used only in complex training examples - 1 signifies that a residue in the non-diffused chain\
    #     i.e. the context, is in contact with the diffused chain.
    #
    #Added extra t1d dimension for contacting or not
    def __init__(self, d_t1d=21+1+1, d_t2d=43+1, d_tor=30, d_pair=128, d_state=32,
                 n_block=4, d_templ=64,d_condition=128,
                 n_head=4, d_hidden=16, p_drop=0.25):
        super(Templ_emb, self).__init__()
        # process 2D features
        self.d_t1d = d_t1d
        self.d_t2d = d_t2d
        self.d_tor = d_tor
        self.d_pair = d_pair
        self.d_state = d_state
        self.d_templ = d_templ
        self.d_condition = d_condition
        self.emb = nn.Linear(d_t1d*2+d_t2d, d_templ)
        self.templ_stack = TemplatePairStack(n_block=n_block, d_templ=d_templ, n_head=n_head,
                                             d_hidden=d_hidden, p_drop=p_drop)

        self.attn = Attention(d_pair, d_templ, n_head, d_hidden, d_templ)

        # process torsion angles
        self.emb_t1d = nn.Linear(d_t1d+d_tor, d_templ)
        self.proj_t1d = nn.Linear(d_templ, d_templ)
        self.tor_stack = TemplateTorsionStack(n_block=n_block, d_templ=d_templ, n_head=n_head,
                                              d_hidden=d_hidden, p_drop=p_drop)
        # self.attn_tor = Attention(d_state, d_templ, n_head, d_hidden, d_state + d_condition)
        self.state_mlp = nn.Sequential(
            nn.Linear(d_templ,d_condition+ d_state),
            nn.ReLU(),
            nn.Linear(d_condition + d_state, d_state + d_condition)
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_templ, d_pair + d_condition),
            nn.ReLU(),
            nn.Linear(d_pair + d_condition, d_pair + d_condition)
        )

        self.reset_parameter()

    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

        nn.init.kaiming_normal_(self.emb_t1d.weight, nonlinearity='relu')
        nn.init.zeros_(self.emb_t1d.bias)

        self.proj_t1d = init_lecun_normal(self.proj_t1d)
        nn.init.zeros_(self.proj_t1d.bias)

    def forward(self, t1d, t2d, alpha_t, xyz_t, pair, state, mask=None, use_checkpoint=False):
        # Input
        #   - t1d: 1D template info (B, L, 23)
        #   - t2d: 2D template info (B, L, L, 44)
        B,L, _ = t1d.shape

        # Prepare 2D template features
        left = t1d.unsqueeze(2).expand(-1,-1,L,-1)
        right = t1d.unsqueeze(1).expand(-1,L,-1,-1)
        templ = torch.cat((t2d, left, right), -1) # (B, L, L, 90)
        templ = self.emb(templ) # Template templures (B, L, L, d_templ)
        # process each template features
        xyz_t = xyz_t.reshape(B, L, -1, 3)
        rbf_feat = rbf(torch.cdist(xyz_t[:,:,1], xyz_t[:,:,1]))
        templ = self.templ_stack(templ, rbf_feat, mask=mask,use_checkpoint=use_checkpoint) # (B, L,L, d_templ)

        # Prepare 1D template torsion angle features
        t1d = torch.cat((t1d, alpha_t), dim=-1) # (B,  L, t1d+30)
        # process each template features
        t1d = self.proj_t1d(F.relu_(self.emb_t1d(t1d)))

        # mixing query state features to template state features
        if use_checkpoint:
            out = checkpoint.checkpoint(create_custom_forward(self.tor_stack,mask=mask),t1d, templ)
        else:
            #out = self.attn_tor(state, t1d, t1d,mask=mask.reshape(B*L,-1)).reshape(B, L, -1)
            out = self.tor_stack(t1d, templ, mask=mask)
        out = self.state_mlp(out)  # (B, L, d_state + d_condition)
        # state = state.reshape(B, L, -1)
        state = state + out[..., :self.d_state] 
        t1d_condition = out[..., self.d_state:]  # condition for attention
        

        # mixing query pair features to template information (Template pointwise attention)
        # mask_2d = mask.unsqueeze(1).bool() & mask.unsqueeze(2).bool()
        # pair = pair.reshape(B,L*L, -1)
        # templ = templ.reshape(B,L*L, -1)
        # if use_checkpoint:
        #     out = checkpoint.checkpoint(create_custom_forward(self.attn), pair, templ, templ,
        #                                 mask=mask_2d.reshape(B,L*L,-1))
        #     out = out.reshape(B, L, L, -1)
        # else:
        #     out = self.attn(pair, templ, templ,mask=mask_2d.reshape(B,L*L,-1)).reshape(B, L, L, -1)
        #
        out = self.pair_mlp(templ)  # (B, L, L, d_pair + d_condition)
        pair = pair.reshape(B, L, L, -1)
        pair = pair + out[..., :self.d_pair]
        t2d_condition = out[..., self.d_pair:]  # condition for attention
        return pair, state, t1d_condition, t2d_condition 

# class Recycling(nn.Module):
#     def __init__(self, d_msa=256, d_pair=128, d_state=32):
#         super(Recycling, self).__init__()
#         self.proj_dist = nn.Linear(36+d_state*2, d_pair)
#         self.norm_state = nn.LayerNorm(d_state)
#         self.norm_pair = nn.LayerNorm(d_pair)
#         self.norm_msa = nn.LayerNorm(d_msa)

#         self.reset_parameter()

#     def reset_parameter(self):
#         self.proj_dist = init_lecun_normal(self.proj_dist)
#         nn.init.zeros_(self.proj_dist.bias)

#     def forward(self, seq, msa, pair, xyz, state,mask=None):
#         B, L = pair.shape[:2]
#         mask = (mask == 0)
#         state = self.norm_state(state)
#         state = state.masked_fill(mask.unsqueeze(-1), 0.0) if mask is not None else state
#         #
#         left = state.unsqueeze(2).expand(-1,-1,L,-1)
#         right = state.unsqueeze(1).expand(-1,L,-1,-1)

#         # three anchor atoms
#         N  = xyz[:,:,0]
#         Ca = xyz[:,:,1]
#         C  = xyz[:,:,2]
#         # recreate Cb given N,Ca,C
#         b = Ca - N
#         c = C - Ca
#         a = torch.cross(b, c, dim=-1)
#         Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

#         dist = rbf(torch.cdist(Cb, Cb))
#         dist = torch.cat((dist, left, right), dim=-1)
#         dist = self.proj_dist(dist)
#         pair = dist + self.norm_pair(pair)
#         msa = self.norm_msa(msa)
#         msa = msa.masked_fill(mask.unsqueeze(-1), 0.0) if mask is not None else msa

#         mask_2d = mask.unsqueeze(1) | mask.unsqueeze(2)
#         pair = pair.masked_fill(mask_2d.unsqueeze(-1), 0.0) 

#         return msa, pair, state

