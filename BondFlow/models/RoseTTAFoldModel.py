import torch
import torch.nn as nn
from BondFlow.models.Embeddings import MSA_emb, Templ_emb
from BondFlow.models.Track_module import IterativeSimulator, SCPred
from BondFlow.models.AuxiliaryPredictor import DistanceNetwork, MaskedTokenNetwork, ExpResolvedNetwork, LDDTNetwork
from opt_einsum import contract as einsum

from layers import *

class RoseTTAFoldModule(nn.Module):
    def __init__(self, 
                 n_main_block, 
                 n_ref_block,
                 n_temp_block,
                 d_msa,
                 d_pair,
                 d_templ,
                 d_condition,
                 n_head_msa,
                 n_head_pair,
                 n_head_templ,
                 d_hidden,
                 d_hidden_templ,
                 p_drop,
                 d_t1d,
                 d_t2d,
                 SE3_param_full={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 SE3_param_topk={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 input_seq_onehot=False,     # For continuous vs. discrete sequence
                 ):

        super(RoseTTAFoldModule, self).__init__()

        # Input Embeddings
        d_state = SE3_param_topk['l0_out_features']
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair, d_state=d_state,
                p_drop=p_drop, input_seq_onehot=input_seq_onehot) # Allowed to take onehotseq
        # self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=25,
        #         p_drop=p_drop, input_seq_onehot=input_seq_onehot) # Allowed to take onehotseq
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ, d_state=d_state,
                                   n_head=n_head_templ,d_condition=d_condition,n_block=n_temp_block,
                                   d_hidden=d_hidden_templ, p_drop=0.25, d_t1d=d_t1d, d_t2d=d_t2d)

        # Update inputs with outputs from previous round
        # self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        #
        self.simulator = IterativeSimulator(
                                            n_main_block=n_main_block,
                                            n_ref_block=n_ref_block,
                                            d_msa=d_msa, 
                                            d_pair=d_pair, d_hidden=d_hidden,
                                            n_head_msa=n_head_msa,
                                            n_head_pair=n_head_pair,
                                            d_condition=d_condition,
                                            SE3_param_full=SE3_param_full,
                                            SE3_param_topk=SE3_param_topk,
                                            p_drop=p_drop)
        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa)
        self.lddt_pred = LDDTNetwork(d_state)
       
        self.exp_pred = ExpResolvedNetwork(d_msa, d_state)
        
        ######
        self.sc_predictor = SCPred(d_msa=d_msa, d_state=d_state, p_drop=p_drop)
        self.bond_pred = BondingNetwork(d_msa, d_state, d_pair)

    def forward(self, msa_latent, seq, xyz, dist_matrix, idx, 
                t1d=None, t2d=None, xyz_t=None, alpha_t=None,
                return_raw=False, return_full=False, return_infer=False,
                use_checkpoint=False, seq_mask=None, bond_mask=None,str_mask=None,batch_mask=None,
                ):


        B, N, L = msa_latent.shape[:3]
        if batch_mask is None:
            batch_mask = torch.ones((B, L), dtype=torch.bool, device=msa_latent.device)
        if seq_mask is None:
            seq_mask = torch.ones((B, L), dtype=torch.bool, device=msa_latent.device)
        if bond_mask is None:
            bond_mask = torch.ones((B, L, L), dtype=torch.bool, device=msa_latent.device)
        if str_mask is None:
            str_mask = torch.ones((B, L), dtype=torch.bool, device=msa_latent.device)

        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx, dist_matrix)
        # msa_full = self.full_emb(msa_full, seq, idx)
        stats_nan(msa_latent, 'msa_latent_emb')
        # stats_nan(msa_full, 'msa_full_emb')
        stats_nan(pair, 'pair_emb')
        print_gpu_memory()


        # Do recycling
        # if msa_prev == None:
        #     msa_prev = torch.zeros_like(msa_latent[:,0])
        #     pair_prev = torch.zeros_like(pair)
        #     state_prev = torch.zeros_like(state)
        # msa_recycle, pair_recycle, state_recycle = self.recycle(seq, msa_prev, pair_prev, xyz, state_prev,mask=batch_mask)
        # msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        # pair = pair + pair_recycle
        # state = state + state_recycle

        stats_nan(msa_latent, 'msa_latent_recycle')
        stats_nan(pair, 'pair_recycle')
        stats_nan(state, 'state_recycle')

        # add template embedding
        pair, state, t1d_condition, t2d_condition = self.templ_emb(t1d, t2d, alpha_t, xyz_t, pair, state, use_checkpoint=False,mask=batch_mask)
        stats_nan(pair, 'pair_templ')
        stats_nan(state, 'state_templ')
        print_gpu_memory()
        # Predict coordinates from given inputs
        msa, pair, R, T, alpha_s, state = self.simulator(seq, msa_latent, pair, xyz[:,:,:3],
                                                         state, idx,t1d_condition, t2d_condition,
                                                         use_checkpoint=use_checkpoint,
                                                         mask=batch_mask)
        
       
        stats_nan(msa, 'msa_simulator')
        stats_nan(pair, 'pair_simulator')
        stats_nan(R, 'R_simulator')
        stats_nan(T, 'T_simulator')
        stats_nan(alpha_s, 'alpha_s_simulator')
        stats_nan(state, 'state_simulator')
        if not alpha_s:
            alpha_s = self.sc_predictor(msa[:,0], state)
            alpha_s = alpha_s.unsqueeze(0)  # Ensure alpha_s is a tensor T,B,L,10,2

        if return_raw:
            # get last structure
            xyz_out = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            return msa[:,0], pair, xyz_out, state, alpha_s[-1]

        # predict masked amino acids
        logits_aa = self.aa_pred(msa)
        
        # Predict LDDT
        # lddt = self.lddt_pred(state)

        # predict bonding matrix
        res_mask_2d = batch_mask.unsqueeze(1) * batch_mask.unsqueeze(2)  # B, L, L
        bond_matrix = self.bond_pred(msa, state, pair,res_mask_2d)

        if return_infer:
            # get last structure
            xyz_out = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            
            # get scalar plddt
            nbin = lddt.shape[1]
            bin_step = 1.0 / nbin
            lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=lddt.dtype, device=lddt.device)
            pred_lddt = nn.Softmax(dim=1)(lddt)
            pred_lddt = torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

            return msa[:,0], pair, xyz_out, state, alpha_s[-1], logits_aa.permute(0,2,1), bond_matrix, pred_lddt
        #
        # # predict distogram & orientograms
        # logits = self.c6d_pred(pair)
        
        # # predict experimentally resolved or not
        # logits_exp = self.exp_pred(msa[:,0], state)
        
        # get all intermediate bb structures
        xyz_out = einsum('rbnij,bnaj->rbnai', R, xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T.unsqueeze(-2)


        return logits_aa.permute(0,2,1), xyz_out.permute(1,0,2,3,4), alpha_s.permute(1,0,2,3,4), bond_matrix
