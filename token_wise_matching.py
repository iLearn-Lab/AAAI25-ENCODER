import torch
import torch.nn.functional as F
import math
import torch.nn as nn


def token_wise_matching(V, T, weighted=None, t_weight=None, v_weight=None):
    # S_local (B x B)    S_local(i,j) = S(t_i,v_j)
    B, N_v, d, N_t= V.shape[0], V.shape[1], V.shape[2], T.shape[1]
    if not weighted:

        V = V / V.norm(dim=-1, keepdim=True)
        T = T / T.norm(dim=-1, keepdim=True)
        M = torch.einsum('atd,bvd->abtv', T, V)
        t2v_m, max_idx1 = M.max(dim=-1)    # bbtv -> bbt  
        v2t_m, max_idx2 = M.max(dim=-2)    # bbtv -> bbv
        t2v_m = torch.sum(t2v_m, dim=2)
        v2t_m = torch.sum(v2t_m, dim=2)
        t2v_m = torch.softmax(t2v_m, dim=-1)
        v2t_m = torch.softmax(v2t_m, dim=-1)
        s = (t2v_m + v2t_m)/2
    else:
        V = V / V.norm(dim=-1, keepdim=True)
        T = T / T.norm(dim=-1, keepdim=True)
        M = torch.einsum('atd,bvd->abtv', T, V)

        #print("M: ",M.shape)
        t2v_m, max_idx1 = M.max(dim=-1)    # bbtv -> bbt
        
        t_weight_output = t_weight(T).squeeze(2)
        t_weight_output = F.normalize(t_weight_output, dim=-1)

        t2v_m = torch.einsum('abt,at->ab', t2v_m, t_weight_output)    
        v2t_m, max_idx2 = M.max(dim=-2)    # bbtv -> bbv
        
        v_weight_output = v_weight(V).squeeze(2)
        v_weight_output = F.normalize(v_weight_output,dim=-1)
        v2t_m = torch.einsum('abv,bv->ab', v2t_m, v_weight_output)
        s = (t2v_m + v2t_m)/2
    return s
