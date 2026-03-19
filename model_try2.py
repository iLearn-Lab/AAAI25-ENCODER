import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Categorical
from torch.autograd import Variable
from transformers import CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor, AutoTokenizer, CLIPTextModelWithProjection
import open_clip
import os
import numbers
from typing import Callable, Optional, Sequence, Tuple
from einops import rearrange
import math
from token_wise_matching import token_wise_matching

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens

class ScoreNet(nn.Module):
    def __init__(self, embed_dim=512, sparse_ratio=0.7, num_keep_token=10):
        super().__init__()
        self.num_keep_token = num_keep_token
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
    
    def forward(self, tokens, attention_x, attention_y):
        B_v, L_v, C = tokens.size()
        score = attention_x + attention_y

        num_keep_token = math.ceil(L_v * self.sparse_ratio)#self.num_keep_token#
        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        keep_policy = score_index[:, :num_keep_token]
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))

        non_keep_policy = score_index[:, num_keep_token:]
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))
        non_keep_score = score_sort[:, num_keep_token:]

        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)

        return select_tokens, extra_token, score_mask
    

class FactorNet(nn.Module):
    def __init__(self, dim=512, keeped_patches=64, dim_ratio=0.2):
        super().__init__()
        
        hidden_dim = int(dim * dim_ratio)

        self.weight = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, keeped_patches)
                        )

        self.scale = nn.Parameter(torch.ones(1, 1, 1))
        
    def forward(self, x, keep_policy=None):
        weight = self.weight(x)
        weight = weight.transpose(2, 1) * self.scale       
        if keep_policy is not None:
            keep_policy = keep_policy.unsqueeze(1)
            weight = weight - (1 - keep_policy) * 1e10
        weight = F.softmax(weight, dim=2)
        x = torch.bmm(weight, x)
        
        return x


class BindingDecoder(nn.Module):
    def __init__(self, dim, hidden_dim, N_p, N_v):
        print(N_p)
        super().__init__()
        self.learnable_relation = nn.Parameter(torch.randn(N_p, dim, dtype=torch.float32)).cuda()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.mlp_p = nn.Sequential(
                        nn.LayerNorm(N_p),
                        nn.Linear(N_p, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, N_p)
                        )
        self.mlp_v = nn.Sequential(
                        nn.LayerNorm(N_v+1),
                        nn.Linear(N_v+1, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, N_v+1)
                        )
        self.mlp = nn.Sequential(
                        nn.LayerNorm(self.dim),
                        nn.Linear(self.dim, self.dim),
                        nn.GELU(),
                        nn.Linear(self.dim, self.dim)
                        )
        
    
    def binding(self, V):
        relation = self.learnable_relation[:, :].unsqueeze(0).cuda()

        S_cross = self.mlp_v(torch.matmul(relation, V.transpose(-2, -1)) / (self.dim ** 0.5))
        S_self = self.mlp_p(torch.matmul(relation, relation.transpose(-2, -1)) / (self.dim ** 0.5))
        
        S_final = self.sigmoid(torch.matmul(S_self, S_cross)) * S_cross
        entity = torch.matmul(self.softmax(S_final), V)
        
        entity = self.softmax(self.mlp(entity))
        return entity, relation
        
        
class Backbone(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.0, local_token_num=8, wc=2, N_p=3, weighted=True):
        super().__init__()
        clip_path = './'#"/root/CLIP-ViT-B-32-laion2B-s34B-b79K" #
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained=os.path.join(clip_path, 'open_clip_pytorch_model.bin'))
        self.clip = self.clip.float()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(768,512)
        self.text_fc = nn.Linear(512,512)
        self.weighted = weighted


        self.weight_chanelImg = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 49),
                        nn.Softmax(dim=-1)
                        )
        self.weight_chanelImg_t = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 49),
        )
        self.cross_t2i = nn.Sequential(
                        nn.Linear(77, 49),
                        nn.Softmax(dim=-1)
                        )
        
        self.weight_chanelText = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 77),
                        nn.Softmax(dim=-1)
                        )
        self.weight_chanelText_i = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 77),
                        
                        )
        self.cross_i2t = nn.Sequential(
                        nn.Linear(49,77),
                        nn.Softmax(dim=-1)
                        )
        

        self.weight_cross = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        nn.Softmax(dim=-1)
                        )

        self.sparse_ratio = 0.6
        self.sparse_net = ScoreNet(embed_dim=self.hidden_dim, 
                                      sparse_ratio=self.sparse_ratio,
                                      num_keep_token=local_token_num
                                      )

        self.sparse_net_text = ScoreNet(embed_dim=self.hidden_dim, 
                                      sparse_ratio=self.sparse_ratio,
                                      num_keep_token=local_token_num
                                      )
        self.keeped_patches = local_token_num#int(self.num_patches * self.aggr_ratio * self.sparse_ratio)

        self.fac_net= FactorNet(dim=self.hidden_dim, 
                                        keeped_patches=self.keeped_patches,
                                        ) 
         
        self.binding_decoder = BindingDecoder(self.hidden_dim, self.hidden_dim, N_p, local_token_num)
        if wc == 1:
            self.t_weight_tokens = nn.Sequential(
                    nn.Linear(self.hidden_dim, 1))
            self.v_weight_tokens = nn.Sequential(
                    nn.Linear(self.hidden_dim, 1))
            self.t_weight_phrases = nn.Sequential(
                    nn.Linear(self.hidden_dim, 1))
            self.v_weight_phrases = nn.Sequential(
                    nn.Linear(self.hidden_dim, 1))
            self.t_weight_batch = nn.Sequential(
                    nn.Linear(self.hidden_dim, 1))
            self.v_weight_batch = nn.Sequential(
                    nn.Linear(self.hidden_dim, 1))
        elif wc == 2:
            self.t_weight_tokens = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
            self.v_weight_tokens = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
            self.t_weight_phrases = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
            self.v_weight_phrases = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
            self.t_weight_batch = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
            self.v_weight_batch = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
        elif wc == 3:
            self.t_weight_tokens = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
            self.v_weight_tokens = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
            self.t_weight_phrases = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
            self.v_weight_phrases = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
            self.t_weight_batch = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
            self.v_weight_batch = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 1),
                        )
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, 16, batch_first=True)
        self.local_token_num = local_token_num
        # self.global_token_num = global_token_num
        

    def visual_out(self, x):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([_expand_token(self.clip.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.patch_dropout(x)
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND(grid**2,batch_size,width)
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD(batch_size,grid**2,width)

        x = self.clip.visual.ln_post(x)
        pooled, tokens = self.clip.visual._global_pool(x)
        # print(tokens.shape)

        pooled = pooled @ self.clip.visual.proj
        
        return pooled, x
    
    
    def text_out(self, text):
        cast_dtype = self.clip.transformer.get_cast_dtype()

        x = self.clip.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        pooled, tokens = text_global_pool(x, text, self.clip.text_pool_type)
        if self.clip.text_projection is not None:
            if isinstance(self.clip.text_projection, nn.Linear):
                pooled = self.clip.text_projection(x)
            else:
                pooled = pooled @ self.clip.text_projection

        return pooled, x

    def extract_img_fea(self, x):
        img_global_fea, img_local_fea = self.visual_out(x)
        img_global_fea = img_global_fea.unsqueeze(1)
        img_local_fea = self.fc(img_local_fea.float())
        # return torch.cat([img_global_fea], dim=1)

        img_spatial_embs = img_local_fea[:, 1:, :]

        img_spatial_glo_norm = torch.matmul(self.weight_chanelImg(img_spatial_embs).transpose(1, 2), img_spatial_embs)
        # (B_v, L_v, C) -> (B_v, L_v)
        img_spatial_self_attention = self.weight_cross(img_spatial_glo_norm * img_spatial_embs).mean(dim=-1)

        
        # (B_v, L_v, C) -> (B_v, L_v)
        img_spatial_cap_i_attention = self.weight_cross(img_spatial_glo_norm * img_spatial_embs).mean(dim=-1)#.sum(dim=-1)

        img_select_tokens, img_extra_token, img_score_mask = self.sparse_net(tokens=img_spatial_embs, 
                                                                attention_x=img_spatial_self_attention, 
                                                                attention_y=img_spatial_cap_i_attention,
                                                                )
        img_select_tokens = self.fac_net(img_select_tokens)
        img_entity, _ = self.binding_decoder.binding(torch.cat([img_global_fea, img_select_tokens], dim=1)[:,0:1+self.local_token_num,:])

        img_select_tokens = torch.cat([img_global_fea, img_entity, img_select_tokens], dim=1)
        return img_select_tokens
    
    def extract_img_fea_patch_selection(self, img_x, txt):
        img_global_fea, img_local_fea = self.visual_out(img_x)
        img_global_fea = img_global_fea.unsqueeze(1)
        img_local_fea = self.fc(img_local_fea.float())

        txt_token = self.tokenizer(txt).cuda()
        text_global_fea, text_local_fea = self.text_out(txt_token)
        text_global_fea = text_global_fea.unsqueeze(1)
        text_local_fea = self.text_fc(text_local_fea.float()) 

        cap_embs = text_local_fea


        img_spatial_embs = img_local_fea[:, 1:, :]

        img_spatial_glo_norm = torch.matmul(self.weight_chanelImg(img_spatial_embs).transpose(1, 2), img_spatial_embs)

        img_spatial_self_attention = self.weight_cross(img_spatial_glo_norm * img_spatial_embs).mean(dim=-1)

        cap_length = text_local_fea.shape[1]
        
        cap_i_glo = torch.matmul(self.cross_t2i(self.weight_chanelImg_t(text_local_fea).transpose(1, 2)), img_spatial_embs) 

        img_spatial_cap_i_attention = self.weight_cross(cap_i_glo * img_spatial_embs).mean(dim=-1)#.sum(dim=-1)
        img_select_tokens, img_extra_token, img_score_mask = self.sparse_net(tokens=img_spatial_embs, 
                                                                attention_x=img_spatial_self_attention, 
                                                                attention_y=img_spatial_cap_i_attention,
                                                                )
        img_select_tokens = self.fac_net(img_select_tokens)
        
        img_entity, relation = self.binding_decoder.binding(torch.cat([img_global_fea, img_select_tokens], dim=1)[:,0:1+self.local_token_num,:])

        img_select_tokens_con = torch.cat([img_global_fea, img_entity, img_select_tokens], dim=1)

        img_2_Text_w = torch.matmul(self.cross_i2t(self.weight_chanelText_i(img_spatial_embs).transpose(1,2)), text_local_fea) 
        text_2_Text_w = torch.matmul(self.weight_chanelText(text_local_fea).transpose(1, 2), text_local_fea)

        cap_self_attention = self.weight_cross(text_2_Text_w * cap_embs).mean(dim=-1)#.sum(dim=-1)
        cap_spatial_img_attention = self.weight_cross(img_2_Text_w * cap_embs).mean(dim=-1)#.sum(dim=-1)

        cap_select_tokens, cap_extra_token, cap_score_mask = self.sparse_net_text(tokens=cap_embs, 
                                                        attention_x=cap_self_attention, 
                                                        attention_y=cap_spatial_img_attention,
                                                        )
        cap_select_tokens = self.fac_net(cap_select_tokens)

        cap_action, _ = self.binding_decoder.binding(torch.cat([text_global_fea, cap_select_tokens], dim=1)[:,0:1+self.local_token_num,:])
        
        cap_select_tokens_con = torch.cat([text_global_fea, cap_action, cap_select_tokens], dim=1)

        sim_entity_action = token_wise_matching(img_entity, cap_action, weighted=self.weighted, t_weight=self.t_weight_phrases, v_weight=self.v_weight_phrases)

        return img_select_tokens_con, cap_select_tokens_con, sim_entity_action, relation, img_entity, cap_action


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level

        self.MLP = nn.Sequential(
                        nn.LayerNorm(in_channels * 2),
                        nn.Linear(in_channels * 2, in_channels),
                        nn.GELU(),
                        nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
                        )

    def forward(self, x, text_embed):
        text_embed_ = torch.cat([x, text_embed], dim=-1)
        batch = x.shape[0]
        chanel = x.shape[1] * 2
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed_).reshape(batch, chanel, -1).chunk(2, dim=1)
            x = gamma * x + (beta) * text_embed
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class Encoder(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.0, local_token_num=8, t=10, wc=2, N_p=3, weighted=True):
        super().__init__()
        self.backbone = Backbone(hidden_dim, dropout, local_token_num, wc, N_p, weighted)
        self.loss_T = nn.Parameter(torch.tensor([10.]))

        self.local_weight = nn.Parameter(torch.tensor([1.0 for _ in range(local_token_num + N_p + 1)]))
        self.t = t
        
        self.affine = FeatureWiseAffine(hidden_dim, hidden_dim, use_affine_level=True)


    def target_fea(self, tag):
        tag_token = self.backbone.extract_img_fea(tag)
        return tag_token#, ref_mask
    
    def compose_feature(self, ref, mod):
        ref_token, mod_token, sim_entity_action, relation, entity, action = self.backbone.extract_img_fea_patch_selection(ref, mod)

        fuse_local = self.affine(ref_token, mod_token)

        return fuse_local, sim_entity_action, relation, entity, action

    def extract_retrieval_compose(self, ref, mod):
        fuse_local,_,_, _,_ = self.compose_feature(ref, mod)
        
        fuse_local = F.normalize(torch.mean(fuse_local, dim=1), p=2, dim=-1)

        return fuse_local

    def extract_retrieval_target(self, tag):
        tag_local = self.target_fea(tag)
        tag_local = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)
        return tag_local

    def compute_loss(self, ref, mod, tag):

        fuse_local, sim_entity_action, relation,entity, action = self.compose_feature(ref, mod)
        
        tag_local = self.target_fea(tag)
        loss = {}
        
        retrieval_query = F.normalize(torch.mean(fuse_local, dim=1), p=2, dim=-1)
        retrieval_target = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)
        entity = F.normalize(torch.mean(entity.transpose(-2,-1), dim=1), p=2, dim=-1)
        action = F.normalize(torch.mean(action.transpose(-2,-1), dim=1), p=2, dim=-1)
        tag_feature = (F.normalize(tag_local, p=2, dim=-1) * self.local_weight.unsqueeze(0).unsqueeze(-1)).flatten(1)

        loss['stu_rank'] = self.info_nce(retrieval_query, retrieval_target)
        loss['kl'] = self.kl_div(retrieval_query, retrieval_target, tag_feature, tag_feature, self.t)
        loss['entity'] = self.info_nce_rpm(sim_entity_action)
        loss['ortho'] =  self.orthogonal_regularization(relation)

        return loss

    
    def mask_constraint(self, mask1, mask2):
        mask = mask1 + mask2
        y = torch.ones_like(mask).float().cuda()
        return F.mse_loss(mask,y)

    def info_nce(self, query, target):
        x = torch.mm(query, target.T)
        labels = torch.arange(query.shape[0]).long().cuda()
        return F.cross_entropy(x * self.loss_T, labels)
    def info_nce_rpm(self, sim):
        labels = torch.arange(sim.shape[0]).long().cuda()
        return (F.cross_entropy(sim*self.loss_T, labels) + F.cross_entropy(sim.T*self.loss_T, labels)) / 2

    
    def kl_div(self, x1, y1, x2, y2, t):
        x1 = F.normalize(x1, p=2, dim=-1)
        y1 = F.normalize(y1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        y2 = F.normalize(y2, p=2, dim=-1)

        x1_y1 = torch.mm(x1, y1.T) / t
        x2_y2 = torch.mm(x2, y2.T) / t

        log_soft_x1 = F.log_softmax(x1_y1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2_y2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')

        return kl
    
    def orthogonal_regularization(self, templates):
        # batch_size, length, dim
        batch_size, length, dim = templates.size()
        device = templates.device
        norm_templates = F.normalize(templates, p=2, dim=-1)
        # (B,L,D) * (B,D,L)
        cosine_score = torch.matmul(norm_templates, norm_templates.permute(0,2,1).contiguous()) # batch_size, length, length 
        eye_matrix = torch.eye(length).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        l2_loss = torch.nn.MSELoss()
        return l2_loss(cosine_score, eye_matrix)


