## Adopted from TransforMatcher: Match-to-Match Attention for Semantic Correspondence. 
## https://github.com/wookiekim/transformatcher


##################################################################
# Inspired by FastAttention implementation of lucidrains         #
# https://github.com/lucidrains/fast-transformer-pytorch         #
# Inspired by model script of Convolutional Hough Matching       #
# https://github.com/juhongm999/chm                              #
##################################################################
 
import math

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce
# import opt_einsum as oe

from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

from .base.geometry import Geometry
from .base.correlation import Correlation

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.Linear(dim, int(dim * mult)),
        nn.GELU(),
        nn.Linear(int(dim * mult), dim)
    )

class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature, dim=1):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), dim) + epsilon, 0.5).unsqueeze(dim).expand_as(feature)
        return torch.div(feature, norm)



class Match2MatchAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64,
        max_seq_len = 30**4,
        pos_emb = None,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.pos_emb = pos_emb
        self.max_seq_len = max_seq_len

        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias = False)  # for projecting queries to query attention logits
        self.to_k_attn_logits = nn.Linear(dim_head // 2, 1, bias = False)  # for projecting keys to key attention logits

        self.to_r = nn.Linear(dim_head // 2, dim_head)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        n, device, h = x.shape[1], x.device, self.heads
        use_rotary_emb = True

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        freqs = self.pos_emb(torch.arange(self.max_seq_len, device = device), cache_key = self.max_seq_len)
        freqs = rearrange(freqs[:n], 'n d -> () () n d')
        q_aggr, k_aggr = map(lambda t: apply_rotary_emb(freqs, t), (q, k))
        
        v_aggr = v
        # calculate query attention logits

        q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
        q_attn = q_attn_logits.softmax(dim = -1)

        # calculate global query token

        global_q = torch.einsum('b h n, b h n d -> b h d', q_attn, q_aggr)
        global_q = rearrange(global_q, 'b h d -> b h () d')

        # bias keys with global query token

        k = k * global_q
        k = reduce(k, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # now calculate key attention logits

        k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
        k_attn = k_attn_logits.softmax(dim = -1)

        # calculate global key token

        global_k = torch.einsum('b h n, b h n d -> b h d', k_attn, k_aggr)
        global_k = rearrange(global_k, 'b h d -> b h () d')

        # bias the values

        u = v_aggr * global_k
        u = reduce(u, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # transformation step

        r = self.to_r(u)

        # add the queries as a residual

        r = r + q

        # combine heads

        r = rearrange(r, 'b h n d -> b n (h d)')
        return self.to_out(r)


class TransforMatcher(nn.Module):

    # def __init__(self, feat_dims,luse):
    def __init__(self, args):
        super(TransforMatcher, self).__init__()

        # input_dim = 16
        input_dim = 6
        layer_num = 6 
        expand_ratio = 4
        # bottlen = 26 # 23 + 3 bottleneck layers
        # bottlen = 6 + 3
        bottlen = 2

        self.flatten_and_project_matches = nn.Sequential(
            Rearrange('b c h1 w1 h2 w2 -> b (h1 w1 h2 w2) c'),
            nn.Linear(bottlen, input_dim)
        )

        layer_pos_emb = RotaryEmbedding(dim=4, freqs_for = 'pixel')

        self.to_correlation = nn.Sequential(
            nn.Linear(input_dim, 1),
            Rearrange('b (h1 w1 h2 w2) c -> b c h1 w1 h2 w2', h1=30, w1=30, h2=30, w2=30),
        )
        
        self.trans_nc = nn.ModuleList([])
        for _ in range(layer_num):
            self.trans_nc.append(nn.ModuleList([
                PreNorm(input_dim, Match2MatchAttention(input_dim, heads = 8, dim_head = 4, pos_emb=layer_pos_emb)),
                PreNorm(input_dim, FeedForward(input_dim)),
            ]))
        
        self.relu = nn.ReLU(inplace=True)

        self.args = args
        self.temp = args.temp
        self.target_size = [60, 60]
        self.l2norm = FeatureL2Norm()
        self.downsample = nn.MaxPool2d(2) # downsample by factor of 2
        # self.downsample_1024 = nn.Conv2d(1024, 1024, (1,1), 2)
        # self.downsample_2048 = nn.Conv2d(2048, 2048, (1,1), 2)


    def cosine_similarity(self, src_feats, trg_feats):
        correlations = []
        for i, (src, trg) in enumerate(zip(src_feats, trg_feats)): 
            # if src.size(dim=1) == 1024:
            #     src = self.downsample_1024(src)
            #     trg = self.downsample_1024(trg)
            # elif src.size(dim=1) == 2048:
            #     src = self.downsample_2048(src)
            #     trg = self.downsample_2048(trg)
            src = self.downsample(src)
            trg = self.downsample(trg)
            src = self.l2norm(src)
            trg = self.l2norm(trg)
            corr = src.flatten(2).transpose(-1, -2) @ trg.flatten(2)

            B, ch, h, w = src.size()
            corr = corr.view(B, -1, h, w, h, w)
            correlations.append(corr)
        return correlations

    def forward(self, src_feats, trg_feats, f_q, v):
        
        correlations = self.cosine_similarity(src_feats, trg_feats)
        correlations = torch.stack(correlations, dim=1) # b, 9, 30, 30, 30, 30
        correlations = correlations.squeeze(2)

        correlations = self.relu(correlations)

        B, ch, side, _, _, _ = correlations.size()
        
        flattened_matches = self.flatten_and_project_matches(correlations)

        for attn, ff in self.trans_nc:
            flattened_matches = attn(flattened_matches) + flattened_matches
            flattened_matches = ff(flattened_matches) + flattened_matches
        
        refined_corr = self.to_correlation(flattened_matches)

        # correlations = Geometry.interpolate4d(refined_corr.squeeze(1), Geometry.upsample_size).unsqueeze(1)
        correlations = Geometry.interpolate4d(refined_corr.squeeze(1), self.target_size).unsqueeze(1)

        side = correlations.size(-1) ** 2
        corr2d = correlations.view(B, side, side).contiguous()

        if v.dim() == 4:
            v = v.flatten(2)
        attn = F.softmax( corr2d*self.temp, dim=-1 )
        weighted_v = torch.bmm(v, attn.permute(0, 2, 1))  # [B, 512, N_s] * [B, N_s, N_q] -> [1, 512, N_q]
        weighted_v = weighted_v.view(B, -1, self.target_size[0], self.target_size[1])
        fq = F.normalize(f_q, p=2, dim=1) + F.normalize(weighted_v, p=2, dim=1) * self.args.att_wt

        return fq, weighted_v