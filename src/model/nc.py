# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .match import MatchNet
from .msm import MSBlock, WeightAverage
from .model_util import SegLoss, get_corr


class NC(nn.Module):
    def __init__(self, args, hyperpixel_ids=list(range(8,17))):
        super().__init__()
        self.args = args     # rmid

        match_ch = len(hyperpixel_ids)
        self.corr_net = MatchNet(temp=args.temp, cv_type='red', sce=False, cyc=False, sym_mode=True, in_channel=match_ch)
        # ======= Attention Branch for 5 shot ===========
        if args.shot > 1:
            self.AttentionBranch = nn.Sequential(
                nn.Conv2d(1024, 256, (3,3), padding=2, dilation=2, bias = True),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
                nn.Conv2d(256, 256, (3,3)),
                nn.MaxPool2d((3,3)),
                nn.Conv2d(256, 1, (3,3)),
                nn.AdaptiveAvgPool2d((1,1))
            ) # output shape: [B, 1, 1, 1] (B=1)

    def forward(self, fq_lst, fs_lst, f_q, f_s): 
        B, ch, h, w = f_q.shape

        corr_lst = []
        for i, (fq_fea, fs_fea) in enumerate(zip(fq_lst, fs_lst)):
            fq_fea = F.normalize(fq_fea, dim=1)
            fs_fea = F.normalize(fs_fea, dim=1)
            corr = get_corr(fq_fea, fs_fea)
            corr4d = corr.view(B, -1, h, w, h, w)
            corr_lst.append(corr4d)

        corr4d = torch.cat(corr_lst, dim=1)   # [B, L, h, w, h, w]

        att_fq = self.corr_net.corr_forward(corr4d, v=f_s)
        fq = F.normalize(f_q, p=2, dim=1) + F.normalize(att_fq, p=2, dim=1) * self.args.att_wt

        comp_feature = torch.cat((f_q, f_s), dim=1)
        weight = self.AttentionBranch(comp_feature).squeeze().unsqueeze(0)

        return fq, att_fq, weight

    def forward_mmn(self, fq_lst, fs_lst, f_q, f_s,):
        fq1, fq2, fq3, fq4 = fq_lst
        fs1, fs2, fs3, fs4 = fs_lst

        if self.sem:
            fq4 = self.msblock4(fq4)   # [B, 32, 60, 60]
            fs4 = self.msblock4(fs4)   # [B, 32, 60, 60]
        if self.wa:
            fq4 = self.wa_4(fq4)
            fs4 = self.wa_4(fs4)

        att_fq4 = self.corr_net(fq4, fs4, f_s, s_mask=None, ig_mask=None, ret_corr=False, use_cyc=False, ret_cyc=False)

        fq = F.normalize(f_q, p=2, dim=1) + F.normalize(att_fq4, p=2, dim=1) * self.args.att_wt

        return fq, att_fq4
