# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .match import MatchNet
from .msm import MSBlock, WeightAverage
from .model_util import SegLoss, get_corr


class MMN(nn.Module):
    def __init__(self, args, agg='cat', wa=False, red_dim=False,):
        super().__init__()
        self.args = args     # rmid
        self.agg = agg
        self.red_dim = red_dim  # Dim(int) or False
        self.wa = wa            # True or False
        match_ch = 1 if agg=='sum' else len(args.rmid)-1
        self.bid_lst = [int(num) - 1 for num in list(args.rmid[1:])]  # [0, 1, 2, 3]

        if self.args.layers == 50:
            self.nbottlenecks = [3, 4, 6, 3]
            self.feature_channels = [256, 512, 1024, 2048]
        if self.wa or (self.red_dim != False):
            for bid in self.bid_lst:
                c_in = self.feature_channels[bid]
                if isinstance(self.red_dim, int):
                    setattr(self, "rd_" + str(bid), nn.Sequential(nn.Conv2d(c_in, red_dim, kernel_size=1, stride=1, padding=1),
                                                                  nn.ReLU(inplace=True)))
                    c_in = red_dim
                setattr(self, "wa_"+str(bid), WeightAverage(c_in))

        self.corr_net = MatchNet(temp=args.temp, cv_type='red', sce=False, cyc=False, sym_mode=True, in_channel=match_ch)

    def forward(self, fq_lst, fs_lst, f_q, f_s):
        B, ch, h, w = f_q.shape

        corr_lst = []
        for idx in self.bid_lst[::-1]:
            fq_fea = fq_lst[idx]
            fs_fea = fs_lst[idx]
            if self.red_dim:
                fq_fea = getattr(self, 'rd_'+str(idx))(fq_fea)
                fs_fea = getattr(self, 'rd_'+str(idx))(fs_fea)
            if self.wa:
                fq_fea = getattr(self, "wa_"+str(idx))(fq_fea)
                fs_fea = getattr(self, 'wa_'+str(idx))(fs_fea)
            fq_fea = F.normalize(fq_fea, dim=1)
            fs_fea = F.normalize(fs_fea, dim=1)
            corr = get_corr(fq_fea, fs_fea)
            corr4d = corr.view(B, -1, h, w, h, w)
            corr_lst.append(corr4d)

        corr4d = torch.cat(corr_lst, dim=1)   # [B, L, h, w, h, w]
        if self.agg == 'sum':
            corr4d = torch.sum(corr4d, dim=1, keepdim=True)  # [B, 1, h, w, h, w]

        att_fq = self.corr_net.corr_forward(corr4d, v=f_s)
        fq = F.normalize(f_q, p=2, dim=1) + F.normalize(att_fq, p=2, dim=1) * self.args.att_wt

        return fq, att_fq

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
