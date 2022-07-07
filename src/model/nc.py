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
        self.red_dim = args.red_dim
        self.bid_lst = [int(num) for num in list(args.rmid[1:])]  # [1, 2, 3, 4]

        if self.args.layers == 50:
            self.nbottlenecks = [3, 4, 6, 3]
            self.feature_channels = [256, 512, 1024, 2048]
        if self.red_dim != False:
            for bid in self.bid_lst:
                c_in = self.feature_channels[bid-1]
                if isinstance(self.red_dim, int):
                    setattr(self, "rd_" + str(bid), nn.Sequential(nn.Conv2d(c_in, self.red_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                                                  nn.ReLU(inplace=True)))
                    c_in = self.red_dim

        match_ch = len(hyperpixel_ids)
        self.corr_net = MatchNet(temp=args.temp, cv_type='red', sce=False, cyc=False, sym_mode=True, in_channel=match_ch)

    def forward(self, fq_lst, fs_lst, f_q, f_s): 
        B, ch, h, w = f_q.shape

        corr_lst = []
        for i, (fq_fea, fs_fea) in enumerate(zip(fq_lst, fs_lst)):
            if self.red_dim: # this is for the no hyperpixel case
                    fq_fea = getattr(self, 'rd_'+str(i+3))(fq_fea)
                    fs_fea = getattr(self, 'rd_'+str(i+3))(fs_fea)
            fq_fea = F.normalize(fq_fea, dim=1)
            fs_fea = F.normalize(fs_fea, dim=1)
            corr = get_corr(fq_fea, fs_fea)
            corr4d = corr.view(B, -1, h, w, h, w)
            corr_lst.append(corr4d)

        corr4d = torch.cat(corr_lst, dim=1)   # [B, L, h, w, h, w]

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
