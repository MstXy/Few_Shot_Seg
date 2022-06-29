# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .match import MatchNet
from .msm import MSBlock, WeightAverage


class MMN(nn.Module):
    def __init__(self, args, inner_channel=32, sem=True, wa=False):
        super().__init__()
        self.args = args     # rmid
        self.sem = sem
        self.wa = wa
        self.temp = args.temp

        if self.args.layers == 50:
            self.nbottlenecks = [3, 4, 6, 3]
            self.feature_channels = [256, 512, 1024, 2048]

        self.bids = [int(num) for num in list(args.rmid[1:])]

        for i, b in enumerate(self.bids):
            if self.sem:
                setattr(self, "msblock"+str(b), MSBlock(self.feature_channels[b-1], rate=4, c_out=32))
            if self.wa:
                setattr(self, "wa_"+str(b), WeightAverage(inner_channel))

        self.corr_net = MatchNet(temp=args.temp, cv_type='red', sce=False, cyc=False, sym_mode=True)

        # self.conv3_scale = nn.Conv2d(2 * inner_channel, inner_channel, (3, 3), stride=1, padding=1)
        self.conv3_scale = nn.Conv2d(2048+1024, 1024, (3, 3), stride=1, padding=1)

    def forward(self, fq_lst, fs_lst, f_q, f_s, padding_mask=None, s_padding_mask=None):

        fq1, fq2, fq3, fq4 = fq_lst
        fs1, fs2, fs3, fs4 = fs_lst

        B, ch, h, w = fq4.shape

        if self.sem:
            fq4 = self.msblock4(fq4)   # [B, 32, 60, 60]
            fs4 = self.msblock4(fs4)

            fq3 = self.msblock3(fq3)   # [B, 32, 60, 60]
            fs3 = self.msblock3(fs3)

        if self.wa:
            fq4 = self.wa_4(fq4)
            fs4 = self.wa_4(fs4)

        att_fq4, corr4 = self.corr_net(fq4, fs4, f_s, s_mask=None, ig_mask=None, ret_corr=False, use_cyc=False, ret_cyc=False, ret_2dcorr=True)
        print(corr4.shape)
        fq3 = torch.cat((fq3, fq4), 1)
        fs3 = torch.cat((fs3, fs4), 1)

        fq3 = self.conv3_scale(fq3)
        fs3 = self.conv3_scale(fs3)

        att_fq34, corr34 = self.corr_net(fq3, fs3, f_s, s_mask=None, ig_mask=None, ret_corr=False, use_cyc=False, ret_cyc=False, ret_2dcorr=True)
        print(corr34.shape)
        refined_corr = corr4 + corr34

        attn = F.softmax( refined_corr*self.temp, dim=-1 )
        if f_s.dim() == 4:
            f_s = f_s.flatten(2)
        weighted_v = torch.bmm(f_s, attn.permute(0, 2, 1))  # [B, 512, N_s] * [B, N_s, N_q] -> [1, 512, N_q]
        weighted_v = weighted_v.view(B, -1, h, w)

        fq = F.normalize(f_q, p=2, dim=1) + \
             F.normalize(weighted_v, p=2, dim=1) * self.args.att_wt

        return fq, weighted_v

