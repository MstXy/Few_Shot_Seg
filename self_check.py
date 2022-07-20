from src.model.pspnet import *
from src.model import MMN, SegLoss
from src.model.nc import NC
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from src.util import load_cfg_from_cfg_file, merge_cfg_from_list, ensure_path, set_log_path, log
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training classifier weight transformer')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


args = parse_args()

model = get_model(args).cuda()
# model = get_model(args)

# Trans = MMN(args, inner_channel=32, sem=args.sem, wa=args.wa).cuda()
Trans = MMN(args, agg=args.agg, wa=args.wa, red_dim=args.red_dim).cuda()
# Trans = NC(args, hyperpixel_ids = args.hyperpixel_ids).cuda()


# pytorch_total_params = sum(p.numel() for p in Trans.parameters() if p.requires_grad)
# print(pytorch_total_params)

spt_imgs = torch.randn(1, 5, 3, 473, 473).cuda()  # [1, n_shot, 3, h, w] 473, 473
s_label = torch.randint(1,5, (1, 1, 473, 473)).cuda()  # [1, n_shot, h, w]
q_label = torch.randint(1,5, (1, 473, 473)).cuda()  # [1, h, w]
qry_img = torch.randn(1, 3, 473, 473).cuda()  # [1, 3, h, w]
# spt_imgs = torch.randn(1, 1, 3, 473, 473)  # [1, n_shot, 3, h, w] 473, 473
# s_label = torch.randn(1, 1, 473, 473)  # [1, n_shot, h, w]
# q_label = torch.randn(1, 473, 473)  # [1, h, w]
# qry_img = torch.randn(1, 3, 473, 473)  # [1, 3, h, w]

# ====== Phase 1: Train the binary classifier on support samples ======

spt_imgs = spt_imgs.squeeze(0)       # [n_shots, 3, img_size, img_size]
s_label = s_label.squeeze(0).long()  # [n_shots, img_size, img_size]

# fine-tune classifier
model.eval()
with torch.no_grad():
    f_s, fs_lst = model.extract_features(spt_imgs)  # f_s为ppm之后的feat, fs_lst为mid_feat
    # fs: [1, 512, 60, 60], fs_lst: [1, 512, 60, 60],[1, 1024, 60, 60],[1, 2048, 60, 60]
# model.inner_loop(f_s, s_label)

# ====== Phase 2: Train the attention to update query score  ======
model.eval()
with torch.no_grad():
    f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
    # pd_q0 = model.classifier(f_q)
    # pred_q0 = F.interpolate(pd_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)

Trans.train()
criterion = SegLoss(loss_type=args.loss_type)
# q_loss1 = 0
# att_fq = []
# for k in range(args.shot):
#     single_fs_lst = {key: [ve[k:k + 1] for ve in value] for key, value in fs_lst.items()}
#     single_f_s = f_s[k:k + 1]
#     _, att_fq_single = Trans(fq_lst, single_fs_lst, f_q, single_f_s,)
#     att_fq.append(att_fq_single)       # [ 1, 512, h, w]

# att_fq = torch.cat(att_fq, dim=0)  # [k, 512, h, w]
# att_fq = att_fq.mean(dim=0, keepdim=True)
# fq = f_q * (1-args.att_wt) + att_fq * args.att_wt

pd_q1 = model.classifier(f_q)
pred_q = F.interpolate(pd_q1, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

print(criterion(pred_q, q_label.long()))
