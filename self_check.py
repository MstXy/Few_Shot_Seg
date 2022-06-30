from src.model.pspnet import *
from src.model import TransforMatcher
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
Trans = TransforMatcher(args).cuda()

spt_imgs = torch.randn(1, 1, 3, 473, 473).cuda()  # [1, n_shot, 3, h, w] 473, 473
s_label = torch.randn(1, 1, 473, 473).cuda()  # [1, n_shot, h, w]
q_label = torch.randn(1, 473, 473).cuda()  # [1, h, w]
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
    # pd_s  = model.classifier(f_s)
    # pred_q0 = F.interpolate(pd_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)

if args.hyperpixel:
    model.eval()
    with torch.no_grad():
        fs_lst = model.extract_hyper_features(spt_imgs.squeeze(1)) ##!! only suits for 1 shot only currently
        fq_lst = model.extract_hyper_features(qry_img)

# Trans.train()
corr = Trans(fq_lst, fs_lst, f_q, f_s)
print(corr.shape)
# pd_q1 = model.classifier(att_fq)
# pred_q1 = F.interpolate(pd_q1, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

# pd_q = model.classifier(fq)
# pred_q = F.interpolate(pd_q, size=q_label.shape[-2:], mode='bilinear', align_corners=True)
