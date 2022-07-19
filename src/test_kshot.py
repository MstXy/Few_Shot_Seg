# encoding:utf-8
import pdb

import os
import time
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from .model import MMN, SegLoss
from .model.pspnet import get_model
from .optimizer import get_optimizer, get_scheduler
from .dataset.dataset import get_val_loader, get_train_loader
from .util import intersectionAndUnionGPU, AverageMeter, CompareMeter
from .util import load_cfg_from_cfg_file, merge_cfg_from_list, ensure_path, set_log_path, log
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main(args: argparse.Namespace) -> None:

    sv_path = 'cross_{}/{}{}/split{}_shot{}/{}'.format(
        args.train_name, args.arch, args.layers, args.train_split, args.shot, args.exp_name)
    sv_path = os.path.join('./results', sv_path)
    ensure_path(sv_path)
    set_log_path(path=sv_path)
    log('save_path {}'.format(sv_path))

    log(args)

    if args.manual_seed is not None:
        cudnn.benchmark = False  # 为True的话可以对网络结构固定、网络的输入形状不变的 模型提速
        cudnn.deterministic = True
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    # ====== Model + Optimizer ======
    model = get_model(args).cuda()

    if args.resume_weights:
        fname = args.resume_weights + args.train_name + '/' + \
                'split={}/pspnet_{}{}/best.pth'.format(args.train_split, args.arch, args.layers)
        if os.path.isfile(fname):
            log("=> loading weight '{}'".format(fname))
            pre_weight = torch.load(fname)['state_dict']
            pre_dict = model.state_dict()

            for index, key in enumerate(pre_dict.keys()):
                if 'classifier' not in key and 'gamma' not in key:
                    if pre_dict[key].shape == pre_weight['module.' + key].shape:
                        pre_dict[key] = pre_weight['module.' + key]
                    else:
                        log('Pre-trained shape and model shape for {}: {}, {}'.format(
                            key, pre_weight['module.' + key].shape, pre_dict[key].shape))
                        continue

            model.load_state_dict(pre_dict, strict=True)
            log("=> loaded weight '{}'".format(fname))
        else:
            log("=> no weight found at '{}'".format(fname))

        # Fix the backbone layers
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.bottleneck.parameters():
            param.requires_grad = False

    # ========= Data  ==========
    episodic_val_loader, _ = get_val_loader(args)

    # ======= Transformer ======= args, inner_channel=32, sem=True, wa=False
    Trans = MMN(args, agg=args.agg, wa=args.wa, red_dim=args.red_dim).cuda()

    # load Trans weights
    if args.ckpt_used is not None:
        root_trans = 'results/msc_{}/{}{}/split{}_shot{}/{}'.format(
                        args.train_name, args.arch, args.layers, args.train_split, args.shot, args.exp_name
                        )
        filepath = os.path.join(root_trans, f'{args.ckpt_used}.pth')
        assert os.path.isfile(filepath), filepath
        print("=> loading transformer weight '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        Trans.load_state_dict(checkpoint['state_dict'])
        print("=> loaded transformer weight '{}'".format(filepath))
    else:
        print("=> Not loading anything")


    # ====== Metrics initialization ======
    max_val_mIoU, max_val_mIoU1 = 0., 0.

    for i in range(1, len(episodic_val_loader)+1):
        val_Iou, val_Iou1, val_loss = validate_epoch(args=args, val_loader=episodic_val_loader, model=model, Net=Trans)

        # Model selection
        if val_Iou.item() > max_val_mIoU:
            max_val_mIoU = val_Iou.item()
            log('----------- Max_mIoU = {:.3f}-----------'.format(max_val_mIoU))

        if val_Iou1.item() > max_val_mIoU1:
            max_val_mIoU1 = val_Iou1.item()
            log('----------- Max_mIoU1 = {:.3f}-----------'.format(max_val_mIoU1))


def validate_epoch(args, val_loader, model, Net):
    log('==> Start testing')

    iter_num = 0
    start_time = time.time()
    loss_meter = AverageMeter()

    cls_intersection = defaultdict(int)  # Default value is 0
    cls_union = defaultdict(int)
    IoU = defaultdict(float)

    cls_intersection0 = defaultdict(int)  # Default value is 0
    cls_union0 = defaultdict(int)
    IoU0 = defaultdict(float)

    cls_intersection1 = defaultdict(int)  # Default value is 0
    cls_union1 = defaultdict(int)
    IoU1 = defaultdict(float)

    val_iou_compare = CompareMeter()

    for e in range(args.test_num):

        iter_num += 1
        try:
            qry_img, q_label, spt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
        except:
            iter_loader = iter(val_loader)
            qry_img, q_label, spt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
        if torch.cuda.is_available():
            spt_imgs = spt_imgs.cuda()
            s_label = s_label.cuda()
            q_label = q_label.cuda()
            qry_img = qry_img.cuda()

        # ====== Phase 1: Train a new binary classifier on support samples. ======
        spt_imgs = spt_imgs.squeeze(0)   # [n_shots, 3, img_size, img_size]
        s_label = s_label.squeeze(0).long()  # [n_shots, img_size, img_size]

        # fine-tune classifier
        model.eval()
        with torch.no_grad():
            f_s, fs_lst = model.extract_features(spt_imgs)
        model.inner_loop(f_s, s_label)

        # ====== Phase 2: Update query score using attention. ======
        with torch.no_grad():
            f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
            pred_q0 = model.classifier(f_q)
            pred_q0 = F.interpolate(pred_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)

        Net.eval()
        with torch.no_grad():
            att_fq = []
            for k in range(args.shot):
                single_fs_lst = {key: [ve[k:k + 1] for ve in value] for key, value in fs_lst.items()}
                single_f_s = f_s[k:k + 1]
                _, att_out = Net(fq_lst, single_fs_lst, f_q, single_f_s, )
                att_fq.append(att_out)  # [ 1, 512, h, w]
            att_fq = torch.cat(att_fq, dim=0)
            att_fq = att_fq.mean(dim=0, keepdim=True)
            fq = f_q * (1 - args.att_wt) + att_fq * args.att_wt

            pd_q1 = model.classifier(att_fq)
            pred_q1 = F.interpolate(pd_q1, size=q_label.shape[-2:], mode='bilinear', align_corners=True)
            pd_q = model.classifier(fq)
            pred_q = F.interpolate(pd_q, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

        # IoU and loss
        curr_cls = subcls[0].item()  # 当前episode所关注的cls
        for id, (cls_intersection_, cls_union_, IoU_, pred) in \
                enumerate( [(cls_intersection0, cls_union0, IoU0, pred_q0), (cls_intersection1, cls_union1, IoU1, pred_q1),
                 (cls_intersection, cls_union, IoU, pred_q)] ):
            intersection, union, target = intersectionAndUnionGPU(pred.argmax(1), q_label, 2, 255)
            intersection, union = intersection.cpu(), union.cpu()
            cls_intersection_[curr_cls] += intersection[1]  # only consider the FG
            cls_union_[curr_cls] += union[1]                # only consider the FG
            IoU_[curr_cls] = cls_intersection_[curr_cls] / (cls_union_[curr_cls] + 1e-10)   # cls wise IoU
            if id==0: iouf0 = intersection[1]/union[1]     # fg IoU for the current episode
            elif id==1: iouf1 = intersection[1]/union[1]
        val_iou_compare.update(iouf1,iouf0)   # compare 当前episode的IoU of att pred and pred0

        criterion_standard = nn.CrossEntropyLoss(ignore_index=255)
        loss1 = criterion_standard(pred_q1, q_label)
        loss_meter.update(loss1.item())

        if (iter_num % 200 == 0):
            mIoU = np.mean([IoU[i] for i in IoU])                                  # mIoU across cls
            mIoU0 = np.mean([IoU0[i] for i in IoU0])
            mIoU1 = np.mean([IoU1[i] for i in IoU1])
            log('Test: [{}/{}] mIoU0 {:.4f} mIoU1 {:.4f} mIoU {:.4f} Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(
                iter_num, args.test_num, mIoU0, mIoU1, mIoU, loss_meter=loss_meter))

    runtime = time.time() - start_time
    mIoU = np.mean(list(IoU.values()))  # IoU: dict{cls: cls-wise IoU}
    log('mIoU---Val result: mIoU0 {:.4f}, mIoU1 {:.4f} mIoU {:.4f} | time used {:.1f}m.'.format(
        mIoU0, mIoU1, mIoU, runtime/60))
    for class_ in cls_union:
        log("Class {} : {:.4f}".format(class_, IoU[class_]))
    log('------Val FG IoU1 compared to IoU0 win {}/{} avg diff {:.2f}'.format(
        val_iou_compare.win_cnt, val_iou_compare.cnt, val_iou_compare.diff_avg))

    return mIoU, mIoU1, loss_meter.avg


if __name__ == "__main__":
    args = parse_args()

    world_size = len(args.gpus)
    args.distributed = (world_size > 1)
    main(args)