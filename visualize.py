from logging import raiseExceptions

import os
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt
import cv2

from collections import defaultdict
from src.model import *
from src.model.pspnet import get_model
from src.model.cats import CATs
from src.optimizer import get_optimizer, get_scheduler
from src.dataset.dataset import get_val_loader, get_train_loader
from src.util import intersectionAndUnionGPU, AverageMeter
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

def main(args: argparse.Namespace) -> None:

    # sv_path = 'fuse_{}/{}{}/split{}_shot{}/{}'.format(
    #     args.train_name, args.arch, args.layers, args.train_split, args.shot, args.exp_name)
    # sv_path = os.path.join('./results', sv_path)
    # ensure_path(sv_path)
    # set_log_path(path=sv_path)
    # log('save_path {}'.format(sv_path))

    print(args)

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
                    'split={}/model/pspnet_{}{}/best.pth'.format(args.train_split, args.arch, args.layers)
            if os.path.isfile(fname):
                print("=> loading weight '{}'".format(fname))
                pre_weight = torch.load(fname)['state_dict']
                pre_dict = model.state_dict()

                for index, key in enumerate(pre_dict.keys()):
                    if 'classifier' not in key and 'gamma' not in key and 'backbone' not in key:
                        if pre_dict[key].shape == pre_weight['module.' + key].shape:
                            pre_dict[key] = pre_weight['module.' + key]
                        else:
                            print('Pre-trained shape and model shape for {}: {}, {}'.format(
                                key, pre_weight['module.' + key].shape, pre_dict[key].shape))
                            continue

                model.load_state_dict(pre_dict, strict=True)
                print("=> loaded weight '{}'".format(fname))
            else:
                print("=> no weight found at '{}'".format(fname))
        
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
    train_loader, train_sampler = get_train_loader(args)
    episodic_val_loader, _ = get_val_loader(args)

    # ======= Transformer =======
    FusionNet = CATs(
                    temp=args.temp,
                    feature_size=args.cats_feature_size, 
                    feature_proj_dim=args.cats_feature_proj_dim, 
                    num_heads=args.num_heads,
                    hyperpixel=args.hyperpixel,
                    hyperpixel_ids=args.hyperpixel_ids
                    ).cuda()

    fname = './results/fuse_pascal/resnet50/split0_shot1/cats_m_32_l2_noig/best1.pth'
    pre_weight = torch.load(fname, map_location=lambda storage, location: storage)['state_dict']
    FusionNet.load_state_dict(pre_weight, strict=True)


    # iterable_train_loader = iter(episodic_val_loader)
    iterable_train_loader = iter(train_loader)

    for e in range(1, 11):
        qry_img, q_label, spt_imgs, s_label, subcls, sl, ql = iterable_train_loader.next()
        spt_imgs = spt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]
        s_label = s_label.squeeze(0).long() # [n_shots, img_size, img_size]

        # ====== 可视化图片 ======
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                                    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                                    ])
        inv_s = invTrans(spt_imgs[0])
        for i in range(1, 473+1, 8):
            for j in range(1, 473+1, 8):
                inv_s[:, i-1, j-1] = torch.tensor([0, 1.0, 0])
        # plt.imshow(inv_s.permute(1, 2, 0))

        inv_q = invTrans(qry_img[0])
        for i in range(1, 473+1, 8):
            for j in range(1, 473+1, 8):
                inv_q[:, i-1, j-1] = torch.tensor([0, 1.0, 0])
        inv_q[:, (34-1)*8, (37-1)*8] = torch.tensor([1.0, 0, 0])

        if torch.cuda.is_available():
            spt_imgs = spt_imgs.cuda()
            s_label = s_label.cuda()
            q_label = q_label.cuda()
            qry_img = qry_img.cuda()

        model.eval()
        with torch.no_grad():
            f_s, fs_lst = model.extract_features(spt_imgs)  # f_s为ppm之后的feat, fs_lst为mid_feat
        model.eval()
        with torch.no_grad():
            f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
            pd_q0 = model.classifier(f_q)
            # pd_s  = model.classifier(f_s)
            pred_q0 = F.interpolate(pd_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)

        if not args.ignore:
            ig_mask = None
        if args.hyperpixel:
            model.eval()
            with torch.no_grad():
                fs_lst = model.extract_hyper_features(spt_imgs.squeeze(1)) ##!! only suits for 1 shot only currently
                fq_lst = model.extract_hyper_features(qry_img)
        weighted_v, refined_corr = FusionNet(fs_lst, s_label, fq_lst, v=f_s.view(f_s.shape[:2] +(-1,)), ig_mask=ig_mask)
        pd_q1 = model.classifier(weighted_v)
        pred_q1 = F.interpolate(pd_q1, size=q_label.shape[-2:], mode='bilinear', align_corners=True)
        out = (weighted_v * 0.2 + f_q) / (1 + 0.2)
        pd_q = model.classifier(out)
        pred_q = F.interpolate(pd_q, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

        IoUb, IoUf = dict(), dict()
        for (pred, idx) in [(pred_q0, 0), (pred_q1, 1), (pred_q, 2)]:
            intersection, union, target = intersectionAndUnionGPU(pred.argmax(1), q_label, args.num_classes_tr, 255)
            IoUb[idx], IoUf[idx] = (intersection / (union + 1e-10)).cpu().numpy()  # mean of BG and FG
        
        m_IoUf0 = IoUf[0]
        m_IoUb0 = IoUb[0]
        m_IoUf1 = IoUf[1]
        m_IoUb1 = IoUb[1]
        m_IoUf = IoUf[2]
        m_IoUb = IoUb[2]

        print('Img{} IoUf0 {:.2f} IoUb0 {:.2f} IoUf1 {:.2f} IoUb1 {:.2f} IoUf {:.2f} IoUb {:.2f}'.format(
                    e, m_IoUf0,m_IoUb0,m_IoUf1, m_IoUb1,m_IoUf,m_IoUb))

        # visualize:
        # 1) original image, support & query
        # 2) pd_0 mask
        # 3) pd_1 mask
        # 4) pd mask
        support_image = inv_s.permute(1, 2, 0)
        query_image = inv_q.permute(1, 2, 0)

        mask0 = torch.argmax(pred_q0, dim=1)
        mask0 = mask0.cpu().squeeze(0)

        mask1 = torch.argmax(pred_q1, dim=1)
        mask1 = mask1.cpu().squeeze(0)

        mask = torch.argmax(pred_q, dim=1)
        mask = mask.cpu().squeeze(0)
        
        # fig, axs = plt.subplots(2, 3)
        # axs[0, 0].imshow(support_image)
        # axs[0, 0].set_title('Support image')
        # axs[0, 1].imshow(query_image)
        # axs[0, 1].set_title('Query image')

        # axs[1, 0].imshow(mask0)
        # axs[1, 0].set_title('IoUf0 {:.2f} IoUb0 {:.2f}'.format(m_IoUf0, m_IoUb0))
        # axs[1, 1].imshow(mask1)
        # axs[1, 1].set_title('IoUf1 {:.2f} IoUb1 {:.2f}'.format(m_IoUf1, m_IoUb1))
        # axs[1, 2].imshow(mask)
        # axs[1, 2].set_title('IoUf2 {:.2f} IoUb2 {:.2f}'.format(m_IoUf, m_IoUb))

        # fig.savefig('vis_result/'+str(e)+'.png')


        # check attention ============================================
        sim = refined_corr
        q_mask = F.interpolate(q_label.unsqueeze(1).float(), size=f_q.shape[-2:], mode='nearest').squeeze(1)  # [1,1,h,w]
        q_mask = (q_mask != 255.0)

        pd_q_mask1 = pd_q1.argmax(dim=1)
        q_mask = q_mask * (pd_q_mask1==1)             # query 根据 pred_mask 选取前景或背景
        q_mask = q_mask.view(q_mask.shape[0], -1, 1)  # [n_shot, 1, hw]
        q_mask = q_mask.expand(sim.shape)             # [1, q_hw, hw]

        sim0 = sim[q_mask].reshape(1, -1, 3600)    # query image 前景或背景 overall sim score 分布
        sim0 = torch.mean(sim0, dim=1)

        sim0 = sim[:,(34-1)*60+37-1]               # 只看query image 单点的 sim score的分布

        a = sim0.detach().cpu().numpy().reshape(60,60)
        print('max {:.4f}, mean {:.4f}, min {:.4f}'.format(np.max(a), np.mean(a), np.min(a)))
        a = np.uint8((a - np.min(a)) / (np.max(a) - np.min(a)) * 255)
        heatmap = cv2.applyColorMap(cv2.resize(a, (473, 473)), cv2.COLORMAP_JET)
        img = inv_s.permute(1, 2, 0).cpu().numpy()*255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = heatmap * 0.3 + img * 0.7
        cv2.imwrite('vis_result/CAM_sp_'+str(e)+'.jpg', result)



if __name__ == "__main__":
    args = parse_args()

    main(args)