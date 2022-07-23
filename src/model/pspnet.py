# encoding:utf-8

from turtle import up
import torch
import numpy as np
from torch import nn
from operator import add
from functools import reduce
import torch.nn.functional as F
from .resnet import resnet50, resnet101
from .vgg import vgg16_bn
from .model_util import get_corr, get_ig_mask
from torch.nn.utils.weight_norm import WeightNorm
from .model_util import SegLoss

from operator import add
from functools import reduce, partial

def get_model(args) -> nn.Module:
    return PSPNet(args, zoom_factor=8, use_ppm=True)


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True))
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


class PSPNet(nn.Module):
    def __init__(self, args, zoom_factor, use_ppm):
        super(PSPNet, self).__init__()
        # assert args.layers in [50, 101, 152]
        assert 2048 % len(args.bins) == 0
        assert args.num_classes_tr > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.m_scale = args.m_scale
        self.bottleneck_dim = args.bottleneck_dim
        self.rmid = args.get('rmid', None)     # 是否返回中间层
        self.args = args                 # all_lr

        # d_kl scalar
        self.l2 = 0

        # 是否hyperpixel，即是否返回resnet每一层conv的结果。
        try:
            self.hyperpixel = args.hyperpixel
        except AttributeError:
            self.hyperpixel = None
        if self.hyperpixel:
            nbottlenecks = [3, 4, 6, 3]
            self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
            self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
            self.hyperpixel_ids = args.hyperpixel_ids

        resnet_kwargs = {}
        if self.rmid == 'nr':
            resnet_kwargs['no_relu'] = True
        if args.arch == 'resnet':
            if args.layers == 50:
                resnet = resnet50(pretrained=args.pretrained, **resnet_kwargs)  # nbottlenecks = [3, 4, 6, 3]   # channels [256, 512, 1024, 2048]
                # self.backbone = resnet
            else:
                resnet = resnet101(pretrained=args.pretrained, **resnet_kwargs) # nbottlenecks = [3, 4, 23, 3]  # channels [256, 512, 1024, 2048]
                # self.backbone = resnet

            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                        resnet.conv2, resnet.bn2, resnet.relu,
                                        resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)

            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        elif args.arch == 'vgg':
            vgg = vgg16_bn(pretrained=args.pretrained)
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_vgg16_layer(vgg)

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        if self.m_scale:
            fea_dim = 1024 + 512
        else:
            if args.arch == 'resnet':
                fea_dim = 2048
            elif args.arch == 'vgg':
                fea_dim = 512
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(args.bins)), args.bins)
            fea_dim *= 2
            self.bottleneck = nn.Sequential(
                nn.Conv2d(fea_dim, self.bottleneck_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=args.dropout)
            )

        if args.get('dist', 'dot') == 'dot':
            self.classifier = nn.Conv2d(self.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False)
            if args.cls_type[0] == 'r':
                WeightNorm.apply(self.classifier, 'weight', dim=0)  # [2, 512, 1, 1]
        elif args.get('dist') in ['cos', 'cosN']:
            self.classifier = CosCls(in_dim=self.bottleneck_dim, n_classes=args.num_classes_tr, cls_type=args.cls_type)

        self.gamma = nn.Parameter(torch.tensor(0.2))

    # reset classifier parameters -> reinitialize all the features
    def reset_parameters(self):
        self.classifier = nn.Conv2d(self.bottleneck_dim, self.args.num_classes_tr, kernel_size=1, bias=False)
        if self.args.cls_type[0] == 'r':
            WeightNorm.apply(self.classifier, 'weight', dim=0)  # [2, 512, 1, 1]
        if torch.cuda.is_available():
            self.classifier.cuda()

    def freeze_bn(self):
        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        H = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        W = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x, fea_lst = self.extract_features(x)
        x = self.classify(x, (H, W))
        if self.rmid:
            return x, fea_lst
        else:
            return x

    def get_feat_backbone(self, x):
        x = self.layer0(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        if self.rmid == 'nr':
            x_4, x4_nr = self.layer4(x_3)
        else:
            x_4 = self.layer4(x_3)
        return [x_1, x_2, x_3, x_4]

    def extract_features(self, x):
        x_4, feat_lst = self.get_feat_list(x)    # feat_lst 其实是 dict

        x = self.ppm(x_4)
        x = self.bottleneck(x)

        if self.rmid is not None and ('l' in self.rmid or 'mid' in self.rmid):
            return x, feat_lst
        else:
            return x, []
    
    def extract_hyper_features(self, img):
        r"""Extract desired a list of intermediate features"""

        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.conv2.forward(feat) # change according to original model
        feat = self.backbone.bn2.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.conv3.forward(feat)
        feat = self.backbone.bn3.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                #if hid + 1 == max(self.hyperpixel_ids):
                #    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # # Up-sample & concatenate features to construct a hyperimage
        # for idx, feat in enumerate(feats):
        #     feats[idx] = F.interpolate(feat, self.feature_size, None, 'bilinear', True)

        return feats

    def to_one_hot(self, mask: torch.tensor,
               num_classes: int) -> torch.tensor:
        """
        inputs:
            mask : shape [n_task, shot, h, w]
            num_classes : Number of classes
        returns :
            one_hot_mask : shape [n_task, shot, num_class, h, w]
        """
        n_tasks, shot, h, w = mask.size()
        one_hot_mask = torch.zeros(n_tasks, shot, num_classes, h, w).cuda()
        new_mask = mask.unsqueeze(2).clone()
        new_mask[torch.where(new_mask == 255)] = 0  # Ignore_pixels are anyways filtered out in the losses
        one_hot_mask.scatter_(2, new_mask, 1).long()
        return one_hot_mask

    def get_probas(self, logits: torch.tensor) -> torch.tensor:
        """
        inputs:
            logits : shape [n_tasks, shot, h, w]
        returns :
            probas : shape [n_tasks, shot, num_classes, h, w]
        """
        # logits_fg = logits - self.bias.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [n_tasks, shot, h, w]
        logits_fg = logits # [n_tasks, shot, h, w]
        probas_fg = torch.sigmoid(logits_fg).unsqueeze(2)
        probas_bg = 1 - probas_fg
        probas = torch.cat([probas_bg, probas_fg], dim=2)
        return probas

    def shannon_entropy(self, pred_q, q_label, iter, reduction='sum'):
        # pred_q: [n_shot, 2, 60, 60]
        # q_label: [B, 473, 473]
        gt_q = q_label.unsqueeze(1) # [B, 1, 473, 473]
        proba_q = F.softmax(pred_q, dim=1)
        proba_q = proba_q.unsqueeze(0) # [B, n_shot, 2, 60, 60]
        proba_q = proba_q.detach()
        
        ds_gt_q = F.interpolate(gt_q.float(), size=proba_q.size()[-2:], mode='nearest').long()
        valid_pixels_q = (ds_gt_q != 255).float()
        cond_entropy = - ((valid_pixels_q.unsqueeze(2) * (proba_q * torch.log(proba_q + 1e-10))).sum(2))
        cond_entropy = cond_entropy.sum(dim=(1, 2, 3))
        cond_entropy /= valid_pixels_q.sum(dim=(1, 2, 3))

        d_kl = 0
        # KL -------
        if self.args.use_kl:
            marginal = (valid_pixels_q.unsqueeze(2) * proba_q).sum(dim=(1, 3, 4))
            marginal /= valid_pixels_q.sum(dim=(1, 2, 3)).unsqueeze(1)

            # should not be using oracle here
            # one_hot_gt_q = self.to_one_hot(ds_gt_q, self.args.num_classes_tr)  # [n_tasks, shot, num_classes, h, w]
            # self.FB_param = (valid_pixels_q.unsqueeze(2) * one_hot_gt_q).sum(dim=(1, 3, 4)) / valid_pixels_q.unsqueeze(2).sum(dim=(1, 3, 4))
            
            if iter == self.args.kl_iter or iter == 0:
                self.l2 += 1
                self.FB_param = (valid_pixels_q.unsqueeze(2) * proba_q).sum(dim=(1, 3, 4))
                self.FB_param /= valid_pixels_q.unsqueeze(2).sum(dim=(1, 3, 4))

            d_kl = (marginal * torch.log(marginal / (self.FB_param + 1e-10))).sum(1)
        # KL -------

        if reduction == 'sum':
            cond_entropy = cond_entropy.sum(0)
            assert not torch.isnan(cond_entropy), cond_entropy
            # KL -------
            if self.args.use_kl:
                d_kl = d_kl.sum(0)
                assert not torch.isnan(d_kl), d_kl
        elif reduction == 'mean':
            cond_entropy = cond_entropy.mean(0)
            # KL -------
            if self.args.use_kl:
                d_kl = d_kl.mean(0)
        return cond_entropy + self.l2 * d_kl # Entropy of predictions [n_tasks,]

    def classify(self, features, shape):
        x = self.classifier(features)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def inner_loop(self, f_s, s_label, f_q, q_label):
        # input: f_s 为feature extractor输出的 feature map
        # self.classifier.reset_parameters()
        self.reset_parameters()
        self.classifier.train()

        # optimizer and loss function
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.args.cls_lr)

        criterion = SegLoss(loss_type=self.args.inner_loss_type)

        # inner loop 学习 classifier的params
        for index in range(self.args.adapt_iter):
            pred_s_label = self.classifier(f_s)  # [n_shot, 2(cls), 60, 60]
            pred_s_label = F.interpolate(pred_s_label, size=s_label.size()[1:],mode='bilinear', align_corners=True)
            s_loss = criterion(pred_s_label, s_label)  # pred_label: [n_shot, 2, 473, 473], label [n_shot, 473, 473]
            if self.args.shannon_loss:
                pred_q = self.classifier(f_q)
                shn_loss = self.shannon_entropy(pred_q, q_label, index)
                s_loss += shn_loss
            optimizer.zero_grad()
            s_loss.backward()
            optimizer.step()

    def outer_forward(self, f_q, f_s, fq_fea, fs_fea, s_label, q_label=None, pd_q0=None, pd_s=None, ret_corr=False):
        # f_q/f_s:[1,512,h,w],  fq_fea/fs_fea:[1,2048,h,w],  s_label: [1,H,w]
        bs, C, height, width = f_q.size()
        proj_v = f_s.view(bs, -1, height * width)

        # 基于attention, refine f_q, 并对query img做prediction
        sim = get_corr(q=fq_fea, k=fs_fea)       # [1, 3600_q, 3600_s]
        corr = torch.clone(sim).reshape(bs, height, width, height, width)                                  # return Corr

        # mask ignored support pixels
        ig_mask = get_ig_mask(sim, s_label, q_label, pd_q0, pd_s)    # [1, hw_s]

        # calculate weighted output
        ig_mask_full = ig_mask.unsqueeze(1).expand(sim.shape)        # [1, hw_q, hw_s]
        sim[ig_mask_full == True] = 0.00001

        if self.args.get('dist','dot')=='cos':
            proj_v = F.normalize(proj_v, dim=1)
            f_q = F.normalize(f_q, dim=1)

        attention = F.softmax(sim * self.args.temp, dim=-1)
        weighted_v = torch.bmm(proj_v, attention.permute(0, 2, 1))  # [1, 512, hw_k] * [1, hw_k, hw_q] -> [1, 512, hw_q]
        weighted_v = weighted_v.view(bs, C, height, width)

        out = (weighted_v * self.gamma + f_q)/(1+self.gamma)
        pred_q_label = self.classifier(out)

        if ret_corr == 'cr':
            return pred_q_label, [corr, weighted_v]
        elif ret_corr == 'cr_ig':
            return pred_q_label, [corr, weighted_v], ig_mask
        else:
            return pred_q_label

    def sampling(self, fq_fea, fs_fea, s_label, q_label=None, pd_q0=None, pd_s = None, ret_corr=False):
        bs, C, height, width = fq_fea.size()

        # 基于attention, refine f_q, 并对query img做prediction
        sim = get_corr(q=fq_fea, k=fs_fea)   # [1, 3600 (q_hw), 3600(k_hw)]
        corr = torch.clone(sim.reshape(bs, height, width, height, width))

        # mask ignored pixels
        ig_mask = get_ig_mask(sim, s_label, q_label, pd_q0, pd_s)  # # [B, 3600_s]

        if ret_corr:
            return ig_mask, corr
        return ig_mask    # [B, 3600]

    def get_feat_list(self, img,):
        feats = dict()

        # Layer 0 and layer1
        feat = self.layer0(img)
        feat = self.layer1(feat)

        # Layer 2,3,4
        for lid in [2, 3, 4]:
            n_bottleneck = len(self.__getattr__('layer'+str(lid)))
            for bid in range(n_bottleneck):
                feat = self.__getattr__('layer'+str(lid))[bid](feat)
                if str(lid) in self.args.get('all_lr', 'l') or bid == n_bottleneck-1:  # to decide whether to to return intermediate layers
                    feats[lid] = feats.get(lid, []) + [feat]

        return feat, feats


class CosCls(nn.Module):
    def __init__(self, in_dim=512, n_classes=2, cls_type = '0000'):
        super(CosCls, self).__init__()
        self.WeightNormR, self.weight_norm, self.bias, self.temp = parse_param_coscls(cls_type)
        self.cls = nn.Conv2d(in_dim, n_classes, kernel_size=1, bias=self.bias)
        if self.WeightNormR:
            WeightNorm.apply(self.cls, 'weight', dim=0) #split the weight update component to direction and norm
        if self.temp:
            self.scale_factor = nn.Parameter(torch.tensor(2.0))
        else:
            self.scale_factor = 2.0

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1, eps=0.00001)  # [B, ch, h, w]
        if self.weight_norm:
            self.cls.weight.data = F.normalize(self.cls.weight.data, p=2, dim=1, eps=0.00001)

        cos_dist = self.cls(x_norm)   #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * (cos_dist)
        return scores

    def reset_parameters(self):   # 与torch自己的method同名
        self.cls.reset_parameters()



def parse_param_coscls(cls_type):
    WeightNormR_dt = {'r': True, '0': False, 'o': False}
    weight_norm_dt = {'n': True, '0': False, 'o': False}
    bias_dt = {'b': True, '0': False, 'o': False}
    temp_dt = {'t': True, '0': False, 'o': False}
    print('weight norm Regular {} weight norm {}, bias {}, temp {}'.format(
        WeightNormR_dt[cls_type[0]], weight_norm_dt[cls_type[1]], bias_dt[cls_type[2]], temp_dt[cls_type[3]] ))
    return WeightNormR_dt[cls_type[0]], weight_norm_dt[cls_type[1]], bias_dt[cls_type[2]], temp_dt[cls_type[3]]


def get_classifier(args, num_classes=None):
    if num_classes is None:
        num_classes = args.num_classes_tr
    in_dim = args.bottleneck_dim

    if args.get('dist', 'dot') == 'dot':
        return nn.Conv2d(in_dim, num_classes, kernel_size=1, bias=False)
    elif args.get('dist') == 'cos' or args.get('dist') == 'cosN':
        return CosCls(in_dim=in_dim, n_classes=num_classes, cls_type=args.cls_type)