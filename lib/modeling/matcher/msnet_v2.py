from torch import nn
import torch
import torch.nn.functional as F

from lib.modeling.backbone import build_backbone
from .evaluators import make_evaluator
from lib.utils.logger import logger
from lib.utils.visualize import draw_paired_img_desc_torch


class DescExtractor(nn.Module):
    def __init__(self, cfg):
        super(DescExtractor, self).__init__()

        self.backbone = build_backbone(cfg)
        self.C3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.regress = nn.Sequential(
            # nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, images, scales):
        """
        images: [B, 3, H, W]
        """
        feats = self.backbone(images)
        feats3 = feats['C3']
        h, w = feats3.shape[2:]

        descs3 = None
        descs4 = None
        descs5 = None

        # import numpy as np
        # descs = []
        # for i in np.arange(1, 3, 0.2):
        #     down_feats3 = F.interpolate(feats3, scale_factor=i, mode='bilinear')
        #     down_descs = F.interpolate(self.regress(self.C3(down_feats3)), (h, w), mode='bilinear')
        #     descs.append(down_descs)
        # descs = torch.stack(descs, dim=1)

        descs = self.regress(self.C3(feats3))
        down_feats3 = F.interpolate(feats3, scale_factor=0.5, mode='bilinear')
        down_descs = F.interpolate(self.regress(self.C3(down_feats3)), (h, w), mode='bilinear')

        # if the pixel scale is bigger than `end`, we pick the downsample descriptor
        scales = F.interpolate(scales, (h, w), mode='bilinear')
        end = 1.5
        masks = (scales > end).float()
        descs = descs * (1 - masks) + down_descs * masks

        descs = dict(
            d3=descs,
            d4=descs4,
            d5=descs5
        )

        return descs


class MultiDescEvaluator(object):
    def __init__(self, cfg):
        self.desc_evaluator = make_evaluator(cfg)

    def __call__(self, desc_maps0, kps0, images0, desc_maps1, kps1, kps2, images1):
        desc_names = desc_maps0.keys()

        loss = []
        distance = []
        similarity = []

        desc_names = ['d3']

        for desc_name in desc_names:
            l, d, s = self.desc_evaluator(
                desc_maps0['d3'], kps0, images0,
                desc_maps1['d3'], kps1, kps2, images1,
                thresh=4, interval=4
            )
            loss.append(l)
            distance.append(d)
            similarity.append(s)

        loss = torch.mean(torch.stack(loss))
        distance = torch.mean(torch.stack(distance))
        similarity = torch.mean(torch.stack(similarity))

        return loss, distance, similarity


class MultiScaleNetV2(nn.Module):
    def __init__(self, cfg):
        super(MultiScaleNetV2, self).__init__()

        self.desc_extractor = DescExtractor(cfg)
        self.desc_evaluator = MultiDescEvaluator(cfg)

    def forward(self, images0, images1, targets):
        """
        images0: [B, 3, H, W]
        images1: [B, 3, H, W]
        targets: {'kps0': [B, N, 2], 'kps1': [B, N, 2]}
        """
        descs0 = self.desc_extractor(images0, targets['left_scale'])
        descs1 = self.desc_extractor(images1, targets['scale'])

        results = dict(
            descs0=descs0,
            descs1=descs1
        )

        loss, distance, similarity = self.desc_evaluator(
            descs0, targets['kps0'], images0,
            descs1, targets['kps1'], targets['kps2'], images1
        )
        losses = dict(loss=loss, distance=distance, similarity=similarity)

        # keep descriptors for visualization
        logger.update(image0=images0[0], image1=images1[0])
        logger.update(kps0=targets['kps0'][0], kps1=targets['kps1'][0])
        logger.update(d03=descs0['d3'][0], d13=descs1['d3'][0])
        logger.update(H=targets['H'][0])

        return losses, results

    def inference(self, images, scale=1, id='left'):
        return self.desc_extractor(images, scale)['d3']

    @staticmethod
    def get_loss_for_bp(loss_dict):
        """
        From loss dict, get loss for backward pass
        :param loss_dict: that returned by self.forward
        :return: tensor, loss
        """
        return loss_dict['loss']

    @staticmethod
    def get_loss_for_log(loss_dict):
        """
        From loss dict, extract info for logging
        :return: a new dictionary
        """
        return loss_dict

