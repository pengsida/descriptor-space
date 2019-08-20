"""
Multi-scale resize, with simple addition
"""
from torch import nn
import torch
import torch.nn.functional as F

from lib.modeling.backbone import build_backbone
from .evaluators import make_evaluator
from lib.utils.logger import logger
from lib.utils.visualize import draw_paired_img_desc_torch


class DescExtractor(nn.Module):
    def __init__(self, cfg, scales):
        """
        DescExtractor
        :param scales: a list of scales, like [0.5, 1, 2]
        """
        super(DescExtractor, self).__init__()
        
        self.scales = scales
        self.backbone = build_backbone(cfg)
        self.regress = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, images, scale=1):
        """
        images: [B, 3, H, W]
        """
        feats = self.backbone(images)
        # feats: (B, D, H', W')
        feats = feats['C3']
        _, _, H, W = feats.size()
        
        # resize feats for to each scale
        descs = []
        for scale in self.scales:
            # scale features
            H_resized, W_resized = int(H * scale), int(W * scale)
            feats_resized = F.interpolate(feats, (H_resized, W_resized), mode='bilinear')
            # extract descriptors, and scale back
            descs_resized = F.interpolate(self.regress(feats_resized), (H, W), mode='bilinear')
            # normalize the descriptor
            descs_resized = F.normalize(descs_resized, p=2, dim=1)
            descs.append(descs_resized)
            
        # (N, B, D, H, W)
        descs = torch.stack(descs)
        # take the sum of all descs
        descs = torch.sum(descs, dim=0)
        
        return F.normalize(descs, p=2, dim=1)
        
        
    def inference(self, images):
        """
        images: [B, 3, H, W]
        """
        return self.forward(images)


class MultiDescEvaluator(object):
    def __init__(self, cfg):
        self.desc_evaluator = make_evaluator(cfg)

    def __call__(self, desc_maps0, kps0, images0, desc_maps1, kps1, kps2, images1):
        # desc_names = desc_maps0.keys()

        loss = []
        distance = []
        similarity = []

        desc_names = ['d3']

        for desc_name in desc_names:
            l, d, s = self.desc_evaluator(
                desc_maps0, kps0, images0,
                desc_maps1, kps1, kps2, images1,
                thresh=4, interval=4
            )
            loss.append(l)
            distance.append(d)
            similarity.append(s)

        loss = torch.mean(torch.stack(loss))
        distance = torch.mean(torch.stack(distance))
        similarity = torch.mean(torch.stack(similarity))

        return loss, distance, similarity


class MultiScaleNetV4(nn.Module):
    """
    This version simply sums descriptors from all scales
    """
    def __init__(self, cfg, scales=[0.5, 1, 2]):
        # super(MultiScaleNetV2, self).__init__()
        nn.Module.__init__(self)

        self.scales = scales
        self.desc_extractor = DescExtractor(cfg, scales)
        self.desc_evaluator = MultiDescEvaluator(cfg)

    def forward(self, images0, images1, targets):
        """
        images0: [B, 3, H, W]
        images1: [B, 3, H, W]
        targets: {'kps0': [B, N, 2], 'kps1': [B, N, 2]}
        """
        descs0 = self.desc_extractor(images0, 1)
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
        logger.update(d03=descs0[0], d13=descs1[0])
        logger.update(H=targets['H'][0])

        return losses, results

    def inference(self, images, scale=1):
        # descs = self.desc_extractor(images, scale)
        descs = self.desc_extractor.inference(images)
        return descs

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
