from torch import nn
import torch
import torch.nn.functional as F

from lib.modeling.backbone import build_backbone
from .evaluators import make_evaluator
from lib.utils.logger import logger
from lib.utils.base import compute_cost_volume


class ScaleEstimator(nn.Module):
    """Estimate scale ratio of right image to left image for each pixel"""

    def __init__(self, cfg):
        super(ScaleEstimator, self).__init__()

        self.regress = nn.Sequential(
            nn.Conv2d(3600, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)
        )
        self.scale_evaluator = nn.CrossEntropyLoss(reduce=False)

    def forward(self, left, right, scale, msk):
        """
        left: [B, D, H0, W0]
        right: [B, D, H1, W1]
        scale: [B, D, H1', W1']
        msk: [B, H1', W1']
        """
        # [b, h0*w0, h1, w1]
        cost_volume = compute_cost_volume(right, left)
        scale_pred = self.regress(cost_volume)

        h1, w1 = right.shape[2:]
        scale = F.interpolate(scale.unsqueeze(1), (h1, w1), mode='bilinear')
        scale[scale > 1.5] = 2
        scale[(scale >= 0.75) * (scale <= 1.5)] = 1
        scale[scale < 0.75] = 0
        scale_loss = self.scale_evaluator(scale_pred, scale.long().squeeze(1))

        msk = F.interpolate(msk.unsqueeze(1), (h1, w1), mode='bilinear').squeeze(1)
        scale_loss = torch.sum(scale_loss * msk) / torch.sum(msk)
        return F.softmax(scale_pred, dim=1), scale_loss

    def inference(self, left, right):
        cost_volume = compute_cost_volume(right, left)
        scale_pred = self.regress(cost_volume)
        return F.softmax(scale_pred, dim=1)


class DescExtractor(nn.Module):
    def __init__(self, cfg):
        super(DescExtractor, self).__init__()

        self.backbone = build_backbone(cfg)

    def forward(self, images):
        """
        images: [B, 3, H, W]
        scale: the pixel scale ratio of right image to left image
        id: identify left image or right image
        """
        feats = self.backbone(images)
        return feats['C3']

    def inference(self, images):
        """
        images: [B, 3, H, W]
        """
        feats = self.backbone(images)
        return feats['C3']


class MultiDescEvaluator(object):
    def __init__(self, cfg):
        self.desc_evaluator = make_evaluator(cfg)

    def __call__(self, desc_maps0, kps0, images0, desc_maps1, kps1, kps2, images1):
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


class MultiScaleNetV9(nn.Module):
    def __init__(self, cfg):
        super(MultiScaleNetV9, self).__init__()

        self.desc_extractor = DescExtractor(cfg)
        self.scale_estimator = ScaleEstimator(cfg)
        self.desc_evaluator = MultiDescEvaluator(cfg)

    def forward(self, images0, images1, targets):
        """
        images0: [B, 3, H, W]
        images1: [B, 3, H, W]
        targets: {'kps0': [B, N, 2], 'kps1': [B, N, 2]}
        """
        descs0 = self.desc_extractor(images0)
        descs1 = self.desc_extractor(images1)

        scale_pred, scale_loss = self.scale_estimator(descs0, descs1, targets['scale'], targets['msk'])

        results = dict(
            descs0=descs0,
            descs1=descs1
        )

        losses = dict(loss=scale_loss)

        # keep descriptors for visualization
        logger.update(image0=images0[0], image1=images1[0])
        logger.update(scale_pred=scale_pred[0])
        logger.update(scale=targets['scale'][0])
        logger.update(msk=targets['msk'][0])

        return losses, results

    def inference(self, images0, images1, left_scales=None, scales=None):
        descs0 = self.desc_extractor(images0)
        descs1 = self.desc_extractor(images1)
        scale_pred = self.scale_estimator.inference(descs0, descs1)
        return scale_pred

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
