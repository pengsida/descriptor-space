from torch import nn
import torch
import torch.nn.functional as F

from lib.modeling.backbone import build_backbone
from .evaluators import make_evaluator, make_scale_evaluator
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
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        )
        self.scale_evaluator = make_scale_evaluator(cfg, kps=True)

    def forward(self, left, right, kps1, images1, scales):
        """
        left: [B, D, H0, W0]
        right: [B, D, H1, W1]
        kps1: [B, N, 2], belongs to images1
        images1: [B, D, H1', W1']
        scales: [B, D, H1', W1']
        """
        # [b, h0*w0, h1, w1]
        cost_volume = compute_cost_volume(right, left)
        scales_pred = self.regress(cost_volume)

        scale_loss = self.scale_evaluator(scales_pred, scales, kps1, images1)

        return scales_pred, scale_loss

    def inference(self, left, right):
        cost_volume = compute_cost_volume(right, left)
        scale_pred = self.regress(cost_volume)
        return scale_pred


class DescExtractor(nn.Module):
    def __init__(self, cfg):
        super(DescExtractor, self).__init__()

        self.backbone = build_backbone(cfg)
        self.regress = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, images, id):
        """
        images: [B, 3, H, W]
        id: identify left image or right image
        """
        feats = self.backbone(images)
        descs = self.regress(feats['C3'])
        descs = F.normalize(descs, p=2, dim=1)
        return descs

    def inference(self, images, scale):
        """
        images: [B, 3, H, W]
        """
        feats = self.backbone(images)
        return F.normalize(self.regress(feats['C3']), p=2, dim=1)


class MultiScaleNetV13(nn.Module):
    def __init__(self, cfg):
        super(MultiScaleNetV13, self).__init__()

        self.desc_extractor = DescExtractor(cfg)
        self.scale_estimator = ScaleEstimator(cfg)
        self.desc_evaluator = make_evaluator(cfg)

    def forward(self, images0, images1, targets):
        """
        images0: [B, 3, H, W]
        images1: [B, 3, H, W]
        targets: {'kps0': [B, N, 2], 'kps1': [B, N, 2]}
        """
        descs0 = self.desc_extractor(images0, 'left')
        descs1 = self.desc_extractor(images1, 'right')

        desc_loss, distance, similarity = self.desc_evaluator(
            descs0, targets['kps0'], images0,
            descs1, targets['kps1'], targets['kps2'], images1,
            thresh=4, interval=4
        )
        scale_pred, scale_loss = self.scale_estimator(
            descs0, descs1,
            targets['kps1'], images1,
            targets['scale']
        )

        results = dict(
            descs0=descs0,
            descs1=descs1
        )

        # losses = dict(loss=scale_loss)
        loss = desc_loss + scale_loss
        losses = dict(loss=loss, desc_loss=desc_loss, distance=distance, similarity=similarity, scale_loss=scale_loss)

        # keep descriptors for visualization
        logger.update(image0=images0[0], image1=images1[0])
        logger.update(kps0=targets['kps0'][0], kps1=targets['kps1'][0])
        logger.update(d03=descs0[0], d13=descs1[0])
        logger.update(H=targets['H'][0])
        logger.update(scale_pred=scale_pred[0])
        logger.update(scale=targets['scale'][0])
        logger.update(msk=targets['msk'][0])

        return losses, results

    def inference(self, images0, images1, left_scales, scales):
        descs0 = self.desc_extractor(images0, 'left')
        descs1 = self.desc_extractor(images1, 'right')
        scale_pred = self.scale_estimator.inference(descs0, descs1)
        return descs0, descs1, scale_pred

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
