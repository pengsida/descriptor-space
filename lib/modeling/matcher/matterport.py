from torch import nn
import torch.nn.functional as F

from lib.modeling.backbone import build_backbone
from .evaluators import make_evaluator
from lib.utils.visualize import desc2RGB, draw_paired_img_desc_torch
from lib.utils.misc import tonumpy, totensor
from lib.utils.logger import logger


class FeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(FeatureExtractor, self).__init__()

        self.backbone = build_backbone(cfg)
        self.conv_out = nn.Conv2d(512, 128, 3, 1, 1)

    def forward(self, images):
        """
        images: [N, 3, H, W]
        """
        h, w = images.shape[2:]
        feats = self.backbone(images)
        descrs = self.conv_out(feats)
        descrs = F.normalize(F.interpolate(descrs, (h, w), mode="bilinear"), dim=1)
        return descrs


class Matterport(nn.Module):
    def __init__(self, cfg):
        super(Matterport, self).__init__()

        self.feature_extractor = FeatureExtractor(cfg)
        # self.feature_extractor = SuperPointFeature()
        self.evaluator = make_evaluator(cfg)

    def forward(self, images0, images1, targets=None):
        """
        images0: [N, 3, H, W]
        images1: [N, 3, H, W]
        targets: {"img2": [N, 3, H, W], "kps0": [N, 3000], "kps1": [N, 3000], "kps2": [N, 3000]}
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        descrs0 = self.feature_extractor(images0)
        descrs1 = self.feature_extractor(images1)

        # img = draw_desc_torch(
        #     images0[0], descrs0[0], targets['kps0'][0],
        #     images1[0], descrs1[0], targets['kps1'][0],
        #     targets['H'][0],
        # )
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()

        results = dict(
            descrs0=descrs0,
            descrs1=descrs1,
        )
        if not self.training:
            return results

        loss, pos_loss, neg_loss = self.evaluator(
            descrs0, targets["kps0"], images0,
            descrs1, targets["kps1"], targets["kps2"], images1
        )
        
        # if targets["iteration"] % 100 == 0:
        #     print(pos_loss, neg_loss)

        losses = dict(loss=loss, distance=pos_loss, similarity=neg_loss)
        
        # keep descriptors for visualization
        logger.update(image0=images0[0], image1=images1[0])
        logger.update(kps0=targets['kps0'][0], kps1=targets['kps1'][0])
        logger.update(desc0=descrs0[0], desc1=descrs1[0])

        return losses, results

    def inference(self, images):
        return self.feature_extractor(images)
    
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
