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
        self.regress = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, images, scale, id):
        """
        images: [B, 3, H, W]
        """
        feats = self.backbone(images)
        feats3 = feats['C3']
        h, w = feats3.shape[2:]

        if id == 'left':
            descs = F.normalize(self.regress(feats3), p=2, dim=1)
        elif id == 'right':
            # descriptors of different receptive fields
            descs = self.regress(feats3)

            up_descs = F.interpolate(feats3, (h*2, w*2), mode='bilinear')
            up_descs = F.interpolate(self.regress(up_descs), (h, w), mode='nearest')

            down_descs = F.interpolate(feats3, (h//2, w//2), mode='bilinear')
            down_descs = F.interpolate(self.regress(down_descs), (h, w), mode='bilinear')

            # combine the descriptors based on the pixel relative scales
            scale = F.interpolate(scale.unsqueeze(0), (h, w), mode='bilinear')
            scale = torch.clamp(scale, min=0.5, max=2.)
            msk = (scale < 1.).float()

            up_weight = 2 * (1 - scale)
            up_descs = up_weight * up_descs + (1 - up_weight) * descs

            down_weight = scale - 1
            down_descs = down_weight * down_descs + (1 - down_weight) * descs

            descs = F.normalize(msk * up_descs + (1 - msk) * down_descs, p=2, dim=1)

        # if scale == 0.5:
        #     feats3 = F.interpolate(feats3, (h*2, w*2), mode='bilinear')
        #     descs = F.interpolate(self.regress(feats3), (h, w), mode='bilinear')
        #     descs = F.normalize(descs, p=2, dim=1)
        # elif scale == 1:
        #     descs = F.normalize(self.regress(feats3), p=2, dim=1)
        # else:
        #     feats3 = F.interpolate(feats3, (h//2, w//2), mode='bilinear')
        #     descs = F.interpolate(self.regress(feats3), (h, w), mode='bilinear')
        #     descs = F.normalize(descs, p=2, dim=1)

        return descs

    def inference(self, images, scale):
        """
        images: [B, 3, H, W]
        """
        feats = self.backbone(images)
        feats3 = feats['C3']
        h, w = feats3.shape[2:]

        x1 = F.interpolate(feats3, (h*2, w*2), mode='bilinear')
        x1 = F.interpolate(self.regress(x1), (h, w), mode='bilinear')
        x2 = self.regress(feats3)
        x3 = F.interpolate(feats3, (h//2, w//2), mode='bilinear')
        x3 = F.interpolate(self.regress(x3), (h, w), mode='bilinear')

        # weighted C3
        scale = torch.clamp(scale, min=0.5, max=2.)
        scale = F.interpolate(scale.unsqueeze(0), (h, w), mode='bilinear').cuda()

        # msk1 = (scale < 0.75).float()
        # msk2 = ((scale > 0.75) * (scale < 1.5)).float()
        # msk3 = (scale > 1.5).float()
        # descs = msk1 * x1 + msk2 * x2 + msk3 * x3

        msk = (scale < 1.).float()
        weight2 = 2 * (scale - 0.5)
        weight3 = scale - 1.
        descs12 = weight2 * x2 + (1 - weight2) * x1
        descs23 = weight3 * x3 + (1 - weight3) * x2
        descs = msk * descs12 + (1 - msk) * descs23
        descs = F.normalize(descs, p=2, dim=1)

        return descs


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


class MultiScaleNetV3(nn.Module):
    def __init__(self, cfg):
        super(MultiScaleNetV3, self).__init__()

        self.desc_extractor = DescExtractor(cfg)
        self.desc_evaluator = MultiDescEvaluator(cfg)

    def forward(self, images0, images1, targets):
        """
        images0: [B, 3, H, W]
        images1: [B, 3, H, W]
        targets: {'kps0': [B, N, 2], 'kps1': [B, N, 2]}
        """
        descs0 = self.desc_extractor(images0, 2, 'left')
        descs1 = self.desc_extractor(images1, targets['scale'], 'right')

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

    def inference(self, images, scale, id):
        if id == 'left':
            descs = self.desc_extractor(images, 2, id)
        else:
            descs = self.desc_extractor(images, scale.cuda(), id)
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
