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

        self.linear = nn.Sequential(
            nn.Conv2d(3600, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.en = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True)
        )
        self.de = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.regress = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.scale_evaluator = make_scale_evaluator(cfg)

    def forward(self, left, right, scale, msk):
        """
        left: [B, D, H0, W0]
        right: [B, D, H1, W1]
        scale: [B, D, H1', W1']
        msk: [B, H1', W1']
        """
        # [b, h0*w0, h1, w1]
        cost_volume = compute_cost_volume(right, left)

        linear = self.linear(cost_volume)
        en = self.en(linear)
        up_en = self.upconv(en)
        de = self.de(torch.cat((up_en, linear), dim=1))
        scale_pred = self.regress(de)

        h1, w1 = right.shape[2:]
        scale = F.interpolate(scale.unsqueeze(1), (h1, w1), mode='bilinear')
        msk = F.interpolate(msk.unsqueeze(1), (h1, w1), mode='bilinear').long().float()
        scale_loss = self.scale_evaluator(scale_pred, scale, msk)

        return scale_pred, scale_loss

    def inference(self, left, right):
        cost_volume = compute_cost_volume(right, left)
        linear = self.linear(cost_volume)
        en = self.en(linear)
        up_en = self.upconv(en)
        de = self.de(torch.cat((up_en, linear), dim=1))
        scale_pred = self.regress(de)
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

    def forward(self, images, scales, id):
        """
        images: [B, 3, H, W]
        scales: the pixel scale ratio of right image to left image
        id: identify left image or right image
        """
        feats = self.backbone(images)

        descs = self.regress(feats['C3'])

        descs = F.normalize(descs, p=2, dim=1)

        # feats3 = feats['C3']
        # h, w = feats3.shape[2:]
        # feats3 = F.max_pool2d(feats3, kernel_size=3, stride=2, padding=1)
        # down_descs = F.interpolate(self.regress(feats3), (h, w), mode='bilinear')
        #
        # # # for each pixel, if its scale is bigger than 1, then it should adopt a downsample one
        # if len(id) != 0:
        #     scales = F.interpolate(scales.unsqueeze(1), (h, w), mode='bilinear')
        #     msk = (scales > 1).float()
        #     scales = torch.clamp(scales, max=2)
        #     weight = scales - 1
        #     down_descs = weight * down_descs + (1 - weight) * descs
        #     # return F.normalize(descs, p=2, dim=1)
        # else:
        #     # for each pixel, if its scale class is 2, then it should adopt a downsample one
        #     cls = torch.argmax(scales, dim=1, keepdim=True)
        #     msk = (cls == 2).float()
        #
        # descs = F.normalize(msk * down_descs + (1 - msk) * descs, p=2, dim=1)

        return descs

    def inference(self, images, scale):
        """
        images: [B, 3, H, W]
        """
        feats = self.backbone(images)

        x1 = F.normalize(self.regress(feats['C3']), p=2, dim=1)

        feats3 = feats['C3']
        h, w = feats3.shape[2:]
        feats3 = F.max_pool2d(feats3, kernel_size=3, stride=2, padding=1)
        feats3 = F.interpolate(self.regress(feats3), (h, w), mode='bilinear')
        x2 = F.normalize(feats3, p=2, dim=1)

        descs = x1
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


class MultiScaleNetV12(nn.Module):
    def __init__(self, cfg):
        super(MultiScaleNetV12, self).__init__()

        self.desc_extractor = DescExtractor(cfg)
        self.scale_estimator = ScaleEstimator(cfg)
        self.desc_evaluator = MultiDescEvaluator(cfg)

    def forward(self, images0, images1, targets):
        """
        images0: [B, 3, H, W]
        images1: [B, 3, H, W]
        targets: {'kps0': [B, N, 2], 'kps1': [B, N, 2]}
        """
        descs0 = self.desc_extractor(images0, targets['left_scale'], 'left')
        descs1 = self.desc_extractor(images1, targets['scale'], 'right')

        desc_loss, distance, similarity = self.desc_evaluator(
            descs0, targets['kps0'], images0,
            descs1, targets['kps1'], targets['kps2'], images1
        )
        scale_pred, scale_loss = self.scale_estimator(descs0, descs1, targets['scale'], targets['msk'])

        results = dict(
            descs0=descs0,
            descs1=descs1
        )

        # losses = dict(loss=scale_loss)
        loss = desc_loss + scale_loss
        losses = dict(loss=loss, desc_loss=desc_loss, distance=distance, similarity=similarity, scale_loss=scale_loss)

        # keep descriptors for visualization
        logger.update(image0=images0[0], image1=images1[0])
        logger.update(kps0=targets['kps0'][0], kps1=targets['kps1'][0], kps2=targets['kps2'][0])
        logger.update(d03=descs0[0], d13=descs1[0])
        logger.update(H=targets['H'][0])
        logger.update(scale_pred=scale_pred[0])
        logger.update(scale=targets['scale'][0])
        logger.update(msk=targets['msk'][0])

        return losses, results

    def inference(self, images0, images1, left_scales, scales):
        descs0 = self.desc_extractor(images0, left_scales, 'left')
        descs1 = self.desc_extractor(images1, scales, 'right')
        # left_scales_pred = self.scale_estimator.inference(descs1, descs0)
        # scales_pred = self.scale_estimator.inference(descs0, descs1)
        # descs0 = self.desc_extractor(images0, left_scales_pred, '')
        # descs1 = self.desc_extractor(images1, scales_pred, '')
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
