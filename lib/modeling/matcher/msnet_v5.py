from torch import nn
import torch
import torch.nn.functional as F

from lib.modeling.backbone import build_backbone
from .evaluators import make_evaluator
from lib.utils.logger import logger
from lib.utils.visualize import draw_paired_img_desc_torch

class AttentionEstimator(nn.Module):
    def __init__(self, cfg, dim_feats, scales):
        """
        DescExtractor
        :param scales: a list of scales, like [0.5, 1, 2]
        :param dim_feat: feature dimension
        """
        nn.Module.__init__(self)
        
        self.scales = scales
        self.regress = nn.Sequential(
            nn.Conv2d(dim_feats, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, feats, which='left'):
        """
        Forward pass of the attention estimator
        
        :param feats: feature of shape (B, D, H, W)
        :param scales: various scales at which the map should be computed
        """
        
        _, _, H, W = feats.size()

        # resize feats for to each scale
        maps = []
        for scale in self.scales:
            # scale features
            H_resized, W_resized = int(H * scale), int(W * scale)
            feats_resized = F.interpolate(feats, (H_resized, W_resized), mode='bilinear')
    
            # extract attention maps, and scale back
            # (B, 1, H, W)
            map_resized = F.interpolate(self.regress(feats_resized), (H, W), mode='bilinear')
            maps.append(map_resized)

        # (B, N, H, W)
        maps = torch.cat(maps, dim=1)
        # softmax application
        maps = F.softmax(maps, dim=1)
        
        # visualize attention map
        args = {
            'map_' + which: maps[0]
        }
        logger.update(**args)

        return maps



class DescExtractor(nn.Module):
    def __init__(self, cfg, scales, backbone):
        """
        DescExtractor
        :param scales: a list of scales, like [0.5, 1, 2]
        """
        super(DescExtractor, self).__init__()
        
        self.backbone = backbone
        self.scales = scales
        # for C3 only
        self.regress = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        )
        # for C4 only
        self.attention = AttentionEstimator(cfg, dim_feats=128, scales=scales)

    def forward(self, images, scale=1, which='left'):
        """
        images: [B, 3, H, W]
        """
        feats = self.backbone(images)
        # feature used for attention maps
        # feats_atten = feats['C4']
        
        # feature used for descriptors and attention
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
            
        # descs: (B, N, D, H, W)
        descs = torch.stack(descs, dim=1)
        # maps: (B, N, H, W)
        maps = self.attention(feats, which)
        # maps: (B, N, 1, H, W)
        maps = maps.unsqueeze(2)
        # apply attention, (B, N, D, H, W) -> (B, D, H, W)
        descs = torch.sum(descs * maps, dim=1)
        
        # normalize descriptors
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


class MultiScaleNetV5(nn.Module):
    """
    This version simply sums descriptors from all scales
    """
    def __init__(self, cfg, scales=[0.5, 1, 1.5, 2]):
        # super(MultiScaleNetV2, self).__init__()
        nn.Module.__init__(self)

        self.backbone = build_backbone(cfg)
        self.scales = scales
        self.desc_extractor = DescExtractor(cfg, scales, self.backbone)
        self.desc_evaluator = MultiDescEvaluator(cfg)

    def forward(self, images0, images1, targets):
        """
        images0: [B, 3, H, W]
        images1: [B, 3, H, W]
        targets: {'kps0': [B, N, 2], 'kps1': [B, N, 2]}
        """
        descs0 = self.desc_extractor(images0, 1, which='left')
        descs1 = self.desc_extractor(images1, targets['scale'], which='right')

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
        logger.update(scales=self.scales)
        logger.update(scale=targets['scale'][0])
        logger.update(mask=targets['msk'][0])

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
