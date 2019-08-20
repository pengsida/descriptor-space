import torch

from lib.utils.base import sample_descriptor, hard_negative_mining, sample_scale


class DescEvaluator(object):
    def __call__(self, desc_maps0, kps0, images0, desc_maps1, kps1, kps2, images1, thresh=16, interval=16):
        """
        desc_maps0: [B, D, H_0', W_0']
        kps0: [B, N, 2], kps0 are the original pixels of images0
        images0: [B, D, H_0, W_0]
        desc_maps1: [B, D, H_1', W_1']
        kps1: [B, N, 2]
        kps2: [B, N, 2], belong to the descr_maps1
        images1: [B, D, H_1, W_1]
        """
        descs0 = sample_descriptor(desc_maps0, kps0, images0)  # [B, N, D]
        descs1 = sample_descriptor(desc_maps1, kps1, images1)  # [B, N, D]
        # descs2 = sample_descriptor(descr_maps1, kps2)
        descs2 = hard_negative_mining(descs0, desc_maps1, kps1, images1, thresh, interval)  # [B, N, D]

        pos_dist = torch.norm(descs0 - descs1, 2, dim=2)
        neg_dist = torch.norm(descs0 - descs2, 2, dim=2)

        distance = torch.sum(pos_dist) / pos_dist.numel()
        # print(distance)

        similarity = 0.5 - neg_dist
        weight = similarity > 0
        similarity = torch.sum(similarity[weight]) / torch.clamp(torch.sum(weight).float(), min=1.)

        loss = distance + similarity

        return loss, distance, similarity


class ScaleEvaluator(object):
    def __call__(self, scales_pred, scales, scale_weights, sigma=1.0):
        """
        scales_pred: [B, 1, H, W]
        scales: [B, 1, H, W]
        scale_weights: [B, 1, H, W]
        """
        sigma_2 = sigma ** 2
        scale_diff = scales_pred - scales
        diff = scale_weights * scale_diff
        abs_diff = torch.abs(diff)
        smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
        in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        in_loss = torch.sum(in_loss) / (torch.sum(scale_weights) + 1e-3)
        return in_loss


class ScaleKpsEvaluator(object):
    def __call__(self, scales_pred, scales, kps, images):
        """
        scales_pred: [B, 1, H', W']
        scales: [B, 1, H, W]
        kps: [B, N, 2]
        images: [B, 3, H, W]
        """
        scales_pred = sample_scale(scales_pred, kps, images)
        scales = sample_scale(scales, kps, images)
        loss = torch.pow(scales_pred - scales, 2) / 2.
        return torch.sum(loss) / loss.numel()


def make_evaluator(cfg):
    return DescEvaluator()


def make_scale_evaluator(cfg, kps=False):
    if kps:
        return ScaleKpsEvaluator()
    else:
        return ScaleEvaluator()
