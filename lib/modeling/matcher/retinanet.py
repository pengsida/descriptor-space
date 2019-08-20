from torch import nn
import torch.nn.functional as F
import torch

from lib.modeling.backbone import build_backbone
from lib.utils.base import hard_negative_mining


class DescriptorExtractor(nn.Module):
    def __init__(self, cfg):
        super(DescriptorExtractor, self).__init__()

        self.backbone = build_backbone(cfg)

    def forward(self, images, kps):
        """
        images: [B, 3, H, W]
        kps: [B, N, 2]
        """
        multi_descriptors = self.backbone(images)

        # normalize kps for grid sample
        b, _, h, w = images.shape
        kps = normalize_kps(kps, h, w).view(b, 1, -1, 2)

        # upsample descriptors
        descriptor_maps = F.upsample(multi_descriptors[0], (h, w), mode="bilinear")
        descriptor_maps = F.normalize(descriptor_maps, p=2, dim=1)

        # extract descriptors given kps
        multi_descriptors = [F.grid_sample(descriptors, kps) for descriptors in multi_descriptors[:3]]
        multi_descriptors = [descriptors[:, :, 0, :].transpose(2, 1) for descriptors in multi_descriptors]
        scale_descriptors = [F.normalize(descriptors, p=2, dim=2) for descriptors in multi_descriptors]
        scale_descriptors = torch.stack(scale_descriptors, dim=2)

        return descriptor_maps, scale_descriptors

    def inference(self, images):
        """
        images: [B, 3, H, W]
        """
        multi_descriptors = self.backbone(images)

        h, w = images[2:]
        multi_descriptors = [F.upsample(descriptors, (h, w), mode="bilinear") for descriptors in multi_descriptors[:3]]
        ds3, ds4, ds5 = [F.normalize(descriptors, p=2, dim=1) for descriptors in multi_descriptors]

        return ds3, ds4, ds5


class DescriptorEvaluator(object):
    def __call__(self, scale_descriptors0, kps0, scale_descriptors1, kps1, descriptor_maps1):
        """
        scale_descriptors0: [B, N, 3, D]
        scale_descriptors1: [B, N, 3, D]
        """
        scale0, scale1 = select_paired_scale(scale_descriptors0, scale_descriptors1)

        b, n = scale_descriptors0.shape[:2]
        dim0 = torch.arange(b).view(b, 1)
        dim1 = torch.arange(n).view(1, n)
        descriptors0 = scale_descriptors0[dim0, dim1, scale0]
        descriptors1 = scale_descriptors1[dim0, dim1, scale1]
        descriptors2 = hard_negative_mining(descriptors0, descriptor_maps1, kps1, thresh=10)

        pos_dist = torch.norm(descriptors0 - descriptors1, 2, dim=2)
        neg_dist = torch.norm(descriptors0 - descriptors2, 2, dim=2)
        loss = pos_dist - neg_dist + 0.5
        weight = loss > 0
        loss = torch.sum(loss[weight]) / torch.clamp(torch.sum(weight).float(), min=1.)

        pos_loss = torch.sum(pos_dist) / pos_dist.numel()
        neg_loss = torch.sum(neg_dist) / neg_dist.numel()

        return descriptors0, descriptors1, loss, pos_loss, neg_loss


class RetinaNet(nn.Module):
    def __init__(self, cfg):
        super(RetinaNet, self).__init__()

        self.desc_extractor = DescriptorExtractor(cfg)
        self.desc_evaluator = DescriptorEvaluator()

    def forward(self, images0, images1, targets):
        """
        images0: [B, 3, H, W]
        images1: [B, 3, H, W]
        """
        _, scale_descriptors0 = self.desc_extractor(images0, targets["kps0"])
        descriptor_maps1, scale_descriptors1 = self.desc_extractor(images1, targets["kps1"])

        kps0 = targets["kps0"]
        kps1 = targets["kps1"]
        descriptors0, descriptors1, loss, pos_loss, neg_loss = self.desc_evaluator(scale_descriptors0, kps0, scale_descriptors1, kps1, descriptor_maps1)

        if targets["iteration"] % 100 == 0:
            print(pos_loss, neg_loss)

        results = dict(
            descrs0=descriptors0,
            descrs1=descriptors1,
        )

        losses = dict(
            triplet=loss
        )

        return losses, results

    def inference(self, images0):
        """
        images: [B, 3, H, w]
        """
        h, w = images0.shape[2:]
        ds03, ds04, ds05 = self.desc_extractor(images0)
        ds03 = F.upsample(ds03, (h, w), mode="bilinear")
        ds04 = F.upsample(ds04, (h, w), mode="bilinear")
        ds05 = F.upsample(ds05, (h, w), mode="bilinear")
        return ds03, ds04, ds05


def select_paired_scale(scale_descriptor0, scale_descriptor1):
    """
    scale_descriptor0: [B, N, 3, D]
    scale_descriptor1: [B, N, 3, D]
    """
    # the max of correlation means the minimal distance
    correlation = torch.matmul(scale_descriptor0, scale_descriptor1.transpose(3, 2))

    # select the row and col of the maximum, where row is scale0 and col is scale1
    b, n = correlation.shape[:2]
    correlation = correlation.view(b, n, 3*3)
    row_col = torch.argmax(correlation, dim=2)  # row_col = 3 * row + col
    scale0 = row_col // 3
    scale1 = row_col % 3
    return scale0, scale1


def normalize_kps(kps, h, w):
    """
    kps: [B, N, 2]
    """
    with torch.no_grad():
        kps = kps.clone().detach()
        kps[:, :, 0] = (kps[:, :, 0] / (float(w)/2.)) - 1.
        kps[:, :, 1] = (kps[:, :, 1] / (float(h)/2.)) - 1.
    return kps
