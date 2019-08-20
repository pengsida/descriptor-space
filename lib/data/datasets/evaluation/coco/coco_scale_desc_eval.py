import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.utils.base import draw_corspd, sample_descriptor, draw_corspd_region, SampleTrainingTarget, draw_scale, \
    sample_scale
from lib.utils.nn_set2set_match.nn_set2set_match_layer import nn_set2set_match_cuda
from lib.utils.visualize import draw_img_desc_torch, draw_corspd_region_torch
from lib.utils.misc import tensor2RGB


class COCOScaleDescEvaluator(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.pcks = []
        self.scales = []
        self.l2s = []

    def evaluate(self, desc0, desc1, index, H, scale_pred, scale, image0, image1, mask):
        """
        desc0: [D, H0, W0]
        desc1: [D, H1, W1]
        index: data order
        H: [3, 3]
        scale_pred: [1, H1, W1]
        scale: [1, H1', W1']
        image0: [3, H0', W0']
        image1: [3, H1', W1']
        mask: [1, H1, W1]
        """
        target = SampleTrainingTarget.sample_torch(image0, image1, H)
        kps0 = target['kps0']
        kps1 = target['kps1']

        if len(desc0.shape) == 4:
            x1 = sample_descriptor(desc0[0].unsqueeze(0), kps0.unsqueeze(0), image0.unsqueeze(0))[0]
            x2 = sample_descriptor(desc0[1].unsqueeze(0), kps0.unsqueeze(0), image0.unsqueeze(0))[0]
            descs0 = torch.stack([x1, x2], dim=1)

            x1 = sample_descriptor(desc1[0].unsqueeze(0), kps1.unsqueeze(0), image1.unsqueeze(0))[0]
            x2 = sample_descriptor(desc1[0].unsqueeze(0), kps1.unsqueeze(0), image1.unsqueeze(0))[0]
            descs1 = torch.stack([x1, x2], dim=1)
        else:
            descs0 = sample_descriptor(desc0.unsqueeze(0), kps0.unsqueeze(0), image0.unsqueeze(0))[0].numpy()
            descs1 = sample_descriptor(desc1.unsqueeze(0), kps1.unsqueeze(0), image1.unsqueeze(0))[0].numpy()

        scales_pred = sample_scale(scale_pred.unsqueeze(0), kps1.unsqueeze(0), image1.unsqueeze(0))[0].numpy()
        scales = sample_scale(scale.unsqueeze(0), kps1.unsqueeze(0), image1.unsqueeze(0))[0].numpy()

        img0 = draw_img_desc_torch(image0, desc0)
        img1 = draw_img_desc_torch(image1, desc1)
        img = np.concatenate([img0, img1], axis=0)
        diff = np.abs(scales_pred - scales)
        import matplotlib.pyplot as plt
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(img)
        ax2.hist(diff, bins=100)
        ax3.hist(scales_pred, bins=100)
        ax4.hist(scales, bins=100)
        plt.show()

        self.pcks.append(compute_pck(descs0, kps0, descs1, kps1))
        self.l2s.append(l1_kps(scales_pred, scales))
        # self.l2s.append(l1(scale_pred, scale, mask))

    def average_precision(self):
        print("pck: {}, l2: {}".format(np.mean(self.pcks), np.mean(self.l2s)))

    def wrong_pixel_scale(self):
        if len(self.scales) == 0:
            return
        import matplotlib.pyplot as plt
        plt.hist(np.concatenate(self.scales), bins=100)
        plt.show()


def nn_match(descs1, descs2):
    """
    Perform nearest neighbor match, using descriptors.

    This function uses OpenCV FlannBasedMatcher

    :param descs1: descriptors from image 1, (N1, D)
    :param descs2: descriptors from image 2, (N2, D)
    :return indices: indices into keypoints from image 2, (N1, D)
    """
    # diff = descs1[:, None, :] - descs2[None, :, :]
    # diff = np.linalg.norm(diff, ord=2, axis=2)
    # indices = np.argmin(diff, axis=1)

    flann = cv2.FlannBasedMatcher_create()
    matches = flann.match(descs1.astype(np.float32), descs2.astype(np.float32))
    indices = [x.trainIdx for x in matches]

    return indices


def nn_set2set_match(descs1, descs2):
    """
    Perform nearest neighbor match on CUDA, using sets of descriptors
    This function uses brute force

    descs1: [N1, 2, D]
    descs2: [N2, 2, D]
    indices: indices into keypoints from image 2, [N1, D]
    """
    idxs = nn_set2set_match_cuda(descs1.unsqueeze(0).cuda(), descs2.unsqueeze(0).cuda()).detach().cpu().long()
    return idxs[0]


def compute_pck(descs0, kps0, descs1, kps1, thresh=3):
    """
    Compute pck given two sets of keypoints and descriptors.

    :param descs0, descs1: descriptors for the images, shape (N0, D) descriptor, or (N0, 2, D) multi-scale descriptor
    :param kps0, kps1: keypoints for the image, shape (N0, 2), each being (x, y)
    :param thresh: threshold in pixel
    :return: matchine score (%)
    """
    # matches points from image 0 to image 1, using NN
    if len(descs0.shape) == 3:
        idxs = nn_set2set_match(descs0, descs1)
    else:
        idxs = nn_match(descs0, descs1)  # matched image 1 keypoint indices
    predicted = kps1[idxs]                                 # matched image 1 keypoints

    correct = np.linalg.norm(predicted - kps1, 2, axis=1) <= thresh

    return np.sum(correct) / len(correct)


def l1_kps(scales_pred, scales):
    """
    scale_pred: [N]
    scale: [N]
    """
    return np.sum(np.abs(scales_pred - scales)) / len(scales_pred)


def l1(scale_pred, scale, mask):
    """
    scale_pred: [1, H', W']
    scale: [1, H, W]
    mask: [1, H, W]
    """
    h0, w0 = scale_pred.shape[1:]
    h, w = scale.shape[1:]
    scale = F.interpolate(scale.view(1, 1, h, w), (h0, w0), mode='bilinear')[0]
    mask = F.interpolate(mask.view(1, 1, h, w), (h0, w0), mode='bilinear')[0].long().float()
    diff = mask * (scale_pred - scale)
    return torch.sum(torch.abs(diff)) / torch.sum(mask)
