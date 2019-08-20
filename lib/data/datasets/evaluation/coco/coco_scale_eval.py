import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.utils.base import draw_corspd, sample_descriptor, draw_corspd_region, SampleTrainingTarget
from lib.utils.nn_set2set_match.nn_set2set_match_layer import nn_set2set_match_cuda
from lib.utils.visualize import draw_img_desc_torch, draw_corspd_region_torch, cls2RGB
from lib.utils.misc import tensor2RGB, tonumpy, touint8


class COCOScaleEvaluator(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.pcks = []
        self.scales = []
        self.accuracies = []

    def evaluate(self, scale_pred, scale, image0, image1, index, mask):
        """
        scale_pred: [3, H', W']
        scale: [1, H, W]
        image0: [3, H, W]
        image1: [3, H, W]
        mask: [1, H, W], boolean tensor
        """
        num_cls = scale_pred.shape[0]
        scale_pred = torch.argmax(scale_pred, dim=0).long()

        scale[scale > 1.5] = 2
        scale[scale < 0.75] = 0
        scale[(scale >= 0.75) * (scale <= 1.5)] = 1
        scale = scale.long()

        h, w = scale_pred.shape
        h0, w0 = scale.shape[1:]
        scale = F.interpolate(scale.view(1, 1, h0, w0).float(), (h, w), mode='bilinear').long()[0, 0]
        mask = F.interpolate(mask.view(1, 1, h0, w0).float(), (h, w), mode='bilinear').byte()[0, 0]

        # correct predictions
        correct = (scale_pred == scale) & mask
        accuracy = torch.sum(correct).float() / torch.sum(mask).float()
        self.accuracies.append(accuracy)
        
        scale_pred = touint8(tonumpy(cls2RGB(scale_pred, num_cls)))
        scale = touint8(tonumpy(cls2RGB(scale, num_cls)))

        img0 = tensor2RGB(image0)
        img1 = tensor2RGB(image1)
        img = np.concatenate([img0, img1], axis=1)

        import matplotlib.pyplot as plt
        _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img)
        ax2.imshow(scale)
        ax3.imshow(scale_pred)
        plt.show()

    def average_precision(self):
        # print("pck: {}".format(np.mean(self.pcks)))
        print("recall: {}".format(np.mean(self.accuracies)))

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
