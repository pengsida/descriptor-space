import cv2
import torch
import numpy as np

def estimate_homography(descs0, descs1, kps0, kps1):
    """
    Estimate a homography from image 0 to 1.
    
    These can either be numpy or Tensor
    :param descs0: shape [D, H, W]
    :param descs1: shape [D, H, W]
    :param kps0: shape [N, 2]
    :param kps1: shape [N, 2]
    :return: H, shape (3, 3)
    """
    def tonumpy(*args):
        return [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in args]
    [descs0, descs1, kps0, kps1] = tonumpy(descs0, descs1, kps0, kps1)
    indices = nn_match(descs0, descs1)
    kps0 = kps0[[x[0] for x in indices]]
    kps1 = kps1[[x[0] for x in indices]]
    H = cv2.findHomography(kps0, kps1)
    
    return H
    


def nn_match(descs1, descs2):
    """
    Perform nearest neighbor match, using descriptors.

    This function uses OpenCV FlannBasedMatcher

    :param descs1: descriptors from image 1, (N1, D)
    :param descs2: descriptors from image 2, (N2, D)
    :return indices: a list of tuples, each is (queryIdx, trainIdx)
    """
    
    flann = cv2.FlannBasedMatcher_create()
    matches = flann.match(descs1.astype(np.float32), descs2.astype(np.float32), crossCheck=True)
    indices = [(x.queryIdx, x.trainIdx) for x in matches]
    
    return indices
