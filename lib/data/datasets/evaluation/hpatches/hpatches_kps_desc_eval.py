import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.utils.base import draw_corspd, sample_descriptor, SampleTrainingTarget, compute_scale, \
    get_homography_correspondence, sample_scale
from lib.utils.nn_set2set_match.nn_set2set_match_layer import nn_set2set_match_cuda
from lib.utils.visualize import draw_img_desc_torch


class HpatchesDescDictEvaluator(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.matching_scores = []
        self.scales = []

    def evaluate(self, desc0, desc1, index, H):
        """
        desc0: [D, H0, W0] descriptor, or [2, D, H0, W0] multi-scale descriptor
        desc1: [D, H1, W1] descriptor, or [2, D, H1, W1] multi-scale descriptor
        """
        image0, image1, _, _ = self.dataset[index]
        target = SampleTrainingTarget.sample_torch(image0, image1, H)
        kps0 = target['kps0']
        kps1 = target['kps1']

        x1 = sample_descriptor(desc0['desc'].unsqueeze(0), kps0.unsqueeze(0), image0.unsqueeze(0))[0]
        x2 = sample_descriptor(desc0['down_desc'].unsqueeze(0), kps0.unsqueeze(0), image0.unsqueeze(0))[0]
        scales0 = sample_scale(target['scale0'].unsqueeze(0), kps0.unsqueeze(0), image0.unsqueeze(0))[0]
        descs0 = linear_combination(x1, x2, scales0).numpy()

        x1 = sample_descriptor(desc1['desc'].unsqueeze(0), kps1.unsqueeze(0), image1.unsqueeze(0))[0]
        x2 = sample_descriptor(desc1['down_desc'].unsqueeze(0), kps1.unsqueeze(0), image1.unsqueeze(0))[0]
        scales1 = sample_scale(target['scale1'].unsqueeze(0), kps1.unsqueeze(0), image1.unsqueeze(0))[0]
        descs1 = linear_combination(x1, x2, scales1).numpy()

        self.matching_scores.append(compute_pck(descs0, kps0, descs1, kps1))

    def average_precision(self):
        print("matching score: {}".format(np.mean(self.matching_scores)))

    def wrong_pixel_scale(self):
        if len(self.scales) == 0:
            return
        import matplotlib.pyplot as plt
        plt.hist(np.concatenate(self.scales), bins=100, range=(0, 2))
        plt.show()
        
    def vis(self, image0, image1, desc0, desc1, descs0, descs1, kps0, kps1, H):
    
        h, w = image1.shape[1:]
        
        # H = H.detach().cpu().numpy()
        scale = compute_scale(H, h, w)
        img0 = draw_img_desc_torch(image0, desc0)
        img1 = draw_img_desc_torch(image1, desc1)
        img = np.concatenate([img0, img1], axis=0)
    
        import matplotlib.pyplot as plt
        _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
        ax1.imshow(img)
        ax2.hist(scale[kps0[:, 1].astype(np.int), kps0[:, 0].astype(np.int)], bins=100)
        ax2.set_ylim((0, 100))
    
        # scales = get_wrong_pixel_scale(descs0, kps0, descs1, kps1, image0, H)
        scales, pix_wrong = get_wrong_pixel_scale(descs0, kps0, descs1, kps1, image1, H)
        ax3.hist(scales, bins=100)
        ax3.set_ylim((0, 100))
        print('<0.75: {}'.format(sum(scales < 0.75)))
        print('>1.5: {}'.format(sum(scales > 1.5)))
    
        # show wrong and right pixel positions
        # mask of all, wrong, right pixels
        mask_all = np.zeros((h, w), dtype=np.bool)
        mask_all[kps0[:, 1].astype(np.int), kps0[:, 0].astype(np.int)] = True
        mask_right = mask_all.copy()
        mask_right[pix_wrong[:, 1], pix_wrong[:, 0]] = False
        mask_wrong = mask_all ^ mask_right
    
        # maps
        map_right = np.zeros((h, w), dtype=np.uint8)
        map_right[mask_right] = 255
        ax4.imshow(map_right, cmap='gray')
    
        map_wrong = np.zeros((h, w), dtype=np.uint8)
        map_wrong[mask_wrong] = 255
        ax5.imshow(map_wrong, cmap='gray')
    
        map_all = np.zeros((h, w), dtype=np.uint8)
        map_all[mask_all] = 255
        ax6.imshow(map_all, cmap='gray')
        
        plt.show()


def linear_combination(descs, down_descs, scales):
    """
    descs: [N, D]
    down_descs: [N, D]
    scales: [N]
    """
    # for each pixel, if its scale is bigger than 1, then it should adopt a downsample descriptor
    scales = scales.unsqueeze(1)
    masks = (scales > 1).float()
    scales = torch.clamp(scales, max=2)
    weights = scales - 1
    down_descs = weights * down_descs + (1 - weights) * descs
    descs = F.normalize(masks * down_descs + (1 - masks) * descs, p=2, dim=1)
    return descs


def keep_valid_keypoints(keypoints, H, height, width):
    """
    Keep only keypoints that is present in the other image.

    :param keypoints: shape (N, 2)
    :param H: shape (3, 3)
    :param height, weight: target image height and width
    :return (keypoints, descriptors): valid ones
    """

    N = keypoints.shape[0]
    mapped = cv2.perspectiveTransform(keypoints.reshape(N, 1, 2).astype(np.float), H).reshape(N, 2)
    indices_valid = (
        (mapped[:, 0] >= 0) & (mapped[:, 0] < width) &
        (mapped[:, 1] >= 0) & (mapped[:, 1] < height)
    )

    return keypoints[indices_valid]


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


def compute_matching_score(descrs0, kps0, descrs1, kps1, H, thresh=3):
    """
    Compute matching score given two sets of keypoints and descriptors.

    :param feats0, feats1: descriptors for the images, shape (N0, D)
    :param kps0, kps1: keypoints for the image, shape (N0, 2), each being (x, y)
    :param H: homography, shape (3, 3), from image 0 to image 1
    :param thresh: threshold in pixel
    :return: matchine score (%)
    """
    N0 = descrs0.shape[0]
    N1 = descrs1.shape[0]

    # matches points from image 0 to image 1, using NN
    idxs = nn_match(descrs0, descrs1)  # matched image 1 keypoint indices
    predicted = kps1[idxs]                                 # matched image 1 keypoints

    # ground truth matched location
    gt = cv2.perspectiveTransform(kps0.reshape(N0, 1, 2).astype(np.float), H).reshape(N0, 2)
    correct0 = np.linalg.norm(predicted - gt, 2, axis=1) <= thresh

    # 1 to 0
    idxs = nn_match(descrs1, descrs0)
    predicted = kps0[idxs]
    gt = cv2.perspectiveTransform(kps1.reshape(N1, 1, 2).astype(np.float), np.linalg.inv(H)).reshape(N1, 2)
    correct1 = np.linalg.norm(predicted - gt, 2, axis=1) <= thresh

    return (np.sum(correct1) + np.sum(correct0)) / (N1 + N0)


def sift_detector(img):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kps = sift.detect(img, None)
    kps_np = np.asarray([kp.pt for kp in kps])
    return np.round(kps_np).astype(np.int32)


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


def get_wrong_pixel_scale(descs0, kps0, descs1, kps1, image, H, thresh=3):
    """
    get the wrongly matched pixels of left image when matching it to right image
    """
    if len(descs0.shape) == 3:
        idxs = nn_set2set_match(descs0, descs1)
    else:
        idxs = nn_match(descs0, descs1)  # matched image 1 keypoint indices
    predicted = kps1[idxs]
    wrong = np.linalg.norm(predicted - kps1, 2, axis=1) > thresh
    # pixels = kps0.detach().cpu().long().numpy()[wrong]
    pixels = kps0.astype(np.int)[wrong]

    h, w = image.shape[1:]
    scale = compute_scale(H, h, w)
    scales = scale[pixels[:, 1], pixels[:, 0]]
    return scales, pixels
