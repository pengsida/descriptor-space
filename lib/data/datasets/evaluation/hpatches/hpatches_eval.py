import cv2
import numpy as np
import torch

from lib.utils.base import draw_corspd, sample_descriptor, SampleTrainingTarget, compute_scale, \
    get_homography_correspondence, sample_pyramid_descriptor, sample_scale
from lib.utils.nn_set2set_match.nn_set2set_match_layer import nn_set2set_match_cuda, nn_set2set_match_v1_cuda, \
    nn_match_cuda
from lib.utils.visualize import draw_img_desc_torch, draw_img_kps_torch
from lib.utils.nn_linear_match.nn_linear_match import nn_linear_interpol_match_numpy


class HpatchesEvaluator(object):
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

        if len(desc0.shape) == 4:
            # self.linear_search_scale(desc0, image0, desc1, image1, target)
            descs0 = sample_pyramid_descriptor(desc0, kps0, image0)
            descs1 = sample_pyramid_descriptor(desc1, kps1, image1)
            # nn_linear_interpol_match_numpy(descs0.detach().cpu().numpy(), descs1.detach().cpu().numpy())
        else:
            descs0 = sample_descriptor(desc0.unsqueeze(0), kps0.unsqueeze(0), image0.unsqueeze(0))[0]
            descs1 = sample_descriptor(desc1.unsqueeze(0), kps1.unsqueeze(0), image1.unsqueeze(0))[0]

        # Visualization
        # def tonumpy(*args):
        #     return [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in args]
        # [kps0, kps1, H] = tonumpy(kps0, kps1, H)
        # self.vis(image0, image1, desc0, desc1, descs0, descs1, kps0, kps1, H)

        # self.scales.append(get_wrong_pixel_scale(descs0, kps0, descs1, kps1, image0, H))
        self.matching_scores.append(compute_pck(descs0, kps0, descs1, kps1))

    def average_precision(self):
        print("matching score: {}".format(np.mean(self.matching_scores)))

    def wrong_pixel_scale(self):
        if len(self.scales) == 0:
            return
        import matplotlib.pyplot as plt
        plt.hist(np.concatenate(self.scales), bins=100, range=(0, 2))
        plt.show()

    def linear_search_scale(self, prd_desc0, image0, prd_desc1, image1, target):
        """
        prd_desc0: [2, D, H0', W0']
        image0: [3, H0, W0]
        prd_desc1: [2, D, H1', W1']
        image1: [3, H1, W1]
        target: ['kps0', 'scale0', 'kps1', 'scale1']
        """
        import matplotlib.pyplot as plt

        kps0 = target['kps0']
        kps1 = target['kps1']
        scale0 = target['scale0']
        scale1 = target['scale1']
        H = target['H']

        descs0 = sample_pyramid_descriptor(prd_desc0, target['kps0'], image0)
        descs1 = sample_pyramid_descriptor(prd_desc1, target['kps1'], image1)
        linear_search_pck = compute_pck(descs0, target['kps0'], descs1, target['kps1'])
        linear_scales_wrong, linear_pix_wrong = get_wrong_pixel_scale(descs0, kps0.numpy(), descs1, kps1.numpy(), image1, H.numpy())
        linear_scales_wrong = 1. / linear_scales_wrong
        scale_idxs = nn_set2set_match_v1(descs0, descs1)

        scales0 = sample_scale(target['scale0'].unsqueeze(0), target['kps0'].unsqueeze(0), image0.unsqueeze(0))[0]
        scales1 = sample_scale(target['scale1'].unsqueeze(0), target['kps1'].unsqueeze(0), image1.unsqueeze(0))[0]

        # compute assigned scale order
        discrete_scales0 = scales0.clone().long()
        end1 = 1.5
        start1 = 1. / end1
        discrete_scales0[(scales0 > start1) * (scales0 < end1)] = 0
        discrete_scales0[scales0 < start1] = 1
        discrete_scales0[scales0 > end1] = 2

        # manually select descriptor
        n = descs0.shape[0]
        s_descs0 = descs0[torch.arange(n), (scales0 > end1).long()]
        s_descs1 = descs1[torch.arange(n), (scales1 > end1).long()]
        manual_assign_pck = compute_pck(s_descs0, target['kps0'], s_descs1, target['kps1'])
        scales_wrong, pix_wrong = get_wrong_pixel_scale(s_descs0, kps0.numpy(), s_descs1, kps1.numpy(), image1, H.numpy())
        scales_wrong = 1. / scales_wrong

        print(linear_search_pck, manual_assign_pck)

        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.hist(linear_scales_wrong, bins=100)
        # ax2.hist(scales_wrong, bins=100)
        # plt.show()
        # import ipdb; ipdb.set_trace()

        # _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        #
        # ax1.hist(scale_idxs, bins=100, range=(0, 2))
        # text = 'pck: {:.2f}'.format(linear_search_pck)
        # ax1.annotate(text, xy=(0, 1), xycoords='axes fraction')
        #
        # ax2.hist(discrete_scales0, bins=100, range=(0, 2))
        # text = 'pck: {:.2f}'.format(manual_assign_pck)
        # ax2.annotate(text, xy=(0, 1), xycoords='axes fraction')
        #
        # text = 'min: {:.2f}, max: {:.2f}'.format(min(scales0), max(scales0))
        # ax3.hist(scales0, bins=100)
        # ax3.annotate(text, xy=(0, 1), xycoords='axes fraction')
        # plt.show()

        
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
    indices = nn_match_cuda(descs1.unsqueeze(0).cuda(), descs2.unsqueeze(0).cuda()).detach().cpu().long()
    # flann = cv2.FlannBasedMatcher_create()
    # matches = flann.match(descs1.astype(np.float32), descs2.astype(np.float32))
    # indices = [x.trainIdx for x in matches]
    return indices[0]


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


def nn_set2set_match_v1(descs1, descs2):
    idxs, scale_idxs = nn_set2set_match_v1_cuda(descs1.unsqueeze(0).cuda(), descs2.unsqueeze(0).cuda())
    idxs = idxs.detach().cpu().long()
    scale_idxs = scale_idxs.detach().cpu().long()
    return scale_idxs


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
    and their scale ratios of kps1 to kps0
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
