import torch
from torch.utils.data import Dataset
import os
import glob
from skimage import io
import cv2
import numpy as np
from scipy.stats import truncnorm

from lib.utils.base import convert_to_rgb, compute_homography_discrete, round_coordinates, random_crop, \
    draw_corspd, draw_kps, draw_corspd_region, compute_scale, draw_scale, get_homography_correspondence


class COCOPerspective(Dataset):
    def __init__(self, cfg, root, transforms=None):
        # super(COCODatasetV3, self).__init__()
        Dataset.__init__(self)
        self.img_paths = get_img_path(root)
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = io.imread(img_path)
        img = convert_to_rgb(img)

        h, w = (240, 240)
        img0 = cv2.resize(img.copy(), (w, h), interpolation=cv2.INTER_LINEAR)
        # img0 = random_crop(img, size=(h, w))
        img1, H, scale = homography_adaption(img0)

        if np.random.uniform(0, 1) > 0.5:
            # swap two images
            img = img0
            img0 = img1
            img1 = img
            H = np.linalg.inv(H)
            h, w = img1.shape[:2]
            scale = compute_scale(np.linalg.inv(H), h, w)
            scale = 1. / scale

        pix_pos0, pix_pos1 = sample_ground_truth(img0, H)
        _, msk = get_homography_correspondence(h, w, np.linalg.inv(H))
        msk = msk.astype(np.float32)

        # img = draw_corspd_region(img0, img1, H)
        # import matplotlib.pyplot as plt
        # print(scale)
        # plt.imshow(np.concatenate([img, draw_scale(scale)], axis=1))
        # plt.show()

        # img = draw_kps(img0, pix_pos0, img1, pix_pos1)
        # import matplotlib.pyplot as plt
        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img)
        # ax2.imshow(msk)
        # plt.show()

        if self.transforms is not None:
            img0, pix_pos0 = self.transforms(img0, pix_pos0)
            img1, pix_pos1 = self.transforms(img1, pix_pos1)

        pix_pos2 = sample_negative(img1, pix_pos1)

        img0 = torch.tensor(img0).permute(2, 0, 1).float()
        img1 = torch.tensor(img1).permute(2, 0, 1).float()

        pix_pos0 = torch.tensor(pix_pos0).float()
        pix_pos1 = torch.tensor(pix_pos1).float()
        target = dict(
            kps0=pix_pos0,
            kps1=pix_pos1,
            kps2=pix_pos2,
            H=H,
            scale=scale,
            msk=msk
        )

        return img0, img1, target, index

    def __len__(self):
        return len(self.img_paths)

def sample_training_targets(img0, img1, H):
    """
    Sample positive correspondences and negative correspondences.

    :param img0: shape (H, W, C)
    :param img1: sahpe (H, W, C)
    :param H: homography from img0 to img1
    :return target: a dictionary {'kps0': kpts0, 'kpts1': kpts1, 'kpts2': kpts2}
    """

    pix_pos0, pix_pos1 = sample_ground_truth(img0, H)

    pix_pos2 = sample_negative(img1, pix_pos1)

    return {
        'kps0': pix_pos0,
        'kps1': pix_pos1,
        'kps2': pix_pos2
    }


def sample_negative(img, pix_pos):
    """
    img: [H, W, 3]
    pix_pos: [N, 2]
    """
    h, w = img.shape[:2]
    pix_pos1 = pix_pos.copy()
    pix_x = np.random.randint(5, w-5, len(pix_pos1))
    pix_y = np.random.randint(5, h-5, len(pix_pos1))
    pix_pos1[:, 0] = (pix_pos1[:, 0] + pix_x) % w
    pix_pos1[:, 1] = (pix_pos1[:, 1] + pix_y) % h
    return pix_pos1


def uniform_sample_correspondence(img, pix_pos, msk):
    hs_ws = np.argwhere(msk)
    sample_num = 3000
    if len(hs_ws) >= sample_num:
        inds = np.random.choice(np.arange(len(hs_ws)), sample_num, replace=False)
    else:
        inds = np.arange(len(hs_ws))
    hs_ws = hs_ws[inds]
    hs = hs_ws[:, 0]
    ws = hs_ws[:, 1]
    pos_num = len(hs)

    if pos_num >= sample_num:
        idxs = np.arange(pos_num)
        np.random.shuffle(idxs)
        idxs = idxs[:sample_num]
    else:
        idxs = np.arange(pos_num)
        idxs = np.append(idxs, np.random.choice(idxs, sample_num - pos_num))

    pix_pos0 = np.concatenate([ws[idxs][:, None], hs[idxs][:, None]], 1)  # sn,2
    pix_pos1 = pix_pos[pix_pos0[:, 1], pix_pos0[:, 0]]

    return pix_pos0, pix_pos1


def sample_correspondence(img, pix_pos, msk):
    val_msk = []

    harris_img = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), 2, 3, 0.04)
    harris_msk = harris_img > np.percentile(harris_img.flatten(), 90)
    val_msk.append(harris_msk)

    val_msk_out = np.zeros_like(msk)
    for item in val_msk:
        val_msk_out = np.logical_or(val_msk_out, item)
    val_msk = val_msk_out
    val_msk = np.logical_and(msk, val_msk)

    hs, ws = np.nonzero(val_msk)
    pos_num = len(hs)

    if pos_num == 0:
        return uniform_sample_correspondence(img, pix_pos, msk)

    sample_num = 3000
    if pos_num >= sample_num:
        idxs = np.arange(pos_num)
        np.random.shuffle(idxs)
        idxs = idxs[:sample_num]
    else:
        idxs = np.arange(pos_num)
        idxs = np.append(idxs, np.random.choice(idxs, sample_num - pos_num))

    pix_pos0 = np.concatenate([ws[idxs][:, None], hs[idxs][:, None]], 1)  # sn,2
    pix_pos1 = pix_pos[pix_pos0[:, 1], pix_pos0[:, 0]]
    return pix_pos0, pix_pos1


def get_homography_correspondence(h, w, H):
    coords = [np.expand_dims(item, 2) for item in np.meshgrid(np.arange(w), np.arange(h))]
    coords = np.concatenate(coords, 2).astype(np.float32)
    coords_target = cv2.perspectiveTransform(np.reshape(coords, [1, -1, 2]), H.astype(np.float32))
    coords_target = np.reshape(coords_target, [h, w, 2])

    source_mask = np.logical_and(np.logical_and(0 <= coords_target[:, :, 0], coords_target[:, :, 0] < w - 0),
                                 np.logical_and(0 <= coords_target[:, :, 1], coords_target[:, :, 1] < h - 0))
    coords_target[np.logical_not(source_mask)] = 0

    return coords_target, source_mask


def sample_ground_truth(img, H):
    h,w,_=img.shape
    pix_pos, msk = get_homography_correspondence(h, w, H)
    pix_pos0, pix_pos1 = uniform_sample_correspondence(img, pix_pos, msk)
    return pix_pos0, pix_pos1


def generate_homography(img):
    h, w = img.shape[:2]
    H, scale_offset = compute_homography_discrete(h, w)
    return H, scale_offset


def perspective(pts2, margin, h, w):
    patch_ratio = 1 - margin * 2

    # perspective distortion
    # when the patch_ratio is near 1, the scale change is low and we allow more perspective distortion
    # when the patch_ratio is near 0.5, the scale change is high and we limit the perspective distortion range
    perspective_amplitude_x = (patch_ratio / 2. - 0.2) * w
    perspective_amplitude_y = (patch_ratio / 2. - 0.2) * h

    perspective_displacement = truncnorm.rvs(-2., 2., loc=0., scale=perspective_amplitude_y/2)
    h_displacement_left = truncnorm.rvs(-2., 2., loc=0., scale=perspective_amplitude_x/2)
    h_displacement_right = truncnorm.rvs(-2., 2., loc=0., scale=perspective_amplitude_x/2)

    # perspective_displacement = -51.95856738659387
    # h_displacement_left = 20.431047717231696
    # h_displacement_right = 36.05068176678458
    pts2 += np.array(
        [[h_displacement_left, perspective_displacement],
         [h_displacement_left, -perspective_displacement],
         [h_displacement_right, perspective_displacement],
         [h_displacement_right, -perspective_displacement]]
    )
    return pts2


def scale(pts2, h, w):
    # scale transformation
    scaling_amplitude = 0.2
    lower = ((1 - scaling_amplitude) - 1) * 2. / scaling_amplitude
    upper = ((1 + scaling_amplitude) - 1) * 2. / scaling_amplitude
    scale_ratio = truncnorm.rvs(lower, upper, loc=1., scale=scaling_amplitude/2.)
    center = np.mean(pts2, axis=0)
    pts2 = (pts2 - center) * scale_ratio + center
    return pts2


def translation(pts2, h, w):
    # translation transformation
    t_min = np.min(pts2, axis=0)
    t_max = np.min(np.array([w, h]) - pts2, axis=0)
    pts2 += [np.random.uniform(-t_min[0], t_max[0]), np.random.uniform(-t_min[1], t_max[1])]
    return pts2


def rotation(pts2, h, w):
    # rotation transformation
    max_angle = np.pi / 6.
    angle = truncnorm.rvs(-2., 2., loc=0., scale=max_angle/2.)
    center = np.mean(pts2, axis=0)
    rot_mat = np.array(
        [[np.cos(angle), -np.sin(angle)],
         [np.sin(angle), np.cos(angle)]]
    )
    pts2 = np.matmul(pts2 - center, rot_mat) + center
    return pts2


def homography(h, w):
    pts1 = np.array(
        [[0., 0.],
         [0., 1.],
         [1., 1.],
         [1., 0.]]
    )

    patch_ratio = np.random.uniform(0.7, 0.9)
    margin = (1 - patch_ratio) / 2.
    patch_ratio = 0.5
    pts2 = margin + np.array(
        [[0, 0],
         [0, patch_ratio],
         [patch_ratio, patch_ratio],
         [patch_ratio, 0]]
    )

    shape = np.array([[w, h]]).astype(np.float32)
    pts1 *= shape
    pts2 *= shape

    pts2 = perspective(pts2, margin, h, w)
    # pts2 = scale(pts2, h, w)
    # pts2 = translation(pts2, h, w)
    # pts2 = rotation(pts2, h, w)

    return cv2.getPerspectiveTransform(pts2.astype(np.float32), pts1.astype(np.float32)), patch_ratio


def homography_adaption(img0):
    h, w = img0.shape[:2]
    H, patch_ratio = homography(h, w)
    img1 = cv2.warpPerspective(img0, H, (w, h), flags=cv2.INTER_LINEAR)
    scale_ratio = compute_scale(np.linalg.inv(H), h, w)
    scale_ratio = 1. / scale_ratio
    return img1, H, scale_ratio


def get_img_path(root):
    img_paths = []
    img_paths += glob.glob(os.path.join(root, "train2017/*"))
    img_paths += glob.glob(os.path.join(root, "val2017/*"))
    return img_paths
