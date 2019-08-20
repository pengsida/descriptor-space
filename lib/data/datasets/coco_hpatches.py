import torch
from torch.utils.data import Dataset
import os
import glob
from skimage import io
import cv2
import numpy as np
from scipy.stats import truncnorm
from PIL import Image

from lib.utils.base import convert_to_rgb, compute_homography_discrete, round_coordinates, random_crop, \
    draw_corspd, draw_kps, draw_corspd_region, compute_scale, draw_scale, draw_scale_cls
from lib.utils.misc import tensor2RGB


class COCODatasetHpatches(Dataset):
    def __init__(self, cfg, root, transforms=None):
        super(COCODatasetHpatches, self).__init__()
        self.img_paths = get_img_path(root)
        self.H_paths = get_H_path('data/HPATCHES')
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        # H_path = self.H_paths[index]
        H_path = np.random.choice(self.H_paths)
        img = io.imread(img_path)
        img = convert_to_rgb(img)

        # img0 = img
        h, w = (240, 240)
        img0 = cv2.resize(img.copy(), (w, h), interpolation=cv2.INTER_LINEAR)
        # img0 = random_crop(img, size=(240, 240))
        H = read_H(H_path, img0)
        img1, scale = homography_adaption(img0, H)

        # the pixel scale change of img0 to img1 for each pixel in img0
        h, w = img0.shape[:2]
        left_scale = compute_scale(H, h, w)
        left_scale = 1. / left_scale

        pix_pos0, pix_pos1 = sample_ground_truth(img0, H)
        _, msk = get_homography_correspondence(h, w, np.linalg.inv(H))
        msk = msk.astype(np.float32)

        # img = draw_corspd_region(img0, img1, H)
        # import matplotlib.pyplot as plt
        # plt.imshow(np.concatenate([img, draw_scale(scale)], axis=1))
        # plt.show()

        # img = draw_kps(img0, pix_pos0, img1, pix_pos1)
        # import matplotlib.pyplot as plt
        # plt.imshow(np.concatenate([img0, img], axis=1))
        # plt.show()

        if self.transforms is not None:
            img0, pix_pos0 = self.transforms(img0, pix_pos0)
            img1, pix_pos1 = self.transforms(img1, pix_pos1)

        pix_pos2 = sample_negative(img1, pix_pos1)

        img0 = torch.tensor(img0).permute(2, 0, 1).float()
        img1 = torch.tensor(img1).permute(2, 0, 1).float()

        pix_pos0 = torch.tensor(pix_pos0).float()
        pix_pos1 = torch.tensor(pix_pos1).float()
        scale = torch.tensor(scale).unsqueeze(0)
        left_scale = torch.tensor(left_scale).unsqueeze(0)
        msk = torch.tensor(msk).unsqueeze(0)
        target = dict(
            kps0=pix_pos0,
            kps1=pix_pos1,
            kps2=pix_pos2,
            H=H,
            scale=scale,
            msk=msk,
            left_scale=left_scale
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
        # idxs = np.append(idxs, np.random.choice(idxs, sample_num - pos_num))

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
    pix_pos0, pix_pos1 = sample_correspondence(img, pix_pos, msk)
    return pix_pos0, pix_pos1


def homography_adaption(img0, H):
    h, w = img0.shape[:2]
    img1 = cv2.warpPerspective(img0, H, (w, h), flags=cv2.INTER_LINEAR)
    scale_ratio = compute_scale(np.linalg.inv(H), h, w)
    scale_ratio = 1. / scale_ratio
    return img1, scale_ratio


def get_img_path(root):
    img_paths = []
    img_paths += glob.glob(os.path.join(root, "train2017/*"))
    img_paths += glob.glob(os.path.join(root, "val2017/*"))
    return img_paths


def get_H_path(root):
    H_paths = glob.glob(os.path.join(root, 'v_*/H_*'))
    return H_paths


def read_H(H_path, img0):
    H_orig = np.loadtxt(H_path).astype(np.float32)
    basename = os.path.basename(H_path)
    i = int(basename[2])
    j = int(basename[-1])
    parent_name = os.path.dirname(H_path)

    img0_orig = Image.open(os.path.join(parent_name, '{}.ppm'.format(i)))
    img1_orig = Image.open(os.path.join(parent_name, '{}.ppm'.format(j)))
    h, w = img0.shape[:2]
    scale_w0, scale_h0 = np.array([w, h]) / img0_orig.size
    scale_w1, scale_h1 = np.array([w, h]) / img1_orig.size
    H = np.diag([scale_w1, scale_h1, 1.0]).dot(H_orig).dot(np.linalg.inv(np.diag([scale_w0, scale_h0, 1.0])))
    return H
