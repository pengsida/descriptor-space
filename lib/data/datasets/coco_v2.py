import torch
from torch.utils.data import Dataset
import os
import glob
from skimage import io
import cv2
import numpy as np
from scipy.stats import truncnorm

from lib.utils.base import convert_to_rgb, compute_homography_discrete, round_coordinates, random_crop, \
    draw_corspd, draw_kps, draw_corspd_region


class COCODatasetV2(Dataset):
    def __init__(self, cfg, root, transforms=None):
        super(COCODatasetV2, self).__init__()
        # self.img_paths = get_img_path(root)[:1] * 100
        self.img_paths = ['data/HPATCHES/v_graffiti/{}.ppm'.format(i) for i in range(1, 7)]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = io.imread(img_path)
        img = convert_to_rgb(img)

        img0 = img
        # img0 = random_crop(img, size=(240, 320))
        img1, H = homography_adaption(img0)
        pix_pos0, pix_pos1 = sample_ground_truth(img0, H)

        # img = draw_corspd_region(img0, img1, H)
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()

        if self.transforms is not None:
            img0, pix_pos0 = self.transforms(img0, pix_pos0)
            img1, pix_pos1 = self.transforms(img1, pix_pos1)

        pix_pos2 = sample_negative(img1, pix_pos1)

        img0 = torch.tensor(img0).permute(2, 0, 1).float()
        img1 = torch.tensor(img1).permute(2, 0, 1).float()

        # img = draw_triplet(img0, pix_pos0, img1, pix_pos1, img1, pix_pos2)
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
        # return

        pix_pos0 = torch.tensor(pix_pos0).float()
        pix_pos1 = torch.tensor(pix_pos1).float()
        target = dict(
            kps0=pix_pos0,
            kps1=pix_pos1,
            kps2=pix_pos2,
            H=H,
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
    inds = np.random.choice(np.arange(len(hs_ws)), 3000, replace=False)
    hs_ws = hs_ws[inds]
    hs = hs_ws[:, 0]
    ws = hs_ws[:, 1]
    pos_num = len(hs)

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


def perspective(pts2, margin):
    patch_ratio = 1 - margin * 2

    # perspective distortion
    # when the patch_ratio is near 1, the scale change is low and we allow more perspective distortion
    # when the patch_ratio is near 0.5, the scale change is high and we limit the perspective distortion range
    perspective_amplitude_x = patch_ratio / 2. - 0.2
    perspective_amplitude_y = patch_ratio / 2. - 0.2

    perspective_displacement = truncnorm.rvs(-2., 2., loc=0., scale=perspective_amplitude_y/2)
    h_displacement_left = truncnorm.rvs(-2., 2., loc=0., scale=perspective_amplitude_x/2)
    h_displacement_right = truncnorm.rvs(-2., 2., loc=0., scale=perspective_amplitude_x/2)
    pts2 += np.array(
        [[h_displacement_left, perspective_displacement],
         [h_displacement_left, -perspective_displacement],
         [h_displacement_right, perspective_displacement],
         [h_displacement_right, -perspective_displacement]]
    )
    return pts2


def scale(pts2):
    # scale transformation
    n_scales = 5
    scaling_amplitude = 0.1
    lower = ((1 - scaling_amplitude) - 1) * 2. / scaling_amplitude
    upper = ((1 + scaling_amplitude) - 1) * 2. / scaling_amplitude
    scales = truncnorm.rvs(lower, upper, loc=1., scale=scaling_amplitude/2., size=n_scales)
    center = np.mean(pts2, axis=0)
    scaled = np.expand_dims(pts2 - center, axis=0) * np.reshape(scales, (n_scales, 1, 1)) + center
    valid = np.all((scaled >= 0.) & (scaled <= 1.), axis=2).T
    idx = np.array([np.random.choice(np.argwhere(v).ravel()) for v in valid])
    pts2 = scaled[idx, np.arange(4)]
    return pts2


def translation(pts2):
    # translation transformation
    t_min = np.min(pts2, axis=0)
    t_max = np.min(1 - pts2, axis=0)
    pts2 += [np.random.uniform(-t_min[0], t_max[0]), np.random.uniform(-t_min[1], t_max[1])]
    return pts2


def rotation(pts2):
    # rotation transformation
    n_angles = 10
    max_angle = np.pi / 3.
    angles = np.linspace(-max_angle, max_angle, n_angles)
    center = np.mean(pts2, axis=0)
    rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)], axis=1), (-1, 2, 2))
    rotated = np.matmul(np.tile(np.expand_dims(pts2 - center, axis=0), (n_angles, 1, 1)), rot_mat) + center
    valid = np.all(((rotated >= 0.) & (rotated <= 1.)).reshape(n_angles, -1), axis=1)
    # idx = np.random.choice(np.argwhere(valid).ravel())
    idx = np.random.randint(0, len(rotated))
    pts2 = rotated[idx]
    return pts2


def homography(h, w):
    pts1 = np.array(
        [[0., 0.],
         [0., 1.],
         [1., 1.],
         [1., 0.]]
    )

    patch_ratio = np.random.uniform(0.5, 1)
    patch_ratio = 1
    margin = (1 - patch_ratio) / 2.
    pts2 = margin + np.array(
        [[0, 0],
         [0, patch_ratio],
         [patch_ratio, patch_ratio],
         [patch_ratio, 0]]
    )

    # pts2 = perspective(pts2, margin)
    # pts2 = scale(pts2)
    # pts2 = translation(pts2)
    pts2 = rotation(pts2)

    shape = np.array([[w, h]]).astype(np.float32)
    pts1 *= shape
    pts2 *= shape

    return cv2.getPerspectiveTransform(pts2.astype(np.float32), pts1.astype(np.float32))


def homography_adaption(img0):
    h, w = img0.shape[:2]
    H = homography(h, w)
    img1 = cv2.warpPerspective(img0, H, (w, h), flags=cv2.INTER_LINEAR)
    return img1, H


def get_img_path(root):
    img_paths = []
    img_paths += glob.glob(os.path.join(root, "train2017/*"))
    img_paths += glob.glob(os.path.join(root, "val2017/*"))
    return img_paths
