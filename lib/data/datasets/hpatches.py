import torch
from torch.utils.data import Dataset
import os
from skimage import io
import numpy as np
import cv2
import random
from lib.utils.base import convert_to_rgb, resize_max_length, draw_corspd, draw_triplet, draw_kps, draw_corspd_region, \
    compute_scale, compute_scale_xy, draw_scale, get_homography_correspondence

from .coco import sample_training_targets




class HpatchesDataset(Dataset):
    def __init__(self, cfg, root, transforms=None):
        super(HpatchesDataset, self).__init__()
        self.img_paths = get_img_path(root)
        # random.shuffle(self.img_paths)
        self.transforms = transforms
        self.max = 0
        self.index = 0

    def __getitem__(self, index):
        """
        Note that unfortunately we have to reisze all images to 640x480 to allow
        batch training
        """
        # index = 11
        img0_path, img1_path, H_path = self.img_paths[index]
        # img0 = convert_to_rgb(io.imread(img0_path))
        # img1 = convert_to_rgb(io.imread(img1_path))
        img0, scale_h0, scale_w0 = read_img(img0_path, (240, 240))
        img1, scale_h1, scale_w1 = read_img(img1_path, (240, 240))

        # this information is needed for rescaling H
        # H = read_H(H_path)
        scale_ratio = scale_h0, scale_w0, scale_h1, scale_w1
        H = read_H(H_path, scale_ratio)

        # compute the pixel scale ratio of img1 to img0 for each pixel in img1
        h, w = img1.shape[:2]
        scale = compute_scale(np.linalg.inv(H), h, w)
        scale = 1. / scale

        # compute the mask that indicates region having corresponding pixels
        _, msk = get_homography_correspondence(h, w, np.linalg.inv(H))
        msk = msk.astype(np.float32)

        # compute the pixel scale ratio of img0 to img1 for each pixel in img0
        h, w = img0.shape[:2]
        left_scale = compute_scale(H, h, w)
        left_scale = 1. / left_scale

        # generate training targets, positive and negative
        targets = sample_training_targets(img0, img1, H)
        targets['H'] = H
        targets['scale'] = torch.tensor(scale).unsqueeze(0)
        targets['left_scale'] = torch.tensor(left_scale).unsqueeze(0)
        targets['msk'] = torch.tensor(msk).unsqueeze(0)

        # img = draw_corspd_region(img0, img1, H)
        # import matplotlib.pyplot as plt
        # _, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.imshow(img)
        # ax2.imshow(draw_scale(left_scale))
        # ax3.imshow(draw_scale(scale))
        # plt.show()

        # img = draw_kps(img0, targets['kps0'], img1, targets['kps1'])
        # import matplotlib.pyplot as plt
        # plt.imshow(np.concatenate([img0, img], axis=1))
        # plt.show()

        # convert target points to tensor
        targets = {k: torch.tensor(v).float() for k, v in targets.items()}

        if self.transforms is not None:
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)

        img0 = torch.tensor(img0).permute(2, 0, 1).float()
        img1 = torch.tensor(img1).permute(2, 0, 1).float()

        return img0, img1, targets, index

    def __len__(self):
        return len(self.img_paths)

    def get_data(self, index, data_name, **kwargs):
        if data_name == "img0":
            img_path = self.img_paths[index][0]
            return read_img(img_path, (240, 240))
        elif data_name == "img1":
            img_path = self.img_paths[index][1]
            return read_img(img_path, (240, 240))
        elif data_name == "H":
            H_path = self.img_paths[index][2]
            scale_ratio = kwargs["scale_ratio"]
            return read_H(H_path, scale_ratio)

        raise NotImplementedError("No action to get the data {}".format(data_name))


class HpatchesViewpoint(HpatchesDataset):
    def __init__(self, *args, **kargs):
        HpatchesDataset.__init__(self, *args, **kargs)
        # only viewpoint changes
        self.img_paths = [x for x in self.img_paths if '/v' in x[0]]

class HpatchesIllum(HpatchesDataset):
    def __init__(self, *args, **kargs):
        HpatchesDataset.__init__(self, *args, **kargs)
        # only illumination changes
        self.img_paths = [x for x in self.img_paths if '/i' in x[0]]



def read_H(H_path, scale_ratio=(1, 1, 1, 1)):
    scale_h0, scale_w0, scale_h1, scale_w1 = scale_ratio
    H = np.loadtxt(H_path).astype(np.float32)
    H = np.diag([scale_w1, scale_h1, 1.0]).dot(H).dot(np.linalg.inv(np.diag([scale_w0, scale_h0, 1.0])))
    return H


def read_img(img_path, scale):
    img = convert_to_rgb(io.imread(img_path))
    oh, ow = img.shape[:2]
    if isinstance(scale, int):
        img = resize_max_length(img, scale)
    elif isinstance(scale, tuple):
        img = cv2.resize(img, scale)
    h, w = img.shape[:2]
    scale_h, scale_w = h / oh, w / ow
    return img, scale_h, scale_w


def get_img_path(root):
    filelist = []
    for subdir in os.scandir(root):
        subdir = subdir.path
        ref = os.path.join(subdir, '1.ppm')
        for i in range(2, 6 + 1):
            H = os.path.join(subdir, 'H_1_{}'.format(i))
            other = os.path.join(subdir, '{}.ppm'.format(i))
            filelist.append((ref, other, H))

    return filelist

