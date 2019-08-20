import os
from skimage import io
import numpy as np
import cv2
import glob

from lib.utils.base import convert_to_rgb, resize_max_length, draw_corspd, draw_triplet, draw_kps, \
    draw_corspd_region, warp_img_by_pts


class Hpatches(object):
    def __init__(self, root):
        super(Hpatches, self).__init__()
        self.img_paths = get_img_path(root)

    def __getitem__(self, index):
        """
        Note that unfortunately we have to reisze all images to 640x480 to allow
        batch training
        """
        img0_path, img1_path, H_path = self.img_paths[index]
        img0 = read_img(img0_path)
        img1 = read_img(img1_path)

        # this information is needed for rescaling H
        H = read_H(H_path)

        return img0, img1, H


    def __len__(self):
        return len(self.img_paths)


def read_H(H_path, scale_ratio=None):
    H = np.loadtxt(H_path).astype(np.float32)

    if scale_ratio is not None:
        scale_h0, scale_w0, scale_h1, scale_w1 = scale_ratio
        H = np.diag([scale_w1, scale_h1, 1.0]).dot(H).dot(np.linalg.inv(np.diag([scale_w0, scale_h0, 1.0])))

    return H


def read_img(img_path, scale=None):
    img = convert_to_rgb(io.imread(img_path))
    if scale is None:
        return img

    oh, ow = img.shape[:2]
    if isinstance(scale, int):
        img = resize_max_length(img, scale)
    else:
        img = cv2.resize(img, scale)
    h, w = img.shape[:2]
    scale_h, scale_w = h / oh, w / ow
    return img, scale_h, scale_w


def get_img_path(root):
    filelist = []
    subdirs = glob.glob(os.path.join(root, 'v_*'))

    for subdir in subdirs:
        ref = os.path.join(subdir, '1.ppm')
        for i in range(2, 6 + 1):
            H = os.path.join(subdir, 'H_1_{}'.format(i))
            other = os.path.join(subdir, '{}.ppm'.format(i))
            filelist.append((ref, other, H))

    return filelist
