import numpy as np
import cv2

import sys
sys.path.extend(['.', '..'])

from lib.config.path import DatasetCatalog
from tools.statistics.hpatches_dataset import Hpatches
from lib.utils.base import warp_img_by_pts, compute_scale, compute_scale_xy, get_homography_correspondence


class DataStatistician(object):
    def __init__(self, dataset_name):
        root = DatasetCatalog.get(dataset_name)['args']['root']
        self.dataset = Hpatches(root)

    @staticmethod
    def compute_control_pts(img0, img1, H):
        h1, w1 = img1.shape[:2]
        pts1 = np.array(
            [[0, 0],
             [0, h1],
             [w1, h1],
             [w1, 0]]
        ).astype(np.float32)
        pts2 = cv2.perspectiveTransform(np.reshape(pts1, [1, -1, 2]), np.linalg.inv(H))[0]
        h0, w0 = img0.shape[:2]
        pts2 /= np.array([[w0, h0]])
        return pts2

    @staticmethod
    def validate_warped_img(img, pts):
        print(pts)
        warped_img = warp_img_by_pts(img, pts)
        img = np.concatenate([img, warped_img], axis=1)
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()

    def get_dataset_control_pts(self):
        pts_set = []
        for img0, img1, H in self.dataset:
            pts = DataStatistician.compute_control_pts(img0, img1, H)
            pts_set.append(pts)
        pts_set = np.array(pts_set)

        for pts in pts_set:
            DataStatistician.validate_warped_img(img0, pts)

    def get_dataset_scale_range(self):
        scale_set = []
        for img0, img1, H in self.dataset:
            # compute the scale changes
            h1, w1 = img1.shape[:2]
            scale = compute_scale(np.linalg.inv(H), h1, w1)
            scale = 1. / scale
            # compute the valid pixels
            _, msk = get_homography_correspondence(h1, w1, np.linalg.inv(H))
            scale_set.append(scale[msk])
        scale_set = np.concatenate(scale_set)

        import matplotlib.pyplot as plt
        plt.hist(scale_set, bins=100)
        plt.show()


if __name__ == '__main__':
    statistician = DataStatistician('HPATCHES')
    statistician.get_dataset_scale_range()

