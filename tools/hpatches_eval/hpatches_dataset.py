import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
from skimage import io, transform


class HpatchesDataset:
    """
    class HpatchesDataset:
    
    Provides evaluation interfaces used by evaluate_detector
    """
    
    IMAGE_SIZE = 640
    
    def __init__(self, root):
        self.root = root
        self.filelist = self.__getfilelist()
    
    def __getitem__(self, index):
        img0_path, img1_path, H_path = self.filelist[index]
        img0 = io.imread(img0_path)
        img1 = io.imread(img1_path)

        # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        # img1 = cv2.cvtcolor(img1, cv2.color_bgr2gray)
        
        H = np.loadtxt(H_path).astype(np.float)
        
        # rescale image such that largest dimension is IMAGE_SIZE
        img0, scale0 = rescale_image(img0, self.IMAGE_SIZE)
        img1, scale1 = rescale_image(img1, self.IMAGE_SIZE)
        
        # rescale H accordingly
        H = rescale_H(H, scale0, scale1)
        
        return img0, img1, H
    
    def __len__(self):
        return len(self.filelist)
    
    def __getfilelist(self):
        filelist = []
        subdirs = [subdir.path for subdir in os.scandir(self.root)]
        subdirs = sorted(subdirs)
        for subdir in subdirs:
            ref = os.path.join(subdir, '1.ppm')
            for i in range(2, 6 + 1):
                H = os.path.join(subdir, 'H_1_{}'.format(i))
                other = os.path.join(subdir, '{}.ppm'.format(i))
                filelist.append((ref, other, H))
        
        return filelist


def rescale_image(image, size):
    """
    Rescale
    :param size: after rescaling, max(h, w) will be size
    :param image: the image to resize, (H, W, C)
    :return: resized image, scale
    """
    
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    return cv2.resize(image, dsize=(new_w, new_h)), scale


def rescale_H(H, scale0, scale1):
    """
    Rescale H
    """
    H = H.copy()
    H = np.diag([scale1, scale1, 1.0]).dot(H).dot(np.diag([1 / scale0, 1 / scale0, 1.0]))
    
    return H
