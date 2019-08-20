import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
from skimage import io

from lib.utils.base import convert_to_rgb, resize_max_length


class HpatchesDataset:
    """
    class HpatchesDataset:

    Provides evaluation interfaces used by evaluate_detector
    """
    def __init__(self, root):
        self.root = root
        self.filelist = self.__getfilelist()

    def __getitem__(self, index):
        img_path = self.filelist[index]
        img = convert_to_rgb(io.imread(img_path))
        img = resize_max_length(img, 240)
        img = img[:, :, 0].astype(np.float32)
        img /= 255.0
        return img

    def __len__(self):
        return len(self.filelist)

    def __getfilelist(self):
        filelist = []
        for subdir in os.scandir(self.root):
            subdir = subdir.path
            for i in range(1, 6 + 1):
                img_path = os.path.join(subdir, '{}.ppm'.format(i))
                filelist.append(img_path)

        return filelist
