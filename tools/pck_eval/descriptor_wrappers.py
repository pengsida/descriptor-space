import cv2
import numpy as np

class DaisyWrapper:
    def __init__(self):
        self.daisy = cv2.xfeatures2d.DAISY_create(norm=cv2.xfeatures2d.DAISY_NRM_FULL)
    
    def __str__(self):
        return 'DAISY'
    
    def compute(self, img):
        """
        Compute dense descriptors
        
        :param img: shape (H, W, C), RGB
        :return: descriptors, shape (H, W, D)
        """
        
        h, w = img.shape[:2]
        # shape (2, H, W)
        keypoints = np.mgrid[:h, :w]
        # reverse x and y, reshape to (H*W, 2)
        keypoints = keypoints[::-1].transpose(1, 2, 0).reshape(-1, 2)
        # opencv keypoints
        keypoints = [cv2.KeyPoint(x, y, 0) for (x, y) in keypoints]
        
        kpts, desc =  self.daisy.compute(img, keypoints)
        return desc.reshape(h, w, -1)
    
if __name__ == '__main__':
    pass
