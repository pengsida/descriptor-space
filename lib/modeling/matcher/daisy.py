import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
from lib.utils.misc import unnormalize, totensor_batch, tonumpy_batch, touint8


class Daisy(nn.Module):
    def __init__(self, cfg=None):
        nn.Module.__init__(self)
        self.feature_extractor = DaisyFeature()
        
    
    def forward(self, images0, images1, targets=None):
        """
        images0: [N, 3, H, W]
        images1: [N, 3, H, W]
        targets: {"img2": [N, 3, H, W], "kps0": [N, 3000], "kps1": [N, 3000], "kps2": [N, 3000]}
        """
        descs0 = self.feature_extractor(images0)
        descs1 = self.feature_extractor(images1)
        
        results = dict(
            descrs0=descs0,
            descrs1=descs1,
        )
        
        return results
        
    
    def inference(self, images, scale=1):
        return self.feature_extractor(images)


class DaisyFeature(nn.Module):
    def __init__(self, cfg=None):
        nn.Module.__init__(self)
        self.daisy = cv2.xfeatures2d.DAISY_create();
    
    
    def forward(self, images):
        """
        images0: [N, 3, H, W]
        images1: [N, 3, H, W]
        targets: {"img2": [N, 3, H, W], "kps0": [N, 3000], "kps1": [N, 3000], "kps2": [N, 3000]}
        """
        transform = lambda x: touint8(tonumpy_batch(unnormalize(x)))
        desc_trans = lambda x: totensor_batch(x)
        
        images = transform(images)
        
        N, H, W, C = images.shape
        
        descs = []
        
        for i in range(N):
            # shape (2, H, W)
            keypoints = np.mgrid[:H, :W]
            # reverse x and y, reshape to (H*W, 2)
            keypoints = keypoints[::-1].transpose(1, 2, 0).reshape(-1, 2)
            # opencv keypoints
            keypoints = [cv2.KeyPoint(x, y, 0) for (x, y) in keypoints]
            
            _, desc =  self.daisy.compute(images[i], keypoints)
            desc = desc.reshape(H, W, -1)
            
            descs.append(desc)
        
        # recombine into batches
        descs = desc_trans(np.stack(descs, axis=0))
        descs = F.normalize(descs)
        
        return descs
    
    
