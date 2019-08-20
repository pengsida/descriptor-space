"""
File detector_wrappers.py

Various detector wrappers for hpatches evaluation.
"""
import cv2
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import os

from tools.hpatches_eval.superpoint.superpoint import SuperPointFrontend
from lib.config import cfg
from lib.modeling.matcher import build_matching_model
from lib.utils.checkpoint import Checkpointer
from lib.data.transforms import build_transforms
from lib.utils.base import convert_to_rgb, sift_detector

class Wrapper:
    def detect(self, img):
        """
        feature detection interface
        
        :param img0, img1: the two images, shape (H, W, C)
        :return: keypoints, descriptors, shape (N, 2) and (N, D)
        """
        raise NotImplementedError
    

class OrbWrapper:
    """
    Wrapper for ORB feature detector and descriptor.
    """
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        
    def detect(self, img):
        kpts, descs = self.orb.detectAndCompute(img, None)
        kpts = np.array([kpt.pt for kpt in kpts])
        return kpts, descs.astype(np.float32)
    
    def __str__(self):
        return 'ORB'
    
class SiftWrapper:
    """
    Wrapper for SIFT feature detector and descriptor.
    """
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
    
    def detect(self, img):
        kpts, descs = self.sift.detectAndCompute(img, None)
        kpts = np.array([kpt.pt for kpt in kpts])

        # sorted_pairs = sorted(zip(kpts, descs), key=lambda x: x[0].response, reverse=True)

        # kpts = np.array([x[0].pt for x in sorted_pairs])
        # descs = np.array([x[1] for x in sorted_pairs])
        
        return kpts, descs.astype(np.float32)
    
    def __str__(self):
        return 'SIFT'
    
class SurfWrapper:
    """
    Wrapper for SURF feature detector and descriptor
    """

    def __init__(self, nfeatures=1000):
        self.surf = cv2.xfeatures2d.SURF_create()
        self.nfeatures = nfeatures

    def detect(self, img):
        kpts, descs = self.surf.detectAndCompute(img, None)
        
        # choose 1000 points
        sorted_pairs = sorted(zip(kpts, descs), key=lambda x: x[0].response, reverse=True)

        kpts = np.array([x[0].pt for x in sorted_pairs])
        descs = np.array([x[1] for x in sorted_pairs])
        
        return kpts[:self.nfeatures], descs[:self.nfeatures].astype(np.float32)
    
    def __str__(self):
        return 'SURF'

class SuperPointWrapper:
    """
    Wrapper for superpoint feature detector and descriptor
    """
    def __init__(
            self,
            weights_path='tools/hpatches_eval/superpoint/superpoint_v1.pth',
            nms_dist=4,
            conf_thresh=0.015,
            nn_thresh=0.7
    ):
        self.superpoint = SuperPointFrontend(weights_path=weights_path,
                                nms_dist=nms_dist,
                                conf_thresh=conf_thresh,
                                nn_thresh=nn_thresh,
                                cuda=True)
        
    def __str__(self):
        return 'SuperPoint'
    
    def detect(self, img):
        img = img.astype(np.float32)
        img /= 255.0
        
        kpts, descs, _ = self.superpoint.run(img)
        kpts, descs = kpts.T, descs.T
        
        return kpts[:, :2], descs
    
        
class RandomDetector:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
    
    def __str__(self):
        return 'random'
    
    def detect(self, img):
        
        kpts = self.sift.detect(img, None)
        kpts = np.array([kpts.pt for kpts in kpts])
        N = kpts.shape[0]
        D = 128
        descs = np.random.rand(N, D)
        
        return kpts, descs
    

    

class MatterportWrapper:
    """
    Wrapper for SIFT feature detector and descriptor.
    """
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)

        parser = argparse.ArgumentParser(description="Dense Correspondence")
        parser.add_argument(
            "--config-file",
            default="",
            metavar="FILE",
            help="path to config file",
        )
        parser.add_argument(
            "opts",
            help="Modify config options using the command-line",
            default=None,
            nargs=argparse.REMAINDER,
        )

        args = parser.parse_args()

        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # cfg.freeze()

        model = build_matching_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        self.model = torch.nn.DataParallel(model)

        model_dir = os.path.join(cfg.MODEL_DIR, cfg.MODEL.NAME)
        checkpointer = Checkpointer(cfg, self.model, save_dir=model_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transform = build_transforms(cfg, is_train=False)

    def detect(self, img):
        kpts = sift_detector(img)
        ori_kpts = kpts.copy()

        img = self.transform(img)
        img = torch.tensor(img).permute(2, 0, 1).float().cuda()

        self.model.eval()
        with torch.no_grad():
            h, w = img.shape[1:]
            img = img.view(1, 3, h, w)
            descr = self.model.module.inference(img)
            descr = F.normalize(F.interpolate(descr, (h, w), mode="bilinear"), dim=1)

            descr = descr[0, :, kpts[:, 1], kpts[:, 0]].transpose(1, 0)
            descr = descr.cpu().numpy()

            # kpts = torch.from_numpy(kpts.copy())
            # kpts[:, 0] = (kpts[:, 0] / (float(w)/2.)) - 1.
            # kpts[:, 1] = (kpts[:, 1] / (float(h)/2.)) - 1.
            # kpts = kpts.view(1, 1, -1, 2)
            # kpts = kpts.float().cuda()
            #
            # descr = F.grid_sample(descr, kpts)
            # descr = descr.squeeze().transpose(1, 0)
            # descr = descr.cpu().numpy()

        kpts = ori_kpts
        descs = descr

        return kpts, descs.astype(np.float32)

    def __str__(self):
        return "Matterport"
