"""
This file defines a logger object to keep everything. This object is like a
dictionary. To use the logger, use

    from lib.utils.logger import logger
    
and then log any data using
    
    logger.update(name0=value, name1=value, ...)

Various getters should also be defined here. These are config-dependent objects,
and should wrap the logger object. It performs config-dependent operations based
on data stored in logger.

To make a getter from a configuration object, use
    
    getter = make_getter(cfg)
    
To use the matterport getter to get data for visualization with Tensorboard,
call
    
    tb_data = getter.get_tensorboard_data()
"""

import torch
from torch.nn import functional as F
import numpy as np

from lib.utils.misc import tonumpy, totensor, unnormalize
from lib.utils.visualize import desc2RGB, draw_kps_torch, draw_corr, draw_corr_torch, draw_paired_img_desc_torch, \
    draw_paired_desc_torch, cls2RGB, draw_scale_torch
from lib.utils.misc import warp_torch, color_scale, toheapmap_torch


class Logger:
    """
    Interface class
    """
    def __init__(self):
        self.things = dict()
        
    def __getitem__(self, key):
        return self.things[key]
        
    def update(self, **kargs):
        # detach any tensor
        for k in kargs:
            if isinstance(kargs[k], torch.Tensor):
                kargs[k] = kargs[k].detach().cpu()
        self.things.update(kargs)
        
        
# global logger to keep literally everything
logger = Logger()

# getter maker
def make_getter(cfg):
    getter = None
    if cfg.GETTER.NAME == 'Matterport':
        return MatterportGetter()
    elif cfg.GETTER.NAME == 'MSNet':
        return MSNetGetter()
    elif cfg.GETTER.NAME == 'MSNetV9':
        return MSNetV9Getter()
    elif cfg.GETTER.NAME == 'MSNetV5':
        return MSNetV5Getter()
    elif cfg.GETTER.NAME == 'MSNetV11':
        return MSNetV11Getter()
    elif cfg.GETTER.NAME == 'MSNetV12':
        return MSNetV12Getter()
    
    return getter

class MatterportGetter:
    """
    Designed for matterport
    """
    
    def __init__(self, logger=logger):
        self.logger = logger
        
    def get_tensorboard_data(self, num_kps=20):
        """
        This processes the data needed for visualization. It expects the follow-
        ing in self.logger
        
        - image0: (C, H, W), Tensor, normalized
        - image1: (C, H, W), Tensor, normalized
        - desc0: (C, H, W)
        - desc1: (C, H, W)
        - kps0: (N, 2), each being (x, y)
        - kps1: (N, 2)
        - kps2: (N, 2), negative ones
        
        And it returns a dictionary
        
        - desc0: descriptor 1, RGB, (3, H, W)
        - desc1: descriptor 2, RGB, (3, H, W)
        - img0: image 1, (3, H, W)
        - img1: image 2, (3, H, W)
        - keypoints: the two images marked with num_kps keypoints
        - neg_keypoints: image 2 marked with negative keypoints
        - corr: ground truth correspondences
        - corr false: false correspondences
        """
        
        # original images
        image0 = self.logger['image0']
        image1 = self.logger['image1']
        
        # descriptors
        desc0 = self.logger['desc0']
        desc1 = self.logger['desc1']

        # keypoints
        kps0 = self.logger['kps0']
        kps1 = self.logger['kps1']
        kps2 = self.logger['kps2']
        
        # process the images
        image0 = unnormalize(image0)
        image1 = unnormalize(image1)
        
        # process the descriptor
        desc0, desc1 = [desc2RGB(tonumpy(x)) for x in [desc0, desc1]]
        desc0, desc1 = [totensor(d) for d in [desc0, desc1]]
        
        # choose keypoints
        N = kps0.shape[0]
        indices = np.random.choice(N, size=num_kps, replace=False)
        
        # draw keypoints
        kps = draw_kps_torch(image0, kps0[indices], image1, kps1[indices])
        
        # draw negative keypoints
        neg_kps = draw_kps_torch(image0, kps0[indices], image1, kps2[indices])
        
        # draw correspondences
        corr_gt = draw_corr_torch(image0, kps0[indices], image1, kps1[indices])

        # draw correspondences
        corr_false = draw_corr_torch(image0, kps0[indices], image1, kps2[indices])
        
        return {
            'img0': image0,
            'img1': image1,
            'desc0': desc0,
            'desc1': desc1,
            'keypoints': kps,
            'neg_keypoints': neg_kps,
            'corr': corr_gt,
            'corr_false': corr_false
        }

class MSNetGetter:
    """
    Designed for matterport
    """

    def __init__(self, logger=logger):
        self.logger = logger

    def get_tensorboard_data(self, num_kps=20):
        """
        This processes the data needed for visualization. It expects the follow-
        ing in self.logger

        - image0: (C, H, W), Tensor, normalized
        - image1: (C, H, W), Tensor, normalized
        - d03: (C, H, W)
        - d13: (C, H, W)
        - kps0: (N, 2), each being (x, y)
        - kps1: (N, 2)
        - kps2: (N, 2), negative ones

        And it returns a dictionary

        - d03: descriptor 1, RGB, (3, H, W)
        - d13: descriptor 2, RGB, (3, H, W)
        - img0: image 1, (3, H, W)
        - img1: image 2, (3, H, W)
        - keypoints: the two images marked with num_kps keypoints
        - neg_keypoints: image 2 marked with negative keypoints
        - corr: ground truth correspondences
        - corr false: false correspondences
        """

        # original images
        image0 = self.logger['image0']
        image1 = self.logger['image1']

        # descriptors
        d03 = self.logger['d03']
        d13 = self.logger['d13']

        # keypoints
        kps0 = self.logger['kps0']
        kps1 = self.logger['kps1']
        kps2 = self.logger['kps2']

        # homography matrix
        H = self.logger['H']

        # img = draw_img_desc_torch(
        #     image0, d03, kps0,
        #     image1, d13, kps1,
        #     H
        # )
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()

        # process the images
        image0 = unnormalize(image0)
        image1 = unnormalize(image1)

        # process the descriptor
        desc = draw_paired_desc_torch(d03, kps0, image0, d13, kps1, image1, H)

        # choose keypoints
        N = kps0.shape[0]
        indices = np.random.choice(N, size=num_kps, replace=False)

        # draw keypoints
        kps = draw_kps_torch(image0, kps0[indices], image1, kps1[indices])

        # draw negative keypoints
        neg_kps = draw_kps_torch(image0, kps0[indices], image1, kps2[indices])

        # draw correspondences
        corr_gt = draw_corr_torch(image0, kps0[indices], image1, kps1[indices])

        # draw correspondences
        corr_false = draw_corr_torch(image0, kps0[indices], image1, kps2[indices])

        return {
            'img0': image0,
            'img1': image1,
            'desc': desc,
            'keypoints': kps,
            'neg_keypoints': neg_kps,
            'corr': corr_gt,
            'corr_false': corr_false
        }
    
class MSNetV5Getter(MSNetGetter):
    def get_tensorboard_data(self, num_kps=20):
        """
        In addition to descritors and images, we also visualize
            - attention map
            - attention map, scale version
            - scale map
        """
        data = MSNetGetter.get_tensorboard_data(self)
        
        
        # map (C, H, W)
        map_left = self.logger['map_left']
        map_right = self.logger['map_right']
        
        # scale map (H, W), value range (0, infty)
        scale_map = self.logger['scale']
        
        # scales, a list of numbers
        scales = self.logger['scales']
        
        # mask (H, W)
        mask = self.logger['mask']
        
        # H, from first to second
        H = self.logger['H'].cpu().detach().numpy()
        
        # interpolate predicted scale map to full size
        h, w = self.logger['image0'].size()[-2:]
        # this weird thing is needed because interpolation accepts batch data
        [map_left, map_right] = [F.interpolate(x[None], size=(w, h), mode='bilinear')[0] for x in [map_left, map_right]]
        [map_left_color, map_right_color] = [color_scale(x) for x in [map_left, map_right]]
        
        
        # warp and devide, and mask
        # map_left_warped = warp_torch(map_left_pred, H)
        # scale_map_pred = map_right_pred / (map_left_warped + 1e-6)
        
        
        # mask not mapped regions
        # scale_map_pred *= mask
        
        # scale everything to (0, 1)
        for x in [scale_map]:
            x /= x.max()
        
        data['map_left'] = map_left
        data['map_left0'] = toheapmap_torch(map_left[0])
        data['map_left1'] = toheapmap_torch(map_left[1])
        data['map_left2'] = toheapmap_torch(map_left[2])
        data['map_left3'] = toheapmap_torch(map_left[3])
        data['map_right'] = map_right
        data['map_left_color'] = map_left_color
        data['map_right_color'] = map_right_color
        data['scale'] = scale_map
        # data['scale_map_pred'] = scale_map_pred
        data['mask'] = mask
        
        return data

class MSNetV9Getter:
    """
    Designed for matterport
    """

    def __init__(self, logger=logger):
        self.logger = logger

    def get_tensorboard_data(self, num_kps=20):
        """
        This processes the data needed for visualization. It expects the follow-
        ing in self.logger

        - image0: (C, H, W), Tensor, normalized
        - image1: (C, H, W), Tensor, normalized
        - scale: (H, W), Tensor
        - scale_pred: (3, H', W'), Tensor

        And it returns a dictionary

        - img0: image 1, (3, H, W)
        - img1: image 2, (3, H, W)
        - scale: (3, H, W)
        """

        # original images
        image0 = self.logger['image0']
        image1 = self.logger['image1']

        # scale ratio of right image to left image
        scale_pred = self.logger['scale_pred']
        num_cls = scale_pred.shape[0]
        scale_pred = torch.argmax(scale_pred, dim=0).long()
        scale_pred = cls2RGB(scale_pred, num_cls)

        scale = self.logger['scale']
        scale[scale > 1.5] = 2
        scale[scale < 0.75] = 0
        scale[(scale >= 0.75) * (scale <= 1.5)] = 1
        scale = scale.long()
        scale = cls2RGB(scale, num_cls)

        # region that has corresponding pixels
        msk = self.logger['msk']

        # process the images
        image0 = unnormalize(image0)
        image1 = unnormalize(image1)

        return {
            'img0': image0,
            'img1': image1,
            'scale_pred': scale_pred,
            'msk': msk,
            'scale': scale
        }


class MSNetV11Getter:
    """
    Designed for matterport
    """

    def __init__(self, logger=logger):
        self.logger = logger

    def get_tensorboard_data(self, num_kps=20):
        """
        This processes the data needed for visualization. It expects the follow-
        ing in self.logger

        - image0: (C, H, W), Tensor, normalized
        - image1: (C, H, W), Tensor, normalized
        - d03: (C, H, W)
        - d13: (C, H, W)
        - kps0: (N, 2), each being (x, y)
        - kps1: (N, 2)
        - kps2: (N, 2), negative ones
        - scale: (H, W), Tensor
        - scale_pred: (3, H', W'), Tensor

        And it returns a dictionary

        - desc0: descriptor 1, RGB, (3, H, W)
        - desc1: descriptor 2, RGB, (3, H, W)
        - img0: image 1, (3, H, W)
        - img1: image 2, (3, H, W)
        - keypoints: the two images marked with num_kps keypoints
        - neg_keypoints: image 2 marked with negative keypoints
        - corr: ground truth correspondences
        - corr false: false correspondences
        - scale: (3, H, W)
        """

        # original images
        image0 = self.logger['image0']
        image1 = self.logger['image1']

        # descriptors
        d03 = self.logger['d03']
        d13 = self.logger['d13']

        # keypoints
        kps0 = self.logger['kps0']
        kps1 = self.logger['kps1']
        kps2 = self.logger['kps2']

        # homography matrix
        H = self.logger['H']

        # process the images
        image0 = unnormalize(image0)
        image1 = unnormalize(image1)

        # process the descriptor
        desc = draw_paired_desc_torch(d03, kps0, image0, d13, kps1, image1, H)

        # choose keypoints
        N = kps0.shape[0]
        indices = np.random.choice(N, size=num_kps, replace=False)

        # draw keypoints
        kps = draw_kps_torch(image0, kps0[indices], image1, kps1[indices])

        # draw negative keypoints
        neg_kps = draw_kps_torch(image0, kps0[indices], image1, kps2[indices])

        # draw correspondences
        corr_gt = draw_corr_torch(image0, kps0[indices], image1, kps1[indices])

        # draw correspondences
        corr_false = draw_corr_torch(image0, kps0[indices], image1, kps2[indices])

        # scale ratio of right image to left image
        scale_pred = self.logger['scale_pred']
        num_cls = scale_pred.shape[0]
        scale_pred = torch.argmax(scale_pred, dim=0).long()
        scale_pred = cls2RGB(scale_pred, num_cls)

        scale = self.logger['scale']
        scale[scale > 1.5] = 2
        scale[scale < 0.75] = 0
        scale[(scale >= 0.75) * (scale <= 1.5)] = 1
        scale = scale.long()
        scale = cls2RGB(scale, num_cls)

        # region that has corresponding pixels
        msk = self.logger['msk']

        return {
            'img0': image0,
            'img1': image1,
            'desc': desc,
            'keypoints': kps,
            'neg_keypoints': neg_kps,
            'corr': corr_gt,
            'corr_false': corr_false,
            'scale_pred': scale_pred,
            'msk': msk,
            'scale': scale
        }


class MSNetV12Getter:
    """
    Designed for matterport
    """

    def __init__(self, logger=logger):
        self.logger = logger

    def get_tensorboard_data(self, num_kps=20):
        """
        This processes the data needed for visualization. It expects the follow-
        ing in self.logger

        - image0: (C, H, W), Tensor, normalized
        - image1: (C, H, W), Tensor, normalized
        - d03: (C, H', W')
        - d13: (C, H', W')
        - kps0: (N, 2), each being (x, y)
        - kps1: (N, 2)
        - kps2: (N, 2), negative ones
        - scale_pred: (1, H', W'), Tensor
        - scale: (1, H, W), Tensor
        - msk: (H, W), Tensor

        And it returns a dictionary

        - desc0: descriptor 1, RGB, (3, H', W')
        - desc1: descriptor 2, RGB, (3, H', W')
        - img0: image 1, (3, H, W)
        - img1: image 2, (3, H, W)
        - keypoints: the two images marked with num_kps keypoints
        - neg_keypoints: image 2 marked with negative keypoints
        - corr: ground truth correspondences
        - corr false: false correspondences
        - scale_pred: (3, H', W')
        - scale: (3, H, W)
        """

        # original images
        image0 = self.logger['image0']
        image1 = self.logger['image1']

        # descriptors
        d03 = self.logger['d03']
        d13 = self.logger['d13']

        # keypoints
        kps0 = self.logger['kps0']
        kps1 = self.logger['kps1']
        kps2 = self.logger['kps2']

        # homography matrix
        H = self.logger['H']

        # scale ratio of right image to left image
        scale_pred = self.logger['scale_pred']
        scale = self.logger['scale']

        # region that has corresponding pixels
        msk = self.logger['msk']

        scale = draw_scale_torch(scale[0] * msk[0])
        msk = F.interpolate(msk.unsqueeze(0), scale_pred.shape[1:], mode='bilinear')[0]
        scale_pred = draw_scale_torch(scale_pred[0] * msk[0])
        msk = msk[0]

        # process the images
        image0 = unnormalize(image0)
        image1 = unnormalize(image1)

        # process the descriptor
        desc = draw_paired_desc_torch(d03, kps0, image0, d13, kps1, image1, H)

        # choose keypoints
        N = kps0.shape[0]
        indices = np.random.choice(N, size=num_kps, replace=False)

        # draw keypoints
        kps = draw_kps_torch(image0, kps0[indices], image1, kps1[indices])

        # draw negative keypoints
        neg_kps = draw_kps_torch(image0, kps0[indices], image1, kps2[indices])

        # draw correspondences
        corr_gt = draw_corr_torch(image0, kps0[indices], image1, kps1[indices])

        # draw correspondences
        corr_false = draw_corr_torch(image0, kps0[indices], image1, kps2[indices])

        return {
            'img0': image0,
            'img1': image1,
            'desc': desc,
            'keypoints': kps,
            'neg_keypoints': neg_kps,
            'corr': corr_gt,
            'corr_false': corr_false,
            'scale_pred': scale_pred,
            'msk': msk,
            'scale': scale
        }
