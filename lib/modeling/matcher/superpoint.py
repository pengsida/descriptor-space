import torch
from torch import nn
from torch.nn import functional as F

from .evaluators import make_evaluator
from lib.utils.logger import logger
from lib.utils.misc import unnormalize


class SuperPoint(nn.Module):
    weight_path = 'tools/hpatches_eval/superpoint/superpoint_v1.pth'
    def __init__(self, cfg=None):
        super(SuperPoint, self).__init__()
        self.feature_extractor = SuperPointFeature()
        self.feature_extractor.load_state_dict(torch.load(self.weight_path))
        self.evaluator = make_evaluator(cfg)

    def forward(self, images0, images1, targets=None):
        """
        images0: [N, 3, H, W]
        images1: [N, 3, H, W]
        targets: {"kps0": [N, 3000], "kps1": [N, 3000], "kps2": [N, 3000]}
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        descrs0 = self.feature_extractor(images0)
        descrs1 = self.feature_extractor(images1)


        results = dict(
            descrs0=descrs0,
            descrs1=descrs1,
        )
        if not self.training:
            return results

        loss, pos_loss, neg_loss = self.evaluator(
            descrs0, targets["kps0"],
            descrs1, targets["kps1"],
            targets["kps2"],
        )


        # if targets["iteration"] % 100 == 0:
        #     print(pos_loss, neg_loss)

        losses = dict(loss=loss, distance=pos_loss, similarity=neg_loss)

        # keep descriptors for visualization
        logger.update(image0=images0[0], image1=images1[0])
        logger.update(kps0=targets['kps0'][0], kps1=targets['kps1'][0])
        logger.update(desc0=descrs0[0], desc1=descrs1[0])

        return losses, results

    def inference(self, images, scale=1, id='left'):
        return self.feature_extractor(images)

    @staticmethod
    def get_loss_for_bp(loss_dict):
        """
        From loss dict, get loss for backward pass
        :param loss_dict: that returned by self.forward
        :return: tensor, loss
        """
        return loss_dict['loss']

    @staticmethod
    def get_loss_for_log(loss_dict):
        """
        From loss dict, extract info for logging
        :return: a new dictionary
        """
        return loss_dict



class SuperPointFeature(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        x = unnormalize(x)
        x = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1).cuda()

        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        # extra things
        # desc = F.normalize(F.interpolate(desc, (h, w), mode="bilinear"), dim=1)
        return  desc
