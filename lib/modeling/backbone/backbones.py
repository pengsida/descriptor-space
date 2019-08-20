from torch import nn
from collections import OrderedDict

from . import resnet
from lib.modeling import registry
from . import retinanet


@registry.BACKBONES.register("R-18-C5")
def build_single_resnet_backbone(cfg):
    body = resnet.SingleResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


@registry.BACKBONES.register('Multi-R-18-C5')
def build_multi_resnet_backbone(cfg):
    body = resnet.MultiResNet(cfg)
    model = nn.Sequential(OrderedDict([('body', body)]))
    return model


@registry.BACKBONES.register("FPN")
def build_fpn_backbone(cfg):
    body = retinanet.resnet18(pretrained=True, cfg=cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
