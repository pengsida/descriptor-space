from .msnet_v14 import MultiScaleNetV14
from .msnet_v13 import MultiScaleNetV13
from .msnet_v12 import MultiScaleNetV12
from .msnet_v11 import MultiScaleNetV11
from .msnet_v10 import MultiScaleNetV10
from .msnet_v6 import MultiScaleNetV6
from .msnet_v9 import MultiScaleNetV9
from .msnet_v8 import MultiScaleNetV8
from .msnet_v7 import MultiScaleNetV7
from .msnet_v5 import MultiScaleNetV5
from .msnet_v4 import MultiScaleNetV4
from .msnet_v3 import MultiScaleNetV3
from .msnet_v2 import MultiScaleNetV2
from .msnet_v1 import MultiScaleNetV1
from .msnet import MultiScaleNet
from .matterport import Matterport
from .retinanet import RetinaNet
from .daisy import Daisy
from .superpoint import SuperPoint


_MATCHING_META_ARCHITECTURES = {
    "MSNetV14": MultiScaleNetV14,
    "MSNetV13": MultiScaleNetV13,
    "MSNetV12": MultiScaleNetV12,
    "MSNetV11": MultiScaleNetV11,
    "MSNetV10": MultiScaleNetV10,
    "MSNetV6": MultiScaleNetV6,
    "MSNetV9": MultiScaleNetV9,
    "MSNetV8": MultiScaleNetV8,
    "MSNetV7": MultiScaleNetV7,
    "MSNetV5": MultiScaleNetV5,
    "MSNetV4": MultiScaleNetV4,
    "MSNetV3": MultiScaleNetV3,
    "MSNetV2": MultiScaleNetV2,
    "MSNetV1": MultiScaleNetV1,
    "MSNet": MultiScaleNet,
    "Matterport": Matterport,
    "RetinaNet": RetinaNet,
    "Daisy": Daisy,
    "SuperPoint": SuperPoint
}


def build_matching_model(cfg):
    meta_arch = _MATCHING_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
