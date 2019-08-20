"""
    Default setting for the network,
    Could be overwrite with training configs
"""

import os
from yacs.config import CfgNode as CN


_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "descriptor-space"
_C.MODEL.DEVICE = "cuda"
_C.MODEL.TARGET = "FLOW"
_C.MODEL.META_ARCHITECTURE = "OneShot"
_C.MODEL.TEST = False

_C.MODEL.WEIGHT = ""
_C.MODEL.IMG_HEIGHT = 480
_C.MODEL.IMG_WIDTH = 640

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.CONV_BODY = "R-18-C4"
_C.MODEL.BACKBONE.STOP_DOWNSAMPLING = "C3"
_C.MODEL.BACKBONE.START_DOWNSAMPLING = ""


# -----------------------------------------------------------------------------
# PATH SETTING
# -----------------------------------------------------------------------------
_C.PATH = CN()
_C.PATH.CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_C.PATH.LIB_DIR = os.path.dirname(_C.PATH.CONFIG_DIR)
_C.PATH.ROOT_DIR = os.path.dirname(_C.PATH.LIB_DIR)
_C.PATH.DATA_DIR = os.path.join(_C.PATH.ROOT_DIR, 'data')
_C.PATH.ShapeNet = os.path.join(_C.PATH.DATA_DIR, 'ShapeNet')


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()

_C.DATASET.TRAIN = ""
_C.DATASET.TEST = ""
_C.DATASET.NEIGHBORHOOD = 5

# -----------------------------------------------------------------------------
# DATASET SPECIFIC
# -----------------------------------------------------------------------------
_C.DATASET.COCO_ANGLE = 45
_C.DATASET.HPATCHES_ANGLE = 45

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = 800  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = False


# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.MAX_ITER = 40000

# use Adam as default
_C.TRAIN.BASE_LR = 0.001
_C.TRAIN.WEIGHT_DECAY = 0.0005

# scheduler, use MultiStepLR as default
_C.TRAIN.MILESTONES = [10000, 40000, 80000]
_C.TRAIN.GAMMA = 0.1

_C.TRAIN.CHECKPOINT_PERIOD = 2500
_C.TRAIN.NUM_CHECKPOINT = 10

_C.TRAIN.IMS_PER_BATCH = 4

_C.TRAIN.RESUME = True


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 1


# ---------------------------------------------------------------------------- #
# Tensorboard
# ---------------------------------------------------------------------------- #
_C.TENSORBOARD = CN()
_C.TENSORBOARD.IS_ON = True
_C.TENSORBOARD.TARGETS = CN()
_C.TENSORBOARD.TARGETS.SCALAR = ["loss"]
_C.TENSORBOARD.TARGETS.IMAGE = []
_C.TENSORBOARD.LOG_DIR = os.path.join(_C.PATH.ROOT_DIR, "logs")


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.MODEL_DIR = os.path.join(_C.PATH.DATA_DIR, "model")
_C.PATHS_CATALOG = os.path.join(_C.PATH.CONFIG_DIR, 'path.py')

# ---------------------------------------------------------------------------- #
# Logger getter options
# ---------------------------------------------------------------------------- #
_C.GETTER = CN()
_C.GETTER.NAME = 'Matterport'
