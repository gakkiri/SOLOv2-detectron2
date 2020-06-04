from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# SOLOV2 Head
# ---------------------------------------------------------------------------- #
_C.MODEL.SOLOV2 = CN()

# This is the number of foreground classes.
_C.MODEL.SOLOV2.NUM_CLASSES = 80
_C.MODEL.SOLOV2.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]  # stride [4, 8, 16, 32, 32 if pooling else 64]
_C.MODEL.SOLOV2.STRIDES = [8, 8, 16, 32, 32]
_C.MODEL.SOLOV2.IN_CHANNELS = 256
_C.MODEL.SOLOV2.STACKED_CONVS = 4
_C.MODEL.SOLOV2.SEG_FEAT_CHANNELS = 256
_C.MODEL.SOLOV2.SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_C.MODEL.SOLOV2.SIGMA = 0.2
_C.MODEL.SOLOV2.NUM_GRIDS = [40, 36, 24, 16, 12]
_C.MODEL.SOLOV2.CATE_DOWN_POS = 0
_C.MODEL.SOLOV2.WITH_DEFORM = False

_C.MODEL.SOLOV2.PRIOR_PROB = 0.01
_C.MODEL.SOLOV2.NMS_PRE = 500
_C.MODEL.SOLOV2.SCORE_TH = 0.1
_C.MODEL.SOLOV2.MASK_TH = 0.5
_C.MODEL.SOLOV2.UPDATE_TH = 0.05
_C.MODEL.SOLOV2.NMS_KERNEL = 'gaussian'
_C.MODEL.SOLOV2.NMS_SIGMA = 2.0
_C.MODEL.SOLOV2.MAX_PER_IMG = 100


_C.MODEL.LOSS = CN()
# ---------------------------------------------------------------------------- #
# FocalLoss Head
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSS.FOCAL_LOSS = CN()

_C.MODEL.LOSS.FOCAL_LOSS.USE_SIGMOID = True
_C.MODEL.LOSS.FOCAL_LOSS.ALPHA = 0.25
_C.MODEL.LOSS.FOCAL_LOSS.GAMMA = 2.0
_C.MODEL.LOSS.FOCAL_LOSS.WEIGHT = 1.0

# ---------------------------------------------------------------------------- #
# DiceLoss Head
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSS.DICE_LOSS = CN()

_C.MODEL.LOSS.DICE_LOSS.USE_SIGMOID = True
_C.MODEL.LOSS.DICE_LOSS.WEIGHT = 3.0
