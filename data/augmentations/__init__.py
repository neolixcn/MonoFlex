import logging
import numpy as np 

from .augmentations import (
    RandomHorizontallyFlip,
    RandomAffineCrop,
    Compose,
)

from config import cfg

aug_list = [RandomHorizontallyFlip,RandomAffineCrop] 
val_list = [RandomAffineCrop]                  
logger = logging.getLogger("monoflex.augmentations")

def get_composed_augmentations():
    aug_params = cfg.INPUT.AUG_PARAMS
    augmentations = []
    for aug, aug_param in zip(aug_list, aug_params):
        if aug_param[0] > 0:
            augmentations.append(aug(*aug_param))
            logger.info("Using {} aug with params {}".format(aug, aug_param))
        
    return Compose(augmentations)

def get_composed_augmentations_val():
    aug_params = cfg.INPUT.VAL_PARAMS
    augmentations = []
    for aug, aug_param in zip(val_list, aug_params):
        if aug_param[0] >= 0:
            augmentations.append(aug(*aug_param))
            logger.info("Using {} aug with params {}".format(aug, aug_param))
        
    return Compose(augmentations)
