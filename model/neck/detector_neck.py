import torch
from torch import nn
import pdb
from .CBAM import CBAM

class Detect_Neck(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Detect_Neck, self).__init__()
        if cfg.MODEL.NECK.USE_TYPE== 'CBAM':
            self.neck = CBAM(in_channels, cfg.MODEL.NECK.CABM_INPUT[0], cfg.MODEL.NECK.CABM_INPUT[1])
       

    def forward(self, features):
        x = self.neck(features)
        return x

def build_neck(cfg, in_channels):
    
    return Detect_Neck(cfg, in_channels)