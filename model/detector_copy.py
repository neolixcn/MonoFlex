import torch
from torch import nn

from structures.image_list import to_image_list

from .backbone import build_backbone
from .head.detector_head import bulid_head
from .head.detector_predictor import _predictor
from .head.detector_predictor_copy import _predictor_test
from model.layers.uncert_wrapper import make_multitask_wrapper
from .head.detector_loss import make_loss_evaluator
from model.head.detector_infer import make_post_processor
class KeypointDetector_v2(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector_v2, self).__init__()

        self.backbone = build_backbone(cfg)
        # self.heads = _predictor(self.backbone.out_channels) for detector predictor copy
        self.test = cfg.DATASETS.TEST_SPLIT == 'test'
        if not self.test:
            self.heads =  _predictor(cfg,self.backbone.out_channels)#bulid_head(cfg, self.backbone.out_channels)#
        else:
            self.heads = _predictor_test(cfg,self.backbone.out_channels)

        self.post_processor = make_post_processor(cfg)
    def forward(self, images,targets=None,test=None):
        
        #images = to_image_list(images)
        features = self.backbone(images)
        if targets is not None :
            output_cls, output_regs = self.heads(features,targets) 
        else:
            output_cls, output_regs = self.heads(features) 
        if test is not None :
            result, eval_utils, visualize_preds = self.post_processor({'cls': output_cls, 'reg': output_regs},targets,test)
            return result, eval_utils, visualize_preds
        return output_cls, output_regs,features
        