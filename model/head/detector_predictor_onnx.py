from typing import Optional
import torch
import pdb
import numpy as np
from torch import nn
from torch.nn import functional as F

from utils.registry import Registry
from model import registry
from model.layers.utils import sigmoid_hm
from model.make_layers import group_norm, _fill_fc_weights
from model.layers.utils import select_point_of_interest

from inplace_abn import InPlaceABN

class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a BatchNorm and an activation function in a single module

    Args:
        num_features: Number of feature channels in the input and output
        eps: Small constant to prevent numerical issues
        momentum: Momentum factor applied to compute running statistics with
            exponential moving average, or `None` to compute running statistics
            with cumulative moving average
        affine: If `True` apply learned scale and shift transformation after normalization
        track_running_stats: a boolean value that when set to `True`, this
            module tracks the running mean and variance, and when set to `False`,
            this module does not track such statistics and uses batch statistics instead
            in both training and eval modes if the running mean and variance are `None`
        activation: Name of the activation functions, one of: `relu`, `leaky_relu`,
            `elu` or `identity`
        activation_param: Negative slope for the `leaky_relu` activation or `alpha`
            parameter for the `elu` activation
    """

    _version = 2
    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "num_features",
        "affine",
        "activation",
        "activation_param",
    ]
    num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool
    activation: str
    activation_param: float

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        activation: str = "leaky_relu",
        activation_param: float = 0.01,
    ):
        super(ABN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.activation = activation
        self.activation_param = activation_param
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _get_momentum_and_training(self):
        if self.momentum is None:
            momentum = 0.0
        else:
            momentum = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    momentum = 1.0 / float(self.num_batches_tracked)
                else:
                    momentum = self.momentum

        if self.training:
            training = True
        else:
            training = (self.running_mean is None) and (self.running_var is None)

        return momentum, training

    def _get_running_stats(self):
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )
        return running_mean, running_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        momentum, training = self._get_momentum_and_training()
        running_mean, running_var = self._get_running_stats()

        x = F.batch_norm(
            x,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            training,
            momentum,
            self.eps,
        )

        if self.activation == "relu":
            return F.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return F.leaky_relu(
                x, negative_slope=self.activation_param, inplace=True
            )
        elif self.activation == "elu":
            return F.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError(f"Unknown activation function {self.activation}")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(ABN, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self):
        rep = "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}"
        if self.activation in ["leaky_relu", "elu"]:
            rep += "[{activation_param}]"
        return rep.format(**self.__dict__)


@registry.PREDICTOR.register("Base_Predictor")
class _predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(_predictor, self).__init__()
        # ("Car", "Cyclist", "Pedestrian")
        classes = len(cfg.DATASETS.DETECT_CLASSES)
        
        self.regression_head_cfg = cfg.MODEL.HEAD.REGRESSION_HEADS
        self.regression_channel_cfg = cfg.MODEL.HEAD.REGRESSION_CHANNELS
        self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
        
        self.head_conv = cfg.MODEL.HEAD.NUM_CHANNEL

        use_norm = cfg.MODEL.HEAD.USE_NORMALIZATION
        if use_norm == 'BN': norm_func = nn.BatchNorm2d
        elif use_norm == 'GN': norm_func = group_norm
        else: norm_func = nn.Identity

        # the inplace-abn is applied to reduce GPU memory and slightly increase the batch-size
        self.use_inplace_abn = cfg.MODEL.INPLACE_ABN
        self.bn_momentum = cfg.MODEL.HEAD.BN_MOMENTUM
        self.abn_activision = 'leaky_relu'

        ###########################################
        ###############  Cls Heads ################
        ########################################### 

        if self.use_inplace_abn:
            self.class_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                InPlaceABN(self.head_conv, momentum=self.bn_momentum, activation=self.abn_activision),
                nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
            )
        else:
            self.class_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                ABN(num_features=self.head_conv, momentum=self.bn_momentum),
                nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
            )
        
        self.class_head[-1].bias.data.fill_(- np.log(1 / cfg.MODEL.HEAD.INIT_P - 1))

        ###########################################
        ############  Regression Heads ############
        ########################################### 
        
        # init regression heads
        self.reg_features = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        # init regression heads
        for idx, regress_head_key in enumerate(self.regression_head_cfg):
            if self.use_inplace_abn:
                feat_layer = nn.Sequential(nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                                    InPlaceABN(self.head_conv, momentum=self.bn_momentum, activation=self.abn_activision))
            else:
                feat_layer = nn.Sequential(nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                                    ABN(num_features=self.head_conv, momentum=self.bn_momentum))

            self.reg_features.append(feat_layer)
            # init output head
            head_channels = self.regression_channel_cfg[idx]
            head_list = nn.ModuleList()
            for key_index, key in enumerate(regress_head_key):
                key_channel = head_channels[key_index]
                output_head = nn.Conv2d(self.head_conv, key_channel, kernel_size=1, padding=1 // 2, bias=True)

                if key.find('uncertainty') >= 0 and cfg.MODEL.HEAD.UNCERTAINTY_INIT:
                    torch.nn.init.xavier_normal_(output_head.weight, gain=0.01)
                
                # since the edge fusion is applied to the offset branch, we should save the index of this branch
                if key == '3d_offset': self.offset_index = [idx, key_index]

                _fill_fc_weights(output_head, 0)
                head_list.append(output_head)

            self.reg_heads.append(head_list)

        ###########################################
        ##############  Edge Feature ##############
        ###########################################

        # edge feature fusion
        self.enable_edge_fusion = cfg.MODEL.HEAD.ENABLE_EDGE_FUSION
        self.edge_fusion_kernel_size = cfg.MODEL.HEAD.EDGE_FUSION_KERNEL_SIZE
        self.edge_fusion_relu = cfg.MODEL.HEAD.EDGE_FUSION_RELU

        if self.enable_edge_fusion:
            trunc_norm_func = nn.BatchNorm1d if cfg.MODEL.HEAD.EDGE_FUSION_NORM == 'BN' else nn.Identity
            trunc_activision_func = nn.ReLU(inplace=True) if self.edge_fusion_relu else nn.Identity()
            
            self.trunc_heatmap_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, classes, kernel_size=1),
            )
            
            self.trunc_offset_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, 2, kernel_size=1),
            )

    def forward(self, features, targets):
        b, c, h, w = features.shape

        # output classification
        feature_cls = self.class_head[:-1](features)
        output_cls = self.class_head[-1](feature_cls)
        # expose for checking intermediate results
        # self.feature_cls = feature_cls
        # self.out_cls = output_cls

        output_regs = []
        # output regression
        for i, reg_feature_head in enumerate(self.reg_features):
            reg_feature = reg_feature_head(features)

            for j, reg_output_head in enumerate(self.reg_heads[i]):
                output_reg = reg_output_head(reg_feature)

                # apply edge feature enhancement
                if self.enable_edge_fusion and i == self.offset_index[0] and j == self.offset_index[1]:
                    # edge_indices = torch.stack([t.get_field("edge_indices") for t in targets]) # B x K x 2
                    # edge_lens = torch.stack([t.get_field("edge_len") for t in targets]) # B
                    # Pyten-20210713-ForConvertingOnnx
                    edge_indices = torch.stack([targets[0].get_field("edge_indices") for _ in range(b)]) # B x K x 2
                    edge_lens = torch.stack([targets[0].get_field("edge_len") for _ in range(b)]) # B

                    # normalize
                    grid_edge_indices = edge_indices.view(b, -1, 1, 2).float()
                    grid_edge_indices[..., 0] = grid_edge_indices[..., 0] / (self.output_width - 1) * 2 - 1
                    grid_edge_indices[..., 1] = grid_edge_indices[..., 1] / (self.output_height - 1) * 2 - 1

                    # apply edge fusion for both offset and heatmap
                    feature_for_fusion = torch.cat((feature_cls, reg_feature), dim=1)
                    edge_features = F.grid_sample(feature_for_fusion, grid_edge_indices, align_corners=True).squeeze(-1)

                    edge_cls_feature = edge_features[:, :self.head_conv, ...]
                    edge_offset_feature = edge_features[:, self.head_conv:, ...]
                    edge_cls_output = self.trunc_heatmap_conv(edge_cls_feature)
                    edge_offset_output = self.trunc_offset_conv(edge_offset_feature)
                    
                    for k in range(b):
                        edge_indice_k = edge_indices[k, :edge_lens[k]]
                        output_cls[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_cls_output[k, :, :edge_lens[k]]
                        output_reg[k, :, edge_indice_k[:, 1], edge_indice_k[:, 0]] += edge_offset_output[k, :, :edge_lens[k]]
                
                output_regs.append(output_reg)
        output_cls = sigmoid_hm(output_cls)
        output_regs = torch.cat(output_regs, dim=1)

        # Pyten-20210713-ForConvertingOnnx
        # return {'cls': output_cls, 'reg': output_regs}
        return output_cls, output_regs

def make_predictor(cfg, in_channels):
    func = registry.PREDICTOR[cfg.MODEL.HEAD.PREDICTOR]
    return func(cfg, in_channels)