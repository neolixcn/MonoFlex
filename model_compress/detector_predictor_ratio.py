import os
import torch
import pdb
import numpy as np
import torch
import pdb
import numpy as np
from torch import imag, nn
from torch.nn import functional as F

from utils.registry import Registry
from model import registry
from model.layers.utils import sigmoid_hm
from model.make_layers import group_norm, _fill_fc_weights
#from model.backbone.DCNv2.dcn_v2 import DCNv2
from model.layers.utils import (
	nms_hm,
	select_topk,
	select_point_of_interest,
)

from inplace_abn import InPlaceABN
from model.layers.utils import Converter_key2channel
PI = np.pi

import pdb

class postprcess(nn.Module):
    def __init__(self,cfg,input_width,input_height):
        super(postprcess, self).__init__()
        self.regression_head_cfg = [['2d_dim'], ['3d_offset'], ['corner_offset'], ['corner_uncertainty'], ['3d_dim'], ['ori_cls', 'ori_offset'], ['depth'], ['depth_uncertainty']]
        self.regression_channel_cfg = [[4, ], [2, ], [20], [3], [3, ], [8, 8], [1, ], [1, ]]
        self.input_width = input_width
        self.input_height = input_height
        self.down_ratio = 4
        self.output_width = self.input_width  // self.down_ratio
        self.output_height = self.input_height // self.down_ratio
        self.max_detection = 50 
        self.head_conv = 256
        self.det_threshold = cfg.TEST.DETECTIONS_THRESHOLD
        self.orien_bin_size = 4
        # dimension related
        # self.dim_mean =  torch.as_tensor(((3.8840, 1.5261, 1.6286),
        #                        (0.8423, 1.7607, 0.6602),
        #                        (1.7635, 1.7372, 0.5968))).to('cuda')

        # # since only car and pedestrian have enough samples and are evaluated in KITTI server 
        # self.dim_std =  torch.as_tensor(((0.4259, 0.1367, 0.1022),
        #                                 (0.2349, 0.1133, 0.1427),
        #                                 (0.1766, 0.0948, 0.1242))).to('cuda')

        self.dim_mean =  torch.as_tensor(((4.83899871 ,1.80778956, 2.11565798),
                               (0.91986743 ,1.75302337, 0.86220807),
                               (1.78652745 ,1.76500989, 0.83395625))).to('cuda')
        self.dim_std =  torch.as_tensor(((1.25442242, 0.43771471, 0.29804158),
								(0.19319452, 0.19077312, 0.15295986),
								(0.29947595, 0.19646908, 0.11966296) )).to('cuda')
        # linear or log ; use mean or not ; use std or not
        self.dim_modes = ['exp', True, False]
        self.key2channel = Converter_key2channel(keys=self.regression_head_cfg, channels=self.regression_channel_cfg)
        self.depth_mode = 'inv_sigmoid'
        self.depth_range =  [0.1, 100]
        self.depth_ref = torch.as_tensor((26.494627, 16.05988)).to('cuda') 
        self.alpha_centers = torch.tensor([0, PI / 2, PI, - PI / 2]).to('cuda') 
        self.EPS = 1e-3
        self.multibin = True

    def decode_box2d_fcos(self,centers, pred_offset, out_size=None):
        box2d_center = centers.view(-1, 2)
        #box2d = box2d_center.new(box2d_center.shape[0], 4).zero_()
        box2d = torch.zeros((centers.shape[0],4),device=centers.device, dtype=centers.dtype)
        # left, top, right, bottom
        box2d[:, :2] = box2d_center - pred_offset[:, :2]
        box2d[:, 2:] = box2d_center + pred_offset[:, 2:]
        box2d = box2d * self.down_ratio
        box2d[:,0:2] = torch.clamp(box2d[:,0:2], min=0, max=self.input_width)
        box2d[:,1:2] = torch.clamp(box2d[:,1:2], min=0, max=self.input_height)
        # box2d[:, 0::2].clamp_(min=0, max=out_size[0].item() - 1)
        # box2d[:, 1::2].clamp_(min=0, max=out_size[1].item() - 1)
        # for inference
        # if pad_size is not None and box2d.shape[0]!=0 :
        #     N = box2d.shape[0]
        #     out_size = out_size[0]
        #     # upscale and subtract the padding
        #     box2d = box2d * self.down_ratio - pad_size.repeat(1, 2)
        #     # clamp to the image bound
        #     # import pdb
        #     # pdb.set_trace()
        #     box2d[:, 0::2].clamp_(min=0, max=out_size[0].item() - 1)
        #     box2d[:, 1::2].clamp_(min=0, max=out_size[1].item() - 1)

        return box2d

    def decode_dimension(self, cls_id, dims_offset):
        cls_id = cls_id.flatten().long()
        if cls_id.is_cuda is False:
            cls_dimension_mean = self.dim_mean[cls_id, :].to('cpu')
        else:
            cls_dimension_mean = self.dim_mean[cls_id, :]

        if self.dim_modes[0] == 'exp':
            dims_offset = dims_offset.exp()

        if self.dim_modes[2]:
            cls_dimension_std = self.dim_std[cls_id, :]
            dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
        else:
            dimensions = dims_offset * cls_dimension_mean
            
        return dimensions

    def decode_depth(self, depths_offset):

        if self.depth_mode == 'exp':
            depth = depths_offset.exp()
        elif self.depth_mode == 'linear':
            depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
        elif self.depth_mode == 'inv_sigmoid':
            depth = 1 / torch.sigmoid(depths_offset) - 1
        else:
            raise ValueError

        if self.depth_range is not None:
            depth = torch.clamp(depth, min=self.depth_range[0], max=self.depth_range[1])

        return depth

    def decode_location_flatten(self, points, offsets, depths, calibs, batch_idxs):
        batch_size = 1# len(calibs)
        gts = torch.unique(batch_idxs, sorted=True).tolist()
        # locations = points.new_zeros(points.shape[0], 3).float()
        locations = torch.zeros((points.shape[0], 3), dtype=points.dtype, device=points.device).float()
        points = (points + offsets) * self.down_ratio #- pad_size[batch_idxs]

        for idx, gt in enumerate(gts):
            corr_pts_idx = torch.nonzero(batch_idxs == gt).squeeze(-1)
            calib = calibs
            # concatenate uv with depth
            corr_pts_depth = torch.cat((points[corr_pts_idx], depths[corr_pts_idx, None]), dim=1)
            locations[corr_pts_idx] = calib.project_image_to_rect(corr_pts_depth)

        return locations
    
    def decode_axes_orientation(self, vector_ori, locations):
    
        if self.multibin:
            pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
            pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
            # orientations = vector_ori.new_zeros(vector_ori.shape[0])
            orientations = torch.zeros(vector_ori.shape[0], device=vector_ori.device, dtype=vector_ori.dtype)
            for i in range(self.orien_bin_size):
                mask_i = (pred_bin_cls.argmax(dim=1) == i)
                s = self.orien_bin_size * 2 + i * 2
                e = s + 2
                pred_bin_offset = vector_ori[mask_i, s : e]
                orientations[mask_i] = torch.atan(pred_bin_offset[:, 0]/ pred_bin_offset[:, 1]) + self.alpha_centers[i]
        else:
            axis_cls = torch.softmax(vector_ori[:, :2], dim=1)
            axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
            head_cls = torch.softmax(vector_ori[:, 2:4], dim=1)
            head_cls = head_cls[:, 0] < head_cls[:, 1]
            # cls axis
            orientations = self.alpha_centers[axis_cls + head_cls * 2]
            sin_cos_offset = F.normalize(vector_ori[:, 4:])
            orientations += torch.atan(sin_cos_offset[:, 0] / sin_cos_offset[:, 1])

        locations = locations.view(-1, 3)
        rays = torch.atan(locations[:, 0] / locations[:, 2])
        alphas = orientations
        rotys = alphas + rays

        larger_idx = (rotys > PI).nonzero()
        small_idx = (rotys < -PI).nonzero()
        if len(larger_idx) != 0:
                rotys[larger_idx] -= 2 * PI
        if len(small_idx) != 0:
                rotys[small_idx] += 2 * PI

        larger_idx = (alphas > PI).nonzero()
        small_idx = (alphas < -PI).nonzero()
        if len(larger_idx) != 0:
                alphas[larger_idx] -= 2 * PI
        if len(small_idx) != 0:
                alphas[small_idx] += 2 * PI

        return rotys, alphas
    
    def decode_orientation(self, vector_ori):

        pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
        pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
        
        orientations = torch.zeros(vector_ori.shape[0], device=vector_ori.device, dtype=vector_ori.dtype)
        
        for i in range(self.orien_bin_size):
            mask_i = (pred_bin_cls.argmax(dim=1) == i)
            s = self.orien_bin_size * 2 + i * 2
            e = s + 2
            pred_bin_offset = vector_ori[mask_i, s : e]
            orientations[mask_i] = torch.atan(pred_bin_offset[:, 0]/ pred_bin_offset[:, 1]) + self.alpha_centers[i]
        alphas = orientations

        larger_idx = (alphas > PI).nonzero()
        small_idx = (alphas < -PI).nonzero()
        if len(larger_idx) != 0:
                alphas[larger_idx] -= 2 * PI
        if len(small_idx) != 0:
                alphas[small_idx] += 2 * PI

        return alphas

    def decode_depth_from_keypoints(self, pred_offsets, pred_keypoints, pred_dimensions, calibs, avg_center=False):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
        assert len(calibs) == 1 # for inference, batch size is always 1
        
        calib = calibs[0]
        # we only need the values of y
        pred_height_3D = pred_dimensions[:, 1]
        pred_keypoints = pred_keypoints.view(-1, 10, 2)
        # center height -> depth
        if avg_center:
            updated_pred_keypoints = pred_keypoints - pred_offsets.view(-1, 1, 2)
            center_height = updated_pred_keypoints[:, -2:, 1]
            center_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (center_height.abs() * self.down_ratio * 2)
            center_depth = center_depth.mean(dim=1)
        else:
            center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
            center_depth = calib.f_u * pred_height_3D / (center_height.abs() * self.down_ratio)
        
        # corner height -> depth
        corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
        corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]
        corner_02_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (corner_02_height * self.down_ratio)
        corner_13_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (corner_13_height * self.down_ratio)
        corner_02_depth = corner_02_depth.mean(dim=1)
        corner_13_depth = corner_13_depth.mean(dim=1)
        # K x 3
        pred_depths = torch.stack((center_depth, corner_02_depth, corner_13_depth), dim=1)

        return pred_depths

    def decode_depth_from_keypoints_batch(self, pred_keypoints, pred_dimensions, calib, batch_idxs=None):
        # pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
        pred_height_3D = pred_dimensions[:, 1].clone()
        batch_size = 1#len(calibs)
        if batch_size == 1:
            # batch_idxs = pred_dimensions.new_zeros(pred_dimensions.shape[0])
            batch_idxs = torch.zeros(pred_dimensions.shape[0], device=pred_dimensions.device,
                                     dtype=pred_dimensions.dtype)

        center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
        corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
        corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]

        pred_keypoint_depths = {'center': [], 'corner_02': [], 'corner_13': []}

        for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):			
            #calib = calibs[idx]
            corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
            center_depth = calib.f_u * pred_height_3D[corr_pts_idx] / (F.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
            corner_02_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_02_height[corr_pts_idx]) * self.down_ratio + self.EPS)
            corner_13_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_13_height[corr_pts_idx]) * self.down_ratio + self.EPS)

            corner_02_depth = corner_02_depth.mean(dim=1)
            corner_13_depth = corner_13_depth.mean(dim=1)

            pred_keypoint_depths['center'].append(center_depth)
            pred_keypoint_depths['corner_02'].append(corner_02_depth)
            pred_keypoint_depths['corner_13'].append(corner_13_depth)

        for key, depths in pred_keypoint_depths.items():
            pred_keypoint_depths[key] = torch.clamp(torch.cat(depths), min=self.depth_range[0], max=self.depth_range[1])

        pred_depths = torch.stack([depth for depth in pred_keypoint_depths.values()], dim=1)

        return pred_depths

    def forward(self,output_cls,output_regs,calib):
        heatmap = nms_hm(output_cls,kernel=3)
        scores, indexs, clses, ys, xs = select_topk(heatmap, K=self.max_detection)
        pred_bbox_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)
        pred_regression_pois = select_point_of_interest(output_regs.shape[0], indexs, output_regs).view(-1, output_regs.shape[1])
        scores = scores.view(-1)
        valid_mask = scores >= self.det_threshold
        # pdb.set_trace()
        if valid_mask.sum() == 0:
            return None,None,None,None,None,None,None
        #     # result = scores.new_zeros(0, 9)
        #     result = torch.zeros((0, 9), device=scores.device, dtype=scores.dtype)
        #     return result, output_cls, output_regs
        clses = clses.view(-1)[valid_mask]
        pred_bbox_points = pred_bbox_points[valid_mask]
        pred_regression_pois = pred_regression_pois[valid_mask]
        pred_2d_reg = F.relu(pred_regression_pois[:, self.key2channel('2d_dim')])
        pred_offset_3D = pred_regression_pois[:, self.key2channel('3d_offset')]
        scores = scores[valid_mask]
        pred_orientation = torch.cat((pred_regression_pois[:, self.key2channel('ori_cls')], pred_regression_pois[:, self.key2channel('ori_offset')]), dim=1)
        #pred_alphas = self.decode_orientation(pred_orientation)
        
        pred_dimensions_offsets = pred_regression_pois[:, self.key2channel('3d_dim')]
        
        img_size = torch.from_numpy(np.array([(self.input_width, self.input_height)])).to('cuda')
        pad_size = torch.from_numpy(np.array([0,0])).to('cuda')
        
        pred_box2d = self.decode_box2d_fcos(pred_bbox_points, pred_2d_reg)
        pred_dimensions = self.decode_dimension(clses, pred_dimensions_offsets)


        # add depth 
        pred_depths_offset = pred_regression_pois[:, self.key2channel('depth')].squeeze(-1)
        pred_direct_depths = self.decode_depth(pred_depths_offset)
        pred_direct_uncertainty = pred_regression_pois[:, self.key2channel('depth_uncertainty')].exp()
        pred_keypoint_offset = pred_regression_pois[:, self.key2channel('corner_offset')]
        pred_keypoint_offset = pred_keypoint_offset.view(-1, 10, 2)
        # solve depth from estimated key-points
        pred_keypoints_depths = self.decode_depth_from_keypoints_batch(pred_keypoint_offset, pred_dimensions, calib)
        pred_combined_depths = torch.cat((pred_direct_depths.unsqueeze(1), pred_keypoints_depths), dim=1)
        pred_keypoint_uncertainty = pred_regression_pois[:, self.key2channel('corner_uncertainty')].exp()
        pred_combined_uncertainty = torch.cat((pred_direct_uncertainty, pred_keypoint_uncertainty), dim=1)
        depth_weights = 1 / pred_combined_uncertainty
        depth_weights = depth_weights / depth_weights.sum(dim=1, keepdim=True)
        pred_depths = torch.sum(pred_combined_depths * depth_weights, dim=1)

        # the uncertainty after combination
        estimated_depth_error = torch.sum(depth_weights * pred_combined_uncertainty, dim=1)
        uncertainty_conf = 1 - torch.clamp(estimated_depth_error, min=0.01, max=1)	
        # scores = scores.view(-1, 1)
        # scores = scores * uncertainty_conf.view(-1, 1)
        scores = scores.view(-1)

        batch_idxs = pred_depths.new_zeros(pred_depths.shape[0]).long()
        pred_locations = self.decode_location_flatten(pred_bbox_points, pred_offset_3D, pred_depths, calib, batch_idxs)
        pred_rotys, pred_alphas = self.decode_axes_orientation(pred_orientation, pred_locations)
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2
        pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)
        return clses,pred_alphas,pred_rotys, pred_box2d, pred_dimensions,scores ,pred_locations



#@registry.PREDICTOR.register("Base_Predictor")
class _predictor(nn.Module):
    def __init__(self, in_channels):
        super(_predictor, self).__init__()
        # ("Car", "Cyclist", "Pedestrian")
        classes = 3 # len(cfg.DATASETS.DETECT_CLASSES)
        
        self.regression_head_cfg = [['2d_dim'], ['3d_offset'], ['corner_offset'], ['corner_uncertainty'], ['3d_dim'], ['ori_cls', 'ori_offset'], ['depth'], ['depth_uncertainty']]
        self.regression_channel_cfg = [[4, ], [2, ], [20], [3], [3, ], [8, 8], [1, ], [1, ]]
        self.input_width = 1280
        self.input_height = 384
        self.down_ratio =4
        self.output_width = self.input_width  // self.down_ratio
        self.output_height = self.input_height // self.down_ratio
        self.max_detection = 50 
        self.head_conv = 256
        self.det_threshold = 0.2
        self.orien_bin_size = 4
        use_norm = "BN"
        if use_norm == 'BN': norm_func = nn.BatchNorm2d
        elif use_norm == 'GN': norm_func = group_norm
        else: norm_func = nn.Identity
        
        # the inplace-abn is applied to reduce GPU memory and slightly increase the batch-size
        self.use_inplace_abn = False
        self.bn_momentum = 0.1
        self.abn_activision = 'leaky_relu'
        self.key2channel = Converter_key2channel(keys=self.regression_head_cfg, channels=self.regression_channel_cfg)
        ###########################################
        ###############  Cls Heads ################
        ########################################### 

        if self.use_inplace_abn:
            self.class_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                InPlaceABN(self.head_conv, momentum=self.bn_momentum, activation=self.abn_activision),
                nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
            )
        elif use_norm =='BN':
            self.class_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
            )
        else:
            self.class_head = nn.Sequential(
                nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                norm_func(self.head_conv), nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
            )
        self.class_head[-1].bias.data.fill_(- np.log(1 / 0.01- 1))

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
                                    norm_func(self.head_conv, momentum=self.bn_momentum), nn.ReLU(inplace=True))
            
            self.reg_features.append(feat_layer)
            # init output head
            head_channels = self.regression_channel_cfg[idx]
            head_list = nn.ModuleList()
            for key_index, key in enumerate(regress_head_key):
                key_channel = head_channels[key_index]
                output_head = nn.Conv2d(self.head_conv, key_channel, kernel_size=1, padding=1 // 2, bias=True)

                if key.find('uncertainty') >= 0 and True:
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
        self.enable_edge_fusion = False
        self.edge_fusion_kernel_size = 3
        self.edge_fusion_relu = False

        if self.enable_edge_fusion:
            trunc_norm_func = nn.BatchNorm1d #if cfg.MODEL.HEAD.EDGE_FUSION_NORM == 'BN' else nn.Identity
            trunc_activision_func = nn.ReLU(inplace=True) if self.edge_fusion_relu else nn.Identity()
            
            self.trunc_heatmap_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, classes, kernel_size=1),
            )
            
            self.trunc_offset_conv = nn.Sequential(
                nn.Conv1d(self.head_conv, self.head_conv, kernel_size=self.edge_fusion_kernel_size, padding=self.edge_fusion_kernel_size // 2, padding_mode='replicate'),
                trunc_norm_func(self.head_conv, momentum=self.bn_momentum), trunc_activision_func, nn.Conv1d(self.head_conv, 2, kernel_size=1),
            )

        if self.enable_edge_fusion:
			# generate edge_indices for the edge fusion module
            self.max_edge_length = (self.output_width + self.output_height) * 2
            input_edge_indices = np.zeros([self.max_edge_length, 2], dtype=np.int64)
            edge_indices = self.get_edge_utils((self.output_width, self.output_height), np.array([0, 0])).numpy()
            input_edge_indices[:edge_indices.shape[0]] = edge_indices
            self.input_edge_count = edge_indices.shape[0] -1 
            self.edge_indices = input_edge_indices 
        
        # dimension related
        self.dim_mean =  torch.as_tensor(((3.8840, 1.5261, 1.6286),
                               (0.8423, 1.7607, 0.6602),
                               (1.7635, 1.7372, 0.5968))).to('cuda')

        # since only car and pedestrian have enough samples and are evaluated in KITTI server 
        self.dim_std =  torch.as_tensor(((0.4259, 0.1367, 0.1022),
                                        (0.2349, 0.1133, 0.1427),
                                        (0.1766, 0.0948, 0.1242))).to('cuda')

        # linear or log ; use mean or not ; use std or not
        self.dim_modes = ['exp', True, False]

        self.depth_mode = 'inv_sigmoid'
        self.depth_range =  [0.1, 100]
        self.depth_ref = torch.as_tensor((26.494627, 16.05988)).to('cuda') 
        self.alpha_centers = torch.tensor([0, PI / 2, PI, - PI / 2]).to('cuda') 
        self.EPS = 1e-3

    def get_edge_utils(self, image_size, pad_size, down_ratio=4):
        img_w ,img_h = image_size 
        x_min, y_min = np.ceil(pad_size[0] / down_ratio), np.ceil(pad_size[1] / down_ratio)
        x_max, y_max = (pad_size[0] + img_w - 1) // down_ratio, (pad_size[1] + img_h - 1) // down_ratio

        step = 1
        # boundary idxs
        edge_indices = []

        # left
        y = torch.arange(y_min, y_max, step)
        x = torch.ones(len(y)) * x_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = torch.arange(x_min, x_max, step)
        y = torch.ones(len(x)) * y_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # right
        y = torch.arange(y_max, y_min, -step)
        x = torch.ones(len(y)) * x_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)

        # top  
        x = torch.arange(x_max, x_min - 1, -step)
        y = torch.ones(len(x)) * y_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)

        # concatenate
        edge_indices = torch.cat([index.long() for index in edge_indices], dim=0)

        return edge_indices

    
    def forward(self, features):
        b, c, h, w = features.shape

        # output classification
        feature_cls = self.class_head[:-1](features)
        output_cls = self.class_head[-1](feature_cls)

        output_regs = []
        # output regression
        for i, reg_feature_head in enumerate(self.reg_features):
            reg_feature = reg_feature_head(features)

            for j, reg_output_head in enumerate(self.reg_heads[i]):
                output_reg = reg_output_head(reg_feature)

                # apply edge feature enhancement
                if self.enable_edge_fusion and i == self.offset_index[0] and j == self.offset_index[1]:
                    edge_indices = torch.from_numpy(self.edge_indices)#torch.stack([t.get_field("edge_indices") for t in self.targets]) # B x K x 2
                    edge_lens =  torch.from_numpy(self.input_edge_count)#torch.stack([t.get_field("edge_len") for t in self.targets]) # B
                    #edge_indices = torch.stack([t.get_field("edge_indices") for t in self.targets])
                    # print(edge_indices)
                    # print(edge_lens)
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
        return output_cls,output_regs 
        clses,pred_alphas,pred_box2d, pred_dimensions,scores,pred_locations = self.postprocess(output_cls,output_regs,calib)
        return clses,pred_alphas,pred_box2d, pred_dimensions,scores ,pred_locations
        
    

def make_predictor(cfg, in_channels):
    func = registry.PREDICTOR[cfg.MODEL.HEAD.PREDICTOR]
    return func(cfg, in_channels)


if __name__ == '__main__':
    #model = build_backbone(num_layers=34).cuda()
    
    model = _predictor(64).cuda()

    x = torch.rand(1, 64, 96, 320).cuda()
    
    o,o1,o2,o3= model(x)

    torch.save(model.state_dict(), "/home/lipengcheng/MonoFlex-llt/output/onnx_model/predictor.pth")
    
    model2 = _predictor(64)

    model2.load_state_dict(torch.load("/home/lipengcheng/MonoFlex-llt/output/onnx_model/predictor.pth"))

    model2.eval()

    temp = torch.rand(1, 64, 96, 320)

    y,y1, y2 = model2(temp)

    torch.onnx.export(model2, temp, "/home/lipengcheng/MonoFlex-llt/output/onnx_model/predictor.onnx", verbose=True)

    print(y2.shape)