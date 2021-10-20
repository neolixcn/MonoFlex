import os,cv2
import torch
from model.backbone.HGFilters import HGFilter
from torch import nn
from config import cfg
from PIL import Image
import numpy as np
import datetime
from utils.check_point import DetectronCheckpointer
from detector_predictor_ratio import _predictor,postprcess
from engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from model.backbone import build_backbone
from utils.visualizer import Visualizer
from skimage import transform as trans 
from model.head.detector_infer import make_post_processor
from data.transforms import build_transforms 
from engine.visualize_infer import show_image_with_boxes_test
from data.datasets.kitti_utils import Calibration


def get_3rd_point(point_a, point_b):
    d = point_a - point_b
    point_c = point_b + np.array([-d[1], d[0]])
    return point_c

def get_transfrom_matrix(center_scale, output_size):
    center, scale = center_scale[0], center_scale[1]
    # todo: further add rot and shift here.
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    src_dir = np.array([0, src_w * -0.5])
    dst_dir = np.array([0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    affine_cv = cv2.getAffineTransform(src,dst)
    #img=  cv2.warpAffine(img, M, (cols, rows))
    get_matrix = trans.estimate_transform("affine", src, dst)
    matrix = get_matrix.params

    return matrix.astype(np.float32),affine_cv

def normalize_img(img, mean,std ):
    #img = img[...,[2, 1, 0]]
    img = np.transpose(img,(2,0,1))

    img = torch.from_numpy(img/255.0).float()
    for t,m,s in zip(img,mean,std):
        t.sub_(m).div_(s)

    return img 

def preprocess_PIL(img):
    input_width = 1280 
    input_height = 384 
    center = np.array([i / 2 for i in img.size], dtype=np.float32)
    size = np.array([i for i in img.size], dtype=np.float32)
    center_size = [center, size]
    trans_affine,affine_opencv = get_transfrom_matrix(
        center_size,
        [input_width, input_height]
    )
    img =cv2.warpAffine(np.array(img), affine_opencv, (input_width, input_height))
    return img

def preprocess(img,mean,std,calib_filename):
    input_width = 1280 
    input_height = 384 
    center = np.array([img.shape[1]//2,img.shape[0]//2], dtype=np.float32)
    size = np.array([img.shape[1],img.shape[0]], dtype=np.float32)

    """
    resize, horizontal flip, and affine augmentation are performed here.
    since it is complicated to compute heatmap w.r.t transform.
    """
    center_size = [center, size]
    trans_affine,affine_opencv = get_transfrom_matrix(
        center_size,
        [input_width, input_height]
    )
    img =cv2.warpAffine(np.array(img), affine_opencv, (input_width, input_height))
    img = img[...,[2,1,0]]
    img_numpy = img.copy()
    #cv2.imwrite("/home/lipengcheng/MonoFlex-llt/test/output_file/save_.png",img)

    img= normalize_img(img, mean,std )

    calib = Calibration(calib_filename)
    calib.matAndUpdate(trans_affine)
    return img ,img_numpy ,calib


def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_work
    cfg.TEST.EVAL_DIS_IOUS = args.eval_iou
    cfg.TEST.EVAL_DEPTH = args.eval_depth 
    
    if args.vis_thre > 0:
        cfg.TEST.VISUALIZE_THRESHOLD = args.vis_thre 
    
    if args.output is not None:
        cfg.OUTPUT_DIR = args.output

    if args.test:
        cfg.DATASETS.TEST_SPLIT = 'test'
        cfg.DATASETS.TEST = ("kitti_test",)

    cfg.START_TIME = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    default_setup(cfg, args)

    return cfg

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
        self.heads = _predictor(self.backbone.out_channels)
        #self.heads = _predictor(256)

    def forward(self, images):
        
        features = self.backbone(images)

        output_cls,  output_regs= self.heads(features)
        
        return output_cls,  output_regs

if __name__ == "__main__":

    args = default_argument_parser().parse_args() 

    cfg = setup(args)
    
    model = KeypointDetector_v2(cfg).cuda()

    path = "/home/lipengcheng/data/neolix/image_2/00.png"
    #path = "/home/lipengcheng/MonoFlex-llt/test/input_file/1.png"


    calibrationPath = "/home/lipengcheng/data/neolix/calib/front_3mm_intrinsics.yaml"

    img = cv2.imread(path)

    pixel_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]))

    pixel_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]))

    img ,img_numpy, calib= preprocess(img,pixel_mean,pixel_std,calibrationPath)

    img =img.unsqueeze(0).to('cuda')

    #delta = img_numpy - img4 
    #img = torch.ones((1, 3, 384, 1280)).to('cuda')

    output_cls,  output_regs =  model(img)

    torch.save(model.state_dict(), "/home/lipengcheng/MonoFlex-llt/test/hg_model_ave_pool.pth")
    
    model.eval()

    model.to('cpu')

    img =img.to('cpu')

    #temp = torch.rand(1, 3, 384, 1280)

    output_cls,  output_regs = model(img)

    torch.onnx.export(model, img, "/home/lipengcheng/MonoFlex-llt/test/hg_model_ave_pool.onnx", verbose=True) 

    print(1)

  
