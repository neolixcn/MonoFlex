import os,cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
#from model.backbone.HGFilters import HGFilter
from torch import nn
from config import cfg
from PIL import Image
import numpy as np
import datetime
from config import TYPE_ID_CONVERSION
from utils.check_point import DetectronCheckpointer
from detector_predictor_ratio import _predictor,postprcess
from engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from shutil import copyfile
from data.datasets.kitti_utils import draw_projected_box3d, \
	draw_box3d_on_top, init_bev_image, draw_bev_box3d
from utils.visualizer import Visualizer
from skimage import transform as trans 
from model.head.detector_infer import make_post_processor
from data.transforms import build_transforms 
from engine.visualize_infer import show_image_with_boxes_test
from data.datasets.kitti_utils import Calibration
from model.backbone import build_backbone
from model.detector_copy import KeypointDetector_v2


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

def box3d_to_corners(locs, dims, roty):
	# 3d bbox template
	h, w, l = dims
	x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
	y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
	z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

	# rotation matirx
	R = np.array([[np.cos(roty), 0, np.sin(roty)],
				  [0, 1, 0],
				  [-np.sin(roty), 0, np.cos(roty)]])

	corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
	corners3d = np.dot(R, corners3d).T
	corners3d = corners3d + locs

	return corners3d

def preprocess(img,mean,std,calib_filename):
    input_width = 1280 
    input_height = 384 
    center = np.array([img.shape[1]//2,img.shape[0]//2], dtype=np.float32)
    #center = np.array([img.shape[1]//2,img.shape[0]//4*3], dtype=np.float32)
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
    #trans_affine [1,2]=-360
    #cv2.imwrite("/home/lipengcheng/MonoFlex-llt/test/output_file/save_ori.png",np.array(img))
    img =cv2.warpAffine(np.array(img), affine_opencv, (input_width, input_height))
    img_numpy = img.copy()
    img = img[...,[2,1,0]]
    
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

# class KeypointDetector_v2(nn.Module):
#     '''
#     Generalized structure for keypoint based object detector.
#     main parts:
#     - backbone
#     - heads
#     '''

#     def __init__(self, cfg):
#         super(KeypointDetector_v2, self).__init__()
#         self.backbone = build_backbone(cfg)
#         self.heads = _predictor(self.backbone.out_channels)

#     def forward(self, images):
        
#         features = self.backbone(images)

#         output_cls,  output_regs= self.heads(features)
        
#         return output_cls,  output_regs

def get_imgs_path(src_dir):
    imgs_path_list = []
    for (root, dirs, files) in os.walk(src_dir):
        if (len(files) != 0):
            for cur_file in files:
                cur_file_path = os.path.join(root, cur_file)
                file_name_prefix, file_name_suffix = os.path.splitext(cur_file)
                if not file_name_suffix.lower() in (".jpg", ".jpeg", ".png"):
                    continue
                else:
                    imgs_path_list.append(cur_file_path)
    return imgs_path_list

if __name__ == "__main__":

    args = default_argument_parser().parse_args() 

    cfg = setup(args)
    cfg.MODEL.BACKBONE.CONV_BODY  = 'dla34'#'hg'#
    cfg.DATASETS.TEST_SPLIT = 'test'
    cfg.MODEL.INPLACE_ABN =False
    cfg.MODEL.HEAD.ENABLE_EDGE_FUSION = True 
    id = 12
    cfg.TEST.DETECTIONS_THRESHOLD = 0.2
    model = KeypointDetector_v2(cfg).cuda()
    
    # state_dict = torch.load("/home/lipengcheng/MonoFlex-llt/test/dla34_randomcrop.pth")
    # #model.load_state_dict(state_dict)
    # for key in list(state_dict.keys()):
    #     print(key)
    #     if "predictor" in key:
    #         if "trunc" in key:
    #             del state_dict[key]
    #         else:
    #             newkey = key.replace("predictor.", "")
    #             state_dict[newkey] = state_dict[key]
    #             del state_dict[key]

    # model.load_state_dict(state_dict)
    checkpointer = DetectronCheckpointer(
            cfg, model, save_dir="/home/lipengcheng/code/fuxian/MonoFlex/output/toy_experiments"#cfg.OUTPUT_DIR
        )
    #ckpt = '/home/lipengcheng/MonoFlex-llt/model_checkpoint-23-17-14.pth'
    #ckpt = "/home/lipengcheng/code/fuxian/MonoFlex/output/toy_experiments/NoAbnNoDcnData7000/model_moderate_best_soft_epoch40.pth"
    #ckpt ='/home/lipengcheng/MonoFlex-llt/test/new_basic_7000.pth'
    ckpt = '/home/lipengcheng/code/fuxian/MonoFlex/output/toy_experiments/centernet2d_crop/model_moderate_best_soft.pth'
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    model_param = torch.load(ckpt)
    #torch.save(model.state_dict(), "/home/lipengcheng/MonoFlex-llt/test/id_8_hg_train_verygood.pth")
    
    model.eval()

    test = 'image'
    test_group = False 
    if test == 'image':
        ID_TYPE_CONVERSION = {k : v for v, k in TYPE_ID_CONVERSION.items()}
        pred_color = (0, 0, 255)
        vis_thd = 0
        
        if test_group :
            src_dir = "/nfs/neolix_data1/neolix_dataset/all_dataset/scenes/China/beijing/yizhuang/original_133-182/d4e7a2b5b87e4b3386d20da23cc95def_front_3mm_png"
            imgs_path_list = get_imgs_path(src_dir)
            imgs_path_list.sort()
            filename  ='/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/Imagesets/test_yz.txt'
            namelist =[]
            save_dir = "/home/lipengcheng/results/yizhuang/3/"
            pp = postprcess(cfg).cuda()
            for k,impath in enumerate(imgs_path_list[:1000]):


                calibrationPath = "/home/lipengcheng/data/neolix/calib/front_3mm_intrinsics.yaml"

                #img = cv2.imread(impath)
                
                name = impath.split('/')[-1]

                namelist.append(name[:-4])

                # pixel_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]))

                # pixel_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]))

                # img ,img_numpy, calib= preprocess(img,pixel_mean,pixel_std,calibrationPath)

                # img =img.unsqueeze(0).to('cuda')    

                # cv2.imwrite("/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/image_yz/"+name[:-4]+ '.png' ,img_numpy)
                # copyfile('/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/calib/0.txt','/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/calib/'+name[:-4]+ '.txt')

                #delta = img_numpy - img4 
                #img = torch.ones((1, 3, 384, 1280)).to('cuda')
                # img3 = img_numpy.copy()

                # output_cls,  output_regs =  model(img)

                
                # #torch.save(model.state_dict(), "/home/lipengcheng/MonoFlex-llt/test/dla_model.pth")

                # clses,alphas,rotys,box2d, dims, score,locs = pp(output_cls,  output_regs, calib)
                # if clses is None:
                #     #cv2.imwrite(save_dir+name[:-4]+ '_2d.jpg' ,img2)
                #     cv2.imwrite(save_dir+name[:-4]+ '_3d_3.jpg' ,img3)
                #     continue
                # clses,alphas,rotys,box2d, dims, score,locs =clses.data.cpu().numpy(),alphas.data.cpu().numpy(),rotys.data.cpu().numpy(),box2d.data.cpu().numpy(), dims.data.cpu().numpy(), score.data.cpu().numpy(),locs.data.cpu().numpy()
                # for i in range(box2d.shape[0]):
                #     #img2.draw_box(box_coord=box2d[i], edge_color='g')
                #     if score[i]>vis_thd:
                #         corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
                #         corners_2d, depth = calib.project_rect_to_image(corners3d)
                #         img3 = draw_projected_box3d(img3, corners_2d, cls=ID_TYPE_CONVERSION[clses[i]], color=pred_color, draw_corner=False)
                #         img3 = Visualizer(img3.copy())
                #         img3.draw_text(text='{},{}, {}, {}, {:.3f},{:.2f},{:.2f},{:.2f}'.format("id :",i, ID_TYPE_CONVERSION[clses[i]],int(clses[i]), score[i], dims[i,0],dims[i,1],dims[i,2]), position=(int(box2d[i, 0]), int(box2d[i, 1])))
                #         img3 = img3.output.get_image().astype(np.uint8)

                # cv2.imwrite(save_dir+name[:-4]+ '_3d_3.jpg' ,img3)


                # img_numpy =img_numpy[...,[2,1,0]]
                # img2 = Visualizer(np.array(img_numpy))

                # for i in range(box2d.shape[0]):
                #     if score[i]>vis_thd:
                #         img2.draw_box(box_coord=box2d[i], edge_color='r')

                # img2 = img2.output.get_image()
                # cv2.imwrite(save_dir+name[:-4]+ '_2d.jpg' ,img2)
                print(name)
                #cv2.imwrite('/home/lipengcheng/MonoFlex-llt/test/output_file/7_draw_2d_box'+"_0{}.jpg".format(str(k)),img2)
            fp = open('test_yz.txt','w')
            for value in namelist:
                fp.write(str(value))
                fp.write('\n')
            fp.close()
            #np.savetxt('/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/ImageSets/test_yz.txt',np.array(namelist))
        else :
            for k in range(0,10):
                path = "/home/lipengcheng/data/neolix/image_2/"+"0{}.png".format(str(k))
                #path = "/home/lipengcheng/MonoFlex-llt/test/input_file/"+"{}.png".format(str(k))
                pp = postprcess(cfg).cuda()

                calibrationPath = "/home/lipengcheng/data/neolix/calib/front_3mm_intrinsics.yaml"

                img = cv2.imread(path)

                pixel_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]))

                pixel_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]))

                img ,img_numpy, calib= preprocess(img,pixel_mean,pixel_std,calibrationPath)

                #cv2.imwrite('/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/image_2/'+"{}.png".format(str(k)),img_numpy)

                img =img.unsqueeze(0).to('cuda')

                #delta = img_numpy - img4 
                #img = torch.ones((1, 3, 384, 1280)).to('cuda')
                img3 = img_numpy.copy()

                output_cls,  output_regs =  model(img)

                #torch.save(model.state_dict(), "/home/lipengcheng/MonoFlex-llt/test/dla_model.pth")

                clses,alphas,rotys,box2d, dims, score,locs = pp(output_cls,  output_regs, calib)
                if clses is None:
                    continue
                clses,alphas,rotys,box2d, dims, score,locs =clses.data.cpu().numpy(),alphas.data.cpu().numpy(),rotys.data.cpu().numpy(),box2d.data.cpu().numpy(), dims.data.cpu().numpy(), score.data.cpu().numpy(),locs.data.cpu().numpy()
                # for i in range(box2d.shape[0]):
                #     #img2.draw_box(box_coord=box2d[i], edge_color='g')
                #     if score[i]>vis_thd:
                #         corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
                #         corners_2d, depth = calib.project_rect_to_image(corners3d)
                #         img3 = draw_projected_box3d(img3, corners_2d, cls=ID_TYPE_CONVERSION[clses[i]], color=pred_color, draw_corner=False)
                #         img3 = Visualizer(img3.copy())
                #         img3.draw_text(text='{},{}, {}, {}, {:.3f},{:.2f},{:.2f},{:.2f}'.format("id :",i, ID_TYPE_CONVERSION[clses[i]],int(clses[i]), score[i], dims[i,0],dims[i,1],dims[i,2]), position=(int(box2d[i, 0]), int(box2d[i, 1])))
                #         img3 = img3.output.get_image().astype(np.uint8)

                # cv2.imwrite('/home/lipengcheng/MonoFlex-llt/test/output_file/7_draw_3d_box'+"_2{}.png".format(str(k)),img3)
                img_numpy =img_numpy[...,[2,1,0]]
                img2 = Visualizer(np.array(img_numpy))

                for i in range(box2d.shape[0]):
                    if score[i]>vis_thd:
                        img2.draw_box(box_coord=box2d[i], edge_color='r')

                img2 = img2.output.get_image()

                cv2.imwrite('/home/lipengcheng/MonoFlex-llt/test/output_file/7_draw_2d_box'+"_3{}.png".format(str(k)),img2)
                
                print(box2d.shape[0])
    if test =='onnx':
        model.to('cpu')

        img = torch.rand((1, 3, 384, 1280))

        #temp = torch.rand(1, 3, 384, 1280)

        output_cls,  output_regs, features = model(img)

        torch.onnx.export(model, img, "/home/lipengcheng/MonoFlex-llt/test/dla34_fusion2.onnx", verbose=True) 

    print(1)

  
