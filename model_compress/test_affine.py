import os,cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('/root/code/cloneServer/MonoFlex')
import torch
from torchstat import stat
from torchsummary import summary
import uuid
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
from data.datasets.kitti_utils import  read_label
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
import data.datasets.evaluation.kitti_object_eval_python.kitti_common as kitti
from data.datasets.evaluation.kitti_object_eval_python.eval import d3_box_overlap
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

def preprocess(img,mean,std,calib_filename,input_width,input_height,bottom_crop = False):
    
    if bottom_crop :
        size = np.array([img.shape[1],input_height*img.shape[1]/input_width], dtype=np.float32)
        center = np.array([img.shape[1]//2,img.shape[0]-size[1]+size[1]//2], dtype=np.float32)
    # shift = np.random.randint(low =0 ,high= (center[1]-size[1]//2)//2, size=1)

    # center[1] = center[1] - shift
    else:
        center = np.array([img.shape[1]//2,img.shape[0]//2], dtype=np.float32)
        size = np.array([img.shape[1],img.shape[0]], dtype=np.float32)
    
    # center = np.array([i / 2 for i in img.size], dtype=np.float32)
    # size = np.array([img.size[0],img.size[1]], dtype=np.float32)
    # center = np.array([img.shape[1]//2,img.shape[0]//2], dtype=np.float32)
    # size = np.array([img.shape[1],img.shape[0]], dtype=np.float32)

    """
    resize, horizontal flip, and affine augmentation are performed here.
    since it is complicated to compute heatmap w.r.t transform.
    """
    center_size = [center, size]
    trans_affine,affine_opencv = get_transfrom_matrix(
        center_size,
        [input_width, input_height]
    )
    trans_affine_inv = np.linalg.inv(trans_affine)
    # img = img.transform(
    #     (input_width, input_height),
    #     method=Image.AFFINE,
    #     data=trans_affine_inv.flatten()[:6],
    #     resample=Image.BILINEAR,
    # )
    
    img =cv2.warpAffine(np.array(img), affine_opencv, (input_width, input_height), cv2.INTER_LINEAR)
    
    # img_t = np.array(img)
    # img_t = cv2.resize(img_t,(int(input_width),int(img.shape[0]*input_width/img.shape[1])),cv2.INTER_LINEAR)
    # img = img_t[img_t.shape[0]//2-input_height//2 :img_t.shape[0]//2+input_height//2,:,:]
 

    # img =np.zeros([input_height,input_width,3])
    # img[max(0,input_height//2 -img_t.shape[0]//2): min(input_height//2 + img_t.shape[0]//2,input_height),input_width//2 -img_t.shape[1]//2:input_width//2 +img_t.shape[1]//2, :] =img_t[max(0,img_t.shape[0]//2-input_height//2):min(img_t.shape[0]//2+input_height//2,img_t.shape[0]),:,:]
    # #trans_affine [1,2]=-360
    #cv2.imwrite("/home/lipengcheng/MonoFlex-llt/test/output_file/resize.png",img)
    
   # dif = img - img2
    img = np.array(img)
    img_numpy = img.copy()
    img = img[...,[2,1,0]]
   
    
    
    #cv2.imwrite("/home/lipengcheng/MonoFlex-llt/test/output_file/save_aff.png",dif)

    img= normalize_img(img, mean,std )

    calib = Calibration(calib_filename)
    calib.matAndUpdate(trans_affine)
    return img ,img_numpy ,calib,trans_affine_inv
def affine_transform(point, matrix):

	point = point.reshape(-1, 2)
	point_exd = np.concatenate((point, np.ones((point.shape[0], 1))), axis=1)

	new_point = np.matmul(point_exd, matrix.T)

	return new_point[:, :2].squeeze()

def update2Dbox2OriIm(boxes,trans_affine,input_width,input_height):
    for i, box2d in enumerate(boxes):
        boxes[i,:2] = affine_transform(box2d[:2], trans_affine)
        boxes[i,2:] = affine_transform(box2d[2:], trans_affine)
        boxes[i,[0, 2]] = boxes[i,[0, 2]].clip(0, input_width - 1)
        boxes[i,[1, 3]] = boxes[i,[1, 3]].clip(0, input_height - 1)
    return boxes 

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

def nms(x, nms_th):
    # x:[p, z, w, h, d]
    if len(x) == 0:
        return x
    sizescore = np.zeros([len(x['name'])])
    for i in np.arange(len(x['name'])):
        sizescore[i] = x['dimensions'][i,0]*x['dimensions'][i,1]*x['dimensions'][i,2]
    sortid = np.argsort(-sizescore)
    target = np.ones([sortid.size])
    
    for i in sortid:
        
        ref_boxes = np.concatenate([x['location'][i], x['dimensions'][i], x['rotation_y'][i][..., np.newaxis]], axis=0)
        ref_boxes = ref_boxes.reshape(1,-1)
       
        for j in sortid:
            if j>i and target[j]==1:
                comp_boxes = np.concatenate([x['location'][j], x['dimensions'][j], x['rotation_y'][j][..., np.newaxis]], axis=0)
                comp_boxes = comp_boxes.reshape(1,-1)
                overlap_part = d3_box_overlap(ref_boxes, comp_boxes).astype(
                    np.float64)
                if overlap_part > nms_th:
                    target[j] = 0
                    print(overlap_part)
                    
                
        #target[i]=0                 
    saveid = np.where(target==1.0)
    return saveid
def read_picture():
    path = '/home/lipengcheng/results/yizhuang/eval3/neolix_1/img/'
    file_list = os.listdir(path)

    fps = 2 # 视频每秒2帧
    height = 1920
    weight = 1280
    size = (int(height), int(weight))  # 需要转为视频的图片的尺寸
    return [path, fps, size, file_list]



def write_video(savefolder, fps, size, file_list):
    #path, fps, size, file_list = read_picture()
    # AVI格式编码输出 XVID
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    save_path = savefolder + '/' + '%s.avi' % str(uuid.uuid1())
    video_writer = cv2.VideoWriter(save_path, four_cc, float(fps), size)
    # 视频保存在当前目录下
    for item in file_list:
        if item.endswith('.jpg') or item.endswith('.png'):
            # 找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
            img = cv2.imread(item)
            re_pics = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)  # 定尺寸
            if len(re_pics):
                video_writer.write(re_pics)

    video_writer.release()


if __name__ == "__main__":

    #write_video()

    args = default_argument_parser().parse_args() 

    cfg = setup(args)
    cfg.MODEL.BACKBONE.CONV_BODY  = 'dla34'#'hg'#
    cfg.DATASETS.TEST_SPLIT = 'test'
    cfg.MODEL.NECK.ADD_NECK =False
    cfg.MODEL.INPLACE_ABN =False
    cfg.MODEL.HEAD.ENABLE_EDGE_FUSION = False 
    id = 12
    cfg.TEST.DETECTIONS_THRESHOLD = 0.2
    model = KeypointDetector_v2(cfg).cuda()

    checkpointer = DetectronCheckpointer(
            cfg, model, save_dir="/home/lipengcheng/code/fuxian/MonoFlex/output/toy_experiments"#cfg.OUTPUT_DIR
        )
    #ckpt = '/home/lipengcheng/MonoFlex-llt/model_checkpoint-23-17-14.pth'
    #ckpt = "/home/lipengcheng/code/fuxian/MonoFlex/output/toy_experiments/NoAbnNoDcnData7000/model_moderate_best_soft_epoch40.pth"
    #ckpt ='/home/lipengcheng/MonoFlex-llt/test/new_basic_7000.pth'
    #ckpt = '/data/lpc_model/nuscense_project2d/model_checkpoint_epoch_80.pth' # monoflex 12  
    #ckpt = '/root/data/lpc_model/monoflex_16/model_checkpoint_epoch_70.pth' #目前最好
    #ckpt = '/root/data/lpc_model/monoflex_18/model_checkpoint_epoch_70.pth' #19最好
    ckpt = '/root/data/lpc_model/monoflex_21/model_moderate_best_soft.pth' #model_moderate_best_soft.pth
    #ckpt = '/data/lpc_model/monoflex_16/model_moderate_best_soft.pth'
    model_param = torch.load(ckpt)
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    
    #torch.save(model.state_dict(), "/home/lipengcheng/MonoFlex-llt/test/id_8_hg_train_verygood.pth")
    input_width = 1280 
    input_height = 384 
    model.eval()

    classes = ["Car", "Pedestrian", "Cyclist"]

    test = 'onnx'
    test_group = True
    compare_gt_det =False 
    lianghua = False 
    test_samllsize = False
    if test == 'image':
        ID_TYPE_CONVERSION = {k : v for v, k in TYPE_ID_CONVERSION.items()}
        pred_color = (0, 0, 255)
        vis_thd = 0

        if lianghua :
            src_dir = '/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training/'
            imageset_txt = src_dir + 'ImageSets/val_7000.txt'
            image_files = []
            save_dir = "/home/lipengcheng/results/yizhuang/eval2/"
            for line in open(imageset_txt, "r"):
                calibrationPath = "/home/lipengcheng/data/neolix/calib/front_3mm_intrinsics.yaml"
                base_name = line.replace("\n", "")
                image_name = base_name + ".png"
                img = cv2.imread(src_dir+'image_2/'+image_name)
                pixel_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]))
                pixel_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]))
                
                img ,img_numpy, calib,trans_affine_inv = preprocess(img,pixel_mean,pixel_std,calibrationPath,input_width,input_height)
                np.save(save_dir + base_name+'.npy',img.cpu().data.numpy())
                image_files.append(image_name)

            

        if test_group :
            
            opendataset = 'neolix'#'kitti'
            
            if opendataset == 'neolix':
                src_dir = "/root/data/neolix_dataset/test_dataset/camera_object_detection/image_2/"
                #src_dir = "/nfs/neolix_data1/neolix_dataset/all_dataset/scenes/China/beijing/yizhuang/original_133-182/d4e7a2b5b87e4b3386d20da23cc95def_front_3mm_png/"
                imgs_path_list = get_imgs_path(src_dir)
                imgs_path_list.sort()
                save_dir = "/root/data/lpc_model/neolix_test/"
                s = 0
                e = len(imgs_path_list)-1 #500#
                subfolderimg = '/img_train{}*{}/'.format(input_width,input_height)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if not os.path.exists(save_dir+subfolderimg):
                    os.makedirs(save_dir+subfolderimg)
                subfoldertxt = '/txt_train{}*{}/'.format(input_width,input_height)
                if not os.path.exists(save_dir+subfoldertxt):
                    os.makedirs(save_dir+subfoldertxt)
            else:
                #'nuscenes'
                image_files = []
                if opendataset=='kitti':
                    imageset_txt_path = '/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training/ImageSets/val_7000.txt'
                    s= 0
                    e=470
                    for line in open(imageset_txt_path, "r"):
                        base_name = line.replace("\n", "")
                        image_files.append('/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training/image_2/'+base_name+'.png')
                elif opendataset == 'waymo':
                    imageset_txt_path = os.path.join('/nfs/neolix_data1/OpenSource_dataset/camera_object_detection/kitti_format/train',"ImageSets", '{}_kitti_{}_img.txt'.format(opendataset,'train'))
                    s = 8000
                    e = 8500
                    for line in open(imageset_txt_path, "r"):
                        base_name = line.replace("\n", "")
                        image_files.append(base_name)
                elif opendataset == 'nuscenes':
                    imageset_txt_path = os.path.join('/data/lpc_data/test/nuscense/train',"ImageSets", '{}_kitti_{}_img.txt'.format(opendataset,'train'))
                    s = 8000
                    e = 8500
                    for line in open(imageset_txt_path, "r"):
                        base_name = line.replace("\n", "")
                        image_files.append(base_name)
                imgs_path_list = image_files
                save_dir = "/home/lipengcheng/results/yizhuang/eval3/{}/".format(opendataset)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    os.makedirs(save_dir+'img')
                    os.makedirs(save_dir+'eval_txt')
            saveimgpath =[]
            pp = postprcess(cfg,input_width,input_height).cuda()
            with torch.no_grad():
                for k,impath in enumerate(imgs_path_list[s:e]):
                    
                    if opendataset =='kitti':
                        calibrationPath = '/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training/calib/'+impath.split('/')[-1].replace('png','txt')
                    elif opendataset =='neolix':
                        calibrationPath = "/root/data/neolix_dataset/test_dataset/camera_object_detection/calib/"+impath.split('/')[-1].replace('png','txt')
                    elif opendataset =='nuscenes':
                        calibrationPath = impath.replace('image_2','calib').replace('png','txt')
                    label_filename_gt = "/root/data/neolix_dataset/test_dataset/camera_object_detection/label_2/"+impath.split('/')[-1].replace('png','txt')
                    if os.path.exists(label_filename_gt):
                        label = read_label(label_filename_gt)
                    fh = open(os.path.join(save_dir+subfoldertxt,impath.split('/')[-1].replace('png','txt')) ,'w')
                    #imgpath ='/nfs/neolix_data1/neolix_dataset/all_dataset/scenes/China/beijing/yizhuang/original_133-182/d4e7a2b5b87e4b3386d20da23cc95def_front_3mm_png/10_7_record.00000_1622538455.344359400.png'
                    img = cv2.imread(impath)
                    #cv2.imwrite("/home/lipengcheng/results/"+ 'ori.png' ,img)
                    img_vis = img.copy()
                    h,w,_  =img.shape
                    #img = Image.open(impath).convert('RGB')
                    name = impath.split('/')[-1]

                    pixel_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]))

                    pixel_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]))

                    img ,img_numpy, calib ,trans_affine_inv = preprocess(img,pixel_mean,pixel_std,calibrationPath,input_width,input_height)

                    img =img.unsqueeze(0).to('cuda')    
                    #cv2.imwrite(save_dir+ 'crop.png' ,img_numpy)
                    #cv2.imwrite("/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/image_yz/"+name[:-4]+ '.png' ,img_numpy)
                    # copyfile('/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/calib/0.txt','/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/calib/'+name[:-4]+ '.txt')

                    #delta = img_numpy - img4 
                    #img = torch.ones((1, 3, 384, 1280)).to('cuda')
                    img3 = img_numpy.copy()

                    output_cls,  output_regs ,features =  model(img)

                    
                    #torch.save(model.state_dict(), "/home/lipengcheng/MonoFlex-llt/test/dla_model.pth")

                    clses,alphas,rotys,box2d, dims, score,locs = pp(output_cls,  output_regs, calib)
                    if clses is None:
                        #cv2.imwrite(save_dir+name[:-4]+ '_2d.jpg' ,img2)
                        cv2.imwrite(save_dir+subfolderimg+name[:-4]+ '_3d_3.jpg' ,img_vis)
                        saveimgpath.append(save_dir+subfolderimg+name[:-4]+ '_3d_3.jpg')
                        continue
                    clses,alphas,rotys,box2d, dims, score,locs =clses.data.cpu().numpy(),alphas.data.cpu().numpy(),rotys.data.cpu().numpy(),box2d.data.cpu().numpy(), dims.data.cpu().numpy(), score.data.cpu().numpy(),locs.data.cpu().numpy()
                    box2d = update2Dbox2OriIm(box2d,trans_affine_inv,w,h)
                    for i in range(box2d.shape[0]):
                        #img2.draw_box(box_coord=box2d[i], edge_color='g')
                        if score[i]>=vis_thd:
                            corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
                            calib = Calibration(calibrationPath)
                            corners_2d, depth = calib.project_rect_to_image(corners3d)
                            img3 = draw_projected_box3d(img_vis, corners_2d, cls=ID_TYPE_CONVERSION[clses[i]], color=pred_color, draw_orientation=True,draw_corner=False)
                            img3 = Visualizer(img3.copy())
                            img3.draw_text(text='{},{}, {}, {}, {:.3f},{:.2f},{:.2f},{:.2f}'.format("id :",i, ID_TYPE_CONVERSION[clses[i]],int(clses[i]), score[i], dims[i,0],dims[i,1],dims[i,2]), position=(int(box2d[i, 0]), int(box2d[i, 1])))
                            img3 = img3.output.get_image().astype(np.uint8)
                            fh.write(classes[int(clses[i])]+' '+ str(0)+' '+str(0)+ ' '+str(alphas[i])+ ' '+str(int(box2d[i,0]))+ ' '+str(int(box2d[i,1]))+' '+ str(int(box2d[i,2]))+ ' '+str(int(box2d[i,3]))+ ' '+str(dims[i,0])+ ' '+str(dims[i,1])+' '+ str(dims[i,2])+ ' '+str(locs[i,0])+ ' '+str(locs[i,1])+ ' '+str(locs[i,2])+ ' '+str(rotys[i])+'\n')
                    cv2.imwrite(save_dir+subfolderimg+name[:-4]+ '_3d_3.jpg' ,img3)
                    saveimgpath.append(save_dir+subfolderimg+name[:-4]+ '_3d_3.jpg')
                    fh.close()
                    # for bb in box2d:
                    #     cv2.rectangle(img_vis, (int(bb[0]),int(bb[1])), (int(bb[2]),int(bb[3])), (0,0,255), thickness = 2)

                    # cv2.imwrite('/home/lipengcheng/results/tmp/a.jpg',img_vis )
                    # img_numpy =img_numpy[...,[2,1,0]]
                    # img2 = Visualizer(np.array(img_numpy))

                    # for i in range(box2d.shape[0]):
                    #     if score[i]>vis_thd:
                    #         img2.draw_box(box_coord=box2d[i], edge_color='r')

                    # img2 = img2.output.get_image()
                    # cv2.imwrite(save_dir+name[:-4]+ '_2d.jpg' ,img2)
                    print(name)
                    #cv2.imwrite('/home/lipengcheng/MonoFlex-llt/test/output_file/7_draw_2d_box'+"_0{}.jpg".format(str(k)),img2)
        write_video(save_dir, 5, (960,540), saveimgpath)
        if compare_gt_det :
            #src_dir = "/nfs/neolix_data1/neolix_dataset/all_dataset/scenes/China/beijing/yizhuang/original_133-182/d4e7a2b5b87e4b3386d20da23cc95def_front_3mm_png/"
            src_dir = "/nfs/neolix_data1//neolix_dataset/test_dataset/camera_object_detection/image_2/"
            imgs_path_list = get_imgs_path(src_dir)
            imgs_path_list.sort()
            save_dir = "/home/lipengcheng/results/yizhuang/eval2/neolix"
            pp = postprcess(cfg).cuda()
            
            for k,impath in enumerate(imgs_path_list[:1000]):
                calibrationPath = "/home/lipengcheng/data/neolix/calib/front_3mm_intrinsics.yaml"
                #fh = open(os.path.join("/nfs/neolix_data1/neolix_dataset/test_dataset/camera_object_detection/eval/eval_python",impath.split('/')[-1].replace('png','txt')) ,'w')
                #imgpath ='/nfs/neolix_data1/neolix_dataset/test_dataset/camera_object_detection/image_2/20210521180702.record.00011_1621592106.259370.png'
                img = cv2.imread(impath)
                img_vis_2d = img.copy()
                img_vis_3d = img.copy()
                h,w,_  =img.shape
                #img = Image.open(impath).convert('RGB')
                name = impath.split('/')[-1]

                pixel_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]))

                pixel_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]))

                img ,img_numpy, calib ,trans_affine_inv = preprocess(img,pixel_mean,pixel_std,calibrationPath)

                img =img.unsqueeze(0).to('cuda')    

                # cv2.imwrite("/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/image_yz/"+name[:-4]+ '.png' ,img_numpy)
                # copyfile('/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/calib/0.txt','/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/testing/calib/'+name[:-4]+ '.txt')

                #delta = img_numpy - img4 
                #img = torch.ones((1, 3, 384, 1280)).to('cuda')
                img3 = img_numpy.copy()

                output_cls,  output_regs ,features =  model(img)

                
                #torch.save(model.state_dict(), "/home/lipengcheng/MonoFlex-llt/test/dla_model.pth")

                clses,alphas,rotys,box2d, dims, score,locs = pp(output_cls,  output_regs, calib)




                if clses is None:
                    #cv2.imwrite(save_dir+name[:-4]+ '_2d.jpg' ,img2)
                    #cv2.imwrite(save_dir+name[:-4]+ '.jpg' ,img_vis)
                    continue
                clses,alphas,rotys,box2d, dims, score,locs =clses.data.cpu().numpy(),alphas.data.cpu().numpy(),rotys.data.cpu().numpy(),box2d.data.cpu().numpy(), dims.data.cpu().numpy(), score.data.cpu().numpy(),locs.data.cpu().numpy()
                box2d = update2Dbox2OriIm(box2d,trans_affine_inv,w,h)
                #########################################
                label_filename_gt =os.path.join('/nfs/neolix_data1/neolix_dataset/test_dataset/camera_object_detection/label_2/',impath.split('/')[-1].replace('png','txt')) 
                annos_gt = kitti.get_label_anno(label_filename_gt)
                #bboxes = nms(annos_gt, 0.3)
                ####################################################
                # for i in range(box2d.shape[0]):
                #     corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
                #     calib = Calibration(calibrationPath)
                #     corners_2d, depth = calib.project_rect_to_image(corners3d)
                #     img3 = draw_projected_box3d(img_vis_3d, corners_2d, cls=ID_TYPE_CONVERSION[clses[i]], color=(0, 0, 255),draw_orientation=True, draw_corner=False)
                #     img3 = Visualizer(img3.copy())
                #     #img3.draw_text(text='{},{}, {}, {}, {:.3f},{:.2f},{:.2f},{:.2f}'.format("id :",i, ID_TYPE_CONVERSION[clses[i]],int(clses[i]), score[i], dims[i,0],dims[i,1],dims[i,2]), position=(int(box2d[i, 0]), int(box2d[i, 1])))
                #     img3 = img3.output.get_image().astype(np.uint8)
                       
                # cv2.imwrite(save_dir+'/vis_3d/'+name[:-4]+ '_det.jpg' ,img3)
                
                # for i,bbox in enumerate(annos_gt['bbox']):
                #     #img2.draw_box(box_coord=box2d[i], edge_color='g')
                #     box3d = annos_gt['dimensions'][i]
                #     corners3d = box3d_to_corners(annos_gt['location'][i], annos_gt['dimensions'][i], annos_gt['rotation_y'][i])
                #     calib = Calibration(calibrationPath)
                #     corners_2d, depth = calib.project_rect_to_image(corners3d)
                #     img3 = draw_projected_box3d(img_vis_3d, corners_2d, cls=annos_gt['name'][i], color=(0,255,0),draw_orientation=True, draw_corner=False)
                #     img3 = Visualizer(img3.copy())
                #     #img3.draw_text(text='{},{}, {}, {}, {:.3f},{:.2f},{:.2f},{:.2f}'.format("id :",i, annos_gt['name'][i] ,int(clses[i]), score[i], dims[i,0],dims[i,1],dims[i,2]), position=(int(box2d[i, 0]), int(box2d[i, 1])))
                #     img3 = img3.output.get_image().astype(np.uint8)

                # cv2.imwrite(save_dir+'/vis_3d/'+name[:-4]+ '_gt.jpg' ,img3)

                #fh.close()
                for bb in box2d:
                    cv2.rectangle(img_vis_2d, (int(bb[0]),int(bb[1])), (int(bb[2]),int(bb[3])), (0,0,255), thickness = 2)
                for bb in annos_gt['bbox']:
                    cv2.rectangle(img_vis_2d, (int(bb[0]),int(bb[1])), (int(bb[2]),int(bb[3])), (0,255,0), thickness = 2)
                
                cv2.imwrite(save_dir+'/vis_2d/'+name[:-4]+ '.jpg' ,img_vis_2d)
                # img_numpy =img_numpy[...,[2,1,0]]
                # img2 = Visualizer(np.array(img_numpy))

                # for i in range(box2d.shape[0]):
                #     if score[i]>vis_thd:
                #         img2.draw_box(box_coord=box2d[i], edge_color='r')

                # img2 = img2.output.get_image()
                # cv2.imwrite(save_dir+name[:-4]+ '_2d.jpg' ,img2)
                print(name)
                #cv2.imwrite('/home/lipengcheng/MonoFlex-llt/test/output_file/7_draw_2d_box'+"_0{}.jpg".format(str(k)),img2)
        else :
            print("end")
            if test_samllsize :
                for k in range(0,9):
                    # path = "/home/lipengcheng/data/neolix/image_2/"+"0{}.png".format(str(k))
                    path = "/home/lipengcheng/data/neolix/image_2/"+"{}.jpg".format(str(k))
                    #path ="/nfs/neolix_data1/neolix_dataset/all_dataset/scenes/China/beijing/yizhuang/original_133-182/d4e7a2b5b87e4b3386d20da23cc95def_front_3mm_png/10_7_record.00022_1622537085.735201000.png"
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

                    output_cls,  output_regs, feature =  model(img)

                    #torch.save(model.state_dict(), "/home/lipengcheng/MonoFlex-llt/test/dla_model.pth")

                    clses,alphas,rotys,box2d, dims, score,locs = pp(output_cls,  output_regs, calib)
                    if clses is None:
                        continue
                    clses,alphas,rotys,box2d, dims, score,locs =clses.data.cpu().numpy(),alphas.data.cpu().numpy(),rotys.data.cpu().numpy(),box2d.data.cpu().numpy(), dims.data.cpu().numpy(), score.data.cpu().numpy(),locs.data.cpu().numpy()
                    for i in range(box2d.shape[0]):
                        #img2.draw_box(box_coord=box2d[i], edge_color='g')
                        if score[i]>=vis_thd:
                            corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
                            corners_2d, depth = calib.project_rect_to_image(corners3d)
                            img3 = draw_projected_box3d(img3, corners_2d, cls=ID_TYPE_CONVERSION[clses[i]], color=pred_color, draw_corner=False)
                            img3 = Visualizer(img3.copy())
                            img3.draw_text(text='{},{}, {}, {}, {:.3f},{:.2f},{:.2f},{:.2f}'.format("id :",i, ID_TYPE_CONVERSION[clses[i]],int(clses[i]), score[i], dims[i,0],dims[i,1],dims[i,2]), position=(int(box2d[i, 0]), int(box2d[i, 1])))
                            img3 = img3.output.get_image().astype(np.uint8)

                    cv2.imwrite('/home/lipengcheng/results/tmp/'+"{}.jpg".format(str(k)),img3)
                    img_numpy =img_numpy[...,[2,1,0]]
                    img2 = Visualizer(np.array(img_numpy))

                    for i in range(box2d.shape[0]):
                        if score[i]>vis_thd:
                            img2.draw_box(box_coord=box2d[i], edge_color='r')

                    img2 = img2.output.get_image()

                    #cv2.imwrite('/home/lipengcheng/MonoFlex-llt/test/output_file/7_draw_2d_box'+"_2{}.png".format(str(k)),img2)
                    
                    print(box2d.shape[0])
    if test =='onnx':
        #summary(model, input_size=(3, 384, 1280), batch_size=-1)
        model.to('cpu')
        # stat(model, (3, 384, 1280))
        # stat(model, (3, 384, 960))
        # stat(model, (3, 416, 960))

        img = torch.rand((1, 3, 384, 960))

        #temp = torch.rand(1, 3, 384, 1280)

        output_cls,  output_regs, features = model(img)

        torch.onnx.export(model, img, "/root/data/lpc_model/monoflex_24/monoflex_24.onnx", input_names = ['input'],output_names =['clses','regs','feature'], verbose=True) 

    print(1)

  
