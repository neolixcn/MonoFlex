import sys
sys.path.append('/root/code/cloneServer/MonoFlex')
import data.datasets.evaluation.kitti_object_eval_python.kitti_common as kitti
from data.datasets.evaluation.kitti_object_eval_python.eval import get_official_eval_result, get_coco_eval_result,get_neolix_eval_result
import os ,cv2 
from utils.visualizer import Visualizer
from data.datasets.kitti_utils import Calibration
from data.datasets.kitti_utils import draw_projected_box3d, \
	draw_box3d_on_top, init_bev_image, draw_bev_box3d
import numpy as np   
from config import TYPE_ID_CONVERSION
from data.datasets.evaluation.kitti_object_eval_python.eval import d3_box_overlap
from data.datasets.kitti_utils import  read_label,show_heatmap, show_image_with_boxes
PI = np.pi

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
def nms(x, nms_th):
    
    if len(x) == 0:
        return x
    sizescore = np.zeros([len(x['name'])])
    for i in np.arange(len(x['name'])):
        sizescore[i] = x['dimensions'][i,0]*x['dimensions'][i,1]*x['dimensions'][i,2]
    sortid = np.argsort(-sizescore)
    target = np.ones([sortid.size])
    
    for ii,i in enumerate(sortid):
        if target[i]==0:
            continue
        ref_boxes = np.concatenate([x['location'][i], x['dimensions'][i], x['rotation_y'][i][..., np.newaxis]], axis=0)
        ref_boxes = ref_boxes.reshape(1,-1)
        #print('+++++++++++++++++++++++')
        for jj, j in enumerate(sortid):
            if jj>ii and target[j]==1:
                comp_boxes = np.concatenate([x['location'][j], x['dimensions'][j], x['rotation_y'][j][..., np.newaxis]], axis=0)
                comp_boxes = comp_boxes.reshape(1,-1)
                overlap_part = d3_box_overlap(ref_boxes, comp_boxes).astype(
                    np.float64)
                #print(i, ' ', j,' ', overlap_part)
                if overlap_part > nms_th:
                    target[j] = 0
                    #print(overlap_part)
                    
                
        #target[i]=0                 
    saveid= np.where(target==1.0)
    return saveid[0]

def get_label_annos_gen(label_folder,det_folder,imageset_txt_path= [],only_genGT =False):
    
    annos_det = []
    annos_gt = []
    label_filenames = os.listdir(label_folder)
    label_filenames.sort()
    if imageset_txt_path != []:
        image_files = []
        
        #imageset_txt_path = os.path.join("/data/lpc_data/test/nuscense/train/","ImageSets", '{}_kitti_{}_label.txt'.format('nuscenes','train'))#_kitti
        #imageset_txt_path = os.path.join("/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training/","ImageSets", '{}_7000.txt'.format('val'))
        for line in open(imageset_txt_path, "r"):
            base_name = line.replace("\n", "") if line.replace("\n", "").split('.')[-1]=='txt' else line.replace("\n", "")+'.txt'
            #if 'kitti' not in base_name:
            image_files.append(base_name)
            
        label_filenames = image_files
    filename = []
    labels =[]
    labels_det =[]
    
    for idx in label_filenames:
        label_filename_det = os.path.join(det_folder, idx.split('/')[-1])
        label_filename_gt = os.path.join(label_folder, idx.split('/')[-1])
        if only_genGT:
            if os.path.exists(label_filename_gt):
                label = read_label(label_filename_gt)
                annos_gt.append(kitti.get_label_anno(label_filename_gt))
                filename.append(idx)
                labels.append(label)
        else:
            if os.path.exists(label_filename_det) and os.path.exists(label_filename_gt):
                label = read_label(label_filename_gt)
                label_det = read_label(label_filename_det)
                annos_gt.append(kitti.get_label_anno(label_filename_gt))
                annos_det.append(kitti.get_label_anno(label_filename_det))
                filename.append(idx)
                labels.append(label)
                labels_det.append(label_det)

    return annos_gt,annos_det,filename,labels,labels_det

def encode_alpha_multibin(alpha, num_bin=2, margin=1 / 6):
    # encode alpha (-PI ~ PI) to 2 classes and 1 regression
    encode_alpha = np.zeros(num_bin * 2)
    bin_size = 2 * np.pi / num_bin # pi
    margin_size = bin_size * margin # pi / 6

    bin_centers = np.array([0, PI / 2, PI, - PI / 2]) 
    range_size = bin_size / 2 + margin_size

    offsets = alpha - bin_centers
    offsets[offsets > np.pi] = offsets[offsets > np.pi] - 2 * np.pi
    offsets[offsets < -np.pi] = offsets[offsets < -np.pi] + 2 * np.pi

    for i in range(num_bin):
        offset = offsets[i]
        if abs(offset) < range_size:
            encode_alpha[i] = 1
            encode_alpha[i + num_bin] = offset

    return encode_alpha

def updateAnno(dt_annos,saveid):
    dt_annos['name'] = dt_annos['name'][saveid]
    dt_annos['truncated'] = dt_annos['truncated'][saveid]
    dt_annos['occluded'] = dt_annos['occluded'][saveid]
    dt_annos['alpha'] = dt_annos['alpha'][saveid]
    dt_annos['bbox'] = dt_annos['bbox'][saveid]
    dt_annos['dimensions'] = dt_annos['dimensions'][saveid]
    dt_annos['location'] = dt_annos['location'][saveid]
    dt_annos['rotation_y'] = dt_annos['rotation_y'][saveid]
    dt_annos['score'] = dt_annos['score'][saveid]


    return dt_annos

def vis_gt(labels,imgfolder,filename,save_dir,vis_neolix= True):
   
    for j,objs in enumerate(labels):
        shotname, extension = os.path.split( filename[j])
        filename[j] = extension
        imgpath = os.path.join(imgfolder , filename[j].replace('txt','png'))
        if not os.path.exists(imgpath):
            continue
        if vis_neolix:
            calibrationPath = "/home/lipengcheng/data/neolix/calib/front_3mm_intrinsics.yaml"
        else:
            calibrationPath = os.path.join(imgfolder.replace('image_2','calib') , filename[j])
        calib = Calibration(calibrationPath)
        img = cv2.imread(imgpath)
        img4 = img.copy()
        reg_mask = np.zeros([len(objs)])
        multibin_size = 4
        img_w, img_h = img.shape[1], img.shape[0]
        orientations =  np.zeros([len(objs), multibin_size * 2], dtype=np.float32)
        cls_ids = np.zeros([len(objs)], dtype=np.int32)
        bboxes = np.zeros([len(objs), 4], dtype=np.float32)
        keypoints = np.zeros([len(objs), 10, 3], dtype=np.float32)
        target_centers = np.zeros([len(objs), 2], dtype=np.int32)
        keypoints_depth_mask = np.zeros([len(objs), 3], dtype=np.float32)
        dimensions = np.zeros([len(objs), 3], dtype=np.float32)
		# 3d location
        locations = np.zeros([len(objs), 3], dtype=np.float32)
		# rotation y
        rotys = np.zeros([len(objs)], dtype=np.float32)
		# alpha (local orientation)
        alphas = np.zeros([len(objs)], dtype=np.float32)
		# offsets from center to expected_center
        offset_3D = np.zeros([len(objs), 2], dtype=np.float32)
        img4 = np.ones((768, 768, 3), dtype=np.uint8) * 230
        down_ratio = 1
        pad_size =[0,0]
        for i,obj in enumerate(objs):
            cls = obj.type
            if not cls == 'Car' and not cls == 'Pedestrian' and not cls == 'Cyclist':
                continue

            cls_id = TYPE_ID_CONVERSION[cls]
            if cls_id < 0: continue
            cls_ids[i] = cls_id
            reg_mask[i] =1 
            bboxes[i] = obj.box2d.copy()
            
            locs = obj.t.copy()
            locs[1] = locs[1] - obj.h / 2

            proj_center, depth = calib.project_rect_to_image(locs.reshape(-1, 3))
            target_centers[i] = proj_center[0]
            
            corners_3d = obj.generate_corners3d()
            bot_top_centers = np.stack((corners_3d[:4].mean(axis=0), corners_3d[4:].mean(axis=0)), axis=0)
            keypoints_3D = np.concatenate((corners_3d, bot_top_centers), axis=0)
            keypoints_2D, _ = calib.project_rect_to_image(keypoints_3D)
            keypoints_x_visible = (keypoints_2D[:, 0] >= 0) & (keypoints_2D[:, 0] <= img_w  - 1)
            keypoints_y_visible = (keypoints_2D[:, 1] >= 0) & (keypoints_2D[:, 1] <= img_h - 1)
            keypoints_z_visible = (keypoints_3D[:, -1] > 0)

            offset_3D = proj_center - target_centers[i]
            # xyz visible
            keypoints_visible = keypoints_x_visible & keypoints_y_visible & keypoints_z_visible

            keypoints[i] = np.concatenate((keypoints_2D - target_centers[i].reshape(1, -1), keypoints_visible[:, np.newaxis]), axis=1)

            orientations[i] = encode_alpha_multibin(obj.alpha, num_bin= 4)

            img4 = draw_bev_box3d(img4, corners_3d[np.newaxis, :], thickness=2, color=(0,255,0), scores=None)
        img3 = show_image_with_boxes(img, cls_ids, target_centers, bboxes.copy(), keypoints, reg_mask, 
                                offset_3D, down_ratio, pad_size, orientations, vis=True)
        img4 = cv2.resize(img4, (img3.shape[0], img3.shape[0]))
        stack_img = np.concatenate([img3, img4], axis=1)
        
        cv2.imwrite(save_dir+'/vis_3d/'+filename[j].replace('txt','jpg') ,stack_img)
        # fh = open(save_dir+'/vis_3d/'+filename[j] ,'w')
        # fh.close()
        #print(save_dir+'/vis_3d/'+filename[j].replace('txt','.jpg'))

def vis_bev(labels,label_det,imgfolder,filename,save_dir):
    if not os.path.exists(save_dir+'/vis_3d/'):
        os.makedirs(save_dir+'/vis_3d/')
    for j,objs in enumerate(zip(labels,label_det)):
        imgpath = imgfolder + filename[j].split('/')[-1].replace('txt','png')
        if not os.path.exists(imgpath):
            continue
        img = cv2.imread(imgpath)

        img4 = np.ones((768, 768, 3), dtype=np.uint8) * 230
        for i,obj in enumerate(objs[0]):
            #print(obj.type)
            if obj.type == 'Van' or obj.type == 'Car'or obj.type == 'Pedestrian'or obj.type == 'Cyclist':
                corners_3d = obj.generate_corners3d()
                img4 = draw_bev_box3d(img4, corners_3d[np.newaxis, :], thickness=2, color=(0,0,255), scores=None)
        for i,obj in enumerate(objs[1]):
            corners_3d = obj.generate_corners3d()
            img4 = draw_bev_box3d(img4, corners_3d[np.newaxis, :], thickness=2, color=(0,255,0), scores=None)
            # cls = obj.type
            # cls_id = TYPE_ID_CONVERSION[cls]
            # cls_ids[i] = cls_id

            # bboxes[i] = obj.box2d.copy()
            # down_ratio = 1

            # locs = obj.t.copy()
            # locs[1] = locs[1] - obj.h / 2

            # proj_center, depth = calib.project_rect_to_image(locs.reshape(-1, 3))
            # target_centers[i] = proj_center[0]
            # pad_size =[0,0]
            
            # bot_top_centers = np.stack((corners_3d[:4].mean(axis=0), corners_3d[4:].mean(axis=0)), axis=0)
            # keypoints_3D = np.concatenate((corners_3d, bot_top_centers), axis=0)
            # keypoints_2D, _ = calib.project_rect_to_image(keypoints_3D)
            # keypoints_x_visible = (keypoints_2D[:, 0] >= 0) & (keypoints_2D[:, 0] <= img_w  - 1)
            # keypoints_y_visible = (keypoints_2D[:, 1] >= 0) & (keypoints_2D[:, 1] <= img_h - 1)
            # keypoints_z_visible = (keypoints_3D[:, -1] > 0)

            # offset_3D = proj_center - target_centers[i]
            # # xyz visible
            # keypoints_visible = keypoints_x_visible & keypoints_y_visible & keypoints_z_visible

            # keypoints[i] = np.concatenate((keypoints_2D - target_centers[i].reshape(1, -1), keypoints_visible[:, np.newaxis]), axis=1)

            # orientations[i] = encode_alpha_multibin(obj.alpha, num_bin= 4)

            
        # img3 = show_image_with_boxes(img, cls_ids, target_centers, bboxes.copy(), keypoints, reg_mask, 
        #                         offset_3D, down_ratio, pad_size, orientations, vis=True)
        img4 = cv2.resize(img4, (img.shape[0], img.shape[0]))
        stack_img = np.concatenate([img, img4], axis=1)
        cv2.imwrite(save_dir+'/vis_3d/'+filename[j].split('/')[-1].replace('txt','_gt_.jpg') ,stack_img)
        print(save_dir+'/vis_3d/'+filename[j].split('/')[-1].replace('txt','_gt_.jpg'))
# def _read_imageset_file(path):
#     with open(path, 'r') as f:
#         lines = f.readlines()
#     return [int(line) for line in lines]
# det_path = "/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training/label_2"

# gt_path = "/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training/label_2"
# gt_split_file = "/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training/ImageSets/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
# val_image_ids = _read_imageset_file(gt_split_file)
# dt_annos = kitti.get_label_annos(det_path, val_image_ids)
# gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
# result_car, ret_dict_car =get_official_eval_result(gt_annos, dt_annos, 0)
# print(result_car) # 6s in my computer
#print(get_coco_eval_result(gt_annos, dt_annos, 0)) 
vis_noelix = True
vis_kitti =False
vis_nusence =False
gt_annos,dt_annos =[],[]
############################### for neolix _data ###############################################
if vis_noelix :
    
    #det_path = "/nfs/neolix_data1/neolix_dataset/test_dataset/camera_object_detection/eval/eval_python"
    #det_path = "/root/data/lpc_model/neolix_test/"+'/txt_train{}*{}/'.format(960,512)
    det_path = "/root/data/lpc_model/monoflex_26/kitti_train/inference_3192/data"
    gt_path = "/root/data/neolix_dataset/test_dataset/camera_object_detection/label_2/"
    gt_annos,dt_annos ,filename,labels,label_det = get_label_annos_gen(gt_path,det_path)

    imgfolder ='/nfs/neolix_data1/neolix_dataset/test_dataset/camera_object_detection/image_2/'
    save_dir = "/home/lipengcheng/results/yizhuang/eval3/"
    if not os.path.exists(save_dir+'vis_3d'):
        os.makedirs(save_dir+'vis_3d')
    #vis_bev(labels,label_det,imgfolder,filename,save_dir)
    vis_gt(labels,imgfolder,filename,save_dir,False)

if vis_kitti :
    det_path = "/home/lipengcheng/results/yizhuang/eval2/kitti/eval_txt/"
    gt_path = "/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training/label_2/"
    imageset_txt_path = os.path.join("/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training/","ImageSets", '{}_7000.txt'.format('val'))
    gt_annos,dt_annos ,filename,labels,label_det = get_label_annos_gen(gt_path,det_path,imageset_txt_path)
    save_dir = "/home/lipengcheng/results/yizhuang/eval3/"
    imgfolder = "/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training/image_2/"
    vis_bev(labels,label_det,imgfolder,filename,save_dir)
    #vis_gt(labels,imgfolder,filename,save_dir,False)

if vis_nusence :
    opendataset = 'nuscenes'
    det_path = "/home/lipengcheng/results/yizhuang/eval2/{}/eval_txt/".format(opendataset)
    gt_path = "/nfs/neolix_data1/OpenSource_dataset/camera_object_detection/kitti_format/nuscense/train/label_2/"
    gt_folder = "/nfs/neolix_data1/OpenSource_dataset/camera_object_detection/kitti_format/nuscense/train/"
    imageset_txt_path = os.path.join(gt_folder,"ImageSets", '{}_{}_label.txt'.format('nuscenes','train'))
    gt_annos,dt_annos ,filename,labels,label_det = get_label_annos_gen(gt_path,det_path,imageset_txt_path)
    save_dir = "/home/lipengcheng/results/yizhuang/nuscenes/"
    if not os.path.exists(save_dir+'vis_3d'):
        os.makedirs(save_dir+'vis_3d')
    imgfolder = "/nfs/neolix_data1/OpenSource_dataset/camera_object_detection/kitti_format/nuscense/train/image_2/"
    #vis_bev(labels,label_det,imgfolder,filename,save_dir)
    vis_gt(labels,imgfolder,filename,save_dir,False)

#for i,annos_gt in enumerate(zip(gt_annos,dt_annos)):
    # if filename[i] == '20210521180702.record.00011_1621592107.259279.txt':
    #     print('set')
    # saveid = nms(annos_gt[1], 0)
    # dt_annos[i] = updateAnno(dt_annos[i],saveid)
    # imgpath = imgfolder + filename[i].replace('txt','png')
    # print(filename[i])
    # if not os.path.exists(imgpath):
    #     continue
    # img_vis_3d = cv2.imread(imgpath)
    # for j,bbox in enumerate(annos_gt[0]['bbox']):
    #     #img2.draw_box(box_coord=box2d[i], edge_color='g')
    #     corners3d = box3d_to_corners(annos_gt[0]['location'][j], annos_gt[0]['dimensions'][j], annos_gt[0]['rotation_y'][j])
        
    #     corners_2d, depth = calib.project_rect_to_image(corners3d)
    #     img3 = draw_projected_box3d(img_vis_3d, corners_2d, cls=annos_gt[0]['name'][j], color=(0,255,0),draw_orientation=True, draw_corner=False)
    #     img3 = Visualizer(img3.copy())
    #     #img3.draw_text(text='{},{}, {}, {}, {:.3f},{:.2f},{:.2f},{:.2f}'.format("id :",i, annos_gt['name'][i] ,int(clses[i]), score[i], dims[i,0],dims[i,1],dims[i,2]), position=(int(box2d[i, 0]), int(box2d[i, 1])))
    #     img3 = img3.output.get_image().astype(np.uint8)
    # #cv2.imwrite(save_dir+'/vis_3d/'+filename[i].replace('txt','_gt_.jpg') ,img3)
    
    # for j,bbox in enumerate(annos_gt[1]['bbox']):
    #     if j not in saveid:
    #         continue
    #     #img2.draw_box(box_coord=box2d[i], edge_color='g')
    #     corners3d = box3d_to_corners(annos_gt[1]['location'][j], annos_gt[1]['dimensions'][j], annos_gt[1]['rotation_y'][j])
    #     corners_2d, depth = calib.project_rect_to_image(corners3d)
    #     img3 = draw_projected_box3d(img_vis_3d, corners_2d, cls=annos_gt[1]['name'][j], color=(0,0,255),draw_orientation=True, draw_corner=False)
    #     img3 = Visualizer(img3.copy())
    #     #img3.draw_text(text='{},{}, {}, {}, {:.3f},{:.2f},{:.2f},{:.2f}'.format("id :",i, annos_gt['name'][i] ,int(clses[i]), score[i], dims[i,0],dims[i,1],dims[i,2]), position=(int(box2d[i, 0]), int(box2d[i, 1])))
    #     img3 = img3.output.get_image().astype(np.uint8)
    #cv2.imwrite(save_dir+'/vis_3d/'+filename[i].replace('txt','jpg') ,img3)
    
    #print (i)

#result_car, ret_dict_car = get_neolix_eval_result(gt_annos, dt_annos, [0,1,2])
# result_ped, ret_dict_ped = get_neolix_eval_result(gt_annos, dt_annos, 1)
# result_cyc, ret_dict_cyc = get_neolix_eval_result(gt_annos, dt_annos, 2)
# print(result_cyc)
# print(result_ped)
#print(result_car)