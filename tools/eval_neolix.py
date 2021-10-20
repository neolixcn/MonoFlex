import sys
sys.path.append('/root/code/MonoFlex')
import data.datasets.evaluation.kitti_object_eval_python.kitti_common as kitti
from data.datasets.evaluation.kitti_object_eval_python.eval import get_official_eval_result, get_coco_eval_result
import os 

def get_label_annos_gen(label_folder,det_folder):
    annos_det = []
    annos_gt = []
    label_filenames = os.listdir(label_folder)
    label_filenames.sort()
    for idx in label_filenames[:3]:
        label_filename_det = os.path.join(det_folder, idx)
        label_filename_gt = os.path.join(label_folder, idx)
        if os.path.exists(label_filename_det) and os.path.exists(label_filename_gt):
            annos_gt.append(kitti.get_label_anno(label_filename_gt))
            annos_det.append(kitti.get_label_anno(label_filename_det))
    return annos_gt,annos_det


det_path = "/root/data/neolix_dataset/test_dataset/camera_object_detection/eval/eval_python"
gt_path = "/root/data/neolix_dataset/test_dataset/camera_object_detection/label_2/"
gt_annos,dt_annos = get_label_annos_gen(gt_path,det_path)
result_car, ret_dict_car = get_official_eval_result(gt_annos, dt_annos, 0)
result_ped, ret_dict_ped = get_official_eval_result(gt_annos, dt_annos, 1)
result_cyc, ret_dict_cyc = get_official_eval_result(gt_annos, dt_annos, 2)
print(result_cyc)
print(result_ped)
print(result_car)
#print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer