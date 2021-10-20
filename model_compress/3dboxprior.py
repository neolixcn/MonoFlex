import numpy as np
import os 
from data.datasets.kitti_utils import Calibration, read_label

if __name__ =="__main__":
    label_files = []
    opendata = 'nuscenes'#waymo
    if opendata =='nuscenes':
        root = '/data/lpc_data/test/nuscense/train/'
        label_files_path = os.path.join(root,"ImageSets", '{}_kitti_{}_label.txt'.format(opendata,'train'))

    if opendata =='waymo':
        root = '/nfs/neolix_data1/OpenSource_dataset/camera_object_detection/kitti_format/train'
        label_files_path = os.path.join(root,"ImageSets", '{}_kitti_{}_label.txt'.format(opendata,'train'))
    classes = ["Car", "Pedestrian", "Cyclist"]
    Car_l, Car_w,Car_h = [],[],[]
    Car,Ped,Cyc =[],[],[]
    Ped_l, Ped_w,Ped_h = [],[],[]
    Cyc_l, Cyc_w,Cyc_h = [],[],[]
    for line in open(label_files_path, "r"):
        base_name = line.replace("\n", "")
        label_files.append(base_name)
        objs = read_label(base_name)
        for obj in objs:
            corners_3d = obj.generate_corners3d()
            h,l,w = obj.h, obj.l,obj.w
            if obj.type == classes[0]:
                Car.append([l,h,w])
                
            if obj.type == classes[1]:
                Ped.append([l,h,w])
                
            if obj.type == classes[2]:
                Cyc.append([l,h,w])
    Car = np.array(Car)  
    Ped = np.array(Ped)
    Cyc = np.array(Cyc)   
    mean_Car , std_Car = np.mean(Car, axis=0) , np.std(Car, axis=0)
    mean_Ped , std_Ped = np.mean(Ped, axis=0) , np.std(Ped, axis=0)
    mean_Cyc , std_Cyc = np.mean(Cyc, axis=0) , np.std(Cyc, axis=0)
    print(mean_Car , ' ', std_Car)
    print(mean_Ped , ' ', std_Ped)
    print(mean_Cyc , ' ', std_Cyc)




            

    

