import os 
import shutil
import numpy as np

if __name__=="__main__":
    savefoldervalid = '/nfs/neolix_data1/neolix_dataset/test_dataset/camera_object_detection/'
    gt_path = "/nfs/neolix_data1/neolix_dataset/test_dataset/camera_object_detection/label_2/"
    label_filenames = os.listdir(gt_path)
    label_filenames.sort()
    valid_image_txt = os.path.join(savefoldervalid,"ImageSets", 'neolix_valid_img.txt')
    valid_label_txt = os.path.join(savefoldervalid,"ImageSets", 'neolix_valid_label.txt')
    valid_calib_txt = os.path.join(savefoldervalid,"ImageSets", 'neolix_valid_calib.txt')
    fh_img_train = open(valid_image_txt ,'w')
    fh_label_train = open(valid_label_txt ,'w')
    fh_calib_train = open(valid_calib_txt ,'w')
    img_folder = "/nfs/neolix_data1//neolix_dataset/test_dataset/camera_object_detection/image_2/"
    calib_folder = "/nfs/neolix_data1//neolix_dataset/test_dataset/camera_object_detection/calib/"
    label_folder = "/nfs/neolix_data1//neolix_dataset/test_dataset/camera_object_detection/label_2/"
    for k,filename in enumerate(label_filenames):
        if os.path.exists(img_folder+ filename.replace('txt','png')) and os.path.exists(calib_folder+ filename) and os.path.exists(label_folder+ filename):
            fh_img_train.write(img_folder+ filename.replace('txt','png') +'\n')
            fh_label_train.write(label_folder+ filename +'\n')
            fh_calib_train.write(calib_folder+ filename +'\n')
    fh_img_train.close()
    fh_label_train.close()
    fh_calib_train.close()


    savefoldertrain = '/nfs/neolix_data1/OpenSource_dataset/camera_object_detection/kitti_format/nuscense/train/'
    kitti_foler ='/nfs/neolix_data1/OpenSource_dataset/camera_object_detection/kitti_format/nuscense/train/'
    imageset_txt = os.path.join(kitti_foler,"ImageSets", '{}_kitti_{}_label.txt'.format('nuscenes','train'))
    image_files =[]
    label_files =[]
    calib_files =[]
    for line in open(imageset_txt, "r"):
        base_name = line.replace("\n", "").split('/')[-1].split('.')[0]
        if 'kitti' not in base_name :
            image_name = os.path.join(kitti_foler, "image_2" , base_name + ".png")
            label_name = os.path.join(kitti_foler, "label_2" , base_name + ".txt")
            calib_path = os.path.join(kitti_foler, "calib" , base_name + ".txt")
            image_files.append(image_name)
            label_files.append(label_name)
            calib_files.append(calib_path)
    
    train_image_txt = os.path.join(savefoldertrain,"ImageSets", 'nuscenes_train_img.txt')

    fh_img_train = open(train_image_txt ,'w')
    for k,filename in enumerate(image_files):
        
        fh_img_train.write(filename +'\n')
    fh_img_train.close()

    train_label_txt = os.path.join(savefoldertrain,"ImageSets", 'nuscenes_train_label.txt')
    fh_img_train = open(train_label_txt ,'w')
    for k,filename in enumerate(label_files):
               
        fh_img_train.write(filename +'\n')
    fh_img_train.close()

    train_calib_txt = os.path.join(savefoldertrain,"ImageSets", 'nuscenes_train_calib.txt')
    fh_img_train = open(train_calib_txt ,'w')
    for k,filename in enumerate(calib_files):
        
        fh_img_train.write(filename +'\n')
    fh_img_train.close()
    

    quantization_data = False 
    if quantization_data:
        savedir_data = "/nfs/neolix_data1/neolix_dataset/develop_dataset/camera_obastcle_detection/monoflex/develop_dataset/date_release/valid_data"
        savedir_label = "/nfs/neolix_data1/neolix_dataset/develop_dataset/camera_obastcle_detection/monoflex/develop_dataset/date_release/valid_label"
        kitti_imageset_txt = "val_7000.txt"
        #NU_imageset_txt = "train.txt"
        #NU_imageset_txt_val = "val.txt"
        #kitti_imageset_txt ="train_7000.txt"
        nuscenes_folder ="/data/lpc_data/test/nuscense/train"
        kitti_foler ="/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training"
        image_files =[]
        label_files =[]
        calib_files =[]
        for imageset_txt in [kitti_imageset_txt]:
            imageset_txt = os.path.join(kitti_foler,"ImageSets", kitti_imageset_txt)
            for line in open(imageset_txt, "r"):
                base_name = line.replace("\n", "")
                image_name = os.path.join(kitti_foler, "image_2" , base_name + ".png")
                label_name = os.path.join(kitti_foler, "label_2" , base_name + ".txt")
                calib_path = os.path.join(kitti_foler, "calib" , base_name + ".txt")
                image_files.append(image_name)
                label_files.append(label_name)
                calib_files.append(calib_path)
        shutil.copyfile(imageset_txt,os.path.join(savedir_data,'ImageSets','valid.txt'))
        if not os.path.exists(os.path.join(savedir_data,'image_2')):
                os.mkdir(os.path.join(savedir_data,'image_2'))

        for filename in image_files:
            shutil.copyfile(filename,os.path.join(savedir_data,'image_2',filename.split('/')[-1]))

        if not os.path.exists(os.path.join(savedir_data,'calib')):
                os.mkdir(os.path.join(savedir_data,'calib'))

        for filename in calib_files:
            shutil.copyfile(filename,os.path.join(savedir_data,'calib',filename.split('/')[-1]))

        if not os.path.exists(os.path.join(savedir_label,'label_2')):
                os.mkdir(os.path.join(savedir_label,'label_2'))

        for filename in label_files:
            shutil.copyfile(filename,os.path.join(savedir_label,'label_2',filename.split('/')[-1]))
           


    NU_imageset_txt = "train.txt"
    NU_imageset_txt_val = "val.txt"
    kitti_imageset_txt ="train_7000.txt"
    waymo_imageset_txt ="train.txt"
    imagesets =[NU_imageset_txt,kitti_imageset_txt,NU_imageset_txt_val,waymo_imageset_txt]
    nuscenes_folder ="/data/lpc_data/test/nuscense/train"
    kitti_foler ="/nfs/neolix_data1/OpenSource_dataset/lidar_object_detection/Kitti/kitti/training"
    waymo_folder = "/nfs/neolix_data1/OpenSource_dataset/camera_object_detection/kitti_format/train"
    
    
    

    image_files =[]
    label_files =[]
    calib_files =[]
    nolabelnum =0
    for id, imageset_txt in enumerate(imagesets):
        if id == 0 or id ==2:
            continue
            imageset_txt = os.path.join(nuscenes_folder,"ImageSets", imagesets[id])
            for line in open(imageset_txt, "r"):
                base_name = line.replace("\n", "")
                image_name = os.path.join(nuscenes_folder, "image_2" , base_name + ".png")
                label_name = os.path.join(nuscenes_folder, "label_2" , base_name + ".txt")
                with open(label_name, 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 0 or len(lines[0]) < 15:
                        content = []
                    else:
                        content = [line.strip().split(' ') for line in lines]
                    name = [x[0] for x in content]
                    if 'Pedestrian' not in name and 'Car' not in name and 'Cyclist' not in name :
                        print("======== no label ======",label_name)
                        nolabelnum=nolabelnum+1
                        continue
                calib_path = os.path.join(nuscenes_folder, "calib" , base_name + ".txt")
                image_files.append(image_name)
                label_files.append(label_name)
                calib_files.append(calib_path)
        if id == 1:
            imageset_txt = os.path.join(kitti_foler,"ImageSets",  imagesets[id])
            for line in open(imageset_txt, "r"):
                base_name = line.replace("\n", "")
                image_name = os.path.join(kitti_foler, "image_2" , base_name + ".png")
                label_name = os.path.join(kitti_foler, "label_2" , base_name + ".txt")
                with open(label_name, 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 0 or len(lines[0]) < 15:
                        content = []
                    else:
                        content = [line.strip().split(' ') for line in lines]
                    name = [x[0] for x in content]
                    if 'Pedestrian' not in name and 'Car' not in name and 'Cyclist' not in name :
                        print("======== no label ======",label_name)
                        nolabelnum =nolabelnum+1
                        continue
                calib_path = os.path.join(kitti_foler, "calib" , base_name + ".txt")
                image_files.append(image_name)
                label_files.append(label_name)
                calib_files.append(calib_path)
        if id == 3:
            plus50 =0 
            imageset_txt = os.path.join(waymo_folder,"ImageSets", imagesets[id])
            for line in open(imageset_txt, "r"):
                base_name = line.replace("\n", "")
                image_name = os.path.join(waymo_folder, "image_2" , base_name + ".png")
                label_name = os.path.join(waymo_folder, "label_1" , base_name + ".txt")
                calib_path = os.path.join(waymo_folder, "calib" , base_name + ".txt")
                if os.path.exists(label_name) and os.path.exists(image_name) and os.path.exists(calib_path):
                    with open(label_name, 'r') as f:
                        lines = f.readlines()
                        if len(lines) == 0 or len(lines[0]) < 15:
                            content = []
                        else:
                            content = [line.strip().split(' ') for line in lines]
                        if len(content)>50:
                            plus50 = plus50 +1
                            print("plus 50 :",plus50 )
                            continue
                        # if np.array([1 for x in content if x[0] in ["Car"] ]).sum()< 2:
                        #     continue
                        name = [x[0] for x in content]
                        if 'Pedestrian' not in name and 'Car' not in name and 'Cyclist' not in name :
                            
                            nolabelnum =nolabelnum+1
                            print("======== no label ======",nolabelnum)
                            continue
                    
                        image_files.append(image_name)
                        label_files.append(label_name)
                        calib_files.append(calib_path)

    train_image_txt = os.path.join(waymo_folder,"ImageSets", 'waymo_kitti_train_img.txt')
    fh_img_train = open(train_image_txt ,'w')
    for k,filename in enumerate(image_files):
        
        if k>7000 and k%3==0:
            print(k)
            fh_img_train.write(filename +'\n')
    fh_img_train.close()


    train_calib_txt = os.path.join(waymo_folder,"ImageSets", 'waymo_kitti_train_calib.txt')
    fh_cb_train = open(train_calib_txt ,'w')
    for k,filename in enumerate(calib_files):
        
        if k>7000 and k%3==0:
            print(k)
            fh_cb_train.write(filename +'\n')
    fh_cb_train.close()


    train_label_txt = os.path.join(waymo_folder,"ImageSets", 'waymo_kitti_train_label.txt')
    fh_label_train = open(train_label_txt ,'w')
    for k,filename in enumerate(label_files):
        
        if k>7000 and k%3==0:
            print(k)
            fh_label_train.write(filename +'\n')
    fh_label_train.close()

#############################for valid ########################################################
    #savedir = "/nfs/neolix_data1/neolix_dataset/develop_dataset/camera_obastcle_detection/monoflex/quantization_data/original_data"
    genVal =False
    if genVal  :
        kitti_imageset_txt = "val_7000.txt"
        image_files =[]
        label_files =[]
        calib_files =[]
        for imageset_txt in [kitti_imageset_txt]:
            imageset_txt = os.path.join(kitti_foler,"ImageSets", kitti_imageset_txt)
            for line in open(imageset_txt, "r"):
                base_name = line.replace("\n", "")
                image_name = os.path.join(kitti_foler, "image_2" , base_name + ".png")
                label_name = os.path.join(kitti_foler, "label_2" , base_name + ".txt")
                calib_path = os.path.join(kitti_foler, "calib" , base_name + ".txt")
                image_files.append(image_name)
                label_files.append(label_name)
                calib_files.append(calib_path)



        val_image_txt = os.path.join(nuscenes_folder,"ImageSets", 'nuscenes_kitti_val_img.txt')
        fh_img_val = open(val_image_txt ,'w')
        for filename in image_files:
            print(filename)
            #shutil.copyfile(filename,os.path.join(savedir,'image_2'))
            fh_img_val.write(filename +'\n')
        fh_img_val.close()


        val_calib_txt = os.path.join(nuscenes_folder,"ImageSets", 'nuscenes_kitti_val_calib.txt')
        fh_cb_val = open(val_calib_txt ,'w')
        for filename in calib_files:
            print(filename)
            #shutil.copyfile(filename,os.path.join(savedir,'calib'))
            fh_cb_val.write(filename +'\n')
        fh_cb_val.close()


        val_label_txt = os.path.join(nuscenes_folder,"ImageSets", 'nuscenes_kitti_val_label.txt')
        fh_label_val = open(val_label_txt ,'w')
        for filename in label_files:
            print(filename)
            #shutil.copyfile(filename,os.path.join(savedir,'label_2'))
            fh_label_val.write(filename +'\n')
        fh_label_val.close()

    
    


