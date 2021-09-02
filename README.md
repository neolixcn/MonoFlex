# MonoFlex
Released code for Objects are Different: Flexible Monocular 3D Object Detection, CVPR21.


**Work in progress.**


## Installation
This repo is tested with Ubuntu 20.04, python==3.7, pytorch==1.4.0 and cuda==10.1

```bash
conda create -n monoflex python=3.7

conda activate monoflex
```

Install PyTorch and other dependencies:

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt
```

Build DCNv2 and the project
```bash
cd models/backbone/DCNv2

. make.sh

cd ../../..

python setup develop
```

## Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT		
  |training/
    |calib/
    |image_2/
    |label/
    |ImageSets/
  |testing/
    |calib/
    |image_2/
    |ImageSets/
```

Then modify the paths in config/paths_catalog.py according to your data path.

## Training & Evaluation

Training with one GPU. (TODO: The multi-GPU training will be further tested.)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size 8 --config runs/monoflex.yaml --output output/exp
```

The model will be evaluated periodically (can be adjusted in the CONFIG) during training and you can also evaluate a checkpoint with

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monoflex.yaml --ckpt YOUR_CKPT  --eval
```

You can also specify --vis when evaluation to visualize the predicted heatmap and 3D bounding boxes. The pretrained model for train/val split and logs are [here](https://drive.google.com/drive/folders/1U60gUYp4JFOkG0VMefc4aVEMxtGM-AMu?usp=sharing).

**Note:** we observe an obvious variation of the performance for different runs and we are still investigating possible solutions to stablize the results, though it may inevitably due to the utilized uncertainties.

## inference
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monoflex.yaml --ckpt ./model_moderate_best_soft.pth --eval --vis

## ConvertOnnx
1. replace some folders and files with those named with suffix of "_onnx":
 - model/backbone/DCNv2_onnx -> model/backbone/DCNv2
 - model/head/detector_head_onnx.py -> model/head/detector_head.py
 - model/head/detector_predictor_onnx.py -> model/head/detector_predictor.py
 - model/detector_onnx.py -> model/detector.py
2. change some files:
 - change import way of DCNv2 in model/backbone/dla_dcn.py:
```
# from model.backbone.DCNv2.dcn_v2 import DCN
from model.backbone.DCNv2.modules.deform_conv import ModulatedDeformConvPack as DCN
```
 - change dcn extension name in setup.py
 ```
  ext_modules = [
        extension(
            "deform_conv_cuda",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros
 ```

3. rebuild dcn module and whole project, then run the following command to convert onnx.
```
CUDA_VISIBLE_DEVICES=2 python tools/convert_onnx.py --config runs/monoflex.yaml --ckpt ./model_moderate_best_soft_noInabn.pth --export-name "monoflex_new.onnx"
```
Note that if "new-ckpt" parameter and "ckpt" parameter are both given, the ckpt will be converted to new state dict without InplaceABN opterator and save to "new-ckpt".
And if you want to compare the results of model with "new-ckpt" and "ckpt", try change the code around line 109.

## Citation

If you find our work useful in your research, please consider citing:

```latex
@InProceedings{MonoFlex,
    author    = {Zhang, Yunpeng and Lu, Jiwen and Zhou, Jie},
    title     = {Objects Are Different: Flexible Monocular 3D Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3289-3298}
}
```

## Acknowlegment

The code is heavily borrowed from [SMOKE](https://github.com/lzccccc/SMOKE) and thanks for their contribution.
