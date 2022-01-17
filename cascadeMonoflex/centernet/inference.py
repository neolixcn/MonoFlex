
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
from utils import read_image

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="projects/CenterNet2/projects/CenterNet2/configs/CenterNet-S4_DLA_8x.yaml",#"configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input",
        default= ["/root/data/neolix_dataset/test_dataset/camera_object_detection/image_2/*.png"], nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        default= "/root/data/lpc_model/centernet/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS","/root/code/detectron2/projects/CenterNet2/model/CenterNet-S4_DLA_8x.pth"],
        nargs=argparse.REMAINDER,
    )
    return parser

args = get_parser().parse_args()

args.input = glob.glob(os.path.expanduser(args.input[0]))

for path in args.input:
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")