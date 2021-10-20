import torch
import sys
sys.path.append('/root/code/MonoFlex')
from model.detector import KeypointDetector
from utils.check_point import DetectronCheckpointer
from engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from tools.plain_train_net import setup

if __name__ =="__main__":
    args = default_argument_parser().parse_args() 
    cfg = setup(args)
    model = KeypointDetector(cfg).cuda()
    checkpointer = DetectronCheckpointer(
            cfg, model, save_dir="/home/lipengcheng/code/fuxian/MonoFlex/output/toy_experiments"#cfg.OUTPUT_DIR
        )
    ckpt = '/root/data/lpc_model/monoflex_15/model_checkpoint_epoch_80.pth'
    model_param = torch.load(ckpt)
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    torch.save(model.state_dict(), '/root/data/lpc_model/monoflex_15/model_checkpoint_epoch_80_cp.pth', _use_new_zipfile_serialization=False) 
