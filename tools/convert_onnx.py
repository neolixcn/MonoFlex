import torch
import pdb
import datetime

from config import cfg

from utils.check_point import DetectronCheckpointer
from engine import (
    default_argument_parser,
    default_setup,
)

from model.detector import KeypointDetector
from data import build_test_loader

import onnxruntime
import numpy as np

def change_ckpt_param(ckpt, new_ckpt):
    """
    change state_dict with inplaceabn to new state_dict and save.
    """
    state_dict = torch.load(ckpt).pop("model")
    eps = 1e-5
    for k,v in state_dict.items():
        if "class_head.1.weight" in k or ("reg_features" in k and ".1.weight" in k):
            print(k)
            state_dict[k] = torch.abs(v) + eps
    torch.save(state_dict, new_ckpt)

def compare_output(old_ckpt, new_ckpt, cfg, images, targets):
    """
    compare output of model with InplaceABN and that of model without it.
    Args:
        old_ckpt: ckpt from model with InplaceABN.
        new_ckpt: ckpt form model without InplaceABN.
        cfg: config
        images: model input images
        targets: targets used in model heads.
    """
    cfg.MODEL.INPLACE_ABN = True
    print("INPLACE_ABN", cfg.MODEL.INPLACE_ABN)

    model = KeypointDetector(cfg, targets)
    model.to(device)
    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    _ = checkpointer.load(old_ckpt, use_latest=args.ckpt is None)
    model.eval()
    # output= model(images)
    out_cls, out_regs= model(images) #model(images, targets)

    cfg.MODEL.INPLACE_ABN = False
    print("INPLACE_ABN", cfg.MODEL.INPLACE_ABN)
    
    model2 = KeypointDetector(cfg, targets)
    model2.to(device)
    checkpointer2 = DetectronCheckpointer(
        cfg, model2, save_dir=cfg.OUTPUT_DIR
    )
    _ = checkpointer2.load(new_ckpt, use_latest=False)
    model2.eval()
    out_cls2, out_regs2= model2(images)

    np.testing.assert_allclose(out_cls.detach().cpu().numpy(), out_cls2.detach().cpu().numpy(), rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(out_regs.detach().cpu().numpy(), out_regs2.detach().cpu().numpy(), rtol=1e-3, atol=1e-5)



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

def main(args):
    cfg = setup(args)

    # input data
    data_loaders_val = build_test_loader(cfg)
    batch = next(iter(data_loaders_val))
    images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]
    device = torch.device(cfg.MODEL.DEVICE)
    images = images.tensors.to(device)
    targets = [target.to(device) for target in targets]

    # Compare output of model with InplaceABN and that of model without it.
    # old_ckpt = cfg.MODEL.WEIGHT
    # new_ckpt = args.new_ckpt
    # compare_output(old_ckpt, new_ckpt, cfg, images, targets)

    # model
    print("INPLACE_ABN", cfg.MODEL.INPLACE_ABN)
    model = KeypointDetector(cfg, targets)
    model.to(device)
    # load checkpoint
    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    if args.new_ckpt:
        ckpt = args.new_ckpt
    elif args.ckpt:
        ckpt = args.ckpt
    else:
        ckpt = cfg.MODEL.WEIGHT
    print("===>loading checkpoint from: {}".format(ckpt))
    _ = checkpointer.load(ckpt, use_latest=args.new_ckpt is None)
    model.eval()

    # export model without postprocess
    input_names = ['images'] #['images', 'targets']
    output_names = ['cls', 'regs']
    torch.onnx.export(
                                        model,
                                        images,
                                        args.export_name,
                                        export_params=True,
                                        opset_version=11, #not supported in torch 1.4:12,13
                                        verbose=True,
                                        do_constant_folding=True,
                                        input_names=input_names,
                                        output_names=output_names,
                                        # dynamic_axes={'images':[0], 'cls':[0], 'regs':[0]},
                                        )

    # # export model with postprocess
    # input_names = ['images'] #['images', 'targets']
    # output_names = ['output']
    # torch.onnx.export(
    #                                     model,
    #                                     images,
    #                                     args.export_name,
    #                                     export_params=True,
    #                                     opset_version=11, # latest version supported in torch 1.4.
    #                                     verbose=True,
    #                                     do_constant_folding=True,
    #                                     input_names=input_names,
    #                                     output_names=output_names,
    #                                     # dynamic_axes={'images':[0], 'output':[0]},
    #                                     )
    # output= model(images)

    out_cls, out_regs= model(images)
    # fea, out_cls, out_regs= model(images) #model(images, targets)

    # # compare pytorch output with onnxruntime output
    # ort_session = onnxruntime.InferenceSession(args.export_name)
    # ort_inputs = {ort_session.get_inputs()[0].name:images.detach().cpu().numpy()}
    # ort_outs = ort_session.run(None, ort_inputs)

    # print(np.testing.assert_allclose(out_cls.detach().cpu().numpy(), ort_outs[0], rtol=1e-3, atol=1e-5))
    # print(np.testing.assert_allclose(out_regs.detach().cpu().numpy(), ort_outs[1], rtol=1e-3, atol=1e-5))
    # print("Export model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument("--new-ckpt", dest="new_ckpt", default=None, help="specify the path for new state_dict with bn and activation converting from state_dict with inplaceabn")
    parser.add_argument("--export-name", dest="export_name", default="./monoflex.onnx", help="specify the full path for exported onnx")
    args = parser.parse_args()

    if args.ckpt and args.new_ckpt:
        change_ckpt_param(ckpt, new_ckpt)
    main(args)