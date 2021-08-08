import argparse
import os
from typing import OrderedDict
import torch

from yacs.config import CfgNode

from models.face_model import Face_Model
from tools.pseudo_face_data import faces_data
from tools.utils import save_tensor_image


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def net_load(model, file_name, strict=True):
    map_loc = {"cuda:0": f"cuda:{model.device}"}
    loaded = torch.load(file_name, map_location=map_loc)
    for n in model.nets:
        model.nets[n].load_state_dict(
            fix_model_state_dict(loaded["nets"][n]), strict=strict
        )
    for o in model.optims:
        model.optims[o].load_state_dict(
            fix_model_state_dict(loaded["optims"][o])
        )
    for lr in model.lr_decays:
        model.lr_decays[lr].load_state_dict(
            fix_model_state_dict(loaded["lr_decays"][lr])
        )


def main():
    parser = argparse.ArgumentParser(description="faces psuedo-sr prediction")
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default="results/faces/nets/nets_302939.pth",
        help='.pth include with {"nets", "optims", "lr_decays"}',
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/faces.yaml",
        help="config yaml file of training",
    )
    args = parser.parse_args()
    img_save_folder = os.path.join(os.getcwd(), "test_imgs")
    os.makedirs(img_save_folder, exist_ok=True)
    with open(args.config_path, "rb") as cf:
        CFG = CfgNode.load_cfg(cf)
        CFG.freeze()

    device = 0
    testset = faces_data(
        data_lr=os.path.join(CFG.DATA.FOLDER, "testset"),
        data_hr=None,
        b_train=False,
        shuffle=False,
        img_range=CFG.DATA.IMG_RANGE,
        rgb=CFG.DATA.RGB,
    )
    model = Face_Model(device, CFG)
    net_load(model, args.trained_model_path)
    model.mode_selector("test")
    for b in range(len(testset)):
        lr = testset[b]["lr"].unsqueeze(0).to(device)
        y, sr, _ = model.test_sample(lr)
        save_tensor_image(
            os.path.join(img_save_folder, f"{b:04d}_y.png"),
            y,
            CFG.DATA.IMG_RANGE,
            CFG.DATA.RGB,
        )
        save_tensor_image(
            os.path.join(img_save_folder, f"{b:04d}_sr.png"),
            sr,
            CFG.DATA.IMG_RANGE,
            CFG.DATA.RGB,
        )
        # save_tensor_image(
        #     os.path.join(img_save_folder, f"{b:04d}_x.png"),
        #     sr,
        #     CFG.DATA.IMG_RANGE,
        #     CFG.DATA.RGB,
        # )
        save_tensor_image(
            os.path.join(img_save_folder, f"{b:04d}_lr.png"),
            lr,
            CFG.DATA.IMG_RANGE,
            CFG.DATA.RGB,
        )


if __name__ == "__main__":
    main()
