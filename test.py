import argparse
import os
from typing import OrderedDict
import torch

from yacs.config import CfgNode

from models.face_model import Face_Model
from models.denoising_sem_model import Sem_Model
from tools.get_dataset import get_dataset
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
        default="results/dsem/202114080006/nets/nets_2816.pth",
        help='.pth include with {"nets", "optims", "lr_decays"}',
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/dsem.yaml",
        help="config yaml file of training",
    )
    args = parser.parse_args()
    with open(args.config_path, "rb") as cf:
        CFG = CfgNode.load_cfg(cf)
        CFG.freeze()

    img_save_folder = os.path.join(os.path.dirname(os.path.pardir(args.trained_model_path)), "test")
    os.makedirs(img_save_folder, exist_ok=True)
    device = 0
    trainset, _ = get_dataset(CFG)
    if CFG.EXP.NAME == "faces":
        model = Face_Model(device, CFG)
    elif CFG.EXP.NAME == "dsem":
        model = Sem_Model(device, CFG)
    else:
        raise Exception("Unexpected error: CFG.EXP.NAME is not defined")
    net_load(model, args.trained_model_path)
    model.mode_selector("test")
    for b in range(len(trainset)):
        if b > 10:
            break
        lr = trainset[b]["lr"].unsqueeze(0).to(device)
        if CFG.EXP.NAME == "faces":
            y, sr, _ = model.test_sample(lr)
            save_tensor_image(
                os.path.join(img_save_folder, f"{b:04d}_sr.png"),
                sr,
                CFG.DATA.IMG_RANGE,
                CFG.DATA.RGB,
            )
        elif CFG.EXP.NAME == "dsem":
            hr = trainset[b]["hr"].unsqueeze(0).to(device)
            if CFG.OPT.NOIZE == 0:
                y, fake_x = model.test_sample(lr, hr, Zs=None)
            else:
                zs = trainset[b]["z"].unsqueeze(0).to(device)
                y, fake_x = model.test_sample(lr, hr, zs)
            save_tensor_image(
                os.path.join(img_save_folder, f"{b:04d}_hr.png"),
                hr,
                CFG.DATA.IMG_RANGE,
                CFG.DATA.RGB,
            )
            save_tensor_image(
                os.path.join(img_save_folder, f"{b:04d}_fake_x.png"),
                fake_x,
                CFG.DATA.IMG_RANGE,
                CFG.DATA.RGB,
            )
        save_tensor_image(
            os.path.join(img_save_folder, f"{b:04d}_y.png"),
            y,
            CFG.DATA.IMG_RANGE,
            CFG.DATA.RGB,
        )
        save_tensor_image(
            os.path.join(img_save_folder, f"{b:04d}_lr.png"),
            lr,
            CFG.DATA.IMG_RANGE,
            CFG.DATA.RGB,
        )


if __name__ == "__main__":
    main()
