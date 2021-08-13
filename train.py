import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from yacs.config import CfgNode as CN

from models.face_model import Face_Model
from models.denoising_sem_model import Sem_Model
from tools.get_dataset import get_dataset
from tools.utils import AverageMeter, save_tensor_image

main_parse = argparse.ArgumentParser()
main_parse.add_argument("--yaml", default="configs/dsem.yaml", type=str)
main_parse.add_argument("--port", type=int, default=5000, required=False)
main_args = main_parse.parse_args()
with open(main_args.yaml, "rb") as cf:
    CFG = CN.load_cfg(cf)
    CFG.freeze()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_args.port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_train_loader(trainset, world_size, batch_size):
    if world_size <= 1:
        return torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=CFG.DATA.NUM_WORKERS,
            pin_memory=True,
        )
    elif world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            trainset, shuffle=True
        )
        return torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=CFG.DATA.NUM_WORKERS,
            pin_memory=True,
        )


def main(rank, world_size, cpu=False):
    if cpu:
        rank = torch.device("cpu")
    elif world_size > 1:
        setup(rank, world_size)
    last_device = world_size - 1
    batch_per_gpu = CFG.SR.BATCH_PER_GPU

    if CFG.EXP.NAME == "faces":
        model = Face_Model(rank, CFG, world_size > 1)
    elif CFG.EXP.NAME == "dsem":
        model = Sem_Model(rank, CFG, world_size > 1)
    else:
        raise Exception("Unexpected error: CFG.EXP.NAME is not defined")
    # summary(model, input_size=(CFG.SR.BATCH_PER_GPU, CFG.SR.CHANEL, CFG.SR.PATCH_SIZE_LR, CFG.SR.PATCH_SIZE_LR),col_names=["output_size", "num_params"])
    output_dir_path = os.path.join(CFG.EXP.OUT_DIR, f"{datetime.now().strftime('%Y%d%m%H%M')}")
    tb_log_path = os.path.join(output_dir_path,"logs")
    print(f"Tensorboard log : tensorboard dev upload --logdir={tb_log_path}")
    os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    trainset, testset = get_dataset(CFG)
    loader = get_train_loader(trainset, world_size, batch_per_gpu)

    end_ep = int(np.ceil(CFG.OPT.MAX_ITER / len(loader))) + 1
    test_freq = max([end_ep // CFG.OPT.NUM_FREQ, 1])

    if rank == last_device:
        net_save_folder = os.path.join(output_dir_path, "nets")
        img_save_folder = os.path.join(output_dir_path, "imgs")
        os.makedirs(net_save_folder, exist_ok=True)
        os.makedirs(img_save_folder, exist_ok=True)
        print("Output dir: ", output_dir_path)
        print(
            f"Batch_size: {batch_per_gpu * world_size}, Batch_size per GPU: {batch_per_gpu}"
        )
        print(
            f"Max epoch: {end_ep - 1}, Total iteration: {(end_ep - 1) * len(loader)}, Iterations per epoch: {len(loader)}, Test & Save epoch: every {test_freq} epoches"
        )

    loss_avgs = dict()
    for ep in range(1, end_ep):
        model.mode_selector("train")
        for b, batch in enumerate(loader):
            lrs = batch["lr"].to(rank)
            hrs = batch["hr"].to(rank)
            if CFG.EXP.NAME == "faces":
                zs = batch["z"].to(rank)
                hr_downs = batch["hr_down"].to(rank)
                losses = model.train_step(hrs, lrs, hr_downs, zs)
            elif CFG.EXP.NAME == "dsem":
                if CFG.OPT.NOIZE == 0:
                    losses = model.train_step(lrs, hrs, Zs=None)
                else:
                    zs = batch["z"].to(rank)
                    losses = model.train_step(lrs, hrs, zs)
            else:
                raise Exception("Unexpected error: CFG.EXP.NAME is not defined")

            info = f"  {model.n_iter}({ep}/{end_ep-1}):"
            for i, itm in enumerate(losses.items()):
                # writer.add_scalar(f" {itm[0]}", itm[1], ep*b)
                if itm[0] not in loss_avgs.keys():
                    loss_avgs[itm[0]] = AverageMeter(itm[1])
                else:
                    loss_avgs[itm[0]].update(itm[1])
                info += (
                    f", {itm[0]}={loss_avgs[itm[0]].get_avg():.3f}"
                    if i > 0
                    else f" {itm[0]}={loss_avgs[itm[0]].get_avg():.3f}"
                )
                writer.add_scalar(f"loss/{itm[0]}", loss_avgs[itm[0]].get_avg(), ep*b)
            print(info + "\r", end="")
            model.lr_decay_step(True)

        if ep % test_freq == 0 and rank == last_device:
            print(f"\nTesting and saving: Epoch {ep}")
            model.net_save(net_save_folder)
            model.mode_selector("eval")
            for b in range(len(trainset)):
                if b > 10:
                    break
                lr = trainset[b]["lr"].unsqueeze(0).to(rank)
                if CFG.EXP.NAME == "faces":
                    y, sr, _ = model.test_sample(lr)
                    sr = save_tensor_image(
                        os.path.join(img_save_folder, f"{b:04d}_sr.png"),
                        sr,
                        CFG.DATA.IMG_RANGE,
                        CFG.DATA.RGB,
                    )
                elif CFG.EXP.NAME == "dsem":
                    hr = trainset[b]["hr"].unsqueeze(0).to(rank)
                    y, fake_x = model.test_sample(lr, hr)
                    hr_img = save_tensor_image(
                        os.path.join(img_save_folder, f"{b:04d}_hr.png"),
                        hr,
                        CFG.DATA.IMG_RANGE,
                        CFG.DATA.RGB,
                    )
                    fake_x_img =save_tensor_image(
                        os.path.join(img_save_folder, f"{b:04d}_fake_x.png"),
                        fake_x,
                        CFG.DATA.IMG_RANGE,
                        CFG.DATA.RGB,
                    )
                y = save_tensor_image(
                    os.path.join(img_save_folder, f"{b:04d}_y.png"),
                    y,
                    CFG.DATA.IMG_RANGE,
                    CFG.DATA.RGB,
                )
                lr = save_tensor_image(
                    os.path.join(img_save_folder, f"{b:04d}_lr.png"),
                    lr,
                    CFG.DATA.IMG_RANGE,
                    CFG.DATA.RGB,
                )
                writer.add_images(tag="images/hr_img", img_tensor=hr_img, global_step=test_freq)
                writer.add_images(tag="images/fake_x_img", img_tensor=fake_x_img, global_step=test_freq)
                writer.add_images(tag="images/y", img_tensor=y, global_step=test_freq)
                writer.add_images(tag="images/lr", img_tensor=lr, global_step=test_freq)
        if world_size > 1:
            dist.barrier()

    if rank == last_device:
        print("\nFinal test and save")
        model.net_save(net_save_folder, True)
        model.mode_selector("test")
        for b in range(len(trainset)):
            lr = trainset[b]["lr"].unsqueeze(0).to(rank)
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
            save_tensor_image(
                os.path.join(img_save_folder, f"{b:04d}_lr.png"),
                lr,
                CFG.DATA.IMG_RANGE,
                CFG.DATA.RGB,
            )
    if world_size > 1:
        dist.barrier()
    if world_size > 1:
        cleanup()


if __name__ == "__main__":
    random_seed = CFG.EXP.SEED
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    n_gpus = 0
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()

    if n_gpus <= 1:
        print("single proc.", f", time: {datetime.now().strftime('%Y-%d-%m_%H:%M:%S')}")
        main(0, n_gpus, cpu=(n_gpus == 0))
    elif n_gpus > 1:
        print(
            f"multi-gpu: {n_gpus}",
            f", time: {datetime.now().strftime('%Y-%d-%m_%H:%M:%S')}",
        )
        mp.spawn(main, nprocs=n_gpus, args=(n_gpus, False), join=True)
    print("fin.")
