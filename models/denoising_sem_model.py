import torch
import torch.nn as nn

from models.geo_loss import geometry_ensemble
from models.losses import GANLoss
from models.pseudo_model import Pseudo_Model


class Sem_Model(Pseudo_Model):
    def __init__(self, device, cfg, use_ddp=False):
        super(Sem_Model, self).__init__(device=device, cfg=cfg, use_ddp=use_ddp)
        self.sr_warmup_iter = cfg.OPT_SR.WARMUP
        del self.U
        del self.opt_U
        del self.lr_U
        del self.D_sr
        del self.opt_Dsr
        del self.lr_Dsr
        del self.nets["U"]
        del self.nets["D_sr"]
        del self.optims["U"]
        del self.optims["D_sr"]
        del self.lr_decays["U"]
        del self.lr_decays["D_sr"]
        del self.discs
        del self.gens
        self.discs = ["D_x", "D_y"]
        self.gens = ["G_xy", "G_yx"]
        self.gan_loss = GANLoss(cfg.OPT_CYC.GAN_TYPE)
        self.l1_loss = nn.L1Loss()
        self.d_gyx_weight = cfg.OPT_CYC.LOSS.D_Gxy_WEIGHT
        self.d_gxy_weight = cfg.OPT_CYC.LOSS.D_Gyx_WEIGHT

    def warmup_checker(self):
        return self.n_iter <= self.sr_warmup_iter

    def lr_decay_step(self, shout=False):
        lrs = "\nLearning rates: "
        changed = False
        for i, n in enumerate(self.lr_decays):
            if self.warmup_checker() and n == "D_esrgan":
                continue
            lr_old = self.lr_decays[n].get_last_lr()[0]
            self.lr_decays[n].step()
            lr_new = self.lr_decays[n].get_last_lr()[0]
            if lr_old != lr_new:
                changed = True
                lrs += (
                    f", {n}={self.lr_decays[n].get_last_lr()[0]}"
                    if i > 0
                    else f"{n}={self.lr_decays[n].get_last_lr()[0]}"
                )
        if shout and changed:
            print(lrs)

    def test_sample(self, Xs, Ys=None, Zs=None):
        """
        Xs: low qualities
        Ys: high qualities
        Zs: noises
        """
        fake_x = None
        with torch.no_grad():
            y = self.nets["G_xy"](Xs)
        if Ys is not None:
            fake_x = self.nets["G_yx"](Ys)
        return y, fake_x

    def train_step(self, Xs, Ys, Zs=None):
        """
        Xs: low qualities
        Ys: high qualities
        Zs: noises
        """
        self.n_iter += 1
        loss_dict = dict()

        # forward
        fake_Xs = self.G_yx(Ys)
        # fake_Xs = self.G_yx(Ys, Zs) ## TODO:Check to confirm if Zs noise is required or not.
        rec_Ys = self.G_xy(fake_Xs)
        fake_Ys = self.G_xy(Xs)
        geo_Ys = geometry_ensemble(self.G_xy, Xs)
        idt_out = self.G_xy(Ys) if self.idt_input_clean else fake_Ys
        # sr_y = self.U(rec_Yds)
        # sr_x = self.U(fake_Yds)

        self.net_grad_toggle(["D_x", "D_y"], True)
        # D_x
        pred_fake_Xs = self.D_x(fake_Xs.detach())
        pred_real_Xs = self.D_x(Xs)
        loss_D_x = (
            self.gan_loss(pred_real_Xs, True, True)
            + self.gan_loss(pred_fake_Xs, False, True)
        ) * 0.5
        self.opt_Dx.zero_grad()
        loss_D_x.backward()
        self.opt_Dx.step()
        loss_dict["D_x"] = loss_D_x.item()

        # D_y
        pred_fake_Ys = self.D_y(fake_Ys.detach())
        pred_real_Ys = self.D_y(Ys)
        loss_D_y = (
            self.gan_loss(pred_real_Ys, True, True)
            + self.gan_loss(pred_fake_Ys, False, True)
        ) * 0.5
        self.opt_Dy.zero_grad()
        loss_D_y.backward()
        self.opt_Dy.step()
        loss_dict["D_y"] = loss_D_y.item()

        self.net_grad_toggle(["D_x", "D_y"], False)

        # G_yx
        self.opt_Gyx.zero_grad()
        self.opt_Gxy.zero_grad()
        pred_fake_Xs = self.D_x(fake_Xs)
        loss_gan_Gyx = self.gan_loss(pred_fake_Xs, True, False)
        loss_dict["G_yx_gan"] = loss_gan_Gyx.item()

        # G_xy
        pred_fake_Ys = self.D_y(fake_Ys)
        loss_gan_Gxy = self.gan_loss(pred_fake_Ys, True, False)
        loss_idt_Gxy = (
            self.l1_loss(idt_out, Ys)
            if self.idt_input_clean
            else self.l1_loss(idt_out, Xs)
        )
        loss_cycle = self.l1_loss(rec_Ys, Ys)
        loss_geo = self.l1_loss(fake_Ys, geo_Ys)
        loss_total_gen = (
            +self.d_gyx_weight * loss_gan_Gyx
            + self.d_gxy_weight * loss_gan_Gxy
            + self.cyc_weight * loss_cycle
            + self.idt_weight * loss_idt_Gxy
            + self.geo_weight * loss_geo
        )
        loss_dict["G_yx_gan"] = loss_gan_Gyx.item()
        loss_dict["G_xy_gan"] = loss_gan_Gxy.item()
        loss_dict["G_xy_idt"] = loss_idt_Gxy.item()
        loss_dict["G_xy_geo"] = loss_geo.item()
        loss_dict["cyc_loss"] = loss_cycle.item()
        loss_dict["G_total"] = loss_total_gen.item()

        # gen loss backward and update
        loss_total_gen.backward()
        self.opt_Gyx.step()
        self.opt_Gxy.step()

        return loss_dict


if __name__ == "__main__":
    from yacs.config import CfgNode

    with open("configs/faces.yaml", "rb") as cf:
        CFG = CfgNode.load_cfg(cf)
        CFG.freeze()
    device = 0
    x = torch.randn(8, 3, 32, 32, dtype=torch.float32, device=device)
    y = torch.randn(8, 3, 64, 64, dtype=torch.float32, device=device)
    yd = torch.randn(8, 3, 32, 32, dtype=torch.float32, device=device)
    z = torch.randn(8, 1, 8, 8, dtype=torch.float32, device=device)
    model = Sem_Model(device, CFG)
    losses = model.train_step(y, x, yd, z)
    file_name = model.net_save(".", True)
    model.net_load(file_name)
    for i in range(110000):
        model.lr_decay_step(True)
    info = f"  1/(1):"
    for i, itm in enumerate(losses.items()):
        info += f", {itm[0]}={itm[1]:.3f}" if i > 0 else f" {itm[0]}={itm[1]:.3f}"
    print(info)
    print("fin")
