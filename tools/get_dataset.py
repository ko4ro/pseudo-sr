import os
from tools.pseudo_face_data import faces_data
from tools.pseudo_dsem_data import dsem_data


def get_dataset(CFG):

    if CFG.EXP.NAME == "faces":

        trainset = faces_data(
            data_lr=os.path.join(CFG.DATA.FOLDER, "LOW/wider_lnew"),
            data_hr=os.path.join(CFG.DATA.FOLDER, "HIGH"),
            img_range=CFG.DATA.IMG_RANGE,
            rgb=CFG.DATA.RGB,
        )

        testset = faces_data(
            data_lr=os.path.join(CFG.DATA.FOLDER, "testset"),
            data_hr=None,
            b_train=False,
            shuffle=False,
            img_range=CFG.DATA.IMG_RANGE,
            rgb=CFG.DATA.RGB,
        )

        return trainset, testset

    elif CFG.EXP.NAME == "dsem":

        trainset = dsem_data(
            data_lr=os.path.join(CFG.DATA.FOLDER, "LOW/saitama"),
            data_hr=os.path.join(CFG.DATA.FOLDER, "HIGH"),
            img_range=CFG.DATA.IMG_RANGE,
            rgb=CFG.DATA.RGB,
            dataset_split = CFG.DATA.HOLDOUT,
            z_size=(CFG.OPT.NOIZE, CFG.OPT.NOIZE)
        )

        testset = None

        return trainset, testset
