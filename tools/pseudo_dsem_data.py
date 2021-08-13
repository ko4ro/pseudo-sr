import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, dataset
from torchvision import transforms
import torch.nn.functional as F
import torch

import glob


class dsem_data(Dataset):
    def __init__(
        self,
        data_lr,
        data_hr=None,
        b_train=True,
        rgb=True,
        dataset_split = False,
        img_range=1,
        shuffle=True,
        z_size=(8, 8),
    ):
        self.dataset_split = dataset_split
        self.ready_hr = data_hr is not None
        if data_hr is not None:
            self.hr_files = [
                p
                for p in glob.glob(os.path.join(data_hr, "**"), recursive=True)
                if os.path.splitext(os.path.basename(p))[1] in [".jpg", ".png", ".bmp"]
            ]
            self.hr_files.sort()
        self.lr_files = [
            p
            for p in glob.glob(os.path.join(data_lr, "**"), recursive=True)
            if os.path.splitext(os.path.basename(p))[1] in [".jpg", ".png", ".bmp"]
        ]
        self.lr_files.sort()
        if shuffle:
            if data_hr is not None:
                np.random.shuffle(self.hr_files)
            np.random.shuffle(self.lr_files)
        if self.dataset_split:
            self.hr_train_indices, self.hr_test_indices = train_test_split(range(len(self.hr_files)),test_size=0.2)
        self.training = b_train
        self.rgb = rgb
        self.z_size = z_size
        self.img_min_max = (0, img_range)
        if self.training:
            self.preproc = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.1, contrast=0.1, saturation=0, hue=0
                            )
                        ],
                        p=0.5,
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.preproc = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        if self.ready_hr:
            return min([len(self.lr_files), len(self.hr_files)])
        else:
            return len(self.lr_files)

    def __getitem__(self, index):
        data = dict()
        if np.prod(self.z_size) > 0:
            data["z"] = torch.randn(1, *self.z_size, dtype=torch.float32)

        lr_idx = index % len(self.lr_files)
        lr = cv2.imread(self.lr_files[lr_idx],0)
        if self.rgb:
            lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        data["lr"] = self.preproc(lr) * self.img_min_max[1]
        data["lr_path"] = self.lr_files[lr_idx]
        if self.ready_hr:
            hr_idx = index % len(self.hr_files)
            if self.dataset_split:
                hr = cv2.imread(self.hr_files[self.hr_train_indices[index]],0)
                hr_test = cv2.imread(self.hr_files[self.hr_test_indices[index]],0)
                data["hr_path"] = self.hr_files[self.hr_train_indices[index]]
                data["hr_test"] = hr_test * self.img_min_max[1]
                data["hr_test_path"] = self.hr_files[self.hr_test_indices[index]]
            else:
                hr = cv2.imread(self.hr_files[hr_idx],0)
                data["hr_path"] = self.hr_files[hr_idx]
            if self.rgb:
                hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            data["hr"] = self.preproc(hr) * self.img_min_max[1]
        return data

    def get_noises(self, n):
        return torch.randn(n, 1, *self.z_size, dtype=torch.float32)

    def permute_data(self):
        if self.ready_hr:
            np.random.shuffle(self.hr_files)
        np.random.shuffle(self.lr_files)


if __name__ == "__main__":
    high_folder = os.path.join(os.environ["DATA_TRAIN"], "HIGH")
    low_folder = os.path.join(os.environ["DATA_TRAIN"], "LOW/wider_lnew")
    test_folder = os.path.join(os.environ["DATA_TEST"])
    img_range = 1
    data = dsem_data(low_folder, high_folder, img_range=img_range)
    for i in range(len(data)):
        d = data[i]
        for elem in d:
            if elem in ["z", "lr_path", "hr_path"]:
                continue
            img = np.around(
                (d[elem].numpy().transpose(1, 2, 0) / img_range) * 255.0
            ).astype(np.uint8)
            cv2.imshow(elem, img[:, :, ::-1])
        cv2.waitKey()
    print("fin.")
