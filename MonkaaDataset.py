# -*- coding=utf8 -*-

"""
@author: CaiTao
@license: Apache Licence
@contact: 1120141815@bit.edu.cn
@file: MonkaaDataset.py
@time: 4/16/18 7:25 PM
"""

from torch.utils.data import Dataset
import torch
from mynet_utils import python_pfm
from PIL import Image
from torchvision import transforms
import numpy as np
from mynet_utils.split_dataset import split_dataset


class MonkaaDataset(Dataset):
    def __init__(self, db_index_file_path, stage, transform, truth_scale):
        super().__init__()
        self.transform = transform
        self.index_file = torch.load(db_index_file_path)[stage]
        self.truth_scale = truth_scale

    def __len__(self):
        return len(self.index_file)

    def __getitem__(self, index):
        img_left = Image.open(self.index_file[index][0])
        img_right = Image.open(self.index_file[index][1])

        img_left = self.transform(img_left)
        img_right = self.transform(img_right)

        truth, _ = python_pfm.readPFM(self.index_file[index][2])
        img = Image.fromarray(truth)
        truth = torch.FloatTensor(np.array(
            transforms.Resize([truth.shape[0] // self.truth_scale, truth.shape[1] // self.truth_scale])(
                img))) / self.truth_scale

        return self.index_file[index], img_left, img_right, truth
