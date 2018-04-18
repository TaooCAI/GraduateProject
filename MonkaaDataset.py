# -*- coding=utf8 -*-

"""
@author: CaiTao
@license: Apache Licence
@contact: 1120141815@bit.edu.cn
@file: MonkaaDataset.py
@time: 4/16/18 7:25 PM
"""

import scipy.misc as scim
import skimage.transform as transform
from torch.utils.data import Dataset
import torch
from mynet_utils import python_pfm


class MonkaaDataset(Dataset):
    def __init__(self, db_index_file_path, stage, transform):
        super().__init__()
        self.transform = transform
        self.index_file = torch.load(db_index_file_path)[stage]

    def __len__(self):
        return len(self.index_file)

    def __getitem__(self, index):
        img_left = scim.imread(self.index_file[index][0])
        img_right = scim.imread(self.index_file[index][1])

        img_left = self.transform(img_left)
        img_right = self.transform(img_right)

        truth, _ = python_pfm.readPFM(self.index_file[index][2])
        truth = transform.downscale_local_mean(truth, (4, 4))
        truth = torch.FloatTensor(truth)

        return img_left, img_right, truth
