# -*- coding=utf8 -*-

"""
@author: CaiTao
@license: Apache Licence
@contact: 1120141815@bit.edu.cn
@file: MonkaaDataset.py
@time: 4/16/18 4:26 PM
"""
import os

import scipy.misc as scim

from torch.utils.data import Dataset
from mynet_utils import utils
import torch

class MonkaaDataset(Dataset):
    def __init__(self, images_root, truth_root, transform):
        super().__init__()
        self.transform = transform
        self.left = []
        self.right = []
        self.truth = []
        for kind in os.listdir(images_root):
            l = os.path.join(images_root, kind, 'left')
            r = os.path.join(images_root, kind, 'right')
            t = os.path.join(truth_root, kind, 'left')
            self.left += [os.path.join(l, image) for image in os.listdir(l) if image.endswith('.png')]
            self.right += [os.path.join(r, image) for image in os.listdir(r) if image.endswith('.png')]
            self.truth += [os.path.join(t, image) for image in os.listdir(t) if image.endswith('.pfm')]
        # 17328
        assert len(self.left) == len(self.right) == len(self.truth)

    def __len__(self):
        return len(self.left)

    def __getitem__(self, index):
        img_left = scim.imread(self.left[index])
        img_right = scim.imread(self.right[index])

        img_left = self.transform(img_left)
        img_right = self.transform(img_right)

        truth, _ = utils.readPFM(self.truth[index])
        truth = torch.FloatTensor(truth.tolist())

        return img_left, img_right, truth
