# -*- coding=utf8 -*-
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

from utils.split_dataset import split_dataset
from utils import python_pfm


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

        # truth, _ = python_pfm.readPFM(self.index_file[index][2])
        truth = np.load(self.index_file[index][2][:-4]+'.npy')

        # img = Image.fromarray(truth)
        # truth = torch.FloatTensor(np.array(
        #     transforms.Resize([truth.shape[0] // self.truth_scale, truth.shape[1] // self.truth_scale])(
        #         img))) / self.truth_scale

        # # check bad pixel point
        # for i in range(truth.size()[0]):
        #     for j in range(truth.size()[1]):
        #         if j - truth[i][j] < 0:
        #             truth[i][j] = 0

        return self.index_file[index], img_left, img_right, torch.from_numpy(truth)
