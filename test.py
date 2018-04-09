# -*- coding=utf8 -*-
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
from PIL import Image
import re
import numpy as np
from torch.utils.data import Dataset
import gc


class MyDataset(Dataset):
    def __init__(self, length):
        super(MyDataset, self).__init__()
        self.length = length

        self.fea = torch.randn(length, 32, 134, 239, 47)
        self.truth = torch.randn(length, 540, 960)

    def __getitem__(self, index):
        if index < 0 or index >= self.__len__():
            raise(Exception("index out of range"))
        return self.fea[index], self.truth[index]

    def __len__(self):
        return self.length


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.deconv37 = nn.ConvTranspose3d(
            32, 32, 3, stride=2, output_padding=(0, 0, 1))
        self.deconv38 = nn.ConvTranspose3d(
            32, 1, 3, stride=2, output_padding=(1, 1, 0))

    def forward(self, out):
        out = self.deconv37(out)
        out = self.deconv38(out)
        out = (nn.Softmax(dim=4))(torch.mul(out, -1))
        res = []
        for i in range(out.size()[4]):
            if len(res) == 0:
                res.append(torch.mul(out[:, :, :, :, i], i))
            else:
                res[0] += torch.mul(out[:, :, :, :, i], i)
        out = torch.squeeze(res[0], dim=1)
        return out


def print_param_count():
    count = 0
    for param in net.parameters():
        ans = 1
        for num in param.size():
            ans = ans * num
        count += ans
    print("parameter's count is {}\n".format(count))


net = MyNet()


def train_gcnet(epoch):
    net.train()
    print_param_count()
    for batch_idx, (vol, truth) in enumerate(train_loader):
        vol, truth = Variable(vol), Variable(truth)
        if cuda_available:
            vol, truth = vol.cuda(), truth.cuda()
        optimizer.zero_grad()
        out = net(vol)
        loss = F.l1_loss(out, truth)
        loss.backward()
        optimizer.step()
        # if (batch_idx + 1) % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_idx+1) * len(truth), len(train_loader.dataset),
            100. * (batch_idx+1) / len(train_loader), loss.data[0]))
        del loss, out, vol, truth


def test_gcnet():
    pass


if __name__ == "__main__":

    train_loader = torch.utils.data.DataLoader(MyDataset(14), batch_size=1)

    if torch.cuda.is_available():
        cuda_available = True
        net = net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
    epochs = 1

    for epoch in range(1, epochs + 1):
        train_gcnet(epoch)
        test_gcnet()
