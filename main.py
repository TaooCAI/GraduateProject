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


MAX_D = 192
root_path = "/home/caitao/Downloads/tmp_data/data/a_rain_of_stones_x2"
truth_path = "/home/caitao/Downloads/tmp_data/groundtruth/a_rain_of_stones_x2/left"
cuda_available = False


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


class MyDataset(Dataset):
    def __init__(self, root_path, truth_path, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        self.root_path = root_path
        self.truth_path = truth_path
        self.transform = transform
        self.target_transform = target_transform
        self.data = self._make_dataset()

    def __getitem__(self, index):
        if index < 0 or index >= self.__len__():
            raise(Exception("index out of range"))
        l = self.data[0]
        r = self.data[1]
        truth = self.data[2]
        return l[index, ...], r[index, ...], truth[index, ...]

    def __len__(self):
        return len(self.data)

    def _make_dataset(self):
        def get_one(child_dir):
            data_dir = os.path.join(self.root_path, child_dir)
            images_list = os.listdir(data_dir)
            ans = []
            for i, filename in enumerate(images_list):
                image_path = os.path.join(data_dir, filename)
                image = Image.open(image_path)
                transform = self.transform
                image = transform(image)
                ans.append(image)
            res = torch.stack(ans, dim=0)
            return res, images_list
        l, imglist = get_one("left")
        r, imglist_r = get_one("right")
        assert imglist == imglist_r

        truth_namelist = [x[:x.rindex('.') + 1] + "pfm" for x in imglist]
        ans = []
        for ii, truth_name in enumerate(truth_namelist):
            abso_path = os.path.join(self.truth_path, truth_name)
            onetruth, _ = readPFM(abso_path)
            onetruth = torch.FloatTensor(onetruth.tolist())
            ans.append(onetruth)

        truth = torch.stack(ans, dim=0)
        print(truth_namelist)
        return l, r, truth


train_loader = torch.utils.data.DataLoader(MyDataset(root_path, truth_path, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])), batch_size=2)


def conv5x5(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=5,
                  stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def conv3x3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2),
        nn.BatchNorm3d(out_channels),
        nn.ReLU()
    )


def conv3x3x3_padding(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels,
                  kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU()
    )


def cost_volume_generation(l, r, max_disparity):
    ans = []
    ans.append(torch.cat([l, r], dim=1))
    for t in range(1, max_disparity + 1):
        ans.append(
            torch.cat([l, torch.cat([r[..., t:], r[..., :t]], dim=3)], dim=1))
    return torch.stack(ans, dim=4)


# Residual Block


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class GCNet(nn.Module):
    def __init__(self):
        super(GCNet, self).__init__()
        self.input_channels = 3

        self.conv1 = conv5x5(self.input_channels, 32)
        # self.conv_down = conv5x5(32, 32)

        self.block1 = ResidualBlock(32, 32)
        self.block2 = ResidualBlock(32, 32)
        self.block3 = ResidualBlock(32, 32)
        self.block4 = ResidualBlock(32, 32)
        self.block5 = ResidualBlock(32, 32)
        self.block6 = ResidualBlock(32, 32)
        self.block7 = ResidualBlock(32, 32)
        self.block8 = ResidualBlock(32, 32)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv19 = conv3x3x3_padding(64, 32)
        self.conv20 = conv3x3x3_padding(32, 32)
        self.conv21 = conv3x3x3(64, 64)
        self.conv22 = conv3x3x3_padding(64, 64)
        self.conv23 = conv3x3x3_padding(64, 64)
        self.conv24 = conv3x3x3(64, 64)
        self.conv25 = conv3x3x3_padding(64, 64)
        self.conv26 = conv3x3x3_padding(64, 64)
        self.conv27 = conv3x3x3(64, 64)
        self.conv28 = conv3x3x3_padding(64, 64)
        self.conv29 = conv3x3x3_padding(64, 64)
        self.conv30 = conv3x3x3(64, 128)
        self.conv31 = conv3x3x3_padding(128, 128)
        self.conv32 = conv3x3x3_padding(128, 128)

        self.deconv33 = nn.ConvTranspose3d(
            128, 64, 3, stride=2, output_padding=(1, 0, 0))
        self.deconv34 = nn.ConvTranspose3d(
            64, 64, 3, stride=2, output_padding=(1, 0, 0))
        self.deconv35 = nn.ConvTranspose3d(
            64, 64, 3, stride=2, output_padding=(1, 0, 0))
        self.deconv36 = nn.ConvTranspose3d(
            64, 32, 3, stride=2, output_padding=(0, 0, 1))

        self.deconv37 = nn.ConvTranspose3d(
            32, 1, 3, stride=2, output_padding=(1, 1, 0))

        # self.deconv38 = nn.ConvTranspose3d(
        #     32, 1, 3, stride=2, output_padding=(1, 1, 0)
        # )

    def forward(self, l, r):
        l = self.conv1(l)
        l = self.block8(self.block7(self.block6(self.block5(
            self.block4(self.block3(self.block2(self.block1(l))))))))
        l = self.conv2(l)

        r = self.conv1(r)
        r = self.block8(self.block7(self.block6(self.block5(
            self.block4(self.block3(self.block2(self.block1(r))))))))
        r = self.conv2(r)

        # v = cost_volume_generation(l, r, 46)
        v = cost_volume_generation(l, r, 95)

        out21 = self.conv21(v)
        out24 = self.conv24(out21)
        out27 = self.conv27(out24)

        out = self.conv29(self.conv28(out27)) + \
            self.deconv33(self.conv32(self.conv31(self.conv30(out27))))
        out = self.conv26(self.conv25(out24)) + self.deconv34(out)

        out = self.conv23(self.conv22(out21)) + self.deconv35(out)

        out = self.conv20(self.conv19(v)) + self.deconv36(out)

        out = self.deconv37(out)

        out = (nn.Softmax(dim=4))(torch.mul(out, -1))
        res = []
        for i in range(out.size()[2]):
            if len(res) == 0:
                res.append(torch.mul(out[:, :, :, :, i], i))
            else:
                res[0] += torch.mul(out[:, :, :, :, i], i)
        out = torch.squeeze(res[0], dim=1)
        return out


net = GCNet()
if torch.cuda.is_available():
    cuda_available = True
    net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)


def train_gcnet(epoch):
    net.train()

    for batch_idx, (l, r, truth) in enumerate(train_loader):
        l, r, truth = Variable(l), Variable(r), Variable(truth)
        if cuda_available:
            l, r, truth = l.cuda(), r.cuda(), truth.cuda()
        optimizer.zero_grad()
        out = net(l, r)

        loss = F.l1_loss(out, truth)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(truth), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test_gcnet():
    pass


epochs = 1

for epoch in range(1, epochs + 1):
    train_gcnet(epoch)
    test_gcnet()
