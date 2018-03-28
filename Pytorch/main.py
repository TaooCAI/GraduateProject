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
import copy


MAX_D = 192
root_path = "/home/caitao/Downloads/tmp_data/data/a_rain_of_stones_x2"
truth_path = "/home/caitao/Downloads/tmp_data/groundtruth/a_rain_of_stones_x2/left"


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
    ss = l.size()
    res = torch.zeros(ss[0], ss[1]*2, max_disparity+1, ss[2], ss[3])
    for i in range(max_disparity + 1):
        res[:, :, i, ...] = torch.cat([l.data, r.data], dim=1)
        r = torch.cat([r[..., -1:], r[..., 0:-1]], dim=3)
    return Variable(res, requires_grad=True)


def cost_volume_generation_new(l, r, max_disparity):
    ss = l.size()
    res = torch.zeros(ss[0], ss[1]*2, max_disparity+1, ss[2], ss[3])
    r_cp = copy.deepcopy(r.data)
    for i in range(max_disparity + 1):
        res[:, :, i, ...] = torch.cat([l.data, r_cp], dim=1)
        r_cp = torch.cat([r_cp[..., -1:], r_cp[..., 0:-1]], dim=3)
    return Variable(res)


def get_data_other():
    data_folder = datasets.ImageFolder(root_path, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    batch = 1000
    loader = torch.utils.data.DataLoader(data_folder, batch_size=batch)

    for ii, images in enumerate(loader):
        if ii == 0:
            l = torch.zeros_like(images[0][0])
            l = torch.unsqueeze(l, dim=0)
            r = torch.zeros_like(l)
        for j in range(len(images[1])):
            if data_folder.class_to_idx.get("left") == images[1][j]:
                l = torch.cat([l, images[0][j:j+1]])
            else:
                r = torch.cat([r, images[0][j:j+1]])

    l = l[1:, ...]
    r = r[1:, ...]
    return l, r


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


def get_data():
    def get_one(child_dir):
        data_dir = os.path.join(root_path, child_dir)
        images_list = os.listdir(data_dir)
        for i, filename in enumerate(images_list):
            image_path = os.path.join(data_dir, filename)
            image = Image.open(image_path)
            transform_list = [transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5))]
            transform = transforms.Compose(transform_list)
            image = transform(image)
            if i == 0:
                res = torch.unsqueeze(image, dim=0)
            else:
                res = torch.cat([res, torch.unsqueeze(image, dim=0)], dim=0)
        return res, images_list
    l, imglist = get_one("left")
    r, imglist_r = get_one("right")
    assert imglist == imglist_r

    truth_namelist = [x[:x.rindex('.') + 1] + "pfm" for x in imglist]
    print(truth_namelist)
    for ii, truth_name in enumerate(truth_namelist):
        abso_path = os.path.join(truth_path, truth_name)
        onetruth, _ = readPFM(abso_path)
        onetruth = torch.from_numpy(np.array(onetruth.tolist()))
        if ii == 0:
            truth = torch.unsqueeze(onetruth, dim=0)
        else:
            truth = torch.cat([truth, torch.unsqueeze(onetruth, dim=0)], dim=0)

    return l, r, truth


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

        self.deconv33 = nn.ConvTranspose3d(128, 64, 3, stride=2)
        self.deconv34 = nn.ConvTranspose3d(64, 64, 3, stride=2)
        self.deconv35 = nn.ConvTranspose3d(64, 64, 3, stride=2)
        self.deconv36 = nn.ConvTranspose3d(
            64, 32, 3, stride=2, output_padding=(1, 0, 0))

        self.deconv37 = nn.ConvTranspose3d(
            32, 1, 3, stride=2, output_padding=(0, 1, 1))

    def forward(self, l, r):
        l = self.conv1(l)
        l = self.block1(l)
        l = self.block2(l)
        l = self.block3(l)
        l = self.block4(l)
        l = self.block5(l)
        l = self.block6(l)
        l = self.block7(l)
        l = self.block8(l)
        l = self.conv2(l)

        r = self.conv1(r)
        r = self.block1(r)
        r = self.block2(r)
        r = self.block3(r)
        r = self.block4(r)
        r = self.block5(r)
        r = self.block6(r)
        r = self.block7(r)
        r = self.block8(r)
        r = self.conv2(r)

        v = cost_volume_generation(l, r, 95)

        out21 = self.conv21(v)
        out24 = self.conv24(out21)
        out27 = self.conv27(out24)

        out30 = self.conv30(out27)
        out31 = self.conv31(out30)
        out32 = self.conv32(out31)
        out33 = self.deconv33(out32)

        out28 = self.conv28(out27)
        out29 = self.conv29(out28)
        out33_s = out29 + out33
        out34 = self.deconv34(out33_s)

        out25 = self.conv25(out24)
        out26 = self.conv26(out25)
        out34_s = out26 + out34
        out35 = self.deconv35(out34_s)

        out22 = self.conv22(out21)
        out23 = self.conv23(out22)
        out35_s = out23 + out35
        out36 = self.deconv36(out35_s)

        out19 = self.conv19(v)
        out20 = self.conv20(out19)
        out36_s = out20 + out36
        out37 = self.deconv37(out36_s)

        out37 = (nn.Softmax(dim=2))(torch.mul(out37, -1))
        res = []
        for i in range(out37.size()[2]):
            if len(res) == 0:
                res.append(torch.mul(out37.data[:, :, i, ...], i))
            else:
                res[0] += torch.mul(out37.data[:, :, i, ...], i)
        out = torch.squeeze(res[0], dim=1)

        return Variable(out, requires_grad=True)


def get_test_data():
    h = 64
    w = 64
    n = 2
    left = torch.randn([n, 3, h, w])
    right = torch.randn([n, 3, h, w])
    truth = torch.randn([n, h, w])
    return left, right, truth


def train_gcnet():
    net = GCNet()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)

    if torch.cuda.is_available():
        net.cuda()

    # l, r, truth = get_test_data()
    # l, r, truth = get_data()
    if torch.cuda.is_available():
        l, r, truth = l.cuda(), r.cuda(), truth.cuda()
    l, r, truth = Variable(l), Variable(r), Variable(truth)
    optimizer.zero_grad()
    out = net(l, r)
    loss = F.l1_loss(out, truth)
    loss.backward()
    optimizer.step()
    print("successfully train once!")


train_gcnet()
