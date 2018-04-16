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
import scipy.io as sio

MAX_D = 192
ROOT_PATH = "/home/caitao/Downloads/tmp_data/data/a_rain_of_stones_x2"
TRUTH_PATH = "/home/caitao/Downloads/tmp_data/groundtruth/a_rain_of_stones_x2/left"
cuda_available = False
epochs = 10


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
    def __init__(self,
                 root_path,
                 truth_path,
                 transform=None,
                 target_transform=None):
        super(MyDataset, self).__init__()
        self.root_path = root_path
        self.truth_path = truth_path
        self.transform = transform
        self.target_transform = target_transform
        self.data = self._make_dataset()

    def __getitem__(self, index):
        if index < 0 or index >= self.__len__():
            raise (Exception("index out of range"))
        l = self.data[0]
        r = self.data[1]
        truth = self.data[2]
        return l[index, ...], r[index, ...], truth[index, ...]

    def __len__(self):
        return (self.data[0]).size()[0]

    def _make_dataset(self):
        def get_one(child_dir):
            data_dir = os.path.join(self.root_path, child_dir)
            images_list = os.listdir(data_dir)
            ans = []
            for _, filename in enumerate(images_list):
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
        for _, truth_name in enumerate(truth_namelist):
            abso_path = os.path.join(self.truth_path, truth_name)
            onetruth, _ = readPFM(abso_path)
            onetruth = torch.FloatTensor(onetruth.tolist())
            # onetruth = sio.loadmat(abso_path)['tmp']
            # onetruth = torch.FloatTensor(onetruth)
            ans.append(onetruth)

        truth = torch.stack(ans, dim=0)
        print(truth_namelist)
        return l, r, truth


def down_sample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=5, stride=2, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU())


def conv5x5(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(out_channels), nn.ReLU())


def conv3x3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2),
        nn.BatchNorm3d(out_channels), nn.ReLU())


def conv3x3x3_padding(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels), nn.ReLU())


def cost_volume_generation(l, r, max_disparity):
    ans = []
    # ans.append(torch.cat([l, r], dim=1))
    for t in range(1, max_disparity + 1):
        ans.append(
            torch.cat([l, torch.cat([r[..., t:], r[..., :t]], dim=3)], dim=1))
    return torch.stack(ans, dim=4)


# Residual Block


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=1, padding=1)
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


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        with torch.cuda.device(2):
            self.fea = conv5x5(3, 32).cuda()
            self.down_sample1 = down_sample(32, 32).cuda()
            self.down_sample2 = down_sample(32, 32).cuda()
            self.block1 = ResidualBlock(32, 32).cuda()
            self.block2 = ResidualBlock(32, 32).cuda()
            self.block3 = ResidualBlock(32, 32).cuda()
            self.block4 = ResidualBlock(32, 32).cuda()
            self.block5 = ResidualBlock(32, 32).cuda()
            self.block6 = ResidualBlock(32, 32).cuda()
            self.block7 = ResidualBlock(32, 32).cuda()
            self.block8 = ResidualBlock(32, 32).cuda()

            self.conv = nn.Conv2d(32, 32, 3, padding=1).cuda()

            self.conv19 = conv3x3x3_padding(64, 32).cuda()
            self.conv20 = conv3x3x3_padding(32, 1).cuda()
            self.enc1 = conv3x3x3(64, 64).cuda()
            self.conv22 = conv3x3x3_padding(64, 64).cuda()
            self.conv23 = conv3x3x3_padding(64, 64).cuda()
            self.enc2 = conv3x3x3(64, 64).cuda()
            self.conv25 = conv3x3x3_padding(64, 64).cuda()
            self.conv26 = conv3x3x3_padding(64, 64).cuda()
            self.enc3 = conv3x3x3(64, 64).cuda()
            self.conv28 = conv3x3x3_padding(64, 64).cuda()
            self.conv29 = conv3x3x3_padding(64, 64).cuda()
            self.enc4 = conv3x3x3(64, 128).cuda()
            self.conv31 = conv3x3x3_padding(128, 128).cuda()
            self.conv32 = conv3x3x3_padding(128, 128).cuda()

            self.dec4 = nn.ConvTranspose3d(
                128, 64, 3, stride=2, output_padding=(0, 0, 0)).cuda()
            self.dec3 = nn.ConvTranspose3d(
                64, 64, 3, stride=2, output_padding=(1, 0, 0)).cuda()
            self.dec2 = nn.ConvTranspose3d(
                64, 64, 3, stride=2, output_padding=(1, 0, 0)).cuda()
            self.dec1 = nn.ConvTranspose3d(
                64, 32, 3, stride=2, output_padding=(1, 0, 0)).cuda()
        # self.output = nn.ConvTranspose3d(
        #     32, 1, 1, stride=1, output_padding=(0, 0, 0)).cuda()

        # self.up_sample2 = nn.ConvTranspose3d(
        #     32, 32, 3, stride=2, output_padding=(0, 0, 1)).cuda(1)
        # self.up_sample1 = nn.ConvTranspose3d(
        #     32, 1, 3, stride=2, output_padding=(1, 1, 0)).cuda(2)
            self.full_resolution2 = nn.ConvTranspose2d(
                32, 32, 3, stride=2, output_padding=(0, 0)).cuda()
            self.full_resolution1 = nn.ConvTranspose2d(
                32, 32, 3, stride=2, output_padding=(1, 1)).cuda()
            self.output = nn.Conv2d(32, 1, 3, padding=1).cuda()

    def forward(self, l, r):
        feature_l = self.fea(l)
        half_l = self.down_sample1(feature_l)
        quarter_l = self.down_sample2(half_l)
        ql = self.block1(quarter_l)
        ql = self.block2(ql)
        ql = self.block3(ql)
        ql = self.block4(ql)
        ql = self.block6(ql)
        ql = self.block6(ql)
        ql = self.block7(ql)
        ql = self.block8(ql)
        ql = self.conv(ql)

        feature_r = self.fea(r)
        half_r = self.down_sample1(feature_r)
        quarter_r = self.down_sample2(half_r)
        qr = self.block1(quarter_r)
        qr = self.block2(qr)
        qr = self.block3(qr)
        qr = self.block4(qr)
        qr = self.block5(qr)
        qr = self.block6(qr)
        qr = self.block7(qr)
        qr = self.block8(qr)
        qr = self.conv(qr)

        v = cost_volume_generation(ql, qr, 47)

        out21 = self.enc1(v)
        out24 = self.enc2(out21)
        out27 = self.enc3(out24)

        residual = self.enc4(out27)
        residual = self.conv31(residual)
        residual = self.conv32(residual)
        residual = self.dec4(residual)
        x1 = self.conv28(out27)
        x1 = self.conv29(x1)
        residual += x1

        residual = self.dec3(residual)
        x2 = self.conv25(out24)
        x2 = self.conv26(x2)
        residual += x2

        residual = self.dec2(residual)
        x3 = self.conv22(out21)
        x3 = self.conv23(x3)
        residual += x3

        residual = self.dec1(residual)
        x4 = self.conv19(v)
        x4 = self.conv20(x4)
        out = residual + x4

        # out = self.output(out)

        # out = out.cuda(1)
        # out = self.up_sample2(out)
        # out = out.cuda(2)
        # out = self.up_sample1(out)

        out = (nn.Softmax(dim=4))(torch.mul(out, -1))
        length = out.data.size()[4]
        res = torch.mul(out[:, :, :, :, 0], 1)
        for i in range(1, length):
            res += torch.mul(out[:, :, :, :, i], i+1)
        # out = torch.squeeze(res, dim=1)

        # GC-Net part ends
        # FS part starts

        out = self.full_resolution2(quarter_l + quarter_r + res)
        out = self.full_resolution1(half_l + half_r + out)
        out = self.output(feature_l + feature_r + out)
        return out


def print_param_count(model):
    count = 0
    for param in model.parameters():
        ans = 1
        for num in param.size():
            ans = ans * num
        count += ans
    print("parameter's count is {}\n".format(count))


def train(model, epoch):
    model.train()
    # print_param_count(model)

    train_loader = torch.utils.data.DataLoader(
        MyDataset(
            ROOT_PATH,
            TRUTH_PATH,
            transform=transforms.Compose([
                # transforms.Resize((270, 480), interpolation=Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])),
        batch_size=1)

    criterion = nn.L1Loss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

    for epoch in range(1, epochs + 1):
        for batch_idx, (l, r, truth) in enumerate(train_loader):
            if cuda_available:
                l, r, truth = l.cuda(2), r.cuda(2), truth.cuda(2)
            l, r, truth = Variable(l), Variable(r), Variable(truth)

            outputs = model(l, r)
            optimizer.zero_grad()
            loss = criterion(outputs, truth)
            loss.backward()
            optimizer.step()
            # if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(truth),
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.data[0]))
            # del l, r, truth, outputs
            # gc.collect()
            # torch.cuda.empty_cache()


def test_gcnet():
    pass


def main():
    model = MyNet()
    train(model, epochs)


if __name__ == "__main__":
    if torch.cuda.is_available():
        cuda_available = True
    main()
