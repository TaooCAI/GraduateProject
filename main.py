# -*- coding=utf8 -*-
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset
from MonkaaDataset import MonkaaDataset

IMAGE_PATH = "/home/caitao/Downloads/tmp_data/data/"
TRUTH_PATH = "/home/caitao/Downloads/tmp_data/groundtruth/"
cuda_available = False
epochs = 10


def down_sample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(out_channels), nn.ReLU())


def conv3x3x3_half(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2),
        nn.BatchNorm3d(out_channels), nn.ReLU())


def conv3x3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels), nn.ReLU())


def cost_volume_generation(l, r, max_disparity):
    ans = []
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


class GCNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_sample1 = down_sample(3, 32)
        self.down_sample2 = down_sample(32, 32)

        self.block1 = ResidualBlock(32, 32)
        self.block2 = ResidualBlock(32, 32)
        self.block3 = ResidualBlock(32, 32)
        self.block4 = ResidualBlock(32, 32)
        self.block5 = ResidualBlock(32, 32)
        self.block6 = ResidualBlock(32, 32)
        self.block7 = ResidualBlock(32, 32)
        self.block8 = ResidualBlock(32, 32)

        self.conv = nn.Conv2d(32, 32, 3, padding=1)

        self.conv19 = conv3x3x3(64, 32)
        self.conv20 = conv3x3x3(32, 1)
        self.enc1 = conv3x3x3_half(64, 64)
        self.conv22 = conv3x3x3(64, 64)
        self.conv23 = conv3x3x3(64, 64)
        self.enc2 = conv3x3x3_half(64, 64)
        self.conv25 = conv3x3x3(64, 64)
        self.conv26 = conv3x3x3(64, 64)
        self.enc3 = conv3x3x3_half(64, 64)
        self.conv28 = conv3x3x3(64, 64)
        self.conv29 = conv3x3x3(64, 64)
        self.enc4 = conv3x3x3_half(64, 128)
        self.conv31 = conv3x3x3(128, 128)
        self.conv32 = conv3x3x3(128, 128)

        self.dec4 = nn.ConvTranspose3d(
            128, 64, 3, stride=2, output_padding=(1, 0, 0))
        self.dec3 = nn.ConvTranspose3d(
            64, 64, 3, stride=2, output_padding=(0, 0, 0))
        self.dec2 = nn.ConvTranspose3d(
            64, 64, 3, stride=2, output_padding=(0, 0, 0))
        self.dec1 = nn.ConvTranspose3d(
            64, 1, 3, stride=2, output_padding=(0, 1, 1))

    def forward(self, l, r):
        l = self.down_sample1(l)
        l = self.down_sample2(l)
        l = self.block1(l)
        l = self.block2(l)
        l = self.block3(l)
        l = self.block4(l)
        l = self.block6(l)
        l = self.block6(l)
        l = self.block7(l)
        l = self.block8(l)
        l = self.conv(l)

        r = self.down_sample1(r)
        r = self.down_sample2(r)
        r = self.block1(r)
        r = self.block2(r)
        r = self.block3(r)
        r = self.block4(r)
        r = self.block5(r)
        r = self.block6(r)
        r = self.block7(r)
        r = self.block8(r)
        r = self.conv(r)

        v = cost_volume_generation(l, r, 48)

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

        out = (nn.Softmax(dim=4))(torch.mul(out, -1))
        length = out.data.size()[4]
        res = torch.mul(out[:, :, :, :, 1], 1)
        for i in range(2, length):
            res += torch.mul(out[:, :, :, :, i], i)
        out = torch.squeeze(res, dim=1)
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
        MonkaaDataset(
            IMAGE_PATH,
            TRUTH_PATH,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])),
        batch_size=2)

    criterion = nn.L1Loss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

    for epoch in range(1, epochs + 1):
        for batch_idx, (l, r, truth) in enumerate(train_loader):
            if cuda_available:
                l, r, truth = l.cuda(1), r.cuda(1), truth.cuda(1)
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


def test_gcnet():
    pass


def main():
    model = GCNet()
    if cuda_available:
        model = torch.nn.DataParallel(model, device_ids=[1])
        model = model.cuda(1)
    train(model, epochs)


if __name__ == "__main__":
    if torch.cuda.is_available():
        cuda_available = True
    main()
