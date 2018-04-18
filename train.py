# -*- coding=utf8 -*-
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset
from MonkaaDataset import MonkaaDataset
import visdom
import os

index_file_path = "/home/caitao/Documents/Monkaa/monkaa_list.pth"
# index_file_path = '/home/caitao/Downloads/tmp_data/db_list.pth'
model_path = '/home/caitao/Documents/Monkaa/model/'
cuda_available = False
epochs = 15


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

        self.dec4 = nn.Sequential(nn.ConvTranspose3d(
            128, 64, 3, stride=2, output_padding=(1, 0, 0)),
            nn.BatchNorm3d(64), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose3d(
            64, 64, 3, stride=2, output_padding=(0, 0, 0)),
            nn.BatchNorm3d(64), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose3d(
            64, 64, 3, stride=2, output_padding=(0, 0, 0)),
            nn.BatchNorm3d(64), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose3d(
            64, 32, 3, stride=2, output_padding=(0, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU())
        self.output = nn.Conv3d(
            32, 1, kernel_size=3, stride=1, padding=1)

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

        out = self.conv19(v)
        out20 = self.conv20(out)
        out = v + out20

        out21 = self.enc1(out)
        out = self.conv22(out21)
        out23 = self.conv23(out)
        out = out21 + out23

        out24 = self.enc2(out)
        out = self.conv25(out24)
        out26 = self.conv26(out)
        out = out24 + out26

        out27 = self.enc3(out)
        out = self.conv28(out27)
        out29 = self.conv29(out)
        out = out27 + out29

        out = self.enc4(out)
        out = self.conv31(out)
        out = self.out32(out)
        out = self.dec4(out)

        out = out + out29
        out = self.dec3(out)

        out = out + out26
        out = self.dec2(out)

        out = out + out23
        out = self.dec1(out)

        out = out + out20

        out = self.output(out)

        out = (nn.Softmax(dim=4))(torch.mul(out, -1))
        length = out.data.size()[4]
        res = torch.mul(out[:, :, :, :, 0], 1)
        for i in range(1, length):
            res += torch.mul(out[:, :, :, :, i], i + 1)
        out = torch.squeeze(res, dim=1)
        return out


def train():
    batch_size = 8
    whether_vis = True
    if whether_vis is True:
        vis = visdom.Visdom()
        loss_window = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1,)).cpu(),
                               opts=dict(xlabel='batches', ylabel='loss', title='Trainingloss', legend=['loss']))
    model = GCNet()
    model.train()
    if cuda_available:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model = model.cuda()

    train_loader = torch.utils.data.DataLoader(
        MonkaaDataset(
            index_file_path,
            stage='train',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])),
        batch_size=batch_size, num_workers=batch_size)

    criterion = nn.L1Loss()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

    best = 100000.0

    x_pos = 0

    whether_load_model = False
    state_file = 'do not load. when whether_load_model is True, please specify this variable'

    epoch_start = 1
    if whether_load_model is True:
        state = torch.load(state_file)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        epoch_start = state['epoch']

    for epoch in range(epoch_start, epochs + 1):
        for batch_idx, (l, r, truth) in enumerate(train_loader):
            if cuda_available:
                l, r, truth = l.cuda(), r.cuda(), truth.cuda()
            l, r, truth = Variable(l), Variable(r), Variable(truth)

            outputs = model(l, r)
            optimizer.zero_grad()
            loss = criterion(outputs, truth)
            loss.backward()
            optimizer.step()
            if whether_vis is True:
                vis.line(
                    X=torch.ones((1,)).cpu() * x_pos,
                    Y=torch.Tensor([loss.data[0]]).cpu(),
                    win=loss_window,
                    update='append')
            x_pos += 1
            if (batch_idx + 1) % 2 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(truth),
                    len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.data[0]))
            if loss.data[0] < best:
                state = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                torch.save(state, os.path.join(
                    model_path, f'best_model_cache_{epoch}_{(batch_idx+1)*batch_size}.pth'))
                continue
            if (batch_idx + 1) % 10 == 0:
                state = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                torch.save(state, os.path.join(
                    model_path, f'model_cache_{epoch}_{(batch_idx+1)*batch_size}.pth'))


if __name__ == "__main__":
    if torch.cuda.is_available():
        cuda_available = True
    train()
