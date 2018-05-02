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
import time
from PIL import Image

index_file_path = "/home/caitao/Documents/Monkaa/monkaa_list.pth"
model_path = '/home/caitao/Documents/Monkaa/model_SGD/'
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

        self.input = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(32), nn.ReLU())
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
        l = self.input(l)
        l = self.block1(l)
        l = self.block2(l)
        l = self.block3(l)
        l = self.block4(l)
        l = self.block6(l)
        l = self.block6(l)
        l = self.block7(l)
        l = self.block8(l)

        l = self.conv(l)

        r = self.input(r)
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
        out = self.conv32(out)
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
    batch_size = 4
    whether_vis = True

    if whether_vis is True:
        vis = visdom.Visdom(port=9999)
        loss_window = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1,)).cpu(),
                               opts=dict(xlabel='batches', ylabel='loss', title='Trainingloss', legend=['loss']))
        A = torch.randn([4, 5])
        A = (A - torch.min(A)) / torch.max(A)
        image_groundtruth = vis.image(A.cpu(), opts=dict(title='groundtruth'))
        image_output = vis.image(A.cpu(), opts=dict(title='output'))

    model = GCNet()
    model.train()
    if cuda_available:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model = model.cuda()

    scale = 4

    train_loader = torch.utils.data.DataLoader(
        MonkaaDataset(
            index_file_path,
            stage='train',
            transform=transforms.Compose([
                transforms.Resize([540 // scale, 960 // scale], interpolation=Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]), truth_scale=scale),
        batch_size=batch_size, num_workers=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        MonkaaDataset(
            index_file_path,
            stage='test',
            transform=transforms.Compose([
                transforms.Resize([540 // scale, 960 // scale], interpolation=Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]), truth_scale=scale),
        batch_size=batch_size, num_workers=batch_size, shuffle=True)

    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

    best = 100000.0
    test_best = 100000.0
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
        # train stage
        for batch_idx, (path_index_tuple, l, r, truth) in enumerate(train_loader):
            if cuda_available:
                l, r, truth = l.cuda(), r.cuda(), truth.cuda()
            l, r, truth = Variable(l), Variable(r), Variable(truth)

            outputs = model(l, r)
            optimizer.zero_grad()
            loss = criterion(outputs, truth)
            loss.backward()
            optimizer.step()

            if loss.data[0] < best:
                best = loss.data[0]
                state = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'loss': best,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                torch.save(state, os.path.join(
                    model_path, f'best_model.pth'))

            if whether_vis:
                vis.line(
                    X=torch.ones((1,)).cpu() * x_pos,
                    Y=torch.Tensor([loss.data[0]]).cpu(),
                    win=loss_window,
                    update='append')
                vis.image(((truth.data[0] - torch.min(truth.data[0])) / torch.max(truth.data[0])).cpu(),
                          win=image_groundtruth, opts=dict(title='groundtruth'))
                vis.image(((outputs.data[0] - torch.min(outputs.data[0])) / torch.max(outputs.data[0])).cpu(),
                          win=image_output, opts=dict(title='output'))

            x_pos += 1
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(truth),
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.data[0]))

        # test stage
        sum_loss = 0.0
        pre_loss = -1.0
        for batch_idx, (path_index_tuple, l, r, truth) in enumerate(test_loader):
            if (batch_idx * 4) >= 400:
                break
            if cuda_available:
                l, r, truth = l.cuda(), r.cuda(), truth.cuda()
            l, r, truth = Variable(l, volatile=True), Variable(
                r, volatile=True), Variable(truth, volatile=True)
            outputs = model(l, r)
            loss = criterion(outputs, truth)
            if batch_idx == 0:
                pre_loss = loss.data[0]
            if pre_loss * 8 <= loss.data[0]:
                ex = {
                    'loss': loss.data[0],
                    'path': path_index_tuple
                }
                torch.save(ex, os.path.join(
                    model_path, f'test_exception_{epoch}_{batch_idx}.pth'))
            else:
                pre_loss = loss.data[0]
            sum_loss += loss.data[0]
        sum_loss = sum_loss / (batch_idx*4)

        # save test best model
        if sum_loss < test_best:
            test_best = sum_loss
            state = {
                'loss': sum_loss,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            torch.save(state, os.path.join(
                model_path, f'test_best_model.pth'))
        print(f'Test Stage: Average loss: {sum_loss}\n')

        # save model and optimizer
        state = {
            'loss': sum_loss,
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }
        torch.save(state, os.path.join(
            model_path, f'model_cache_{epoch}.pth'))


if __name__ == "__main__":
    if torch.cuda.is_available():
        cuda_available = True
    start = time.time()
    os.makedirs(model_path, exist_ok=True)
    train()
    end = time.time()
    print(f'cost time: {end - start}')
