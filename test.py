# -*- coding=utf8 -*-

"""
@author: CaiTao
@license: Apache Licence
@contact: 1120141815@bit.edu.cn
@file: test.py
@time: 4/17/18 4:03 PM
"""

# test_loader = torch.utils.data.DataLoader(
#     MonkaaDataset(
#         index_file_path,
#         stage='test',
#         transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])),
#     batch_size=batch_size, drop_last=True, num_workers=8)
#
#
# def test():
#     model.eval()
#     for batch_idx, (l, r, truth) in enumerate(test_loader):
#         if cuda_available:
#             l, r, truth = l.cuda(), r.cuda(), truth.cuda()
#         l, r, truth = Variable(l, volatile=True), Variable(
#             r, volatile=True), Variable(truth, volatile=True)
#
#         outputs = model(l, r)
#         loss = criterion(outputs, truth)
#         print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#             epoch, (batch_idx + 1) * len(truth),
#             len(test_loader.dataset),
#                    100. * (batch_idx + 1) / len(test_loader), loss.data[0]))
#
#         if loss.data[0] < best:
#             state = {
#                 'epoch': epoch,
#                 'model_state': model.state_dict(),
#                 'optimizer_state': optimizer.state_dict()
#             }
#             torch.save(state, os.path.join(model_path, 'best_model.pth'))


import torch
A  = torch.load('/home/caitao/Documents/Monkaa/monkaa_list.pth')
import pprint
pprint.pprint(A)