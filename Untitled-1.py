import torch
from torch.autograd import Variable

# v = Variable((torch.randn([2, 3, 4])*100).int().float(), requires_grad=True)

# print(v)
# print("+"*60)
# # for i in range(v.size()[0]):
# #     v.data[i, ...] = v.data[i, ...] * 2

# # v = Variable(v, requires_grad=True)
# v1 = torch.mul(v, 0.5)
# print(v1)
# print("+"*60)
# grad = (torch.randn([2, 3, 4]) * 10).int()
# grad = grad.float()
# print(grad)

# print("+"*60)
# v1.backward(grad)
# print(v.grad.data)

# print("+"*60)
# print(v.grad_fn)
# print("+"*60)
# print(v1.grad_fn)

x = torch.cuda.FloatTensor(1)
# x.get_device() == 0
y = torch.FloatTensor(1).cuda()
# y.get_device() == 0

with torch.cuda.device(1):
    # allocates a tensor on GPU 1
    a = torch.cuda.FloatTensor(1)

    # transfers a tensor from CPU to GPU 1
    b = torch.FloatTensor(1).cuda()
    # a.get_device() == b.get_device() == 1

    c = a + b
    # c.get_device() == 1

    z = x + y
    # z.get_device() == 0

    # even within a context, you can give a GPU id to the .cuda call
    d = torch.randn(2).cuda(2)
    # d.get_device() == 2
