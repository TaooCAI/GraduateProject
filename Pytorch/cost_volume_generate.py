import torch


def costVolumeGeneration(l, r, max_disparity):
    res = torch.cat([l, r], dim=1)
    res = torch.unsqueeze(res, dim=1)

    for i in range(max_disparity):
        indices = torch.LongTensor([r.size()[-1] - 1])
        z = torch.index_select(r, 3, indices)

        z = torch.cat([z, r], dim=3)
        seq = [ii for ii in range(0, z.size()[-1] - 1)]
        indices = torch.LongTensor(seq)
        z = torch.index_select(z, 3, indices)

        out = torch.cat([l, z], dim=1)

        res = torch.cat([res, torch.unsqueeze(out, dim=1)], dim=1)

        r = z
    return res


def costVolumeGeneration_one(l, r, max_disparity):
    res = torch.cat([l, r], dim=1)
    res = torch.unsqueeze(res, dim=1)

    for i in range(max_disparity):
        tmp = r[:, :, :, -1]
        tmp = torch.unsqueeze(tmp,dim=3)
        r = torch.cat([tmp, r[:, :, :, 0:-1]], dim=3)
        out = torch.cat([l, r], dim=1)

        res = torch.cat([res, torch.unsqueeze(out, dim=1)], dim=1)
    return res


def costVolumeGeneration_another(l, r, max_disparity):
    ss = l.size()
    res = torch.zeros([ss[0], max_disparity+1, ss[1]*2, ss[2], ss[3]])
    for i in range(max_disparity + 1):
        res[:, i, :, :, :] = torch.cat([l, r], dim=1)
        r = torch.cat([r[:, :, :, -1:], r[:, :, :, 0:-1]], dim=3)
    return res