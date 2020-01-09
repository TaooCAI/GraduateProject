# -*- coding=utf8 -*-
import os
import random
import torch


def split_dataset(images_root, truth_root, save_path, scale=4):
    left = []
    right = []
    truth = []
    for kind in os.listdir(images_root):
        l = os.path.join(images_root, kind, 'left')
        r = os.path.join(images_root, kind, 'right')
        t = os.path.join(truth_root, kind, 'left')
        left += [os.path.join(l, item) for item in sorted(os.listdir(l)) if item.endswith('.png')]
        right += [os.path.join(r, item) for item in sorted(os.listdir(r)) if item.endswith('.png')]
        truth += [os.path.join(t, item) for item in sorted(os.listdir(t)) if item.endswith('.pfm')]
    length = len(left)
    tmp_list = list(range(length))
    random.shuffle(tmp_list)
    database = [(left[i], right[i], truth[i]) for i in tmp_list]
    print('database size: {}'.format(length))
    length = length // scale
    test = database[:length]
    train = database[length:]
    print('test size: {}'.format(len(test)))
    print('train size: {}'.format(len(train)))

    db = {
        'test': test,
        'train': train
    }
    torch.save(db, save_path)


if __name__ == "__main__":
    image = "/home/caitao/Documents/Monkaa/frames_cleanpass/"
    truth = "/home/caitao/Documents/Monkaa/disparity/"
    save_path = "/home/caitao/Documents/Monkaa/monkaa_list.pth"
    split_dataset(image, truth, save_path, scale=4)
