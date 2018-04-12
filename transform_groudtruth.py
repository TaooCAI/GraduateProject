import numpy as np
import os
import scipy.io as sio
import re


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


if __name__ == "__main__":
    mat_path = "ground_truth.mat"
    truth_path = "/home/caitao/Downloads/tmp_data/groundtruth/a_rain_of_stones_x2/left"
    mat_path = "/home/caitao/Downloads/tmp_data/groundtruth/a_rain_of_stones_x2/mat"

    if not os.path.exists(mat_path):
        os.makedirs(mat_path)
    for f in os.listdir(truth_path):
        if f.endswith('pfm'):
            data, _ = readPFM(os.path.join(truth_path, f))
            sio.savemat(os.path.join(mat_path, f), {'mat': data})