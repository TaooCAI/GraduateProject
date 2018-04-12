import numpy as np
import scipy.io as sio

mat_file = "matlab_mat.mat"

# save_array = np.random.rand(4,5)
#
# sio.savemat(mat_file, {'mat':save_array})

save_array = sio.loadmat(mat_file)
nparray = np.array(save_array['mat'])

print(nparray)