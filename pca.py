import os
import numpy as np
import scipy.io

from os import path

# get the current working directory
ROOT_DIR = path.abspath(os.curdir)
DATA_DIR = path.join(ROOT_DIR, 'digits')

mat = scipy.io.loadmat(path.join(DATA_DIR, 'digits.mat'))

x = mat['labels']
print(x.shape)

y = mat['digits']
print(y.shape)

# concatenate the data
data = np.concatenate((x, y), axis=1)
print(data)
print(data.shape)
