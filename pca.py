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

test_data = []
train_data = []
for i in range(0, 10):
    class_i = data[data[:, 0] == i]
    np.random.shuffle(class_i)

    mid = (len(class_i) // 2)
    test_i = class_i[0:mid, :]
    train_i = class_i[mid: , :]

    test_data.append(test_i)
    train_data.append(train_i)

# concatenate and shuffle the data to form the test data
test_data = np.concatenate(test_data, axis=0)
np.random.shuffle(test_data)
print(test_data.shape)

# concatenate and shuffle the data to form the train data
train_data = np.concatenate(train_data, axis=0)
np.random.shuffle(train_data)
print(train_data.shape)