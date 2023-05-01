import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from os import path
from sklearn.decomposition import PCA
from numpy.linalg import eig

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
test_labels = test_data[:, 0]
test_data = test_data[:, 1:]
print(test_labels.shape)
print(test_data.shape)

# concatenate and shuffle the data to form the train data
train_data = np.concatenate(train_data, axis=0)
np.random.shuffle(train_data)
train_labels = train_data[:, 0]
train_data = train_data[:, 1:]
print(train_labels.shape)
print(train_data.shape)

pca = PCA()
pca.fit(train_data)

print(pca.components_)
print(pca.explained_variance_)
print(pca.singular_values_)

"""
plt.figure()
plt.plot(pca.explained_variance_)
plt.axhline(y = 0.1, color = 'r', linestyle = '-')
plt.show()
"""

# returns the index where the cumulative explained variance exceeds f, which is between 0 and 1
def cumulative_explained_variance(eigenVals, f : float):
    if f < 0 or f > 1:
        print('f is out of the interval [0, 1]')
        return -1

    # get the cumulative ratios
    cum_ratios = np.cumsum(eigenVals) / np.sum(eigenVals)
    return np.searchsorted(cum_ratios, f)

# Capture %90 variance
print(cumulative_explained_variance(pca.explained_variance_, 0.9))

# Question 1.2
train_mean = np.mean(train_data, axis=0)
print(train_mean)

# plot the image
plt.figure()
plt.axis('off')
plt.title('The Mean Image From Training Data')
train_mean = train_mean.reshape((20, 20, -1))
plt.imshow(train_mean, cmap='gray')
plt.show()