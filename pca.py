import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from os import path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.manifold import Isomap
from numpy.linalg import eig
from tqdm import tqdm

# get the current working directory
ROOT_DIR = path.abspath(os.curdir)
DATA_DIR = path.join(ROOT_DIR, 'digits')

mat = scipy.io.loadmat(path.join(DATA_DIR, 'digits.mat'))
np.random.seed(2023)

x = np.loadtxt(path.join(DATA_DIR, 'labels.txt'), dtype=np.int32)
y = np.loadtxt(path.join(DATA_DIR, 'digits.txt'))
x = x.reshape((5000, 1))

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

# Get the mean of the whole dataset
digit_data = data[:, 1:]
mean_data = np.mean(digit_data, axis=0)

# Center the test and train data for PCA
pca_train_data = train_data - mean_data
pca_test_data = test_data - mean_data

pca = PCA()
pca.fit(pca_train_data)

print(pca.components_)
print(pca.explained_variance_)

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

def plot_image(data : np.ndarray):
    # plot the image
    plt.figure()
    plt.axis('off')
    plt.title('The Mean Image From Training Data')
    data = data.reshape((20, 20, -1))
    plt.imshow(data, cmap='gray')
    plt.show()

def plot_image_min_max_scaled(data : np.ndarray):
    # plot the image
    plt.figure()
    plt.axis('off')
    plt.title('The Mean Image From Training Data')
    data = (data - data.min()) / (data.max() - data.min())
    data = data.reshape((20, 20, -1))
    plt.imshow(data, cmap='gray')
    plt.show()

# Plot the mean image before centering
plot_image(train_mean)

# display the first PC
plot_image(pca.components_[0, :])
plot_image(pca.components_[1, :])
plot_image(pca.components_[2, :])

X = np.asarray(pca_train_data).copy()
mean = np.mean(X, axis=0, keepdims=True)

# normalize the data
X = X - mean

#calculate the covariance matrix
covX = (X.T @ X) / X.shape[0]

# calculate the eigenvalues and eigenvectors
eigenVals, eigenVectors = eig(covX)

# sort the eigenvalues in descending order
idx = eigenVals.argsort()[::-1]   
eigenVals = eigenVals[idx]
eigenVectors = eigenVectors[:,idx]

print(eigenVectors)

# the number of components picked
kVals = [10 * k for k in range(1, 21)]
trainAccHistory = []
testAccHistory = []

for k in kVals:
    pcaK = PCA(n_components=k)
    pcaK.fit(pca_train_data)
    train_transformed = pcaK.transform(pca_train_data)
    test_transformed = pcaK.transform(pca_test_data)
    print(train_transformed.shape)

    gaussianK = QuadraticDiscriminantAnalysis()
    gaussianK.fit(train_transformed, train_labels)
    train_preds = gaussianK.predict(train_transformed)
    test_preds = gaussianK.predict(test_transformed)

    trainAccK = accuracy_score(train_labels, train_preds)
    testAccK = accuracy_score(test_labels, test_preds)

    trainAccHistory.append(trainAccK)
    testAccHistory.append(testAccK)

plt.figure(figsize=(18, 12))
plt.xlabel('Number Of First Principal Components')
plt.ylabel('Train Accuracy')
plt.xticks(kVals)
plt.plot(kVals, trainAccHistory, label='Train Accuracy')
plt.legend()
plt.title('Train Accuracy For The Quadratic Gaussian Model')
plt.show()

plt.figure(figsize=(18, 12))
plt.xlabel('Number Of First Principal Components')
plt.ylabel('Test Accuracy')
plt.xticks(kVals)
plt.plot(kVals, testAccHistory, label='Test Accuracy', color='green')
plt.legend()
plt.title('Test Accuracy For The Quadratic Gaussian Model')
plt.show()

# Question 2
trainAccHistory = []
testAccHistory = []
kVals = [10 * k for k in range(1, 21)]

for k in tqdm(kVals):
    iso = Isomap(n_components=k)
    iso.fit(train_data, train_labels)
    iso_transformed_train = iso.transform(train_data)
    iso_transformed_test = iso.transform(test_data)

    gaussianK = QuadraticDiscriminantAnalysis()
    gaussianK.fit(iso_transformed_train, train_labels)
    train_preds = gaussianK.predict(iso_transformed_train)
    test_preds = gaussianK.predict(iso_transformed_test)

    trainAccK = accuracy_score(train_labels, train_preds)
    testAccK = accuracy_score(test_labels, test_preds)  

    trainAccHistory.append(trainAccK)
    testAccHistory.append(testAccK)

plt.figure(figsize=(18, 12))
plt.xlabel('Number Of Dimensions in Isomap')
plt.ylabel('Train Accuracy')
plt.xticks(kVals)
plt.plot(kVals, trainAccHistory, label='Train Accuracy')
plt.legend()
plt.title('Train Accuracy For The Quadratic Gaussian Model')
plt.show()

plt.figure(figsize=(18, 12))
plt.xlabel('Number Of Dimensions in Isomap')
plt.ylabel('Test Accuracy')
plt.xticks(kVals)
plt.plot(kVals, testAccHistory, label='Test Accuracy', color='green')
plt.legend()
plt.title('Test Accuracy For The Quadratic Gaussian Model')
plt.show()