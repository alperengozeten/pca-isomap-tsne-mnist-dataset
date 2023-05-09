import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

from os import path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.manifold import Isomap, TSNE
from numpy.linalg import eig
from tqdm import tqdm

# get the current working directory
ROOT_DIR = path.abspath(os.curdir)
DATA_DIR = path.join(ROOT_DIR, 'digits')

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
centered_train_data = train_data - mean_data
centered_test_data = test_data - mean_data

pca = PCA()
pca.fit(centered_train_data)

# returns the index where the cumulative explained variance exceeds f, which is between 0 and 1
def cumulative_explained_variance(eigenVals, f : float):
    if f < 0 or f > 1:
        print('f is out of the interval [0, 1]')
        return -1

    # get the cumulative ratios
    cum_ratios = np.cumsum(eigenVals) / np.sum(eigenVals)
    return (np.searchsorted(cum_ratios, f) + 1) # +1 since this function returns the index

# Capture %80 variance with heuristics
threshold_index = cumulative_explained_variance(pca.explained_variance_, 0.8)
print('The minimum number of principal components to ensure %80 variance: ' + str(threshold_index)) # the output is 39

plt.figure()
plt.plot(pca.explained_variance_, label='eigenvalues')
plt.xlabel('Index Of The Eigenvalue')
plt.ylabel('Magnitude Of The Eigenvalue')
plt.legend()
plt.fill_between([x for x in range(threshold_index)], pca.explained_variance_[0:threshold_index], step="pre", alpha=0.4)
plt.title('Eigenvalues In Descending Order')
plt.show()

# Question 1.2
train_mean = np.mean(train_data, axis=0)

def plot_image(data : np.ndarray):
    # plot the image
    plt.figure()
    plt.axis('off')
    plt.title('The Mean Image From Training Data')
    data = data.reshape((20, 20)).T
    plt.imshow(data, cmap='gray')
    plt.show()

def plot_image_min_max_scaled(data : np.ndarray):
    # plot the image
    plt.figure()
    plt.axis('off')
    plt.title('The Mean Image From Training Data')
    data = (data - data.min()) / (data.max() - data.min())
    data = data.reshape((20, 20)).T
    plt.imshow(data, cmap='gray')
    plt.show()

# Plot the mean image before centering
plot_image(train_mean)

# display the first 40 PC's since it captures %80 variance
fig, ax = plt.subplots(nrows=5, ncols=8, figsize=(18, 12))
for index in range(40):
    row = index // 8
    col = index % 8
    ax[row, col].set_axis_off()
    component = pca.components_[index, :].reshape((20, 20)).T
    ax[row, col].imshow(component, cmap='gray')
plt.suptitle('The First 40 Principal Components')
plt.show()

X = np.asarray(centered_train_data).copy()
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
trainErrHistory = []
testErrHistory = []

for k in tqdm(kVals):
    pcaK = PCA(n_components=k)
    pcaK.fit(centered_train_data)
    train_transformed = pcaK.transform(centered_train_data)
    test_transformed = pcaK.transform(centered_test_data)

    gaussianK = QuadraticDiscriminantAnalysis()
    gaussianK.fit(train_transformed, train_labels)
    train_preds = gaussianK.predict(train_transformed)
    test_preds = gaussianK.predict(test_transformed)

    trainErrK = 1 - accuracy_score(train_labels, train_preds)
    testErrK = 1 - accuracy_score(test_labels, test_preds)

    trainErrHistory.append(trainErrK)
    testErrHistory.append(testErrK)

plt.figure(figsize=(18, 12))
plt.xlabel('Number Of First Principal Components')
plt.ylabel('Train Classification Error')
plt.xticks(kVals)
plt.plot(kVals, trainErrHistory, label='Train Classification Error')
plt.legend()
plt.title('Train Classification Error For The Quadratic Gaussian Model')
plt.show()

plt.figure(figsize=(18, 12))
plt.xlabel('Number Of First Principal Components')
plt.ylabel('Test Classification Error')
plt.xticks(kVals)
plt.plot(kVals, testErrHistory, label='Test Classification Error', color='green')
plt.legend()
plt.title('Test Classification Error For The Quadratic Gaussian Model')
plt.show()

# Question 2
trainErrHistory = []
testErrHistory = []
kVals = [10 * k for k in range(1, 21)]

# get the centered full data
centered_full_data = digit_data - mean_data

for k in tqdm(kVals):
    iso = Isomap(n_components=k) # n_neighbors default which is 5
    iso.fit(centered_full_data, x)
    iso_transformed_train = iso.transform(centered_train_data)
    iso_transformed_test = iso.transform(centered_test_data)

    gaussianK = QuadraticDiscriminantAnalysis()
    gaussianK.fit(iso_transformed_train, train_labels)
    train_preds = gaussianK.predict(iso_transformed_train)
    test_preds = gaussianK.predict(iso_transformed_test)

    trainErrK = 1 - accuracy_score(train_labels, train_preds)
    testErrK = 1 - accuracy_score(test_labels, test_preds)  

    trainErrHistory.append(trainErrK)
    testErrHistory.append(testErrK)

plt.figure(figsize=(18, 12))
plt.xlabel('Number Of Dimensions in Isomap')
plt.ylabel('Train Classification Error')
plt.xticks(kVals)
plt.plot(kVals, trainErrHistory, label='Train Classification Error')
plt.legend()
plt.title('Train Classification Error For The Quadratic Gaussian Model')
plt.show()

plt.figure(figsize=(18, 12))
plt.xlabel('Number Of Dimensions in Isomap')
plt.ylabel('Test Classification Error')
plt.xticks(kVals)
plt.plot(kVals, testErrHistory, label='Test Classification Error', color='green')
plt.legend()
plt.title('Test Classification Error For The Quadratic Gaussian Model')
plt.show()

tsne = TSNE(n_components=2, verbose=1)

embedded_full_data = tsne.fit_transform(centered_full_data)

df = pd.DataFrame()
df['first-dim'] = embedded_full_data[:,0]
df['second-dim'] = embedded_full_data[:,1]
df['labels'] = x # set to initial labels

plt.figure(figsize=(16,10))
seaborn.scatterplot(
    x="first-dim", y="second-dim",
    hue="labels",
    palette=seaborn.color_palette("hls", 10),
    data=df,
    alpha=0.3,
    legend="full"
)
plt.title('Digit Data Mapped to 2D')
plt.show()