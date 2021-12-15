from utils.utils import loadImagesDatabase
from keras.datasets import cifar10, mnist
import tensorflow as tf
import numpy as np

def loadDataset(name):
    if name == "MNIST":
        return loadDatasetMnist()
    elif name == "CIFAR10":
        return loadDatasetCifar10()
    else:
        path = 'dataset/'
        return loadImagesDatabase(path, name)


def loadDatasetMnist():
    (trainX, _), (testX, _) = mnist.load_data()
    dataset = np.concatenate((trainX, testX))
    return dataset.reshape(dataset.shape + (1,))

def loadDatasetCifar10(valueFilter = 3):
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    filterTrain = np.where(trainY == valueFilter)
    filterTest = np.where(testY == valueFilter)
    trainX = trainX[filterTrain[0]]
    testX = testX[filterTest[0]]
    dataset = np.concatenate((trainX, testX))
    return dataset


def loadFaceDataset(folder):
    return loadImagesDatabase(folder)