import time
from matplotlib import pyplot
from keras.datasets import cifar10, mnist
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import random
import numpy as np
from math import *
import os
import glob


def loadDatasetCifar10():
    (trainX, trainY), (testX, testY) = cifar10.load_data()

    dataX = np.concatenate((trainX, testX))
    dataY = np.concatenate((trainY, testY))

    dataset = tf.data.Dataset.from_tensor_slices((dataX, dataY))
    dataset = dataset.filter(lambda img, label: label[0] == 8)

    return np.concatenate([[x] for x, y in dataset], axis=0)

def loadDatasetMnist():
    (trainX, _), (testX, _) = mnist.load_data()
    dataset = np.concatenate((trainX, testX))
    return dataset.reshape(dataset.shape + (1,))

def findMaxDivisorBy2(number):
    cptDivisor=0
    number = float(number)
    while number.is_integer():
        number = number/2
        cptDivisor+=1
    return (cptDivisor-1), int(number*2)

def discriminatorModel(sizeImage, lossDis, coefFilters=128):
    numbreLayers, sizeReduce = findMaxDivisorBy2(sizeImage[0])

    model = Sequential()
    for i in range(numbreLayers):
        if i == 0:
            model.add(Conv2D(coefFilters * (i + 1), (3, 3), strides=(2, 2), padding='same', input_shape=sizeImage))
        else:
            model.add(Conv2D(coefFilters * (i + 1), (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(coefFilters, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=lossDis, optimizer='Adam')
    model.summary()

    return model


def generatorModel(sizeSeedGenerator, sizeImage, coefFilters=128):
    numbreLayers, sizeReduce = findMaxDivisorBy2(sizeImage[0])

    model = Sequential()
    model.add(Dense(sizeReduce * sizeReduce * coefFilters * (numbreLayers), input_shape=(sizeSeedGenerator,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((sizeReduce, sizeReduce, coefFilters * (numbreLayers))))
    for i in range(numbreLayers):
        model.add(Conv2DTranspose(coefFilters * (numbreLayers - i), (4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(coefFilters/2, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(sizeImage[-1], (5, 5), activation='tanh', padding='same'))
    model.summary()

    return model

def ganModel(discriminator, generator, lossGan):
    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss=lossGan, optimizer='Adam')

    return model

def generateFakeSample(generator, halfBatch, sizeSeedGenerator):
    fakeSampleX = generator(generateSeedSample(halfBatch, sizeSeedGenerator))
    fakeSampleY = np.zeros((halfBatch, 1))
    return fakeSampleX, fakeSampleY


def generateRealSample(dataset, halfBatch):
    realSample = dataset[np.random.randint(0, len(dataset), halfBatch)]
    realSampleX = (realSample.astype('float32') - 127.5) / 127.5
    realSampleY = np.ones((halfBatch, 1))
    return realSampleX, realSampleY


def generateSeedSample(halfBatch, sizeSeedGenerator):
    return np.random.rand(halfBatch, sizeSeedGenerator)


def displaySampleImages(images, numberImages):
    axeNumber = ceil(sqrt(len(images)))
    sampleImages = random.sample(images, numberImages)
    for i in range(len(sampleImages)):
        pyplot.subplot(axeNumber * 100 + axeNumber * 10 + 1 + i)
        pyplot.imshow(images[i])
    pyplot.show()


def generateAndDisplaySampleImages(images, display=True, save=False, path=None, nameFile=None, step=None):
    images = images.numpy() * 127.5 + 127.5
    images = images.astype(int)
    axeNumber = ceil(sqrt(len(images)))
    for i in range(len(images)):
        pyplot.subplot(axeNumber * 100 + axeNumber * 10 + 1 + i)
        pyplot.imshow(images[i])
    if display:
        pyplot.show()
    if save:
        if not os.path.exists('./imagesSample/' + path):
            os.makedirs('./imagesSample/' + path)
        pyplot.savefig('./imagesSample/' + path + '/' + nameFile + '-' + str(step) + '.png')
        pyplot.close()


def saveModelGan(model, path=None, nameFile=None, step=None):
    if not os.path.exists('./checkpointsGan/' + path):
        os.makedirs('./checkpointsGan/' + path)
    model.save('./checkpointsGan/' + path + '/' + nameFile + '-' + str(step))


def saveModels(modelGan, modelDiscriminater, modelGenerator, path=None, nameFile=None, step=None):
    if not os.path.exists('./checkpointsGan/' + path):
        os.makedirs('./checkpointsGan/' + path)
    modelGan.save('./checkpointsGan/' + path + '/' + nameFile + '-Gan-' + str(step))
    modelDiscriminater.save('./checkpointsDiscriminator/' + path + '/' + nameFile + '-Discriminater-' + str(step))
    modelGenerator.save('./checkpointsGenerator/' + path + '/' + nameFile + '-Generator-' + str(step))


def loadModelGan(path=None, nameFile=None, step=None):
    if nameFile is not None:
        return load_model('./checkpointsGan/' + path + '/' + nameFile + '-Gan-' + str(step))
    else:
        listFiles = glob.glob('./checkpointsGan/' + path + '/*')
        lastFile = max(listFiles, key=os.path.getctime)
        return load_model('./checkpointsGan/' + path + '/' + lastFile)


def loadModels(path=None, nameFile=None):
    listFiles = glob.glob('./checkpointsGan/' + path + '/*')
    lastFile = max(listFiles, key=os.path.getctime)
    step = lastFile.split(nameFile + '-')[-1]

    modelGan = load_model('./checkpointsGan/' + path + '/' + nameFile + '-Gan-' + step)
    modelDiscriminater = load_model('./checkpointsDiscriminator/' + path + '/' + nameFile + '-Discriminater-' + step)
    modelGenerator = load_model('./checkpointsGenerator/' + path + '/' + nameFile + '-Generator-' + step)
    return modelGan, modelDiscriminater, modelGenerator, (int(step) + 1)


if __name__ == '__main__':
    batchSize = 128
    epochs = 200
    sizeSeedGenerator = 100
    stepToSave = 10
    numberImagesToSave = 9
    metricLossDiscriminator = 'binary_crossentropy'
    metricLossGan = 'binary_crossentropy'
    saveModel = False
    saveImage = True
    displayImage = False
    path = 'model3'
    nameFile = 'gan'

    dataset = loadDatasetMnist()

    numberValue = dataset.shape[0]
    sizeImage = dataset.shape[1:]
    batchPerEpoch = ceil(numberValue / batchSize)
    numberStep = batchPerEpoch * epochs
    halfBatch = int(batchSize / 2)
    listLossDis = []
    listLossGan = []

    if os.path.exists('./checkpointsGan/' + path):
        gan, discriminator, generator, initalStep = loadModels(path, nameFile)
    else:
        discriminator = discriminatorModel(sizeImage, metricLossDiscriminator)
        generator = generatorModel(sizeSeedGenerator, sizeImage)
        gan = ganModel(discriminator, generator, metricLossGan)
        initalStep = 0

    for step in range(initalStep, numberStep):
        fakeSampleX, fakeSampleY = generateFakeSample(generator, halfBatch, sizeSeedGenerator)
        realSampleX, realSampleY = generateRealSample(dataset, halfBatch)

        lossDisFake = discriminator.train_on_batch(fakeSampleX, fakeSampleY)
        lossDisReal = discriminator.train_on_batch(realSampleX, realSampleY)
        listLossDis.append((lossDisFake + lossDisReal) / 2)

        seedSampleX = generateSeedSample(halfBatch, sizeSeedGenerator)
        seedSampleY = np.ones((halfBatch, 1))

        lossGan = gan.train_on_batch(seedSampleX, seedSampleY)
        listLossGan.append(lossGan)

        print("Iteration : {}/{}, Loss GAN : {:.3f}, Loss Discriminator : {:.3f}".format(step, numberStep, lossGan,
                                                                                         (lossDisFake + lossDisReal)
                                                                                         /2))

        if step % stepToSave == 0:
            fakeSample, _ = generateFakeSample(generator, numberImagesToSave, sizeSeedGenerator)
            generateAndDisplaySampleImages(fakeSample, displayImage, saveImage, path, nameFile, step)
            if saveModel:
                saveModels(gan, discriminator, generator, path, nameFile, step)
