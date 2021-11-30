from PIL import Image
from matplotlib import pyplot
from keras.datasets import cifar10, mnist
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.backend import clear_session
import random
import numpy as np
from math import *
import os
import glob
import cv2


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

def loadFaceDataset():
    # folder='dataset/dataset_face_celeba'
    # for i in os.listdir(folder):
    #     file = f"{folder}\\{i}"
    #     im = Image.open(file)
    #     im = im.resize((64, 64))
    #     im.save('dataset/dataset_face_celeba_64x64/'+i)
    #
    # folder='dataset/dataset_face_celeba_64x64'
    # X_data = []
    # files = glob.glob(folder+'/*')
    # for file in files:
    #     image = cv2.imread(file)
    #     X_data.append(image)
    #
    # print('X_data shape:', np.array(X_data).shape)
    #
    # np.save('dataset/dataset_face_celeba', np.array(X_data))

    X_data = np.load('dataset/dataset_face_celeba.npy')
    return X_data


def findMaxDivisorBy2(number, minimalSize):
    cptDivisor=0
    number = float(number)
    while number.is_integer():
        number = number/2
        cptDivisor+=1
        if number < minimalSize:
            break
    return (cptDivisor-1), int(number*2)

def discriminatorModel(sizeImage, dropoutValue, lossDis, addLastLayer=False, coefFilters=128, minimalSize=8):
    numbreLayers, sizeReduce = findMaxDivisorBy2(sizeImage[0], minimalSize)

    model = Sequential()
    for i in range(numbreLayers):
        if i == 0:
            model.add(Conv2D(coefFilters * (i + 1), (3, 3), strides=(2, 2), padding='same', input_shape=sizeImage))
        else:
            model.add(Conv2D(coefFilters * (i + 1), (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropoutValue))
    model.add(Flatten())
    if addLastLayer:
        model.add(Dense(coefFilters, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=lossDis, optimizer='Adam')
    model.summary()

    return model


def generatorModel(sizeSeedGenerator, sizeImage, addLastLayer=False, coefFilters=32, minimalSize=8):
    numbreLayers, sizeReduce = findMaxDivisorBy2(sizeImage[0], minimalSize)

    model = Sequential()
    model.add(Dense(sizeReduce * sizeReduce * coefFilters * (numbreLayers), input_shape=(sizeSeedGenerator,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((sizeReduce, sizeReduce, coefFilters * (numbreLayers))))
    for i in range(numbreLayers):
        model.add(Conv2DTranspose(coefFilters * (numbreLayers - i), (4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
    if addLastLayer:
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


def generateAndDisplaySampleImages(images, display=True, save=False, path=None, nameFile=None, index=None, step=None):
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
        pyplot.savefig('./imagesSample/' + path + '/' + nameFile + '-' + str(index) + '-' + str(step) + '.png')
        pyplot.close()


def saveModelGan(model, path=None, nameFile=None, step=None):
    if not os.path.exists('./checkpointsGan/' + path):
        os.makedirs('./checkpointsGan/' + path)
    model.save('./checkpointsGan/' + path + '/' + nameFile + '-' + str(step))

def saveGeneratorModel(model, path=None, index=None):
    if not os.path.exists('./checkpointsGenerator/' + path):
        os.makedirs('./checkpointsGenerator/' + path)
    model.save('./checkpointsGenerator/' + path + '/generator-' + str(index), save_format='h5')

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

def loadGeneratorModel(path=None):
    listFiles = glob.glob('./checkpointsGenerator/' + path + '/*')
    lastFile = max(listFiles, key=os.path.getctime)
    return load_model(lastFile, compile ="False")


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
    epochsPerDiscriminator = 50
    sizeSeedGenerator = 100
    stepToSave = 10
    numberImagesToSave = 9
    metricLossDiscriminator = 'binary_crossentropy'
    metricLossGan = 'binary_crossentropy'
    saveModel = False
    saveImage = True
    displayImage = False
    path = 'model8'
    nameFile = 'gan'

    defaultValue=0.3

    activeDropoutDiscriminator = True
    initialDropoutDiscriminator = 0.5
    finalDropoutDiscriminator = 0.1
    stepDropoutDiscriminator = 0.1
    multiplyingCoefficientLastDiscriminator = 4

    dataset = loadFaceDataset()

    numberValue = dataset.shape[0]
    sizeImage = dataset.shape[1:]
    batchPerEpoch = ceil(numberValue / batchSize)
    numberStep = batchPerEpoch * epochsPerDiscriminator
    halfBatch = int(batchSize / 2)
    listLossDis = []
    listLossGan = []
    index = 0
    multiplyingCoefficient = 1

    if activeDropoutDiscriminator:
        rangeDiscriminator = [round(elem, 1) for elem in np.arange(initialDropoutDiscriminator,
                                                                   finalDropoutDiscriminator - stepDropoutDiscriminator,
                                                                   -stepDropoutDiscriminator)]
    else:
        rangeDiscriminator =[defaultValue]
        initialDropoutDiscriminator = defaultValue
        finalDropoutDiscriminator = -1

    for dropoutValue in rangeDiscriminator:
        print('Discriminator droupout value : {:.1f}'.format(dropoutValue))
        if dropoutValue == initialDropoutDiscriminator:
            discriminator = discriminatorModel(sizeImage, dropoutValue, metricLossDiscriminator)
            generator = generatorModel(sizeSeedGenerator, sizeImage)
        else:
            discriminator = discriminatorModel(sizeImage, dropoutValue, metricLossDiscriminator)
            generator = loadGeneratorModel(path)
        gan = ganModel(discriminator, generator, metricLossGan)
        if dropoutValue == finalDropoutDiscriminator:
            multiplyingCoefficient = multiplyingCoefficientLastDiscriminator

        for step in range(numberStep*multiplyingCoefficient):
            fakeSampleX, fakeSampleY = generateFakeSample(generator, halfBatch, sizeSeedGenerator)
            realSampleX, realSampleY = generateRealSample(dataset, halfBatch)

            lossDisFake = discriminator.train_on_batch(fakeSampleX, fakeSampleY)
            lossDisReal = discriminator.train_on_batch(realSampleX, realSampleY)
            listLossDis.append((lossDisFake + lossDisReal) / 2)

            seedSampleX = generateSeedSample(halfBatch, sizeSeedGenerator)
            seedSampleY = np.ones((halfBatch, 1))

            lossGan = gan.train_on_batch(seedSampleX, seedSampleY)
            listLossGan.append(lossGan)

            print('Disciminator : {}/{}, Iteration : {}/{}, Loss GAN : {:.3f}, Loss Discriminator : {:.3f}'.format(index, len(rangeDiscriminator),step, numberStep*multiplyingCoefficient, lossGan,
                                                                                             (lossDisFake + lossDisReal)
                                                                                             /2))
            if step % stepToSave == 0:
                fakeSample, _ = generateFakeSample(generator, numberImagesToSave, sizeSeedGenerator)
                generateAndDisplaySampleImages(fakeSample, displayImage, saveImage, path, nameFile, index, step)
        saveGeneratorModel(generator, path, index)
        clear_session()
        index += 1
