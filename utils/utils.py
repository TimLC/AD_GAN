import cv2
from matplotlib import pyplot
from tensorflow.keras.models import *
import numpy as np
from math import *
import os
import glob
import shutil

from tensorflow.python.keras.backend import clear_session

from gan_model.discrimineur import discriminatorModel
from gan_model.gan import ganModel
from gan_model.generator import generatorModel
from processing.processing_data import generateFakeSample


def generateListSetpsAndEpochs(epochs, initialDropoutDiscriminator, finalDropoutDiscriminator, stepDropoutDiscriminator,
                               rateLastDiscriminator):
    if finalDropoutDiscriminator is not None and stepDropoutDiscriminator is not None and rateLastDiscriminator is not None:
        listNumberEpochsPerDiscriminator = []

        listRangeDropout = [round(elem, 1) for elem in np.arange(initialDropoutDiscriminator,
                                                                 finalDropoutDiscriminator - stepDropoutDiscriminator,
                                                                 -stepDropoutDiscriminator)]

        listNumberEpochsPerDiscriminator.append(round(epochs * rateLastDiscriminator))
        for i in range(len(listRangeDropout) - 1):
            listNumberEpochsPerDiscriminator.append(
                round(epochs * (1 - rateLastDiscriminator) / (len(listRangeDropout) - 1)))
        listNumberEpochsPerDiscriminator[1] = listNumberEpochsPerDiscriminator[-1] + epochs - sum(
            listNumberEpochsPerDiscriminator)
        listNumberEpochsPerDiscriminator.reverse()
    else:
        listRangeDropout = [initialDropoutDiscriminator]
        listNumberEpochsPerDiscriminator = [epochs]
    return list(zip(listRangeDropout, listNumberEpochsPerDiscriminator))


def checkIfFileExist(path, name):
    return os.path.isfile(path + name)


def getNameLastFileInDirectory(path):
    listFiles = glob.glob(path + '*')
    if listFiles:
        return os.path.basename(max(listFiles, key=os.path.getctime))
    else:
        return None


def generateVaraibles(dataset, batchSize):
    numberValue = dataset.shape[0]
    sizeImage = dataset.shape[1:]
    batchPerEpoch = ceil(numberValue / batchSize)
    halfBatch = int(batchSize / 2)

    return sizeImage, batchPerEpoch, halfBatch

def checkStepsDiscriminatorAndGenerator(fileNameDiscriminator,fileNameGenerator):
    if fileNameDiscriminator.split('-')[1] == fileNameGenerator.split('-')[1] and fileNameDiscriminator.split('-')[2].split('.')[0] == fileNameGenerator.split('-')[2].split('.')[0]:
        return True
    else:
        return False

def saveKerasModel(model, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isfile(path + name):
        model.save(path + name, save_format='h5')
    else:
        print('Error : file already exist')


def loadKerasModel(path, name):
    return load_model(path + name)


def saveModel(modelDiscriminator, modelGenerator, modelName, stepDiscriminator, step=None, checkpoints=False, clearSession=False):
    if checkpoints:
        pathFileDiscriminator = './gan/checkpointsDiscriminator/' + modelName + '/'
        pathFileGenerator = './gan/checkpointsGenerator/' + modelName + '/'
        fileNameDiscriminator = 'discriminatorModel-' + str(stepDiscriminator) + '-' + str(step)
        fileNameGenerator = 'generatorModel-' + str(stepDiscriminator) + '-' + str(step)

    else:
        pathFileDiscriminator = './gan/model/' + modelName + '/'
        pathFileGenerator = './gan/model/' + modelName + '/'
        fileNameDiscriminator = 'discriminatorModel-' + str(stepDiscriminator)
        fileNameGenerator = 'generatorModel-' + str(stepDiscriminator)

    saveKerasModel(modelDiscriminator, pathFileDiscriminator, fileNameDiscriminator)
    saveKerasModel(modelGenerator, pathFileGenerator, fileNameGenerator)

    if clearSession:
        clear_session()

    return 0


def checkpointModel(discriminator, generator, path, stepDiscriminator, step, numberStepToSaveModel):
    if step % numberStepToSaveModel == 0:
        saveModel(discriminator, generator, path, stepDiscriminator, step, checkpoints=True)


def loadModel(modelName, lossGan, stepDiscriminator=None, step=None, checkpoints=False):
    if checkpoints:
        pathFileDiscriminator = './gan/checkpointsDiscriminator/' + modelName + '/'
        pathFileGenerator = './gan/checkpointsGenerator/' + modelName + '/'
    else:
        pathFileDiscriminator = './gan/model/' + modelName + '/'
        pathFileGenerator = './gan/model/' + modelName + '/'

    if stepDiscriminator is None and step is None:
        fileNameDiscriminator = getNameLastFileInDirectory(pathFileDiscriminator)
        fileNameGenerator = getNameLastFileInDirectory(pathFileGenerator)
    elif stepDiscriminator is not None and step is None and not checkpoints:
        fileNameDiscriminator = 'discriminatorModel-' + str(stepDiscriminator)
        fileNameGenerator = 'generatorModel-' + str(stepDiscriminator)
    elif stepDiscriminator is not None and step is not None:
        fileNameDiscriminator = 'discriminatorModel-' + str(stepDiscriminator) + '-' + str(step)
        fileNameGenerator = 'generatorModel-' + str(stepDiscriminator) + '-' + str(step)
    else:
        exit()

    modelDiscriminator = loadKerasModel(pathFileDiscriminator, fileNameDiscriminator)
    modelGenerator = loadKerasModel(pathFileGenerator, fileNameGenerator)
    modelGan = ganModel(modelDiscriminator, modelGenerator, lossGan)

    return modelGan, modelDiscriminator, modelGenerator


def loadGenerator(modelName, stepDiscriminator=None, step=None, checkpoints=False):
    if checkpoints:
        pathFileGenerator = './gan/checkpointsGenerator/' + modelName + '/'
    else:
        pathFileGenerator = './gan/model/' + modelName + '/'

    if stepDiscriminator is None and step is None:
        fileNameGenerator = getNameLastFileInDirectory(pathFileGenerator)
    elif stepDiscriminator is not None and step is None and not checkpoints:
        fileNameGenerator = 'generatorModel-' + str(stepDiscriminator)
    elif stepDiscriminator is not None and step is not None:
        fileNameGenerator = 'generatorModel-' + str(stepDiscriminator) + '-' + str(step)
    else:
        exit()

    return loadKerasModel(pathFileGenerator, fileNameGenerator)


def restoreModel(modelName, listDiscriminator, batchPerEpoch,lossGan):
    pathFileDiscriminator = './gan/checkpointsDiscriminator/' + modelName + '/'
    pathFileGenerator = './gan/checkpointsGenerator/' + modelName + '/'

    if os.path.isdir(pathFileDiscriminator) and os.path.isdir(pathFileGenerator) and os.listdir(pathFileDiscriminator) and os.listdir(pathFileGenerator):
        fileNameDiscriminator = getNameLastFileInDirectory(pathFileDiscriminator)
        fileNameGenerator = getNameLastFileInDirectory(pathFileGenerator)
        if checkStepsDiscriminatorAndGenerator(fileNameDiscriminator, fileNameGenerator):
            stepDiscriminator = int(fileNameDiscriminator.split('-')[1])
            stepStart = int(fileNameDiscriminator.split('-')[2].split('.')[0])

            if stepStart < listDiscriminator[stepDiscriminator][1]*batchPerEpoch:
                modelDiscriminator = loadKerasModel(pathFileDiscriminator, fileNameDiscriminator)
                modelGenerator = loadKerasModel(pathFileGenerator, fileNameGenerator)
                modelGan = ganModel(modelDiscriminator, modelGenerator, lossGan)
                hasModelToRestore = True
                stepStart += 1
                return hasModelToRestore, stepDiscriminator, stepStart, listDiscriminator, modelGan, modelDiscriminator, modelGenerator
            else:
                if len(listDiscriminator) - 1 >= stepDiscriminator + 1:
                    stepDiscriminator += 1
                    modelDiscriminator = loadKerasModel(pathFileDiscriminator, fileNameDiscriminator)
                    modelGenerator = loadKerasModel(pathFileGenerator, fileNameGenerator)
                    modelGan = ganModel(modelDiscriminator, modelGenerator, lossGan)
                    hasModelToRestore = True
                    stepStart = 0
                    return hasModelToRestore, stepDiscriminator, stepStart, listDiscriminator, modelGan, modelDiscriminator, modelGenerator

    hasModelToRestore = False
    stepStart = 0
    stepDiscriminator = None
    return hasModelToRestore, stepDiscriminator,stepStart, listDiscriminator, None, None, None


def createModel(metricLossGan, sizeImage, dropoutValue, sizeSeedGenerator, metricLossDiscriminator, addDenseLayersInDiscriminator,
                coefFiltersDiscriminator, addCon2dLayersInGenerator, coefFiltersGenerator):
    modelDiscriminator = discriminatorModel(sizeImage, dropoutValue, metricLossDiscriminator, addDenseLayersInDiscriminator,
                                            coefFiltersDiscriminator)
    modelGenerator = generatorModel(sizeSeedGenerator, sizeImage, addCon2dLayersInGenerator, coefFiltersGenerator)
    modelGan = ganModel(modelDiscriminator, modelGenerator, metricLossGan)
    return modelGan, modelDiscriminator, modelGenerator


def updateDiscriminatorInModel(metricLossGan, modelGenerator, sizeImage, dropoutValue, metricLossDiscriminator, addDenseLayersInDiscriminator,
                coefFiltersDiscriminator):
    modelDiscriminator = discriminatorModel(sizeImage, dropoutValue, metricLossDiscriminator, addDenseLayersInDiscriminator,
                                            coefFiltersDiscriminator)
    modelGan = ganModel(modelDiscriminator, modelGenerator, metricLossGan)
    return modelGan, modelDiscriminator, modelGenerator


def loadImagesDatabase(path, name):
    X_data = []
    files = glob.glob(path + '/*')
    for file in files:
        image = cv2.imread(file)
        X_data.append(image)
    return np.load(path + name)


def generateAndSaveSampleImages(images, modelName=None, stepDiscriminator=None, step=None):
    images = images.numpy() * 127.5 + 127.5
    images = images.astype(int)
    axeNumber = ceil(sqrt(len(images)))
    f, axarr = pyplot.subplots(3, 3, figsize=(images.shape[1], images.shape[1]))

    for i in range(axeNumber):
        for j in range(axeNumber):
            if len(images) > axeNumber*i+j:
                axarr[i, j].imshow(images[3*i+j])
                axarr[i, j].axis('off')
    pyplot.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    if not os.path.exists('./gan/imagesSample/' + modelName):
        os.makedirs('./gan/imagesSample/' + modelName)
    pyplot.savefig('./gan/imagesSample/' + modelName + '/' + str(stepDiscriminator) + '-' + str(step) + '.png', dpi=axeNumber*5)
    pyplot.close()


def getIndexOfElementInListDiscriminator(listDiscriminator, dropoutValue):
    listRangeDropout = list(zip(*listDiscriminator))
    return listRangeDropout[0].index(dropoutValue)


def generateAndSaveFakeSample(path, generator, numberImagesToSave, numberStepToSaveImage, sizeSeedGenerator, stepDiscriminator, step):
    if step % numberStepToSaveImage == 0:
        fakeSample, _ = generateFakeSample(generator, numberImagesToSave, sizeSeedGenerator)
        generateAndSaveSampleImages(fakeSample, path, stepDiscriminator, step)


def logTraningStep(listDiscriminator, dropoutValue, step, totalStep, lossGan, lossDisFake, lossDisReal):
    listRangeDropout = list(zip(*listDiscriminator))
    stepDiscriminator = listRangeDropout[0].index(dropoutValue)

    print('Disciminator : {}/{}, Iteration : {}/{}, Loss GAN : {:.3f}, Loss Discriminator : {:.3f}'
          .format(stepDiscriminator, len(listRangeDropout[0]), step, totalStep, lossGan,
                  (lossDisFake + lossDisReal) / 2))

def deleteFolder(pathFolder):
    if os.path.isdir(pathFolder):
        shutil.rmtree(pathFolder)

def deleteModel(modelName):
    deleteFolder('./gan/checkpointsDiscriminator/' + modelName)
    deleteFolder('./gan/checkpointsGenerator/' + modelName)
    deleteFolder('./gan/imagesSample/' + modelName)
    deleteFolder('./gan/model/' + modelName)

def updatePostion(position, valueToAdd, sizeSeedGenerator):
    newPosition = position + valueToAdd
    if 0 <= newPosition < sizeSeedGenerator:
        return newPosition

    else:
        return position

def updateValue(seedSample, position, valueToAdd):
    newValue = seedSample[0][position] + valueToAdd
    if newValue < 0:
        newValue = 0
    elif newValue > 1:
        newValue = 1
    seedSample[0][position] = newValue
    return seedSample

def displayImage(imageGenerated, seedSample, position):
    text1ToDisplay = 'e : exit   p : new image   q : left   d : right   z : +0,1   s : -0,1'
    text2ToDisplay = 'Position :' + str(position) + '   Value :' + str(round(seedSample[0][position],2))

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText1 = (20, 20)
    bottomLeftCornerOfText2 = (20, 40)
    fontScale = 0.4
    fontColor = 125
    thickness = 1
    lineType = 2

    image = imageGenerated[0].numpy() * 127.5 + 127.5
    imageResize = cv2.resize(image, (500,500), interpolation=cv2.INTER_AREA)

    text = np.zeros((50, 500,imageGenerated.shape[-1])) * 255
    cv2.putText(text, text1ToDisplay, bottomLeftCornerOfText1, font, fontScale, fontColor, thickness, lineType)
    cv2.putText(text, text2ToDisplay, bottomLeftCornerOfText2, font, fontScale, fontColor, thickness, lineType)

    imageAndDescription = np.concatenate((imageResize, text), axis=0)

    cv2.imshow('GAN', imageAndDescription)


def generateVideoWithImages(modelName):
    imagePath = './gan/imagesSample/' + modelName + '/'
    videoPath = './gan/video/' + modelName + '/'
    videoName = 'video-' + modelName + '.avi'

    if not os.path.exists(videoPath):
        os.makedirs(videoPath)

    images = [img for img in os.listdir(imagePath) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(imagePath, images[0]))
    height, width, _ = frame.shape
    fourccMp4v = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(videoPath + videoName, fourccMp4v, 24, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(imagePath, image)))

    cv2.destroyAllWindows()
    video.release()