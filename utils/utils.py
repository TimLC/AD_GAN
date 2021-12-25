import gc

import cv2
import matplotlib
from matplotlib import pyplot
from tensorflow.keras.models import *
import numpy as np
from math import *
import os
import glob
import shutil

from tensorflow.python.keras.backend import clear_session
from tensorflow.python.training.checkpoint_management import CheckpointManager
from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow.keras.optimizers import Adam

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


def initCheckpointModel(modelGan, modelDiscriminator, modelGenerator, modelName):
    maxToKeep = 10
    pathCheckpoints = './gan/checkpoints/' + modelName + '/'
    if not os.path.exists(pathCheckpoints):
        os.makedirs(pathCheckpoints)

    checkpoint = Checkpoint(gan=modelGan,
                            generator=modelGenerator,
                            discriminator=modelDiscriminator)
    checkpointManager = CheckpointManager(checkpoint, pathCheckpoints, max_to_keep=maxToKeep)

    return checkpoint, checkpointManager


def checkpointModelSave(checkpointManager, modelName, stepDiscriminator, step):
    maxToKeep = 10
    pathCheckpoints = './gan/checkpoints/' + modelName + '/'
    checkpointManager.save()
    if checkIfFileExist(pathCheckpoints, 'step'):
        with open(pathCheckpoints + 'step', 'r') as stepFile:
            listRows = stepFile.readlines()
        with open(pathCheckpoints + 'step', 'w') as stepFile:
            if len(listRows) < maxToKeep:
                stepFile.writelines(listRows)
            elif len(listRows) >= maxToKeep:
                stepFile.writelines(listRows[-(maxToKeep-1):])
            stepFile.write('\n' + str(stepDiscriminator) + '-' + str(step))
    else:
        with open(pathCheckpoints + 'step', 'w') as stepFile:
            stepFile.write(str(stepDiscriminator) + '-' + str(step))


def checkpointModelLoad(checkpoint, checkpointManager, modelName, listDiscriminator, batchPerEpoch):
    pathCheckpoints = './gan/checkpoints/' + modelName + '/'
    if checkpointManager.latest_checkpoint:
        print(checkpointManager.latest_checkpoint)
        checkpoint.restore(checkpointManager.latest_checkpoint)
    if checkIfFileExist(pathCheckpoints, 'step'):
        with open(pathCheckpoints + 'step', 'r+') as stepFile:
            listRows = stepFile.readlines()
            stepInformations = listRows[-1].split('-')
            stepDiscriminatorRestore = int(stepInformations[0])
            stepStart = int(stepInformations[1])
            if stepStart < listDiscriminator[stepDiscriminatorRestore][1] * batchPerEpoch:
                stepStart += 1
            else:
                if len(listDiscriminator) - 1 >= stepDiscriminatorRestore + 1:
                    stepDiscriminatorRestore += 1
                    stepStart = 0
    else:
        stepStart = 0
        stepDiscriminatorRestore = None

    return stepDiscriminatorRestore, stepStart


def checkpointModel(generator, modelName, stepDiscriminator, numberStepToSaveModel, checkpointManager, step, totalStep, finalDropoutDiscriminator, dropoutValue):
    if step % numberStepToSaveModel == 0:
        checkpointModelSave(checkpointManager, modelName, stepDiscriminator, step)
        if finalDropoutDiscriminator == dropoutValue:
            if step > int(totalStep*0.75):
                saveGenerator(generator, modelName, stepDiscriminator, step, checkpoints=True)


def saveGenerator(modelGenerator, modelName, stepDiscriminator, step=None, checkpoints=False, clearSession=False):
    if checkpoints:
        pathFileGenerator = './gan/checkpointsGenerator/' + modelName + '/'
        fileNameGenerator = 'generatorModel-' + str(stepDiscriminator) + '-' + str(step)

    else:
        pathFileGenerator = './gan/model/' + modelName + '/'
        fileNameGenerator = 'generatorModel-' + str(stepDiscriminator)

    saveKerasModel(modelGenerator, pathFileGenerator, fileNameGenerator)

    if clearSession:
        clear_session()

    return 0


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
    matplotlib.use('Agg')
    images = images * 127.5 + 127.5
    images = images.astype(int)
    axeNumber = ceil(sqrt(len(images)))

    fig, axarr = pyplot.subplots(axeNumber, axeNumber, figsize=(images.shape[1], images.shape[1]))

    for i in range(axeNumber):
        for j in range(axeNumber):
            if len(images) > axeNumber*i+j:
                axarr[i, j].imshow(images[axeNumber*i+j])
                axarr[i, j].axis('off')
    pyplot.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    if not os.path.exists('./gan/imagesSample/' + modelName):
        os.makedirs('./gan/imagesSample/' + modelName)
    pyplot.savefig('./gan/imagesSample/' + modelName + '/' + str(stepDiscriminator) + '-' + str(step) + '.png', dpi=axeNumber*5)

    pyplot.close('all')


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
    deleteFolder('./gan/checkpoints/' + modelName)
    deleteFolder('./gan/checkpointsGenerator/' + modelName)
    deleteFolder('./gan/imagesSample/' + modelName)
    deleteFolder('./gan/model/' + modelName)
    deleteFolder('./gan/video/' + modelName)


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

    image = imageGenerated[0] * 127.5 + 127.5
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