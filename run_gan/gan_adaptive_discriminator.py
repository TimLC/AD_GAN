import numpy as np

from dataset.dataset import loadDataset
from processing.processing_data import generateFakeSample, generateRealSample, generateSeedSample
from utils.utils import generateListSetpsAndEpochs, createModel, logTraningStep, saveGenerator, \
    generateAndSaveFakeSample, generateVaraibles, checkpointModel, initCheckpointModel, checkpointModelLoad, \
    updateDiscriminatorInModel


def adGan(modelName, imagesPath, batchSize=128, epochs=100, sizeSeedGenerator=100, addDenseLayersInDiscriminator=2, coefFiltersDiscriminator=64,
          addCon2dLayersInGenerator=2, coefFiltersGenerator=128, numberStepToSaveImage=10, numberStepToSaveModel=50, numberImagesToSave=9,
          initialDropoutDiscriminator=0.5, finalDropoutDiscriminator=0.0, stepDropoutDiscriminator=0.1, rateLastDiscriminator=0.75,
          metricLossDiscriminator='binary_crossentropy', metricLossGan='binary_crossentropy'):

    listLossGan = []

    dataset = loadDataset(imagesPath)
    sizeImage, batchPerEpoch, halfBatch = generateVaraibles(dataset, batchSize)
    listDiscriminator = generateListSetpsAndEpochs(epochs, initialDropoutDiscriminator, finalDropoutDiscriminator, stepDropoutDiscriminator, rateLastDiscriminator)
    gan, discriminator, generator = createModel(metricLossGan, sizeImage, initialDropoutDiscriminator, sizeSeedGenerator, metricLossDiscriminator, addDenseLayersInDiscriminator, coefFiltersDiscriminator, addCon2dLayersInGenerator, coefFiltersGenerator)
    checkpoint, checkpointManager = initCheckpointModel(gan, discriminator, generator, modelName)
    stepDiscriminatorRestore, stepStart = checkpointModelLoad(checkpoint, checkpointManager, modelName, listDiscriminator, batchPerEpoch)

    for stepDiscriminator, elementDiscriminator in enumerate(listDiscriminator):
        if stepDiscriminatorRestore is not None and stepDiscriminator < stepDiscriminatorRestore:
            continue
        dropoutValue = elementDiscriminator[0]
        epochsPerDiscriminator = elementDiscriminator[1]

        if not dropoutValue == initialDropoutDiscriminator:
            gan, discriminator, generator = updateDiscriminatorInModel(metricLossGan, generator, sizeImage, dropoutValue, metricLossDiscriminator,
                                                                       addDenseLayersInDiscriminator, coefFiltersDiscriminator)

        for step in range(stepStart, epochsPerDiscriminator * batchPerEpoch):
            fakeSampleX, fakeSampleY = generateFakeSample(generator, halfBatch, sizeSeedGenerator)
            realSampleX, realSampleY = generateRealSample(dataset, halfBatch)

            lossDisFake = discriminator.train_on_batch(fakeSampleX, fakeSampleY)
            lossDisReal = discriminator.train_on_batch(realSampleX, realSampleY)

            seedSampleX = generateSeedSample(halfBatch, sizeSeedGenerator)
            seedSampleY = np.ones((halfBatch, 1))

            lossGan = gan.train_on_batch(seedSampleX, seedSampleY)
            listLossGan.append(lossGan)

            logTraningStep(listDiscriminator, dropoutValue, step, epochsPerDiscriminator * batchPerEpoch, lossGan, lossDisFake, lossDisReal)
            checkpointModel(generator, modelName, stepDiscriminator, numberStepToSaveModel, checkpointManager, step, epochsPerDiscriminator * batchPerEpoch, finalDropoutDiscriminator, dropoutValue)
            generateAndSaveFakeSample(modelName, generator, numberImagesToSave, numberStepToSaveImage, sizeSeedGenerator, stepDiscriminator, step)

        stepStart = saveGenerator(generator, modelName, stepDiscriminator, clearSession=True)

