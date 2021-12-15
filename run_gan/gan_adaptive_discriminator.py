import numpy as np

from dataset.dataset import loadDataset
from processing.processing_data import generateFakeSample, generateRealSample, generateSeedSample
from utils.utils import generateListSetpsAndEpochs, createModel, updateDiscriminatorInModel, logTraningStep, \
    saveModel, generateAndSaveFakeSample, generateVaraibles, restoreModel, checkpointModel

def adGan(modelName, imagesPath, batchSize=128, epochs=100, sizeSeedGenerator=100, addDenseLayersInDiscriminator=2, coefFiltersDiscriminator=128,
          addCon2dLayersInGenerator=2, coefFiltersGenerator=32, numberStepToSaveImage=10, numberStepToSaveModel=50, numberImagesToSave=9,
          initialDropoutDiscriminator=0.5, finalDropoutDiscriminator=0.0, stepDropoutDiscriminator=0.1, rateLastDiscriminator=0.75,
          metricLossDiscriminator='binary_crossentropy', metricLossGan='binary_crossentropy'):


    initialDropoutDiscriminator = 0.5
    finalDropoutDiscriminator = 0.0
    stepDropoutDiscriminator = 0.1
    rateLastDiscriminator = 0.75

    listLossGan = []

    dataset = loadDataset(imagesPath)
    sizeImage, batchPerEpoch, halfBatch = generateVaraibles(dataset, batchSize)

    listDiscriminator = generateListSetpsAndEpochs(epochs, initialDropoutDiscriminator, finalDropoutDiscriminator, stepDropoutDiscriminator, rateLastDiscriminator)

    hasModelToRestore, stepDiscriminatorRestore, stepStart, listDiscriminator, gan, discriminator, generator = restoreModel(modelName, listDiscriminator, batchPerEpoch, metricLossGan)

    for stepDiscriminator, elementDiscriminator in enumerate(listDiscriminator):
        if stepDiscriminatorRestore is not None and stepDiscriminator < stepDiscriminatorRestore:
            continue
        dropoutValue = elementDiscriminator[0]
        epochsPerDiscriminator = elementDiscriminator[1]

        if not hasModelToRestore:
            if dropoutValue == initialDropoutDiscriminator:
                gan, discriminator, generator = createModel(metricLossGan, sizeImage, dropoutValue, sizeSeedGenerator, metricLossDiscriminator, addDenseLayersInDiscriminator,
                    coefFiltersDiscriminator, addCon2dLayersInGenerator, coefFiltersGenerator)
            else:
                gan, discriminator, generator = updateDiscriminatorInModel(metricLossGan, generator, sizeImage, dropoutValue, metricLossDiscriminator, addDenseLayersInDiscriminator,
                    coefFiltersDiscriminator)
        else:
            hasModelToRestore = False

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
            checkpointModel(discriminator, generator, modelName, stepDiscriminator, step, numberStepToSaveModel)
            generateAndSaveFakeSample(modelName, generator, numberImagesToSave, numberStepToSaveImage, sizeSeedGenerator, stepDiscriminator, step)

        stepStart = saveModel(discriminator, generator, modelName, stepDiscriminator, clearSession=True)

