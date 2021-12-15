import numpy as np

def generateFakeSample(generator, batchSize, sizeSeedGenerator):
    fakeSampleX = generator(generateSeedSample(batchSize, sizeSeedGenerator))
    fakeSampleY = np.zeros((batchSize, 1))
    return fakeSampleX, fakeSampleY

def generateRealSample(dataset, batchSize):
    realSample = dataset[np.random.randint(0, len(dataset), batchSize)]
    realSampleX = (realSample.astype('float32') - 127.5) / 127.5
    realSampleY = np.ones((batchSize, 1))
    return realSampleX, realSampleY

def generateSeedSample(batchSize, sizeSeedGenerator):
    return np.random.rand(batchSize, sizeSeedGenerator)