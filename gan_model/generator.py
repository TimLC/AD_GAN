from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import numpy as np

from utils.utils_model import findMaxDivisorBy2

def generatorModel(sizeSeedGenerator, sizeImage, addCon2dLayers=0, coefFilters=128):
    minimalSize = int(round(sizeImage[0] / 8))
    numberLayers, sizeReduce = findMaxDivisorBy2(sizeImage[0], minimalSize)

    model = Sequential()
    model.add(Dense(sizeReduce * sizeReduce * coefFilters, activation=LeakyReLU(alpha=0.2),
                    input_shape=(sizeSeedGenerator,)))
    model.add(BatchNormalization())
    model.add(Reshape((sizeReduce, sizeReduce, coefFilters)))
    for i in range(numberLayers, 0, -1):
        numberFilters = coefFilters * i
        model.add(Conv2DTranspose(numberFilters, (3, 3), strides=(2, 2), activation=LeakyReLU(alpha=0.2), padding='same'))
        model.add(BatchNormalization())

    if addCon2dLayers > 0:
        for i in range(addCon2dLayers):
            numberFilters = int(np.exp(-i) * coefFilters)
            if numberFilters < sizeImage[-1]:
                numberFilters = sizeImage[-1]
            model.add(Conv2DTranspose(numberFilters, (3, 3), strides=(1, 1), activation=LeakyReLU(alpha=0.2), padding='same'))
            model.add(BatchNormalization())
    model.add(Conv2D(sizeImage[-1], (3, 3), activation='tanh', padding='same'))
    model.summary()

    return model
