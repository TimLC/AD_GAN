from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from utils.utils_model import findMaxDivisorBy2


def discriminatorModel(sizeImage, dropoutValue, lossDiscriminator, addDenseLayers=None, coefFilters=128):
    minimalSize = int(round(sizeImage[0]/8))
    numberLayers, sizeReduce = findMaxDivisorBy2(sizeImage[0], minimalSize)

    model = Sequential()
    model.add(Input(shape=sizeImage))
    for i in range(numberLayers):
        numberFilters = coefFilters * (i + 1)
        model.add(Conv2D(numberFilters, (3, 3), strides=(2, 2), padding='same', activation=LeakyReLU(alpha=0.2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropoutValue))
    model.add(Flatten())

    if addDenseLayers is not None:
        for i in range(addDenseLayers, 0, -1):
            numberNeural = 16 * i
            model.add(Dense(numberNeural, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=lossDiscriminator, optimizer='Adam')
    # model.summary()

    return model

