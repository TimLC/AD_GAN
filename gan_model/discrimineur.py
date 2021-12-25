from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam

from utils.utils_model import findMaxDivisorBy2


def discriminatorModel(sizeImage, dropoutValue, lossDiscriminator, addDenseLayers=0, coefFilters=64):
    minimalSize = int(round(sizeImage[0]/8))
    numberLayers, sizeReduce = findMaxDivisorBy2(sizeImage[0], minimalSize)

    model = Sequential()
    model.add(Input(shape=sizeImage))
    model.add(Conv2D(coefFilters, (3, 3), padding='same', activation=LeakyReLU(alpha=0.2)))
    for i in range(numberLayers):
        numberFilters = coefFilters * (i + 2)
        model.add(Conv2D(numberFilters, (3, 3), strides=(2, 2), padding='same', activation=LeakyReLU(alpha=0.2)))
        model.add(LeakyReLU(alpha=0.2))
        if dropoutValue > 0:
            model.add(Dropout(dropoutValue))
    model.add(Flatten())

    if addDenseLayers > 0:
        for i in range(addDenseLayers, 0, -1):
            numberNeural = 16 * i
            model.add(Dense(numberNeural, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=lossDiscriminator, optimizer=Adam(lr=0.0001, beta_1=0.5))
    model.summary()

    return model

# def discriminatorModel(sizeImage, dropoutValue, lossDiscriminator, addDenseLayers=0, coefFilters=64):
# 	in_shape=(32,32,3)
# 	model = Sequential()
# 	# normal
# 	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
# 	model.add(LeakyReLU(alpha=0.2))
# 	# downsample
# 	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
# 	model.add(LeakyReLU(alpha=0.2))
# 	# downsample
# 	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
# 	model.add(LeakyReLU(alpha=0.2))
# 	# downsample
# 	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
# 	model.add(LeakyReLU(alpha=0.2))
# 	# classifier
# 	model.add(Flatten())
# 	model.add(Dropout(0.4))
# 	model.add(Dense(1, activation='sigmoid'))
# 	# compile model
# 	opt = Adam(lr=0.0002, beta_1=0.5)
# 	model.compile(loss='binary_crossentropy', optimizer=opt)
# 	return model