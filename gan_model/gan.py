from tensorflow.keras.models import *

def ganModel(discriminator, generator, lossGan):
    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss=lossGan, optimizer='Adam')

    return model