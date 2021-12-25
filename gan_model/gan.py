from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam

def ganModel(discriminator, generator, lossGan):
    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss=lossGan, optimizer=Adam(lr=0.0001, beta_1=0.5))
    model.summary()

    return model

