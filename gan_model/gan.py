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


# def ganModel(discriminator, generator, lossGan):
# 	# make weights in the discriminator not trainable
# 	discriminator.trainable = False
# 	# connect them
# 	model = Sequential()
# 	# add generator
# 	model.add(generator)
# 	# add the discriminator
# 	model.add(discriminator)
# 	# compile model
# 	opt = Adam(lr=0.0002, beta_1=0.5)
# 	model.compile(loss='binary_crossentropy', optimizer=opt)
# 	return model