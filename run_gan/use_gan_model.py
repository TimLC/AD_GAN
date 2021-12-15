from processing.processing_data import generateSeedSample
from utils.utils import loadGenerator, displayImage, updatePostion, updateValue
import cv2

def useGanModel(modelName, stepDiscriminator, step):

    if step is None:
        generator = loadGenerator(modelName, stepDiscriminator=stepDiscriminator, step=step, checkpoints=False)
    else:
        generator = loadGenerator(modelName, stepDiscriminator=stepDiscriminator, step=step, checkpoints=True)

    sizeSeedGenerator = generator.input_shape[1]
    seedSample = generateSeedSample(1,sizeSeedGenerator)
    position = round(len(seedSample[0])/2)
    image = generator(seedSample)
    display = True

    while True:
        key = cv2.waitKey(1)
        if display:
            displayImage(image, seedSample, position)
            display = False
        if key == ord('e'):
            exit()
            cv2.destroyAllWindows()
        if key == ord('p'):
            seedSample = generateSeedSample(1, sizeSeedGenerator)
            position = round(len(seedSample[0]) / 2)
            image = generator(seedSample)
            display = True
        if key == ord('d'):
            position = updatePostion(position, 1, sizeSeedGenerator)
            display = True
        if key == ord('q'):
            position = updatePostion(position, -1, sizeSeedGenerator)
            display = True
        if key == ord('z'):
            seedSample = updateValue(seedSample, position, 0.1)
            image = generator(seedSample)
            display = True
        if key == ord('s'):
            seedSample = updateValue(seedSample, position, -0.1)
            image = generator(seedSample)
            display = True
