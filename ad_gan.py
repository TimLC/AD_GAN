import argparse
import sys
import tensorflow as tf

from run_gan.gan_adaptive_discriminator import adGan
from run_gan.use_gan_model import useGanModel
from utils.utils import deleteModel, generateVideoWithImages


def trainAdGan():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', type=str)
    parser.add_argument('--imagesPath', type=str)
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--sizeSeedGenerator', type=int, default=100)

    parser.add_argument('--addDenseLayersInDiscriminator', type=int, default=2)
    parser.add_argument('--coefFiltersDiscriminator', type=int, default=64)
    parser.add_argument('--addCon2dLayersInGenerator', type=int, default=2)
    parser.add_argument('--coefFiltersGenerator', type=int, default=128)

    parser.add_argument('--numberStepToSaveImage', type=int, default=10)
    parser.add_argument('--numberStepToSaveModel', type=int, default=50)
    parser.add_argument('--numberImagesToSave', type=int, default=9)

    parser.add_argument('--initialDropoutDiscriminator', type=float, default=0.5)
    parser.add_argument('--finalDropoutDiscriminator', type=float, default=0.0)
    parser.add_argument('--stepDropoutDiscriminator', type=float, default=0.1)
    parser.add_argument('--rateLastDiscriminator', type=float, default=0.75, choices=[i/100 for i in range(0, 100)])

    parser.add_argument('--metricLossDiscriminator', type=str, default='binary_crossentropy')
    parser.add_argument('--metricLossGan', type=str, default='binary_crossentropy')

    args, _ = parser.parse_known_args()
    adGan(**vars(args))


def useAdGan():
    print(argparse.ArgumentParser())
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', type=str)
    parser.add_argument('--stepDiscriminator', type=int)
    parser.add_argument('--step', type=int, default=None)

    args, _ = parser.parse_known_args()
    useGanModel(**vars(args))
    print('end run')


def deleteModelGenerated():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', type=str)

    args, _ = parser.parse_known_args()
    deleteModel(**vars(args))

def generateVideo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', type=str)

    args, _ = parser.parse_known_args()
    generateVideoWithImages(**vars(args))


def getActionArg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str)
    action_arg, _ = parser.parse_known_args()

    return action_arg


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')

    if len(sys.argv) > 2:
        arg = getActionArg()
        globals()[arg.action]()
    else:
        exit()
