from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import argparse

from keras.models import load_model
from keras.utils import CustomObjectScope
from sklearn.model_selection import train_test_split
from scipy.misc import imresize
from scipy.ndimage import imread

from data import seed, standardize
from loss import np_dice_coef
from nets.MobileUNet import custom_objects
import re

SAVED_MODEL1 = 'artifacts/model.h5'

img_size = 128


def main(img_file):
    with CustomObjectScope(custom_objects()):
        model1 = load_model(SAVED_MODEL1)

    img = imread(img_file)
    img = imresize(img, (img_size, img_size))

    mask_file = re.sub('img2.jpg$', 'msk1.png',img_file)

    mask = imread(mask_file)
    mask = imresize(mask, (img_size, img_size))

    batched1 = img.reshape(1, img_size, img_size, 3).astype(float)
    pred1 = model1.predict(standardize(batched1)).reshape(img_size, img_size)

    if True:
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.subplot(2, 2, 2)
        plt.imshow(pred1)
        plt.subplot(2, 2, 3)
        plt.imshow(mask)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_file',
        type=str,
        default='data/test.jpg',
    )

    args, _ = parser.parse_known_args()
    main(**vars(args))
